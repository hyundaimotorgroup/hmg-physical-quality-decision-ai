"""CLI: run weld detection + quality judgment on a video, with cross-frame
tracking so the same physical weld across many sampled frames collapses into
one stable verdict instead of being counted (and flicker-flipped) per frame.

Every sampled frame is judged by the UNCHANGED photo pipeline
(pqdai.core.pipeline.QualityPipeline.judge_image -- tiled detection + WBF +
ROI-redetection + gap-bridging + NMS-dedup cascade, exactly as for a photo).
Only the cross-frame identity (video_tracker.py) and the track-level
aggregation (track_aggregation.py) are new. See docs/architecture.md for the
design this implements.

Usage:
  pqdai video --video <path> --out runs/<run_name> [--domain weld] [--max-seconds 30]
  python -m pqdai.video.run --video <path> --out runs/<run_name>
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from ..core.runtime import build_pipeline
from .video_chunking import probe_video, n_chunks_for, iter_sampled_frames
from .video_tracker import Tracker
from .camera_motion import EgoMotionEstimator
from .track_aggregation import compute_track_verdict, compute_video_verdict
from .bead_clustering import cluster_tracks_into_beads
from .video_visualize import draw_frame_overlay, draw_live_preview_overlay, generate_html_report, build_overlay_video
from .dedup_audit import generate_dedup_audit



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--out', default=None, help='Output dir; default runs/<video_stem>')
    parser.add_argument('--sample-fps', type=float, default=5.0)
    parser.add_argument('--chunk-duration', type=float, default=20.0, help='seconds; reporting/traceability grouping only, see video_chunking.py')
    parser.add_argument('--lost-grace-sec', type=float, default=1.5, help='seconds a track may go unmatched before being finalized')
    parser.add_argument('--conf', type=float, default=0.1, help='stage-1 detector confidence (extract-all)')
    parser.add_argument('--verify-conf', type=float, default=0.2, help='ROI re-detection confirmation threshold')
    parser.add_argument('--alpha', type=float, default=0.1, help='entropy gate for retrieval, same default as photo pipeline')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--domain', default='weld', help='domain plugin (see `pqdai domains`)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--reference-bank', default=None, help='default: the domain\'s bundled bank')
    parser.add_argument('--detector-weights', default=None, help='default: the domain\'s released weights')
    parser.add_argument('--classifier-weights', default=None)
    parser.add_argument('--max-seconds', type=float, default=None, help='Smoke-test: only process N seconds of the video (from --start-seconds)')
    parser.add_argument('--start-seconds', type=float, default=0.0, help='Smoke-test: skip to this timestamp before processing (see video_chunking.py)')
    parser.add_argument('--no-overlays', action='store_true', help='Skip writing per-frame overlay JPGs (faster, less disk)')
    parser.add_argument('--no-bead-clustering', action='store_true',
                         help='Report raw per-segment tracks instead of merging same-bead identity-handoff segments '
                              '(see bead_clustering.py). Clustering is ON by default; disable to compare against '
                              'the unclustered per-segment track list.')
    parser.add_argument('--progress-every', type=int, default=20, help='Print progress every N sampled frames')
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error('--progress-every must be >= 1')

    video_path = str(Path(args.video).resolve())
    out_dir = Path(args.out) if args.out else Path('runs') / Path(video_path).stem
    frames_dir = out_dir / 'frames'
    overlays_dir = out_dir / 'overlays'
    live_preview_dir = out_dir / 'live_preview'
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)
        live_preview_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(video_path)
    remaining_sec = max(0.0, meta.duration_sec - args.start_seconds)
    processed_sec = min(remaining_sec, args.max_seconds) if args.max_seconds else remaining_sec
    n_chunks = n_chunks_for(processed_sec, args.chunk_duration)
    est_sampled_frames = int(processed_sec * args.sample_fps)

    print("=" * 70)
    print(f"video:            {video_path}")
    print(f"resolution/fps:   {meta.width}x{meta.height} @ {meta.fps:.2f}fps, {meta.total_frames} frames, {meta.duration_sec:.1f}s")
    print(f"sample_fps:       {args.sample_fps} (~{est_sampled_frames} frames will be processed)")
    print(f"chunk_duration:   {args.chunk_duration}s (~{n_chunks} chunks)")
    print(f"detector conf:    conf={args.conf} verify_conf={args.verify_conf}")
    print(f"lost_grace:       {args.lost_grace_sec}s (~{round(args.lost_grace_sec * args.sample_fps)} sampled frames)")
    if args.start_seconds:
        print(f"start_seconds:    {args.start_seconds} (SMOKE TEST window, not from the video's start)")
    if args.max_seconds:
        print(f"max_seconds:      {args.max_seconds} (SMOKE TEST -- not the full video)")
    print(f"out_dir:          {out_dir}")
    print("=" * 70)

    print("Loading models...")
    pipeline = build_pipeline(
        args.domain, args.device, args.detector_weights, args.classifier_weights, args.reference_bank,
        conf=args.conf, verify_conf=args.verify_conf, alpha=args.alpha, k=args.k,
    )
    if list(pipeline.class_names) != ['BAD', 'GOOD']:
        parser.error("video mode currently requires a domain whose classes are ['BAD', 'GOOD'] "
                     f"(domain {args.domain!r} has {list(pipeline.class_names)})")

    lost_grace_frames = max(1, round(args.lost_grace_sec * args.sample_fps))
    tracker = Tracker(lost_grace_frames=lost_grace_frames)
    ego_motion = EgoMotionEstimator()

    t_start = time.time()
    n_processed = 0
    n_with_welds = 0
    # Per-frame "still tracked but not reconfirmed this exact sample"
    # positions (see Tracker.update's `lost` return value) -- captured live
    # since it reflects each frame's actual tracker state, then merged into
    # the post-process overlay render below (keyed by track_id, remapped to
    # bead_id once clustering has run) so the video visibly shows a weld
    # being HELD during its grace period instead of vanishing from frame to
    # frame (2026-09-07 user report -- this was a visualization gap, not an
    # actual tracking loss; see video_tracker.py::update's docstring).
    lost_by_frame = {}
    last_timestamp_sec = 0.0
    for frame_idx, timestamp_sec, chunk_id, frame_bgr in iter_sampled_frames(
        video_path, args.sample_fps, args.chunk_duration,
        max_seconds=args.max_seconds, start_seconds=args.start_seconds,
    ):
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise IOError(f"Failed to write frame to {frame_path}")

        # Global camera ego-motion for THIS frame (cheap, CPU-only ORB+RANSAC
        # against the previous sampled frame -- see camera_motion.py) -- every
        # weld is static, so this one transform is what predicts every
        # active track's expected position, replacing a per-track motion
        # model entirely.
        cumulative_transform, motion_confidence = ego_motion.update(frame_bgr)

        image_verdict = pipeline.judge_image(str(frame_path))
        # Live preview for the web UI's "processing" screen: this frame's
        # RAW judge_image() output, overwritten each frame (single file, not
        # one-per-frame) so a poller always fetches the latest without the
        # dir growing unbounded. Uses only data already computed above --
        # no effect on tracking/classification, see draw_live_preview_overlay's
        # docstring for why this can't show final bead ids yet.
        if not args.no_overlays:
            preview = draw_live_preview_overlay(frame_bgr, image_verdict.objects)
            cv2.imwrite(str(live_preview_dir / 'latest.jpg'), preview)
        frame_tracks, lost_tracks = tracker.update(
            frame_idx, timestamp_sec, chunk_id, image_verdict.objects,
            cumulative_transform, frame_bgr, motion_confidence,
        )
        if frame_tracks:
            n_with_welds += 1
        if lost_tracks:
            lost_by_frame[frame_idx] = lost_tracks
        last_timestamp_sec = timestamp_sec

        n_processed += 1
        if n_processed % args.progress_every == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed
            remaining = (est_sampled_frames - n_processed) / rate if rate > 0 else float('nan')
            print(f"  [{n_processed}/{est_sampled_frames}] t={timestamp_sec:6.1f}s chunk={chunk_id} "
                  f"active_tracks={len(tracker.active_tracks)} finalized={len(tracker.finalized_verdicts)} "
                  f"| {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining ({rate:.2f} frame/s)")

    tracker.finalize_all(end_timestamp_sec=last_timestamp_sec)
    total_processing_sec = time.time() - t_start

    # Bead clustering (see bead_clustering.py): merges raw per-segment
    # tracks that are actually an identity handoff on the SAME physical weld
    # bead (temporally disjoint/near-disjoint + geometrically collinear)
    # into one reported track. Distinct, simultaneously-visible welds are
    # never touched by this -- see that module's safety-gate docstring.
    if args.no_bead_clustering:
        report_tracks = tracker.finalized_tracks
        track_id_to_bead_id = {t.track_id: t.track_id for t in tracker.finalized_tracks}
    else:
        report_tracks, track_id_to_bead_id = cluster_tracks_into_beads(
            tracker.finalized_tracks, max_overlap_sec=args.lost_grace_sec,
        )
    track_verdicts = [compute_track_verdict(t) for t in report_tracks]

    video_verdict = compute_video_verdict(
        track_verdicts, video_path=video_path, fps=meta.fps,
        duration_sec=(args.start_seconds + processed_sec),
        sample_fps=args.sample_fps, chunk_duration_sec=args.chunk_duration, n_chunks=n_chunks,
    )

    results_path = out_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(video_verdict), f, ensure_ascii=False, indent=2)

    # Every accepted match's decision basis this run -- see Tracker.match_log's
    # docstring. Dumped unconditionally (cheap, ~1 entry per real detection)
    # so a later "why did this id claim that detection" question can be
    # answered from this exact run instead of a non-reproducible replay.
    with open(out_dir / 'match_log.json', 'w', encoding='utf-8') as f:
        json.dump(tracker.match_log, f, ensure_ascii=False, indent=2)

    report_path = out_dir / 'report.html'
    generate_html_report(video_verdict, str(report_path), total_processing_sec=total_processing_sec)

    # Overlays are rendered here (post-process), not inline during the main
    # loop above, so they can show the FINAL (post-clustering) bead id for
    # each weld instead of the raw per-frame segment track_id -- clustering
    # can only run once every track has been finalized, so this is
    # necessarily a second pass over the already-saved frame JPGs.
    overlay_video_path = None
    if not args.no_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)
        frame_overlay_data = {}
        for track in tracker.finalized_tracks:
            bead_id = track_id_to_bead_id[track.track_id]
            for tf in track.history:
                frame_overlay_data.setdefault(tf.frame_idx, []).append((bead_id, tf.verdict))
        frame_lost_data = {}
        for frame_idx, lost_tracks in lost_by_frame.items():
            for tid, predicted_polygon, last_verdict, position_std in lost_tracks:
                # A track only ever appears in track_id_to_bead_id once
                # FINALIZED -- one still active_tracks-only edge case (the
                # very last frame(s) before finalize_all(), if it never gets
                # reconfirmed again) has no bead_id yet; fall back to its
                # own raw id so it's still drawn instead of silently dropped.
                bead_id = track_id_to_bead_id.get(tid, tid)
                frame_lost_data.setdefault(frame_idx, []).append((bead_id, predicted_polygon, last_verdict, position_std))

        # Defensive dedup: bead_clustering.py now requires merged tracks to
        # be strictly time-disjoint precisely so one bead_id never has two
        # simultaneous presences in the same frame (2026-09-07, confirmed
        # concretely: two source tracks merged into one bead under the old
        # 15%-overlap allowance both showed up "held" in the same frame,
        # rendering as the same id duplicated at two positions). Keeping
        # this as a second, independent safety net rather than trusting the
        # upstream invariant alone: prefer a real match over a held
        # placeholder for the same bead_id, and never emit more than one
        # box per bead_id per frame regardless.
        for frame_idx in set(frame_overlay_data) | set(frame_lost_data):
            matched_bead_ids = {bead_id for bead_id, _ in frame_overlay_data.get(frame_idx, [])}
            if frame_idx in frame_lost_data:
                frame_lost_data[frame_idx] = [
                    (bead_id, poly, v, std) for bead_id, poly, v, std in frame_lost_data[frame_idx]
                    if bead_id not in matched_bead_ids
                ]
                seen = set()
                deduped = []
                for bead_id, poly, v, std in frame_lost_data[frame_idx]:
                    if bead_id in seen:
                        continue
                    seen.add(bead_id)
                    deduped.append((bead_id, poly, v, std))
                frame_lost_data[frame_idx] = deduped
        all_frame_indices = set(frame_overlay_data) | set(frame_lost_data)

        # Sticky-BAD display (2026-09-11, mirrors track_aggregation.py's
        # sticky-BAD verdict policy at the per-frame overlay level): once a
        # bead has shown BAD in any frame (matched or held), every later
        # frame displays it as BAD too, even if that exact frame's own
        # classification says GOOD -- otherwise the overlay visibly "cures"
        # a flagged defect from frame to frame, which reads as the weld
        # having been fixed when nothing about the underlying judgment
        # changed. Walked chronologically (ascending frame_idx) so frames
        # BEFORE the first real BAD detection are not retroactively
        # recolored -- this is a forward-only latch, not a global override.
        bead_seen_bad = set()
        for frame_idx in sorted(all_frame_indices):
            for i, (bead_id, verdict) in enumerate(frame_overlay_data.get(frame_idx, [])):
                if verdict.class_name == 'BAD':
                    bead_seen_bad.add(bead_id)
                elif bead_id in bead_seen_bad:
                    frame_overlay_data[frame_idx][i] = (bead_id, replace(verdict, class_name='BAD'))
            for i, (bead_id, poly, verdict, std) in enumerate(frame_lost_data.get(frame_idx, [])):
                if verdict.class_name == 'BAD':
                    bead_seen_bad.add(bead_id)
                elif bead_id in bead_seen_bad:
                    frame_lost_data[frame_idx][i] = (bead_id, poly, replace(verdict, class_name='BAD'), std)

        generate_dedup_audit(
            frame_overlay_data, frame_lost_data, track_id_to_bead_id, tracker.finalized_tracks,
            frames_dir, out_dir, video_path=video_path,
        )

        for frame_idx in all_frame_indices:
            frame_bgr = cv2.imread(str(frames_dir / f"frame_{frame_idx:06d}.jpg"))
            if frame_bgr is None:
                continue
            overlay = draw_frame_overlay(
                frame_bgr, frame_overlay_data.get(frame_idx, []), frame_lost_data.get(frame_idx, []),
            )
            cv2.imwrite(str(overlays_dir / f"frame_{frame_idx:06d}.jpg"), overlay)
        overlay_video_path = build_overlay_video(str(overlays_dir), str(out_dir / 'overlay.mp4'), args.sample_fps)

    print("=" * 70)
    print(f"Processed {n_processed} sampled frames in {total_processing_sec:.1f}s "
          f"({n_processed / total_processing_sec:.2f} frame/s), {n_with_welds} had >=1 weld")
    print(f"Raw segment-tracks: {len(tracker.finalized_verdicts)} -> "
          f"{'bead-clustered ' if not args.no_bead_clustering else ''}reported tracks: {len(video_verdict.tracks)} "
          f"({sum(1 for t in video_verdict.tracks if t.class_name == 'BAD')} BAD, "
          f"{sum(1 for t in video_verdict.tracks if t.class_name == 'GOOD')} GOOD)")
    print(f"OVERALL VIDEO VERDICT: {video_verdict.overall_class_name}")
    print(f"results.json: {results_path}")
    print(f"report.html:  {report_path}")
    if overlay_video_path:
        print(f"overlay.mp4:  {overlay_video_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
