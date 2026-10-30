"""Overlay drawing (per sampled frame, with track IDs) and the final HTML
timeline report -- the data-driven version of the mockup in
docs/VIDEO.md's timeline section, now
populated from a real Tracker run instead of hand-picked example numbers."""

import html
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.pipeline import ObjectVerdict

from .video_types import VideoVerdict

COLOR_GOOD = (0, 200, 0)   # BGR green, matches pqdai/webapp image_visualize
COLOR_BAD = (0, 0, 255)    # BGR red, matches pqdai/webapp image_visualize


def _draw_dashed_polygon(canvas: np.ndarray, pts: np.ndarray, color, thickness: int, dash_len: int = 10) -> None:
    pts = pts.reshape(-1, 2)
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        seg_len = float(np.linalg.norm(p2 - p1))
        n_dashes = max(1, int(seg_len // dash_len))
        for d in range(0, n_dashes, 2):
            a = p1 + (p2 - p1) * (d / n_dashes)
            b = p1 + (p2 - p1) * (min(d + 1, n_dashes) / n_dashes)
            cv2.line(canvas, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def draw_frame_overlay(
    frame_bgr: np.ndarray,
    tracked: List[Tuple[int, ObjectVerdict]],
    lost: Optional[List[Tuple[int, np.ndarray, ObjectVerdict, float]]] = None,
) -> np.ndarray:
    """`tracked` is [(track_id, ObjectVerdict), ...] for welds actually
    matched THIS frame (solid box). `lost` is
    [(track_id, predicted_polygon, last_verdict, position_std), ...] for
    tracks still being held inside their grace period but not reconfirmed
    this exact frame -- drawn at their predicted position with the SAME id
    and GOOD/BAD color/label as a real detection (from `last_verdict`, its
    most recent real classification -- nothing about that changed just
    because this frame didn't reconfirm the position), just dashed instead
    of solid so held vs. freshly-confirmed stays visually distinguishable.
    Without this, a held track simply doesn't appear for a few frames and
    then reappears with no indication it's the same object, which reads as
    "lost and re-found" even though the tracker never actually dropped its
    identity (2026-09-07 user report -- see
    video_tracker.py::Tracker.update's docstring).

    `position_std` is accepted for interface compatibility with
    Tracker.update()'s return shape but not rendered -- drawing it as an
    uncertainty ring was tried and rejected (2026-09-09): showing the drift
    honestly is not the goal, eliminating it is. See video_tracker.py for
    the actual fix."""
    canvas = frame_bgr.copy()
    any_bad = False
    for track_id, weld in tracked:
        color = COLOR_BAD if weld.class_name == 'BAD' else COLOR_GOOD
        any_bad = any_bad or weld.class_name == 'BAD'
        pts = weld.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
        label = f"id={track_id} {weld.class_name}"
        origin = tuple(pts[0, 0])
        cv2.putText(canvas, label, (origin[0], max(origin[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for track_id, predicted_polygon, last_verdict, _position_std in (lost or []):
        color = COLOR_BAD if last_verdict.class_name == 'BAD' else COLOR_GOOD
        any_bad = any_bad or last_verdict.class_name == 'BAD'
        pts = predicted_polygon.astype(np.int32).reshape(-1, 2)
        _draw_dashed_polygon(canvas, pts, color, 3)
        label = f"id={track_id} {last_verdict.class_name}"
        origin = tuple(pts[0])
        cv2.putText(canvas, label, (origin[0], max(origin[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    banner_color = COLOR_BAD if any_bad else COLOR_GOOD
    banner = f"frame objects: {len(tracked)}" + (f" (+{len(lost)} held)" if lost else "")
    cv2.putText(canvas, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2, cv2.LINE_AA)
    return canvas


def draw_live_preview_overlay(frame_bgr: np.ndarray, objects: List[ObjectVerdict]) -> np.ndarray:
    """Cheap per-frame preview for the web UI's "processing" screen: draws
    THIS frame's raw judge_image() output (no track id, no cross-frame
    identity -- that only exists once bead-clustering/sticky-BAD runs after
    the whole video is processed, see pqdai/video/run.py's main loop
    docstring). Purely a "something is happening, here's what it currently
    sees" preview, not a substitute for the final draw_frame_overlay pass."""
    canvas = frame_bgr.copy()
    any_bad = False
    for weld in objects:
        color = COLOR_BAD if weld.class_name == 'BAD' else COLOR_GOOD
        any_bad = any_bad or weld.class_name == 'BAD'
        pts = weld.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
        origin = tuple(pts[0, 0])
        cv2.putText(canvas, weld.class_name, (origin[0], max(origin[1] - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    banner_color = COLOR_BAD if any_bad else COLOR_GOOD
    cv2.putText(canvas, f"detections: {len(objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2, cv2.LINE_AA)
    return canvas


def build_overlay_video(overlays_dir: str, out_path: str, sample_fps: float) -> Optional[str]:
    """Stitches the per-frame overlay JPGs (already drawn by
    draw_frame_overlay and written by pqdai/video/run.py) into a single
    playable .mp4, encoded at `sample_fps` so its runtime matches the
    original clip's real-world duration (each overlay frame stands in for
    the ~1/sample_fps seconds of source video it was sampled from). Frames
    with zero welds are never written by the caller, so this only includes
    frames where the pipeline actually flagged >=1 weld -- fine for a quick
    visual sanity check, not a frame-complete video.

    Returns the output path, or None if there were no overlay frames to
    stitch (e.g. --no-overlays was passed, or nothing was ever detected)."""
    frame_paths = sorted(Path(overlays_dir).glob('frame_*.jpg'))
    if not frame_paths:
        return None

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, sample_fps, (w, h))
    try:
        for p in frame_paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()
    return str(out_path)


_HTML_HEAD = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Video quality decision report — {video_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 0 0 80px 0; line-height: 1.6; }}
  header {{ background: #252526; border-bottom: 1px solid #3e3e42; padding: 28px 40px; }}
  header h1 {{ font-size: 22px; margin: 0 0 6px 0; color: #fff; }}
  header .sub {{ font-size: 13px; color: #9d9d9d; }}
  main {{ max-width: 1150px; margin: 0 auto; padding: 0 40px; }}
  section {{ margin-top: 40px; }}
  h2 {{ font-size: 17px; color: #4fc3f7; border-bottom: 1px solid #3e3e42; padding-bottom: 8px; margin-bottom: 18px; }}
  code {{ background: #2d2d30; color: #ce9178; padding: 1px 6px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 12.5px; }}
  .summary-banner {{ background: linear-gradient(135deg,#1d3a2a,#1a2e3a); border: 1px solid #2f5a3f; border-radius: 8px; padding: 18px 22px; font-size: 14px; color: #d7ffe6; }}
  .summary-banner.bad {{ background: linear-gradient(135deg,#3a1d1d,#3a1a1a); border-color: #5a2f2f; color: #ffd7d7; }}
  .summary-banner b {{ color: #7cffc4; }}
  .summary-banner.bad b {{ color: #ff9a9a; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 12px 0; }}
  th, td {{ border: 1px solid #3e3e42; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #2d2d30; color: #fff; }}
  tr:nth-child(even) td {{ background: #232324; }}
  .cls-badge {{ display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px; font-weight: bold; }}
  .cls-badge.good {{ background: #1d4a3a; color: #7cffc4; }}
  .cls-badge.bad {{ background: #5a1d1d; color: #ff8080; }}
  .timeline-wrap {{ background: #18181a; border: 1px solid #3e3e42; border-radius: 8px; padding: 20px 24px; margin: 20px 0; overflow-x: auto; }}
  .tl-axis {{ position: relative; height: 26px; margin-left: 150px; border-bottom: 1px solid #3e3e42; margin-bottom: 14px; min-width: 800px; }}
  .tl-tick {{ position: absolute; top: 0; bottom: 0; border-left: 1px solid #5a5a5a; font-size: 10.5px; color: #777; padding-left: 3px; }}
  .tl-row {{ display: flex; align-items: center; margin-bottom: 8px; min-width: 950px; }}
  .tl-label {{ width: 150px; flex-shrink: 0; font-size: 11.5px; color: #9d9d9d; padding-right: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .tl-track {{ position: relative; flex: 1; height: 20px; background: #232324; border-radius: 4px; }}
  .tl-seg {{ position: absolute; top: 0; height: 100%; border-radius: 3px; opacity: 0.9; min-width: 2px; }}
  .tl-seg.good {{ background: #2f9e5f; }}
  .tl-seg.bad {{ background: #b23b3b; }}
  .legend {{ display: flex; gap: 18px; font-size: 11.5px; color: #9d9d9d; margin-top: 12px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
  footer {{ max-width: 1150px; margin: 60px auto 0; padding: 0 40px; font-size: 12.5px; color: #888; }}
  a {{ color: #4fc3f7; }}
</style>
</head>
<body>
<header>
  <h1>Video quality decision report</h1>
  <div class="sub">{video_name} · generated by pqdai video · see docs/VIDEO.md</div>
</header>
<main>
"""

_HTML_TAIL = """
</main>
<footer>Auto-generated report (video_visualize.py::generate_html_report). Source video: <code>{video_path}</code></footer>
</body>
</html>
"""


def generate_html_report(
    video_verdict: VideoVerdict, out_path: str,
    total_processing_sec: Optional[float] = None,
) -> None:
    video_name = html.escape(Path(video_verdict.video_path).name)
    duration = video_verdict.duration_sec
    n_bad = sum(1 for t in video_verdict.tracks if t.class_name == 'BAD')
    n_good = len(video_verdict.tracks) - n_bad

    parts = [_HTML_HEAD.format(video_name=video_name)]

    banner_cls = 'bad' if video_verdict.overall_class_name == 'BAD' else ''
    proc_line = (
        f" · processing time: {total_processing_sec:.1f} s" if total_processing_sec is not None else ""
    )
    parts.append(f"""
<section>
  <div class="summary-banner {banner_cls}">
    <b>Overall verdict: {video_verdict.overall_class_name}</b> — {len(video_verdict.tracks)} track(s) (unique objects)
    (BAD {n_bad} / GOOD {n_good}) · duration {duration:.1f} s · source {video_verdict.fps:.1f} fps
    · sampling {video_verdict.sample_fps:.1f} fps · chunks {video_verdict.chunk_duration_sec:.0f} s x {video_verdict.n_chunks}{proc_line}
  </div>
</section>
""")

    parts.append("""
<section>
  <h2>Tracks (unique objects)</h2>
  <table>
    <tr><th>track_id</th><th>Verdict</th><th>Span</th><th>Frames</th><th>Consistency</th><th>Mean detection confidence</th><th>Representative frame</th></tr>
""")
    for t in sorted(video_verdict.tracks, key=lambda x: x.start_ts):
        badge = 'bad' if t.class_name == 'BAD' else 'good'
        consistency_flag = ' ⚠' if t.consistency_score < 0.7 else ''
        parts.append(
            f"    <tr><td>track_{t.track_id:03d}</td>"
            f"<td><span class=\"cls-badge {badge}\">{t.class_name}</span></td>"
            f"<td>{t.start_ts:.1f}s – {t.end_ts:.1f}s</td>"
            f"<td>{t.n_frames}</td>"
            f"<td>{t.consistency_score * 100:.0f}%{consistency_flag}</td>"
            f"<td>{t.mean_detection_confidence:.2f}</td>"
            f"<td>frame_{t.representative_frame_idx:06d}</td></tr>\n"
        )
    parts.append("  </table>\n</section>\n")

    parts.append("""
<section>
  <h2>Timeline</h2>
  <div class="timeline-wrap">
    <div class="tl-axis">
""")
    if duration > 0:
        chunk_dur = video_verdict.chunk_duration_sec
        n_ticks = int(duration // chunk_dur) + 1
        for i in range(n_ticks + 1):
            t_sec = i * chunk_dur
            if t_sec > duration:
                break
            left_pct = (t_sec / duration) * 100
            parts.append(f'      <div class="tl-tick" style="left:{left_pct:.2f}%">{t_sec:.0f}s</div>\n')
    parts.append("    </div>\n")

    for t in sorted(video_verdict.tracks, key=lambda x: x.start_ts):
        badge = 'bad' if t.class_name == 'BAD' else 'good'
        # Clamped: cv2's CAP_PROP_FRAME_COUNT-derived duration_sec can
        # slightly under-count actual decodable frames for VFR/H.264 .MOV
        # footage (a known cv2/ffmpeg quirk, confirmed relevant by code
        # review 2026-09-06), which can otherwise push a late track's
        # end_ts just past duration_sec and overflow the timeline bar past
        # its container's right edge.
        left_pct = min((t.start_ts / duration) * 100, 100.0) if duration > 0 else 0.0
        width_pct = max(min(((t.end_ts - t.start_ts) / duration) * 100, 100.0 - left_pct), 0.3) if duration > 0 else 0.3
        label = f"track_{t.track_id:03d} ({t.class_name})"
        parts.append(
            f'    <div class="tl-row"><div class="tl-label">{html.escape(label)}</div>'
            f'<div class="tl-track"><div class="tl-seg {badge}" style="left:{left_pct:.2f}%;width:{width_pct:.2f}%"></div></div></div>\n'
        )

    parts.append("""
    <div class="legend">
      <span><i style="background:#2f9e5f"></i> GOOD track</span>
      <span><i style="background:#b23b3b"></i> BAD track</span>
      <span>┊ vertical ticks = chunk boundaries</span>
    </div>
  </div>
</section>
""")

    parts.append(_HTML_TAIL.format(video_path=html.escape(video_verdict.video_path)))

    Path(out_path).write_text(''.join(parts), encoding='utf-8')
