"""Collapse a Track's per-frame ObjectVerdict history into one stable
TrackVerdict, and collapse a video's finalized tracks into one VideoVerdict.

Final class is a STICKY-BAD latch, not a vote (2026-09-11, user-mandated
safety policy): once ANY frame in a track's real history was classified
BAD, that track's reported verdict is BAD, permanently -- GOOD -> BAD is
always possible, BAD -> GOOD never is. Explicitly NOT a confidence-weighted
majority vote (an earlier version of this function was): from a
missed-defect-is-costly standpoint, a single confident BAD frame catching a
real defect must never be outvoted by many GOOD frames that simply didn't
happen to see it (e.g. a different viewing angle, or the defect only
visible when the classifier's entropy gate triggered retrieval). The
weights are still used for `consistency_score`/`representative_frame_idx`,
which are informational, not the safety-critical decision.
"""

import math
from typing import List

from .video_types import Track, TrackVerdict, VideoVerdict

LOG2 = math.log(2)  # max entropy for a 2-class softmax; see src/uncertainty.py


def _frame_weight(entropy: float, detection_confidence: float) -> float:
    normalized_entropy = min(max(entropy / LOG2, 0.0), 1.0)
    return detection_confidence * (1.0 - normalized_entropy)


def compute_track_verdict(track: Track) -> TrackVerdict:
    if not track.history:
        raise ValueError(f"track {track.track_id} has no history to aggregate")

    weights = [
        _frame_weight(tf.verdict.entropy, tf.verdict.detection_confidence)
        for tf in track.history
    ]
    total_weight = sum(weights)

    class_weight = {'BAD': 0.0, 'GOOD': 0.0}
    for tf, w in zip(track.history, weights):
        class_weight[tf.verdict.class_name] += w

    if total_weight <= 0.0:
        # Every frame was maximally uncertain and/or zero-confidence -- fall
        # back to an unweighted frame count for consistency_score/representative
        # frame purposes below (final_class itself no longer depends on this).
        class_weight = {'BAD': 0.0, 'GOOD': 0.0}
        for tf in track.history:
            class_weight[tf.verdict.class_name] += 1.0

    # Sticky-BAD latch, not a vote -- see module docstring. Any single BAD
    # frame anywhere in this track's real history is enough.
    final_class = 'BAD' if any(tf.verdict.class_name == 'BAD' for tf in track.history) else 'GOOD'

    n_frames = len(track.history)
    agreeing = sum(1 for tf in track.history if tf.verdict.class_name == final_class)
    consistency_score = agreeing / n_frames

    # Secondary key (raw detection_confidence) breaks ties -- matters mainly
    # in the all-zero-weight fallback above, where every primary key would
    # otherwise tie at 0.0 and max() would silently always pick frame 0.
    representative_idx = max(
        range(n_frames),
        key=lambda i: (weights[i], track.history[i].verdict.detection_confidence),
    )
    representative_frame_idx = track.history[representative_idx].frame_idx

    mean_detection_confidence = sum(
        tf.verdict.detection_confidence for tf in track.history
    ) / n_frames

    return TrackVerdict(
        track_id=track.track_id,
        class_name=final_class,
        consistency_score=consistency_score,
        n_frames=n_frames,
        start_ts=track.first_seen_sec,
        end_ts=track.last_seen_sec,
        mean_detection_confidence=mean_detection_confidence,
        representative_frame_idx=representative_frame_idx,
    )


def compute_video_verdict(
    track_verdicts: List[TrackVerdict],
    video_path: str, fps: float, duration_sec: float,
    sample_fps: float, chunk_duration_sec: float, n_chunks: int,
) -> VideoVerdict:
    # Same safety-first rule as pqdai/core/pipeline.py::judge_image, applied to
    # tracks (physical welds) instead of raw per-frame detections -- this is
    # the fix for the single-threshold-per-frame flicker problem: a track is
    # only BAD if ITS OWN weighted vote across all its frames says BAD, not
    # because any one noisy frame said so.
    overall = 'BAD' if any(tv.class_name == 'BAD' for tv in track_verdicts) else 'GOOD'
    return VideoVerdict(
        video_path=video_path, fps=fps, duration_sec=duration_sec,
        sample_fps=sample_fps, chunk_duration_sec=chunk_duration_sec, n_chunks=n_chunks,
        tracks=track_verdicts, overall_class_name=overall,
    )
