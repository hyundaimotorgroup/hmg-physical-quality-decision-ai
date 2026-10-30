"""Sequential frame decoding + sampling + chunk-id bookkeeping.

Deliberately a SINGLE sequential cv2.VideoCapture pass over the whole video
rather than per-chunk seeks: seeking (CAP_PROP_POS_FRAMES) on compressed
h264/.MOV footage is only reliable to the nearest keyframe, which would
silently shift which frame gets sampled right at a chunk boundary -- exactly
where the tracker's carry-over matters most. Chunk boundaries are therefore a
property DERIVED from each frame's timestamp (chunk_id = timestamp //
chunk_duration_sec) for reporting/traceability, not a separate decode
operation; the Tracker object itself just stays alive across the boundary
in-process, so there is no actual state to serialize/hand off for a single
run like this one (see docs/architecture.md's chunking
rationale -- resumability/parallelism matters more for much longer videos
than this 208s clip).
"""

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


def _open_capture(video_path: str) -> cv2.VideoCapture:
    """cv2.VideoCapture with rotation metadata applied. iPhone (and most
    phone camera) portrait footage stores pixels landscape and relies on an
    embedded rotation hint (90 deg here) for playback -- CAP_PROP_ORIENTATION_AUTO
    defaults to OFF on some OpenCV/FFmpeg builds (confirmed: opencv-python
    4.11.0 in this project's GPU venv defaults to 0, vs 1 on the 5.0.0 build
    used elsewhere on this machine), so frames were being read pre-rotation
    (1920x1080 landscape) instead of the actual upright 1080x1920 -- every
    downstream detection polygon and the overlay video were then computed/
    drawn in that wrong, sideways orientation. Setting this explicitly makes
    the behavior consistent regardless of which OpenCV build runs this."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def probe_video(video_path: str) -> VideoMeta:
    cap = _open_capture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        return VideoMeta(fps=fps, total_frames=total_frames, width=width, height=height, duration_sec=duration_sec)
    finally:
        cap.release()


def n_chunks_for(duration_sec: float, chunk_duration_sec: float) -> int:
    return max(1, math.ceil(duration_sec / max(chunk_duration_sec, 1e-6)))


def iter_sampled_frames(
    video_path: str, sample_fps: float, chunk_duration_sec: float,
    max_seconds: Optional[float] = None, start_seconds: float = 0.0,
) -> Iterator[Tuple[int, float, int, np.ndarray]]:
    """Yields (frame_idx, timestamp_sec, chunk_id, frame_bgr) for every
    frame selected by the sampling stride, in ascending time order.

    `max_seconds` truncates the pass early -- used for smoke-testing a short
    clip before committing to the full video's runtime. `start_seconds`
    skips (decodes but does not yield) every frame before that timestamp --
    used to smoke-test a short window elsewhere in the video, e.g. to spot-
    check a tracking fix beyond the always-tested first 30s. Still a single
    sequential decode pass (no CAP_PROP_POS_FRAMES seek) for the same
    keyframe-accuracy reason as the module docstring -- cheap relative to
    the per-frame detector/classifier cost this skips for the discarded
    frames, since only cv2.read() runs on them, not the pipeline.
    `frame_idx`/`timestamp_sec` stay absolute (relative to the video's own
    start), not rebased to 0 at `start_seconds` -- so frame filenames and
    tracker bookkeeping remain consistent with a start_seconds=0 run over
    the same stretch.
    """
    meta = probe_video(video_path)
    stride = max(1, round(meta.fps / sample_fps))
    end_seconds = (start_seconds + max_seconds) if max_seconds is not None else None

    cap = _open_capture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_sec = frame_idx / meta.fps
            if end_seconds is not None and timestamp_sec > end_seconds:
                break
            if timestamp_sec >= start_seconds and frame_idx % stride == 0:
                chunk_id = int(timestamp_sec // chunk_duration_sec)
                yield frame_idx, timestamp_sec, chunk_id, frame
            frame_idx += 1
    finally:
        cap.release()
