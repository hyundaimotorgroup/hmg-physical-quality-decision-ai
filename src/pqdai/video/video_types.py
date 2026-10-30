"""Shared dataclasses for the video weld inspection pipeline.

Extends (does not modify) pqdai/core/pipeline.py's per-frame ObjectVerdict/ImageVerdict
with the identity/aggregation layer video needs: a ObjectVerdict has no notion
of "the same weld as last frame" -- Track ties a sequence of per-frame
ObjectVerdicts together into one physical weld, and TrackVerdict/VideoVerdict
collapse that sequence into a single stable judgment. See
docs/architecture.md for the full design rationale.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


from ..core.pipeline import ObjectVerdict  # (reused unchanged)


@dataclass
class TrackedFrame:
    """One frame's observation of a track -- the per-frame ObjectVerdict plus
    the video-time bookkeeping needed to place it on a timeline.

    `cumulative_transform` is the camera_motion.py::EgoMotionEstimator
    global transform in effect AT THIS FRAME (3x3 homogeneous affine,
    mapping the first-ever-processed frame's pixel coordinates into this
    frame's). Recorded per-observation (not just once on the Track) so a
    track's LAST observation always carries the exact reference needed to
    project it forward to any later frame: see
    video_tracker.py::_predicted_polygon."""
    frame_idx: int
    timestamp_sec: float
    chunk_id: int
    verdict: ObjectVerdict
    cumulative_transform: np.ndarray


@dataclass
class Track:
    """A physical weld followed across frames via tracking-by-detection
    (video_tracker.py::Tracker). `status` is one of 'active' (matched
    recently), 'lost' (missed some frames but still inside its grace
    period, kept alive so a temporarily-occluded/blurred weld isn't split
    into two tracks), or 'finalized' (grace period exceeded or video ended;
    no longer updated, its TrackVerdict has been computed)."""
    track_id: int
    status: str = 'active'
    missed_count: int = 0
    # Consecutive held frames since the last REAL match, counted through
    # degraded frames too (unlike missed_count, which video_tracker.py
    # deliberately freezes during a degraded frame so a brief unreliable
    # patch doesn't burn through the grace period). Added 2026-09-11 after
    # finding, on a rougher-motion 30s clip than the one this whole
    # tracking model was validated on, that 58.7% of its frames were
    # flagged degraded -- letting missed_count freeze stack across many
    # non-contiguous degraded runs kept tracks alive 4-8 REAL seconds past
    # their last actual detection (vs. the intended 1.6s grace period),
    # long enough for their Kalman covariance to grow into the thousands of
    # px and, combined with UNCERTAINTY_GATE_STD_FACTOR's unbounded
    # widening, hijack an unrelated real detection over 2000px away (see
    # video_tracker.py's UNCERTAINTY_GATE_MAX_PX and
    # MAX_HELD_WITHOUT_MATCH_FACTOR docstrings for the concrete numbers).
    # This counter is what actually bounds that -- missed_count alone
    # cannot, by design.
    frames_since_real_match: int = 0
    history: List[TrackedFrame] = field(default_factory=list)
    verdict: Optional['TrackVerdict'] = None
    # Timestamp this track actually stopped being tracked (grace period
    # exceeded, or the video ended) -- NOT the same as last_seen_sec
    # (its last REAL match). A track can sit "held" on screen (see
    # video_tracker.py::Tracker.update's `lost` return value) for up to a
    # full grace period AFTER its last real match, still occupying that
    # screen space the whole time. bead_clustering.py's temporal-overlap
    # safety gate needs THIS, not last_seen_sec, to correctly detect that
    # two tracks were simultaneously on screen (2026-09-07, third
    # deep-dive: confirmed a merge that used last_seen_sec alone let a
    # still-being-held track and a genuinely different, already-active new
    # track get merged into one reported bead, which then visibly
    # duplicated on screen since both were really present at once). None
    # until the track is finalized.
    finalized_at_sec: Optional[float] = None
    # Per-track Kalman filter state: mean=[cx, cy, vx, vy], cov=4x4 -- see
    # video_tracker.py's KF_* constants and _kf_predict_and_warp/_kf_update.
    # History of this position model on this project (2026-09-08, all in
    # one day): (1) pure global CMC composition -- broke down because this
    # footage has real depth parallax (a corner post vs. a farther beam)
    # that one global transform can't fit for both at once, confirmed by a
    # held track's error GROWING over time (137px -> 259px) instead of
    # staying flat; (2) per-track KLT optical-flow, chained one frame-step
    # at a time and warped nowhere -- fixed the parallax problem (matches
    # SAMOFT/Flow-Tracker's published pattern of isolating per-object
    # motion from global scene motion) but KLT's status flag only means the
    # LK iteration numerically converged, not that it converged to the
    # RIGHT feature -- this footage's repetitive rivet pattern let points
    # confidently lock onto the next rivet over, confirmed as a 149px
    # single-step jump that satisfied every KLT success check. A
    # CMC-disagreement outlier gate patched that specific case but is a
    # symptom-level fix, not a structural one -- runs back into the SAME
    # unreliable global-CMC fallback the moment KLT is (correctly)
    # distrusted. (3) THIS: a real Kalman filter per track, replacing both
    # of the above -- the standard architecture in production trackers
    # (BoT-SORT, StrongSORT; 2026-09-08 web research): global CMC only ever
    # WARPS the filter's predicted mean/covariance (rotation+scale+
    # translation on position, rotation+scale only on velocity -- see
    # _kf_predict_and_warp), it is never trusted as the position estimate
    # by itself. The velocity component is what a naive CMC-only or
    # KLT-only model doesn't have: it's fit via the Kalman gain from this
    # SPECIFIC track's own real re-detections over time, so a track that
    # keeps landing persistently offset from its CMC-warped prediction
    # (exactly the corner-post parallax bias) has that bias absorbed into
    # its own velocity state automatically -- no per-region/local CMC
    # bookkeeping needed, and no per-point feature-identity risk the way
    # KLT had. An earlier, pre-CMC per-track LEAST-SQUARES velocity
    # estimate (2026-09-07, see git history) was rejected for amplifying
    # 2-observation noise into a spurious velocity -- a proper Kalman
    # filter's process/measurement noise covariances are exactly the
    # standard fix for that specific failure mode (a raw 2-point slope has
    # no way to express "low confidence", a KF's covariance does).
    kf_mean: Optional[np.ndarray] = None
    kf_cov: Optional[np.ndarray] = None

    @property
    def last_polygon(self) -> np.ndarray:
        return self.history[-1].verdict.polygon

    @property
    def last_embedding(self) -> Optional[np.ndarray]:
        return self.history[-1].verdict.embedding

    @property
    def presence_end_sec(self) -> float:
        """When this track actually stopped occupying screen space --
        `finalized_at_sec` if known, else `last_seen_sec` as a fallback for
        a track inspected before finalization."""
        return self.finalized_at_sec if self.finalized_at_sec is not None else self.last_seen_sec

    @property
    def first_seen_sec(self) -> float:
        return self.history[0].timestamp_sec

    @property
    def last_seen_sec(self) -> float:
        return self.history[-1].timestamp_sec

    @property
    def last_cumulative_transform(self) -> np.ndarray:
        """The global camera-ego-motion transform in effect when this track
        was last observed -- see camera_motion.py and
        video_tracker.py::_predicted_polygon. Superseded the earlier
        per-track velocity/scale_rate least-squares estimators entirely
        (2026-09-07): those modeled EACH TRACK'S motion independently from
        just its own few observations, which is the wrong level for a
        target that has zero true motion of its own -- every weld's
        apparent motion is 100% camera ego-motion, so one global transform
        (estimated from hundreds of background feature matches, not a
        handful of noisy per-track points) is both simpler and far more
        robust, and -- unlike the old model -- gives a usable prediction
        starting from a track's very FIRST frame, no minimum observation
        count required."""
        return self.history[-1].cumulative_transform


@dataclass
class TrackVerdict:
    """A track's per-frame ObjectVerdict history collapsed into one stable
    judgment -- see track_aggregation.py for the confidence-weighted vote
    that produces this."""
    track_id: int
    class_name: str
    consistency_score: float  # fraction of frames agreeing with class_name
    n_frames: int
    start_ts: float
    end_ts: float
    mean_detection_confidence: float
    representative_frame_idx: int  # the frame with the highest per-frame weight, for thumbnailing


@dataclass
class VideoVerdict:
    video_path: str
    fps: float
    duration_sec: float
    sample_fps: float
    chunk_duration_sec: float
    n_chunks: int
    tracks: List[TrackVerdict] = field(default_factory=list)
    overall_class_name: str = 'GOOD'  # no tracks -> nothing to flag as BAD
