"""Global camera ego-motion estimation (Camera Motion Compensation, CMC).

Why this replaces video_tracker.py's earlier per-track velocity/scale
model: our "objects" (welds) are 100% static in the world -- unlike
autonomous-driving/general MOT targets (other cars, pedestrians), a weld
never moves. Every pixel of apparent motion between sampled frames is
therefore pure camera ego-motion, and it is the SAME transform for every
track in the frame simultaneously. Estimating that transform ONCE per frame
pair from the whole background (hundreds of matched feature points) is a
far better-conditioned problem than each track independently estimating
"its own" velocity from 2-5 of its own noisy observations (confirmed as a
real failure mode on this project's footage: a 2-point per-track velocity
estimate amplified ordinary detection-box jitter into a spurious
-398px/sec "velocity" that stole a track from its true continuation, see
video_types.py's MIN_OBSERVATIONS_FOR_MOTION_ESTIMATE history -- CMC
sidesteps this class of bug entirely instead of just raising the min-sample
floor).

This is the standard technique in modern tracking-by-detection literature
for moving-camera scenes (2026-09-07 web research): BoT-SORT and OC-SORT
extensions both add a "CMC" stage that warps each track's Kalman-predicted
box by a global affine/homography estimated via ORB/SIFT feature matching +
RANSAC (or ECC image alignment) BEFORE computing IoU against new
detections -- without it, "a panning or moving camera can make stationary
... targets appear to jump, degrading IoU overlap and causing false
unmatched tracks" (exactly this project's symptom). Autonomous-driving
tracking is the more general version of the same idea (ego-motion
compensation into a world/road frame so a static object's position doesn't
change) -- our case is the clean special case where EVERY tracked object is
static, so a single global transform is a complete and sufficient motion
model (no separate per-object dynamics on top of it are needed at all).
"""

from typing import Optional, Tuple

import cv2
import numpy as np

MIN_FEATURES = 20  # below this many detected keypoints in either frame, don't even attempt matching
MIN_MATCHES = 10  # below this many raw matches, the affine fit would be underdetermined/unreliable
RANSAC_REPROJ_THRESHOLD = 3.0  # pixels; standard default for this kind of frame-to-frame registration

# Per-track LOCAL motion estimate (2026-09-09, replacing the "one global
# transform predicts every track" assumption documented above -- that
# assumption is what this project's own footage disproves: a near corner
# post and a farther beam have genuinely different apparent motion under
# real camera translation, no single global affine fits both. Rather than
# fight that with more/fewer global features or region-restricted grids
# (both tried 2026-09-08, see video_tracker.py's Track.kf_mean docstring --
# region-restricted local CMC on a FIXED grid region measured WORSE, 188px
# vs 179px), this fits a fresh local affine EVERY STEP, in a crop centered
# on THAT TRACK's own current position and sized to THAT TRACK's own box --
# i.e. genuinely local to the object, not a static image region, and
# RANSAC-voted across many local keypoints rather than trusting one point
# the way the earlier per-track KLT attempt did (KLT's single-point-lock
# failure mode: a repetitive rivet pattern let one point converge to the
# WRONG rivet, and its "converged" status flag couldn't tell the
# difference). Measured on real footage (2026-09-09, 1574 real re-detection
# events across a 30s clip): global-CMC+KF prediction error median 10.5px /
# p90 50.6px vs local median 4.1px / p90 31.9px -- local wins 72.6% of
# events, including the exact frame ranges independently reported as
# visibly drifting (e.g. fid 258 track 47: 29.8px -> 3.1px). Falls back to
# the global transform (video_tracker.py) when a track's local crop can't
# support a reliable fit -- e.g. a largely featureless close-up crop, or a
# track near the frame edge with little crop area left.
LOCAL_MAX_FEATURES = 1000
LOCAL_MIN_FEATURES = 15  # below this many local keypoints in either crop, don't attempt a local fit
LOCAL_MIN_MATCHES = 8
LOCAL_RANSAC_REPROJ_THRESHOLD = 3.0
LOCAL_MIN_PAD = 250.0  # px -- floor on crop half-size so a small weld still gets enough surrounding context
LOCAL_PAD_DIAG_FACTOR = 1.5  # crop half-size scales with the track's own box diagonal above the floor
_local_orb = cv2.ORB_create(LOCAL_MAX_FEATURES)
_local_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _to_homogeneous(affine_2x3: np.ndarray) -> np.ndarray:
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = affine_2x3
    return h


def apply_affine(transform_3x3: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Applies a 3x3 homogeneous affine transform to a (4,2) polygon."""
    pts = np.asarray(polygon, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    transformed = homogeneous @ transform_3x3.T
    return transformed[:, :2].astype(np.float32)


def local_affine_transform(prev_gray: np.ndarray, curr_gray: np.ndarray,
                            center_xy: np.ndarray, box_diagonal: float,
                            ) -> Optional[Tuple[np.ndarray, float]]:
    """Per-object motion estimate: ORB+RANSAC affine fit ONLY to keypoints in
    a crop centered on `center_xy` (a track's current predicted position),
    sized relative to `box_diagonal` -- see the LOCAL_* constants' docstring
    above for why this replaces trusting the single global transform for
    every track. Returns (transform_3x3, RANSAC inlier fraction), where the
    transform maps a point in `prev_gray`'s FULL-FRAME coordinates to
    `curr_gray`'s, or None if this crop doesn't support a reliable fit (too
    few keypoints/matches, or the track is near enough to the frame edge
    that the crop is too small) -- the caller should fall back to the
    global transform in that case."""
    pad = max(LOCAL_MIN_PAD, LOCAL_PAD_DIAG_FACTOR * box_diagonal)
    h, w = prev_gray.shape[:2]
    cx, cy = center_xy
    x0, x1 = int(max(0, cx - pad)), int(min(w, cx + pad))
    y0, y1 = int(max(0, cy - pad)), int(min(h, cy + pad))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    crop_prev = prev_gray[y0:y1, x0:x1]
    crop_curr = curr_gray[y0:y1, x0:x1]
    kp1, des1 = _local_orb.detectAndCompute(crop_prev, None)
    kp2, des2 = _local_orb.detectAndCompute(crop_curr, None)
    if des1 is None or des2 is None or len(des1) < LOCAL_MIN_FEATURES or len(des2) < LOCAL_MIN_FEATURES:
        return None
    matches = _local_matcher.match(des1, des2)
    if len(matches) < LOCAL_MIN_MATCHES:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=LOCAL_RANSAC_REPROJ_THRESHOLD,
    )
    if affine is None or inlier_mask is None:
        return None
    confidence = float(inlier_mask.sum()) / len(matches)
    transform = _to_homogeneous(affine)
    # affine was fit in CROP-local coordinates (same crop origin both
    # frames), so translate it back into full-frame coordinates: since the
    # crop offset is identical on both sides, only the translation term
    # needs correcting, not the rotation/scale part.
    R = transform[:2, :2]
    t = transform[:2, 2]
    offset = np.array([x0, y0], dtype=np.float64)
    transform[:2, 2] = R @ (-offset) + offset + t
    return transform, confidence


class EgoMotionEstimator:
    """Call `update(frame_bgr)` once per sampled frame, in order. Returns
    (cumulative_transform, confidence) where cumulative_transform is a 3x3
    homogeneous affine mapping a point in the FIRST processed frame's pixel
    coordinates into the CURRENT frame's pixel coordinates, and confidence
    is the RANSAC inlier fraction for this frame's registration against the
    previous one (1.0 on the very first frame, 0.0 if registration failed
    outright -- e.g. too few features, usually from severe motion blur or a
    largely featureless frame).

    To predict where a track last seen at an earlier frame should be NOW:
    `relative = current_cumulative @ inv(cumulative_at_track's_last_seen_frame)`,
    then `apply_affine(relative, track.last_polygon)` -- see
    video_tracker.py::_predicted_polygon."""

    def __init__(self, max_features: int = 800):
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_des = None
        self.cumulative = np.eye(3, dtype=np.float64)

    def update(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self._prev_des is None:
            # First frame ever -- nothing to register against; cumulative
            # starts at identity and this frame defines the reference origin.
            self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
            return self.cumulative.copy(), 1.0

        confidence = 0.0
        if des is not None and len(des) >= MIN_FEATURES and len(self._prev_des) >= MIN_FEATURES:
            matches = self.matcher.match(self._prev_des, des)
            if len(matches) >= MIN_MATCHES:
                src_pts = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])
                affine, inlier_mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
                )
                if affine is not None and inlier_mask is not None:
                    confidence = float(inlier_mask.sum()) / len(matches)
                    self.cumulative = _to_homogeneous(affine) @ self.cumulative

        # Registration failure for this one pair (severe blur, featureless
        # frame, etc.) leaves `cumulative` unchanged -- equivalent to
        # assuming zero motion for just this step rather than corrupting
        # every later frame's absolute transform with a bad estimate. The
        # caller (video_tracker.py) uses the returned low confidence to
        # treat this frame as degraded instead of trusting its (now stale)
        # motion prediction.
        self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
        return self.cumulative.copy(), confidence
