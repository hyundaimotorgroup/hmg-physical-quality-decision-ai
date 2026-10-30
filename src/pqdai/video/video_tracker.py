"""Cross-frame tracking-by-detection for video weld inspection.

Answers the question the photo pipeline never had to: "the same physical
weld was found again in this frame -- is it the same one as before, or a
new one?" Each sampled frame is judged independently and completely by the
UNCHANGED photo pipeline (pqdai/core/pipeline.py::QualityPipeline.judge_image,
including its full SAHI-tiling + WBF + ROI-redetection + gap-bridging +
NMS-dedup cascade -- see docs/architecture.md section 1). This
module only decides which of THIS frame's already-confirmed ObjectVerdicts
continue an existing track vs. start a new one.

Motion model (2026-09-09, fourth revision -- see video_types.py's
Track.kf_mean docstring for the full history of what was tried before and
why each was replaced): a per-track Kalman filter (constant-velocity model
on the box center), warped every frame by a PER-TRACK LOCAL motion estimate
(camera_motion.py::local_affine_transform -- ORB+RANSAC restricted to a
crop centered on that track's own position, falling back to the global CMC
transform only when a track's local crop can't support a reliable fit),
then corrected by the Kalman gain whenever a real detection lands -- the
standard architecture in production trackers (BoT-SORT, StrongSORT), with
the GMC step made per-object instead of global. A single global transform
assumes every static object's apparent motion is identical, which this
footage disproves (real depth parallax between a near corner post and a
farther beam) -- measured directly (2026-09-09, 1574 real re-detection
events): global-CMC prediction error median 10.5px vs per-track-local
median 4.1px, local better in 72.6% of cases. The Kalman filter's velocity
state still absorbs whatever residual bias remains after the (now much
smaller) per-track motion error -- see _kf_predict_and_warp/_kf_update.

Deliberate v1 simplification kept from the original design (an open
question the design doc flagged explicitly -- see
docs/architecture.md's "open questions before finalizing"): no ByteTrack-style
low-confidence recovery tier. Every ObjectVerdict this module ever sees
already cleared the detector's own verify_conf gate (the caller constructs
TiledDetector with verify_conf set), so there is no leftover "weak but real"
detection pool to recover from within judge_image's output -- that would
require also classifying the detector's rejected candidates
(TiledDetector.last_rejected_candidates), which is deferred. The grace-period
(missed-frame tolerance) below is what absorbs a track's occasional
single-frame non-detection instead.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


from ..core.pipeline import ObjectVerdict

from .video_types import Track, TrackedFrame, TrackVerdict
from .track_aggregation import compute_track_verdict
from .camera_motion import local_affine_transform

import cv2

IOU_WEIGHT = 0.6
EMBEDDING_WEIGHT = 0.4
# Below this combined similarity, a Hungarian-assigned pair is rejected and
# both sides are treated as unmatched -- Hungarian always returns a complete
# assignment even when every pair is a terrible match, so this gate is what
# actually implements "no match" for tracking-by-detection (see module docs
# in docs/architecture.md's tracker table, "matching" row).
MIN_MATCH_SIMILARITY = 0.2
# Hard geometric precondition, checked BEFORE embedding similarity is even
# consulted (see _combined_similarity). Code review (2026-09-06) confirmed
# that without this, two physically distinct welds with zero spatial overlap
# could still clear MIN_MATCH_SIMILARITY on embedding similarity alone --
# the classifier's embedding is a GOOD/BAD-classification feature, not an
# identity/re-ID feature, so unrelated weld crops routinely have a
# significant positive cosine similarity. Embedding is only a signal among
# spatially-plausible candidates, never a standalone path to a match.
# Raised 0.01 -> 0.05 (2026-09-11, user-reported: a track's box drifting
# onto a visually-similar NEIGHBORING rivet in a row of closely-spaced,
# near-identical welds -- exactly the "distractor" scenario MOT literature
# flags as the hard case for appearance-based Re-ID). Measured on the
# offset_65s clip's match_log (590 real matches): 34 had iou<0.10, heavily
# clustered on ONE degraded/blurry frame (fid=2064, 7 different tracks
# simultaneously) where geometry was barely informative and the match was
# effectively decided by IOU_WEIGHT/EMBEDDING_WEIGHT blending an emb_sim
# that, per the docstring above, isn't discriminative enough to trust at
# this evidence level -- of those 34, only 14 would also clear the strict
# fallback path's emb_sim>=0.85 bar. 0.01 was letting IoU as low as 1-2%
# (barely more than touching) receive the SAME full blended-score treatment
# as a confident 60%+ overlap; 0.05 still comfortably passes routine
# registration jitter (median observed iou on confirmed-good matches is
# 0.64) while pushing the genuinely-marginal cases through the fallback
# path's stricter embedding+tight-distance bar instead of the primary
# path's more permissive blend.
MIN_IOU_FOR_MATCH = 0.05
# Fallback gate for when a track's CMC-PREDICTED position has ZERO polygon
# overlap with a candidate detection -- meant to bridge the small residual
# registration error CMC (camera_motion.py) can still have, NOT to serve as
# a wide "did the object move far" catch-all the way it had to before CMC
# existed. Shrunk from an original 2.0 (2026-09-07 deep-dive, second pass):
# once global ego-motion compensation made position prediction genuinely
# accurate (confirmed on real data: a true continuation landed within ~3px
# of its CMC-predicted position), a generous 2.0-diagonal gate started
# actively causing wrong matches -- a detection ~130px away (a DIFFERENT,
# merely similar-looking nearby weld) was passing the old gate on embedding
# similarity alone and, because of the scoring asymmetry fixed below,
# OUTSCORING the true match sitting almost exactly at the predicted spot.
# 0.5 diagonals is generous enough for leftover registration jitter but
# firmly excludes a genuinely different object one weld-spacing away.
PROXIMITY_GATE_DIAG_FACTOR = 0.5
MIN_EMBEDDING_SIM_FOR_PROXIMITY_MATCH = 0.85
# Distractor-aware discrimination (2026-09-11, adopted from established MOT
# literature at the user's explicit direction -- this codebase should use
# what the field already validated, not another ad-hoc constant). The
# literature term for this project's exact recurring failure -- a track's
# predicted position, imperfect at 5fps sampling with real camera shake,
# lands ambiguously between the true weld and a near-IDENTICAL-looking
# neighbor (a repeated bolt/rivet on the same beam) -- is a "distractor":
# see Zhu et al., "Distractor-aware discrimination learning for online
# multiple object tracking," Pattern Recognition 2021 (foundational; cited
# by APPTracker+, IJCV 2024/2025, which addresses this SAME low-frame-rate
# regime this project runs in). Hungarian assignment alone doesn't protect
# against this: it always returns the GLOBALLY best one-to-one pairing even
# when the winning score barely beats a second, equally-plausible track for
# the same detection -- exactly the geometry of two adjacent, near-identical
# rivets. The fix is the same "ratio test" idea Lowe's SIFT matching uses
# for the identical ambiguous-correspondence problem: don't accept a match
# just because it's the BEST option, only accept it if it's CLEARLY better
# than the runner-up -- an ambiguous case should be rejected (the detection
# starts a fresh track, the track stays held) rather than guessed at,
# because guessing wrong silently corrupts an existing id's GOOD/BAD
# history while rejecting only costs a fragmented (but individually
# correct) extra track -- a strictly safer failure direction per this
# project's own established priority (missing/splitting a defect report is
# recoverable; silently merging two different welds' histories is not).
AMBIGUITY_MARGIN = 0.1
# Hard precondition, checked before either match path -- rejects a
# candidate whose box size is drastically different from the track's own
# established size, regardless of how well its position/embedding score.
# Added 2026-09-09 after tracing two independently user-reported "box
# jump" incidents (both confirmed via visual inspection AND raw detection
# data) to the SAME root cause: src/detector.py's WBF/gap-bridging cascade
# occasionally emits one oversized box that spatially spans several real,
# separate welds (a detector-side defect, not a motion-prediction one --
# out of this module's scope to fix directly). Without this gate, plain
# IoU-based matching has no way to tell that apart from a legitimate
# same-object continuation: the track's small predicted box can sit
# entirely WITHIN the oversized detection's footprint, scoring a
# perfectly valid positive IoU, so the track's identity gets "teleported"
# to the giant box's centroid (measured concretely: one track in a
# sample clip jumped 55.7px -> 535.1px box diagonal in one 0.2s step,
# then 535.1px -> 91.0px the very next real match -- a 9.6x and 5.9x
# ratio respectively). Measured the natural per-step size-ratio
# distribution across a full run for calibration (1583 consecutive real
# matches): p99=2.56, p99.9=5.27, vs. the two confirmed-defective events
# at 9.6x and 4.8x -- 3.5 sits above the legitimate tail (fast zoom can
# genuinely change box size 2-3x in one step) while still catching both
# confirmed incidents. A rejected match still lets the oversized detection
# start its own (separate, short-lived) track rather than corrupting an
# established one's identity and GOOD/BAD history.
SIZE_RATIO_MAX = 3.5
# Confidence-aware association (2026-09-09, per Sentinel's "Confidence
# Aware Association" -- arxiv/nature.com/articles/s41598-026-43938-2 --
# and the broader uncertainty-aware-MOT literature, UncertaintyTrack/UAST):
# TRIED, then REJECTED 2026-09-11 -- kept here (unused below) as the
# record of why. The idea was to widen the proximity-gate fallback above BY
# a track's own Kalman covariance, growing the search radius the longer a
# track went unconfirmed (tracking_architecture_v2.html section 6 option
# D). First measured problem (2026-09-11): with no cap, position_std could
# grow into the hundreds of px on a rougher clip, once producing a gate of
# 2808px that matched a real detection 2028px away -- a different physical
# weld entirely. Capping the gate at UNCERTAINTY_GATE_MAX_PX=300 was tried
# next, but a SECOND concrete incident (same clip, user-reported) showed
# the cap wasn't the fix either: track 64, unconfirmed for 2.2s, matched a
# candidate 245px away at embedding similarity 0.86 (clearing
# MIN_EMBEDDING_SIM_FOR_PROXIMITY_MATCH) -- and that candidate was a
# PERSON'S HEAD in the background, not a weld at all, confirmed by
# directly comparing the two frames' crops. This is the exact failure mode
# DeepSORT/StrongSORT's matching CASCADE exists to prevent: those
# architectures never let appearance similarity justify a match against a
# candidate with zero motion/geometric support -- appearance only picks
# among candidates that ALREADY passed a tight, motion-based gate, it
# never substitutes for one. Any nonzero widening based on "how long
# unconfirmed" repeats the same mistake at a different radius, because
# this classifier's embedding is a GOOD/BAD-classification feature, not a
# re-ID feature (see MIN_EMBEDDING_SIM_FOR_PROXIMITY_MATCH) -- it is not
# discriminative enough to trust past the small, OBJECT-SIZE-based radius
# PROXIMITY_GATE_DIAG_FACTOR already provides. The per-track local motion
# estimate and MAX_HELD_WITHOUT_MATCH_FACTOR's absolute hold cap (both
# added the same day) address the position-uncertainty-growth problem this
# was originally meant to compensate for, from the causes instead.
UNCERTAINTY_GATE_STD_FACTOR = 3.0  # kept for the diagnostic _position_std still logged in match_log/lost_result; NOT used in the gate below
UNCERTAINTY_GATE_MAX_PX = 300.0    # unused -- see docstring above; a cap wasn't the fix, removing the term was

# Minimum RANSAC inlier fraction for a per-track local_affine_transform
# result to actually be trusted over the global fallback (2026-09-11).
# local_affine_transform returning non-None only means it cleared
# LOCAL_MIN_MATCHES -- it says nothing about fit QUALITY, and a crop with
# mostly outlier correspondences can still "succeed" with a low inlier
# ratio. Confirmed concretely on the offset_65s clip's fid=2058 (a frame
# where the global registration ALSO independently measured low
# confidence, 0.17): every one of 11 active tracks' local crops returned a
# "successful" fit, but every single one had confidence 0.13-0.29 -- and
# using them anyway sent boxes in inconsistent, independently-wrong
# directions relative to each other (the user's report: "the boxes
# drift" -- boxes visibly detach from their real object). Below this floor,
# fall back to the single GLOBAL transform instead: even when the global
# fit is ALSO imperfect, it moves every track by the SAME amount, which
# looks like a coherent camera pan rather than each box independently
# drifting off in its own wrong direction -- a real accuracy loss in
# exchange for no longer visibly incoherent motion. Reuses
# MIN_MOTION_CONFIDENCE's threshold value for consistency (same
# underlying question -- "is this RANSAC fit's inlier fraction high enough
# to trust" -- just asked per-track instead of frame-wide), not because
# the two thresholds are mathematically related.
LOCAL_MIN_CONFIDENCE = 0.3

# Per-track Kalman filter (2026-09-08 -- see video_types.py's Track.kf_mean
# docstring for the full history: this replaced both a pure-global-CMC
# composition and a per-track KLT optical-flow refinement, in that order,
# each rejected after concrete measurement). Standard constant-velocity
# model on the box CENTER (4-dim state: cx, cy, vx, vy); box SHAPE (corner
# offsets from center) stays fixed at the last real observation and is
# just translated by the filter's current center estimate -- see
# _predicted_polygon. This is BoT-SORT/StrongSORT's architecture (2026-09-08
# web research): global CMC only ever WARPS the filter's predicted mean/
# covariance (see _kf_warp), it is never itself trusted as the position
# estimate -- the velocity component is fit per-track from that track's
# OWN real re-detections via the Kalman gain, so a track that keeps landing
# offset from its CMC-warped prediction in a consistent direction (this
# footage's corner-post parallax bias) has that bias absorbed into its own
# velocity automatically, with no per-point feature-identity risk the way
# KLT had (confirmed failure mode: KLT's status flag only meant the LK
# iteration numerically converged, not that it converged to the CORRECT
# feature -- this footage's repetitive rivet pattern let points lock onto
# the next rivet over, a 149px single-step jump that passed every KLT
# per-point check).
KF_INITIAL_POS_VAR = 100.0   # (10px)^2 -- a fresh detection's position is exact; small initial uncertainty
KF_INITIAL_VEL_VAR = 400.0   # (20px/step)^2 -- velocity is completely unknown for a brand-new track
KF_PROCESS_VAR_POS = 4.0     # (2px/step)^2 -- process noise added to position every predict step
KF_PROCESS_VAR_VEL = 25.0    # (5px/step)^2 -- keeps velocity able to keep adapting to a persistent per-track bias instead of freezing after a few observations
# (2026-09-08, seventh deep-dive -- user re-reported a BAD-classified weld
# vanishing outright): confirmed on real data that a SINGLE 65px residual
# (this footage's known corner-post CMC bias, not real motion) got read by
# the filter as a genuine ~(-11,+8)px/step velocity, which then kept being
# extrapolated, unconfirmed, for 6+ held frames -- drifting the prediction
# from within 12px of the real (still-static) weld all the way to 200px+
# away, well past any plausible match gate, so the track aged out and
# finalized instead of ever reconnecting. This is the EXACT failure mode
# the pre-CMC per-track least-squares velocity model was rejected for
# (git history, 2026-09-07: "amplify ordinary detection jitter into a
# spurious -398px/sec velocity") -- a Kalman filter is not automatically
# immune to it; it still needs (a) measurement noise large enough to not
# read a single known-noisy residual as confirmed motion, and (b) a way to
# stop trusting an unconfirmed velocity estimate more the longer it goes
# without a real re-detection. Both fixed below.
KF_MEASUREMENT_VAR = 1600.0  # (40px)^2 -- was 64 (8px); this footage's real single-step CMC-warp residual runs 30-90px (see above), not 8px -- treating that scale of mismatch as "real velocity" instead of noise is what caused the runaway
KF_VELOCITY_DECAY = 0.85     # applied to velocity every predict step (matched or held) -- an unconfirmed velocity estimate decays ~85% per step instead of extrapolating forever, so a spurious single-observation velocity fades out over a few held frames rather than compounding into a permanent drift; a velocity CONTINUOUSLY reconfirmed by real matches every step is unaffected in practice (each real match's Kalman update re-asserts it faster than one step's 15% decay removes it)


def _kf_init(polygon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = polygon.mean(axis=0).astype(np.float64)
    mean = np.array([center[0], center[1], 0.0, 0.0])
    cov = np.diag([KF_INITIAL_POS_VAR, KF_INITIAL_POS_VAR, KF_INITIAL_VEL_VAR, KF_INITIAL_VEL_VAR])
    return mean, cov


_KF_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
_KF_Q = np.diag([KF_PROCESS_VAR_POS, KF_PROCESS_VAR_POS, KF_PROCESS_VAR_VEL, KF_PROCESS_VAR_VEL])
_KF_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
_KF_R = np.diag([KF_MEASUREMENT_VAR, KF_MEASUREMENT_VAR])


def _kf_predict_and_warp(mean: np.ndarray, cov: np.ndarray, step_relative_transform: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """One frame-step: constant-velocity predict, then BoT-SORT's GMC warp
    by this step's camera-motion transform (prev sampled frame -> current).
    Position gets the full affine (rotation+scale+translation); velocity,
    being a vector not a location, gets only the rotation+scale (linear)
    part -- translation cannot apply to a vector. Called every frame for
    every active track, matched or held, so this is what a track's display
    position reflects when not reconfirmed this exact frame."""
    mean = _KF_F @ mean
    mean[2:] *= KF_VELOCITY_DECAY  # see KF_VELOCITY_DECAY docstring -- an unconfirmed velocity must fade, not compound
    cov = _KF_F @ cov @ _KF_F.T + _KF_Q
    R = step_relative_transform[:2, :2]
    t = step_relative_transform[:2, 2]
    mean = np.concatenate([R @ mean[:2] + t, R @ mean[2:]])
    G = np.zeros((4, 4))
    G[:2, :2] = R
    G[2:, 2:] = R
    cov = G @ cov @ G.T
    return mean, cov


def _kf_update(mean: np.ndarray, cov: np.ndarray, measured_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Kalman correction from a REAL detection's center."""
    y = measured_center - _KF_H @ mean
    S = _KF_H @ cov @ _KF_H.T + _KF_R
    K = cov @ _KF_H.T @ np.linalg.inv(S)
    mean = mean + K @ y
    cov = (np.eye(4) - K @ _KF_H) @ cov
    return mean, cov


def _iou(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """True rotated-polygon IoU via cv2.rotatedRectangleIntersection, NOT an
    axis-aligned-box approximation. Code review (2026-09-06) confirmed on
    a sample video (via its overlay frames) that it has stretches
    with 20-30 densely-packed, individually-rotated weld/bolt detections
    along a narrow seam -- an axis-aligned-box IoU there systematically
    inflates apparent overlap between two DISTINCT nearby rotated welds
    (the standard AABB-of-rotated-rectangle effect), which can make
    Hungarian steal a track's correct match in favor of a merely-nearby
    unrelated weld, silently corrupting both tracks' identity."""
    rect_a = cv2.minAreaRect(polygon_a.astype(np.float32))
    rect_b = cv2.minAreaRect(polygon_b.astype(np.float32))
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    retval, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if retval == cv2.INTERSECT_NONE or intersection is None or len(intersection) < 3:
        return 0.0
    inter_area = cv2.contourArea(intersection)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return None
    cos = float(np.dot(a, b) / denom)
    # Clamp to [0,1] WITHOUT the +1 offset a naive [-1,1]->[0,1] remap would
    # use: "no correlation" (cos<=0) must contribute similarity 0, not 0.5 --
    # code review (2026-09-06) confirmed the old (cos+1)/2 remap alone made
    # EMBEDDING_WEIGHT*0.5 == MIN_MATCH_SIMILARITY, i.e. two crops with zero
    # meaningful correlation could already clear the match gate on embedding
    # alone. MIN_IOU_FOR_MATCH above is the primary fix; this is defense in
    # depth for the same finding.
    return max(0.0, min(1.0, cos))


def _polygon_diagonal(polygon: np.ndarray) -> float:
    return float(np.linalg.norm(polygon[0] - polygon[2]))


def _position_std(track: Track) -> float:
    """Average per-axis standard deviation of this track's current Kalman
    position estimate (sqrt of the mean of the x/y variance diagonal
    entries) -- the filter's own, continuously-updated measure of how
    confident it is in `track.kf_mean[:2]` right now. See
    UNCERTAINTY_GATE_STD_FACTOR and video_visualize.py's uncertainty
    rendering for the two places this feeds into."""
    return float(np.sqrt(max(0.0, (track.kf_cov[0, 0] + track.kf_cov[1, 1]) / 2.0)))


def _predicted_polygon(track: Track) -> np.ndarray:
    """Where this track's box is EXPECTED to be in the CURRENT frame --
    see video_types.py's Track.kf_mean docstring. `track.kf_mean` must
    already reflect this frame's predict+warp step (done once per track
    per frame, before matching -- see Tracker.update()), so this function
    itself is just applying the filter's current center to the box's
    fixed SHAPE (its corner offsets from center, from the last real
    detection)."""
    shape_offsets = track.last_polygon - track.last_polygon.mean(axis=0)
    return (shape_offsets + track.kf_mean[:2]).astype(np.float32)


def _combined_similarity(track: Track, verdict: ObjectVerdict) -> float:
    predicted = _predicted_polygon(track)
    diag_predicted = _polygon_diagonal(predicted)
    diag_candidate = _polygon_diagonal(verdict.polygon)
    size_ratio = max(diag_predicted, diag_candidate) / max(min(diag_predicted, diag_candidate), 1e-6)
    if size_ratio > SIZE_RATIO_MAX:
        return 0.0  # see SIZE_RATIO_MAX -- almost certainly a detector-side oversized/merged box, not this track

    iou = _iou(predicted, verdict.polygon)
    if iou > MIN_IOU_FOR_MATCH:
        emb_sim = _cosine_sim(track.last_embedding, verdict.embedding)
        if emb_sim is None:
            return iou  # embedding hook unavailable for this model -- fall back to geometry only
        return IOU_WEIGHT * iou + EMBEDDING_WEIGHT * emb_sim

    # No polygon overlap at all even against the CMC-predicted position --
    # only bridge this gap (leftover registration jitter, not real
    # unmodeled motion -- see PROXIMITY_GATE_DIAG_FACTOR) when the candidate
    # is both close by (relative to the objects' own size) AND very
    # confidently the same identity by embedding.
    emb_sim = _cosine_sim(track.last_embedding, verdict.embedding)
    if emb_sim is None or emb_sim < MIN_EMBEDDING_SIM_FOR_PROXIMITY_MATCH:
        return 0.0
    center_a = predicted.mean(axis=0)
    center_b = verdict.polygon.mean(axis=0)
    dist = float(np.linalg.norm(center_a - center_b))
    # Deliberately NOT widened by the track's own uncertainty -- see
    # UNCERTAINTY_GATE_STD_FACTOR's docstring for the two confirmed wrong
    # matches (one 2028px, one onto a person's head) that widening caused.
    # Object-size-based only, matching DeepSORT/StrongSORT's matching
    # cascade: appearance never bridges a gap motion/geometry didn't
    # already make plausible.
    gate = PROXIMITY_GATE_DIAG_FACTOR * (_polygon_diagonal(predicted) + _polygon_diagonal(verdict.polygon)) / 2.0
    if dist > gate:
        return 0.0
    # Scored the SAME way the primary IoU path would weight a real match
    # (EMBEDDING_WEIGHT * emb_sim, zero IoU contribution) -- NOT the raw
    # emb_sim. Deliberately capped below what any genuine IoU-backed match
    # can reach: a zero-overlap candidate must never outscore a candidate
    # actually sitting at the predicted position, however good its
    # embedding looks. Confirmed on real data (2026-09-07, second deep-dive)
    # that returning raw emb_sim let a DIFFERENT weld ~130px away outscore
    # (0.933 raw vs 0.6*0.88+0.4*emb ~0.87 blended) the true continuation
    # sitting almost exactly on the predicted spot, purely because this
    # path wasn't discounted the way a real match's IoU term would be.
    return EMBEDDING_WEIGHT * emb_sim


# Degraded-frame detection (2026-09-07, user-reported "mass drop" during
# camera transitions -- confirmed on real data: per-frame weld count
# crashed from 10 to 1 in a single 0.2s sample step, while a Laplacian
# blur-variance metric checked as an alternative signal was NOT reliably
# correlated -- a densely-detailed area scores high variance regardless of
# focus, so a smooth panel in perfect focus can score just as "blurry" as a
# genuine motion-blurred frame. The per-frame detection COUNT itself is the
# direct symptom (not a content-confounded proxy) and needs no new
# detector/GPU work to compute -- it's already the output of the unchanged
# per-frame cascade this tracker consumes every frame anyway.
#
# Method, per Budigi et al.'s ByteTrack extension and BoT-SORT/OCSORT's own
# camera-motion-compensation literature (2026-09-07 web research): when a
# frame is judged unreliable, DON'T let it consume any track's grace-period
# budget, and don't trust it to seed brand-new tracks -- freeze the clock
# until a frame's count recovers close to the pre-drop baseline, then
# resume normal bookkeeping. This is what the user described as "discard a
# few frames, then re-detect once the transition settles" -- the tracks
# that were already established survive the gap for free; only the
# (unreliable, likely spurious or fragmentary) mid-transition detections
# are prevented from either falsely aging out good tracks or seeding
# throwaway new ones.
# Calibrated by backtest against real footage (2026-09-07): 0.35 only
# caught the single sharpest dip (n_det=1-2) but missed the several-second
# GRADUAL recovery tail that followed (n_det=4-7 while the scene is still
# only partially back in view) -- and that tail, not just the initial
# spike, is what caused the bulk of the mass-finalization event, since ~30
# tracks established pre-transition still can't all find a match once
# recovery is only partial. 0.6 covers the whole recovery tail (21/36
# frames flagged vs. 4/36 at 0.35 on the same window) while only adding a
# small false-positive cost on ordinary footage (3/51 vs. 0/51 frames on an
# unrelated clean 10s window). Going to 0.7 catches marginally more of the
# transition but starts costing more false positives for little extra gain
# -- see this constant's backtest log for the full sweep.
DEGRADED_DROP_RATIO = 0.6  # current n_det must fall below this fraction of the rolling baseline to flag
DEGRADED_MIN_BASELINE = 3.0  # don't evaluate degraded-ness until a real baseline has been established
BASELINE_EMA_ALPHA = 0.3  # baseline only updates on NON-degraded frames -- see docstring: it must hold at the last-known-good level throughout a whole bad patch, not track the patch downward
# Second, independent degraded-frame signal (2026-09-07, added alongside the
# weld-count-crash check once camera_motion.py existed to provide it): if
# EgoMotionEstimator couldn't confidently register this frame against the
# last one (too few ORB matches / RANSAC inliers -- typically severe motion
# blur or a near-featureless view), every track's predicted position this
# frame is built on a stale/unchanged cumulative transform and cannot be
# trusted either. This complements the weld-count signal rather than
# replacing it: a frame can have plausible detection counts but genuinely
# unregistrable motion (or vice versa).
MIN_MOTION_CONFIDENCE = 0.3
# Absolute ceiling on how long a track may go without a REAL match, counted
# through degraded frames too -- unlike missed_count (frozen during a
# degraded frame by design, see above), this one never freezes. Added
# 2026-09-11: on a rougher-motion clip than this degraded-frame mechanism
# was validated against, 58.7% of sampled frames were flagged degraded, and
# missed_count's freeze stacked across many non-contiguous degraded runs --
# not just one long one -- letting tracks survive 4 to 8 REAL seconds past
# their last actual detection (measured on that clip: 48 of its ~60
# finalized tracks exceeded a 3s held gap, several past 7s), vs. the
# intended 1.6s (lost_grace_frames) grace period. That many extra seconds
# of unconfirmed predict+warp steps is what let position uncertainty grow
# large enough to trigger the UNCERTAINTY_GATE_MAX_PX incident above in the
# first place -- capping the HOLD duration directly is the fix for the
# cause, capping the GATE above is the fix for that one consequence.
# 2.5x lost_grace_frames sits comfortably above the longest single
# continuous degraded run measured on that same clip (16 samples = 3.2s),
# so a genuine brief burst -- what degraded-frame handling was designed
# for -- is never cut short by this; only the multi-burst stacking case is.
MAX_HELD_WITHOUT_MATCH_FACTOR = 2.5

# Standard SORT/DeepSORT/ByteTrack/BoT-SORT track state machine (2026-09-11,
# adopted wholesale instead of another ad-hoc threshold -- user's explicit
# direction: stop inventing project-specific patches, use the methodology
# these production trackers already settled on). Every one of that lineage
# distinguishes "Tracked" (confirmed recently -- render it) from "Lost"
# (unconfirmed -- keep the identity alive for possible re-identification,
# but STOP RENDERING it) from "Removed" (exceeded max age -- drop for
# good). This project has the Removed threshold (MAX_HELD_WITHOUT_MATCH_FACTOR
# above) and the Kalman/GMC prediction a Lost track needs to still be
# positioned reasonably WHEN it reconnects.
#
# A Tracked/Lost DISPLAY split (hide the box while merely Lost, DISPLAY_HELD_MAX_FRAMES=2)
# was added here 2026-09-11 after a held box was confirmed drifted completely
# off its weld -- then REVERTED the same day per explicit user direction:
# hiding a track is not what safety-critical MOT/AV perception does when a
# real object goes briefly unconfirmed (occlusion, blur) -- the object stays
# in the world model and keeps being tracked/rendered with the filter's own
# best estimate; it is not deleted from view. Silently hiding it here just
# swaps one visible failure (a wrong position) for a different one (an
# object that should still be visible reads as "lost"), and this project's
# very first reported complaint in this whole investigation was exactly
# that disappearing-then-reappearing behavior. The actual fix for a wrong
# held position is making the position estimate MORE TRUSTWORTHY, not
# hiding it when it might be wrong -- see AMBIGUITY_MARGIN (distractor-aware
# matching) and MIN_IOU_FOR_MATCH's increase, both added the same day,
# which target the actual causes of bad reconnections instead of the
# symptom of showing one.


class Tracker:
    """Stateful multi-object tracker. Call `update()` once per sampled frame
    in timestamp order, then `finalize_all()` once after the last frame."""

    def __init__(self, lost_grace_frames: int):
        self.lost_grace_frames = lost_grace_frames
        self._next_track_id = 0
        # Every accepted match's decision basis (2026-09-11) -- which path
        # (real geometric overlap vs. zero-IoU embedding-only fallback)
        # accepted it, and the actual iou/embedding/distance values, so a
        # later "why did this id claim that detection" question can be
        # answered from THIS run's own numbers instead of a separate replay
        # (RANSAC's per-process randomness in camera_motion.py means a
        # fresh replay's track/bead numbering does NOT reliably line up
        # with a previous run's -- confirmed the hard way 2026-09-11).
        self.match_log: List[dict] = []
        self.active_tracks: Dict[int, Track] = {}
        self.finalized_verdicts: List[TrackVerdict] = []
        # Full Track objects (not just their TrackVerdict), kept for
        # post-processing steps that need per-frame history -- e.g.
        # bead_clustering.py's identity-handoff merge, or overlay
        # re-rendering with merged bead ids.
        self.finalized_tracks: List[Track] = []
        self._baseline_count = 0.0
        self.last_frame_degraded = False  # exposed for logging/diagnostics
        # A "freeze the prediction transform during degraded frames" idea
        # was tried here (2026-09-08) and REVERTED after direct measurement
        # against real footage disproved it: comparing every fid=228 weld's
        # actual re-detected position at fid=246 (the first frame with a
        # healthy weld count after the t=7.6-8.4s blur burst) against (a) the
        # frozen/static prediction vs (b) just trusting each frame's own live
        # CMC transform, the LIVE transform was closer on average (109px vs
        # 153px error, worse for freeze the closer/lower in frame the weld
        # was: 158px vs 56px on the bottom half) -- the camera was still
        # genuinely moving fast throughout the "degraded" stretch, so
        # freezing prediction made held boxes lag further behind the real
        # object than just continuing to apply the (still real, if
        # lower-confidence) live per-frame CMC estimate. Low motion_confidence
        # here means "this frame's transform has more residual error than
        # usual", not "trust zero motion instead" -- the degraded-frame gate
        # (below) already handles the actual risk (freezing grace-period
        # consumption and new-track seeding); prediction itself should keep
        # using the live transform every frame.
        self._prev_cumulative_transform: Optional[np.ndarray] = None  # fallback GMC warp, see update()
        self._prev_gray: Optional[np.ndarray] = None  # previous frame, for local_affine_transform

    def _new_track(self, frame_idx: int, timestamp_sec: float, chunk_id: int,
                    verdict: ObjectVerdict, cumulative_transform: np.ndarray) -> int:
        track = Track(track_id=self._next_track_id)
        self._next_track_id += 1
        track.history.append(TrackedFrame(frame_idx, timestamp_sec, chunk_id, verdict, cumulative_transform))
        track.kf_mean, track.kf_cov = _kf_init(verdict.polygon)
        self.active_tracks[track.track_id] = track
        return track.track_id

    def _finalize(self, track_id: int, timestamp_sec: float) -> None:
        track = self.active_tracks.pop(track_id)
        track.status = 'finalized'
        track.finalized_at_sec = timestamp_sec
        track.verdict = compute_track_verdict(track)
        self.finalized_verdicts.append(track.verdict)
        self.finalized_tracks.append(track)

    def update(self, frame_idx: int, timestamp_sec: float, chunk_id: int,
               weld_verdicts: List[ObjectVerdict], cumulative_transform: np.ndarray,
               frame_bgr: np.ndarray, motion_confidence: float = 1.0,
               ) -> Tuple[List[Tuple[int, ObjectVerdict]], List[Tuple[int, np.ndarray, ObjectVerdict, float]]]:
        """Returns (matched, lost):
        - matched: [(track_id, ObjectVerdict), ...] for every weld matched or
          newly created THIS frame.
        - lost: [(track_id, predicted_polygon, last_verdict, position_std), ...]
          for EVERY track that's still alive (inside MAX_HELD_WITHOUT_MATCH_FACTOR's
          identity-retention window) but wasn't matched this frame -- always
          rendered, matching how SORT/DeepSORT/BoT-SORT-style trackers (and
          AV perception stacks generally) keep a briefly-occluded/unconfirmed
          object in the world model and continue displaying the filter's
          best estimate, rather than removing it from view (a Tracked/Lost
          display split that HID this was tried 2026-09-11 and reverted the
          same day -- see the class docstring's "Tracked/Lost DISPLAY split"
          note for why). `last_verdict` is that track's most recent REAL
          ObjectVerdict (so its known GOOD/BAD still displays while held, not
          a bare placeholder). `position_std` is _position_std(track) --
          carried for diagnostics/matching only, deliberately not rendered
          (tried and rejected 2026-09-09, see video_visualize.py).

        `cumulative_transform`/`motion_confidence` come from
        camera_motion.EgoMotionEstimator.update() for this same frame, used
        as the FALLBACK warp for a track whose own local crop can't support
        a reliable fit (see camera_motion.local_affine_transform's
        docstring for why per-track local is tried first) -- see
        _kf_predict_and_warp for how a transform is applied, and
        MIN_MOTION_CONFIDENCE for how confidence feeds degraded-frame
        detection. `frame_bgr` is this frame's raw image, needed for the
        per-track local motion crop."""
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_cumulative_transform is not None:
            global_step_transform = cumulative_transform @ np.linalg.inv(self._prev_cumulative_transform)
            for track in self.active_tracks.values():
                diag = _polygon_diagonal(track.last_polygon)
                local_result = None
                if self._prev_gray is not None:
                    local_result = local_affine_transform(self._prev_gray, curr_gray, track.kf_mean[:2], diag)
                trust_local = local_result is not None and local_result[1] >= LOCAL_MIN_CONFIDENCE
                step_transform = local_result[0] if trust_local else global_step_transform
                track.kf_mean, track.kf_cov = _kf_predict_and_warp(track.kf_mean, track.kf_cov, step_transform)

        track_ids = list(self.active_tracks.keys())
        n_det, n_trk = len(weld_verdicts), len(track_ids)

        is_degraded = (
            self._baseline_count >= DEGRADED_MIN_BASELINE
            and n_det < DEGRADED_DROP_RATIO * self._baseline_count
        ) or motion_confidence < MIN_MOTION_CONFIDENCE
        self.last_frame_degraded = is_degraded
        if not is_degraded:
            self._baseline_count = (
                (1 - BASELINE_EMA_ALPHA) * self._baseline_count + BASELINE_EMA_ALPHA * n_det
                if self._baseline_count > 0 else float(n_det)
            )

        matched_det = set()
        matched_trk = set()
        frame_result: List[Tuple[int, ObjectVerdict]] = []

        if n_det > 0 and n_trk > 0:
            # Bipartite assignment via Hungarian algorithm: solves the full
            # N-detection x M-track candidate matrix globally (optimal
            # one-to-one pairing) rather than greedily nearest-first, so one
            # frame's several simultaneous welds (see architecture doc
            # section 3) are matched against multiple active tracks
            # correctly as a set. Each detection matches at most one track
            # and vice versa within this call -- it cannot fuse two
            # same-frame detections onto one track or split one track's slot
            # across two; any such upstream duplicate-detection case is
            # expected to already be resolved by TiledDetector's own
            # NMS-dedup before verdicts reach this tracker.
            cost = np.zeros((n_det, n_trk))  # fully overwritten below before use
            sim = np.zeros((n_det, n_trk))
            for i, wv in enumerate(weld_verdicts):
                for j, tid in enumerate(track_ids):
                    s = _combined_similarity(self.active_tracks[tid], wv)
                    sim[i, j] = s
                    cost[i, j] = 1.0 - s
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if sim[i, j] < MIN_MATCH_SIMILARITY:
                    continue  # Hungarian still had to assign *something* here -- reject weak pairs
                if n_trk > 1:
                    # Distractor-aware ratio test -- see AMBIGUITY_MARGIN's
                    # docstring. Hungarian's global optimum can still be a
                    # narrow, ambiguous win; reject rather than guess.
                    second_best = max(sim[i, j2] for j2 in range(n_trk) if j2 != j)
                    if sim[i, j] - second_best < AMBIGUITY_MARGIN:
                        continue
                tid = track_ids[j]
                track = self.active_tracks[tid]
                predicted_before = _predicted_polygon(track)
                match_iou = _iou(predicted_before, weld_verdicts[i].polygon)
                match_emb_sim = _cosine_sim(track.last_embedding, weld_verdicts[i].embedding)
                match_dist = float(np.linalg.norm(predicted_before.mean(axis=0) - weld_verdicts[i].polygon.mean(axis=0)))
                self.match_log.append({
                    'frame_idx': frame_idx, 'track_id': tid,
                    'path': 'primary_iou' if match_iou > MIN_IOU_FOR_MATCH else 'proximity_fallback',
                    'sim': float(sim[i, j]), 'iou': match_iou, 'emb_sim': match_emb_sim,
                    'dist_px': match_dist, 'frames_since_real_match_before': track.frames_since_real_match,
                })
                track.history.append(TrackedFrame(frame_idx, timestamp_sec, chunk_id, weld_verdicts[i], cumulative_transform))
                track.missed_count = 0
                track.frames_since_real_match = 0
                track.status = 'active'
                track.kf_mean, track.kf_cov = _kf_update(track.kf_mean, track.kf_cov, weld_verdicts[i].polygon.mean(axis=0))
                matched_det.add(i)
                matched_trk.add(tid)
                frame_result.append((tid, weld_verdicts[i]))

        # Unmatched detections: every ObjectVerdict reaching this tracker
        # already cleared the detector's verify_conf gate (see module
        # docstring simplification #2), so no extra confidence check is
        # needed here -- any unmatched detection is either a genuinely new
        # weld entering view or the reappearance of a track that was already
        # dropped (which then correctly starts a fresh track_id). EXCEPT
        # during a degraded frame: a handful of detections surviving a
        # transition/blur are unreliable seeds for a brand-new identity --
        # better to wait for the scene to stabilize and let a real,
        # confident re-detection start the track once _baseline_count-based
        # normal operation resumes.
        if not is_degraded:
            for i, wv in enumerate(weld_verdicts):
                if i not in matched_det:
                    new_tid = self._new_track(frame_idx, timestamp_sec, chunk_id, wv, cumulative_transform)
                    frame_result.append((new_tid, wv))

        # Unmatched tracks: bump their miss counter; finalize once the grace
        # period is exceeded so a temporarily-occluded/blurred weld isn't
        # split into two tracks by one bad frame. During a degraded frame,
        # freeze THIS clock entirely (don't increment, don't finalize) --
        # the whole point is that a transition/blur frame's absence of a
        # match says nothing about whether the weld is still there.
        # frames_since_real_match, below, never freezes -- see
        # MAX_HELD_WITHOUT_MATCH_FACTOR's docstring for why missed_count
        # alone isn't sufficient on footage with sustained degraded stretches.
        max_held_without_match = round(self.lost_grace_frames * MAX_HELD_WITHOUT_MATCH_FACTOR)
        lost_result: List[Tuple[int, np.ndarray, ObjectVerdict]] = []
        for tid in track_ids:
            if tid in matched_trk:
                continue
            track = self.active_tracks[tid]
            track.frames_since_real_match += 1
            if not is_degraded:
                track.missed_count += 1
                track.status = 'lost'
                if track.missed_count > self.lost_grace_frames:
                    self._finalize(tid, timestamp_sec)
                    continue
            if track.frames_since_real_match > max_held_without_match:
                self._finalize(tid, timestamp_sec)
                continue
            # Always rendered while alive (see the class docstring above,
            # "Tracked/Lost DISPLAY split" -- tried and reverted 2026-09-11):
            # a held track keeps showing its predicted position and known
            # GOOD/BAD, same as any object that stays in a tracker's world
            # model through a brief occlusion/blur. `position_std` is
            # carried for diagnostics/matching only, never itself drawn.
            lost_result.append((tid, _predicted_polygon(track), track.history[-1].verdict, _position_std(track)))

        self._prev_cumulative_transform = cumulative_transform
        self._prev_gray = curr_gray
        return frame_result, lost_result

    def finalize_all(self, end_timestamp_sec: Optional[float] = None) -> None:
        """`end_timestamp_sec` should be the video's actual last processed
        timestamp (or duration) -- a track still active when the video ends
        was occupying screen space right up to that moment, not just up to
        its own last real match; falls back to each track's own
        last_seen_sec if not given (only correct for a track that happened
        to be matched on the very last frame)."""
        for tid in list(self.active_tracks.keys()):
            track = self.active_tracks[tid]
            self._finalize(tid, end_timestamp_sec if end_timestamp_sec is not None else track.last_seen_sec)
