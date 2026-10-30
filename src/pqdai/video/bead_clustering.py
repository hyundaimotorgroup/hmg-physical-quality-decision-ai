"""Merge same-bead segment-tracks into one persistent "bead" identity.

Root cause this addresses (2026-09-07 deep-dive, confirmed on real frames):
a single continuous weld bead is independently re-segmented into a
DIFFERENT number/arrangement of small boxes by the (unchanged) per-frame
detection cascade -- e.g. 5 short segments this frame, 4 differently-drawn
segments next frame. Neighboring segments of the same bead sit close
together and look nearly identical (position + classifier embedding), so
video_tracker.py's frame-to-frame matching occasionally swaps identity
between them even with correct motion prediction. The result: many SHORT
tracks that are each individually fine (their own GOOD/BAD verdict is
correct) but fragment what a human would call "one weld" into several
report rows with churning IDs.

This module does NOT touch video_tracker.py's frame-to-frame matching --
that logic is already the right tool for "is this the same box as last
frame". Instead it's a POST-PROCESSING step over already-FINALIZED tracks:
group tracks that look like they're segments of the same physical bead that
got a fresh track_id after a brief identity swap, and report them as one
merged "bead" track.

Safety-critical design choice -- why this is NOT the same as
src/detector.py's `merge_collinear` (which defaults OFF there): that
feature merges collinear boxes WITHIN one frame, with no way to tell "two
adjacent segments of one bead" apart from "two separate, genuinely distinct
adjacent welds" other than geometry -- confirmed in this project's own
history to nearly double the missed-weld count by wrongly fusing distinct
welds (see the `merge_collinear` ablation note in core/tiled_detector.py). This module adds a
signal single-frame merging can never have: TIME. Two tracks that were
BOTH visible at the same time for any meaningful duration are essentially
certain to be two different physical objects (we directly observed frames
with 25-30 simultaneous, individually distinct welds) and are never merged
here regardless of how collinear they look. Only tracks whose time windows
are disjoint or barely overlap -- i.e. an identity HANDOFF, not
co-existence -- are even considered, and only then does geometry decide.
"""

from typing import Dict, List, Tuple

import numpy as np

from .video_types import Track, TrackedFrame
from .camera_motion import apply_affine
from .video_tracker import _iou, _cosine_sim  # single source of truth for polygon overlap / embedding similarity

# Ported from src/detector.py's _fragments_mergeable/_obb_axis_info (same
# values, same rationale -- see that module's docstrings). Reused here only
# under the additional temporal-disjointness gate above, which that
# single-frame feature never had available.
ANGLE_TOL_DEG = 15.0
WIDTH_RATIO_MAX = 2.0
GAP_RATIO_MAX = 0.6
PERP_OFFSET_RATIO_MAX = 0.75

# Temporal gate. ANY positive overlap disqualifies a merge -- tightened
# from an original 0.15-of-shorter-track-duration allowance (2026-09-07,
# third deep-dive) after confirming a concrete failure it allowed: two raw
# tracks that overlapped in time by less than 15% still both counted as
# "present" (active or held) during that overlap, so the overlay drew the
# SAME bead id twice in one frame once merged.
#
# A grace-period-sized allowance (`max_overlap_sec > 0`, letting a held
# track and its real-detection successor overlap by up to one full grace
# period) was tried and REJECTED (2026-09-08, fourth deep-dive,
# internal profiling notes vs. actual re-test): backtesting
# it against the same cached footage immediately reintroduced concrete
# duplicate-bead-in-one-frame cases (tracks 128/185 in the 450-660s cache
# window) where the two source tracks had genuinely overlapping REAL
# (non-held) matches for many consecutive frames -- i.e. two distinct,
# simultaneously-visible welds, exactly what this gate exists to protect
# against. Separately, tracing the specific pair the relaxation was meant
# to fix (tracks 55/57 around the t=7.6-8.4s motion-blur burst) showed they
# never actually lost their identity in the first place once
# video_tracker.py's CMC-freeze-during-degraded-frames fix (see that
# module's `_last_trusted_transform`) was in place -- both tracks survived
# the whole burst under their original ids with no re-matching needed. So
# the CMC freeze is the real fix for that failure mode; this gate stays at
# strict disjointness.
DEFAULT_MAX_OVERLAP_SEC = 0.0
# Two disjoint tracks separated by more than this many seconds are treated
# as unrelated -- a genuine identity handoff (segment mis-swap) happens
# within roughly one grace-period cycle, not tens of seconds later.
MAX_TIME_GAP_SEC = 3.0

# Re-identification path (2026-09-07, added after user report that
# previously-tracked welds still visibly got "lost and re-found" as new
# IDs even with CMC-based live tracking): a track that goes quiet for a few
# seconds (grace period exceeded, e.g. through a rough patch of frames) and
# then reappears essentially AT THE SAME PHYSICAL SPOT is a much more
# common case than the narrow "adjacent segment of one continuous bead"
# scenario _collinear_mergeable targets -- but comparing their raw
# (unprojected) polygon coordinates for this is wrong whenever the camera
# moved during the gap, which live tracking's own CMC (camera_motion.py)
# has already been correcting for on every frame the whole time. So: project
# the earlier track's last polygon forward through the SAME accumulated
# camera transform (stored per-observation in TrackedFrame.cumulative_transform)
# to what it would look like at the later track's first frame, THEN compare.
# A high compensated-IoU is strong same-object evidence on its own; combined
# with an embedding check (which the old collinear-only path never had) this
# is both broader (catches non-adjacent re-detections of the exact same
# segment) and safer (rejects a merge the geometry alone might have allowed).
REID_IOU_THRESHOLD = 0.15
REID_MIN_EMBEDDING_SIM = 0.6


def _project_polygon(polygon: np.ndarray, from_transform: np.ndarray, to_transform: np.ndarray) -> np.ndarray:
    """Where `polygon` (observed under `from_transform`, i.e. some earlier
    frame's camera ego-motion state) would appear under `to_transform`
    (some later frame's state) -- the SAME composition
    video_tracker.py::_predicted_polygon uses for live per-frame matching,
    applied here across a much longer, already-elapsed gap between two
    finalized tracks instead of one frame-step."""
    relative = to_transform @ np.linalg.inv(from_transform)
    return apply_affine(relative, polygon)


def _obb_axis_info(polygon: np.ndarray):
    """(center, unit_long_axis, length, width) -- see src/detector.py's
    identical helper for why edges 0/1 are used instead of cv2.minAreaRect's
    angle (consistent perimeter-order corners from this project's OBB
    output, not dependent on OpenCV's angle-convention quirks)."""
    pts = np.asarray(polygon, dtype=np.float64)
    edge0 = pts[1] - pts[0]
    edge1 = pts[2] - pts[1]
    len0, len1 = float(np.linalg.norm(edge0)), float(np.linalg.norm(edge1))
    if len0 >= len1:
        long_vec, length, width = edge0, len0, len1
    else:
        long_vec, length, width = edge1, len1, len0
    long_unit = long_vec / (length + 1e-9)
    center = pts.mean(axis=0)
    return center, long_unit, length, width


def _collinear_mergeable(polygon_a: np.ndarray, polygon_b: np.ndarray) -> bool:
    """True if these two OBBs look like disjoint fragments of the same
    elongated weld bead: near-identical orientation and width, small gap
    along their shared long axis, small perpendicular offset. See module
    docstring for the constants' provenance."""
    center_a, dir_a, len_a, width_a = _obb_axis_info(polygon_a)
    center_b, dir_b, len_b, width_b = _obb_axis_info(polygon_b)

    cos_angle = min(1.0, abs(float(np.dot(dir_a, dir_b))))
    if np.degrees(np.arccos(cos_angle)) > ANGLE_TOL_DEG:
        return False

    if max(width_a, width_b) / max(min(width_a, width_b), 1e-6) > WIDTH_RATIO_MAX:
        return False

    if np.dot(dir_a, dir_b) < 0:
        dir_b = -dir_b
    axis = dir_a + dir_b
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    perp = np.array([-axis[1], axis[0]])

    proj_a = np.asarray(polygon_a, dtype=np.float64) @ axis
    proj_b = np.asarray(polygon_b, dtype=np.float64) @ axis
    gap = max(proj_b.min() - proj_a.max(), proj_a.min() - proj_b.max(), 0.0)
    avg_length = (len_a + len_b) / 2.0
    if gap > GAP_RATIO_MAX * avg_length:
        return False

    perp_offset = abs(float(np.dot(center_a - center_b, perp)))
    avg_width = (width_a + width_b) / 2.0
    if perp_offset > PERP_OFFSET_RATIO_MAX * avg_width:
        return False

    return True


def _can_merge(a: Track, b: Track, max_overlap_sec: float) -> bool:
    # presence_end_sec, NOT last_seen_sec: a track stays visibly "present"
    # (held, see video_tracker.py's `lost` mechanism) for up to a full grace
    # period AFTER its last real match, not just until that last match.
    # Using last_seen_sec alone let a still-being-held track and a
    # genuinely different, already-active new track pass this gate as
    # "disjoint" purely because the OLDER track's last REAL match predated
    # the NEWER track's first real match -- even though the older one was
    # still on screen the whole time the newer one started. Confirmed
    # concretely (2026-09-07, third deep-dive): merging on that basis
    # produced a reported bead whose two source tracks were both visibly
    # on screen at once, duplicating the same id at two positions in one
    # frame.
    a_end, b_end = a.presence_end_sec, b.presence_end_sec
    overlap = min(a_end, b_end) - max(a.first_seen_sec, b.first_seen_sec)

    if overlap > 0:
        if overlap > max_overlap_sec:
            return False  # co-existed longer than one grace period -- almost certainly two distinct welds
    elif -overlap > MAX_TIME_GAP_SEC:
        return False  # too far apart in time to plausibly be one handoff

    if a_end <= b.first_seen_sec:
        earlier, later = a, b
    elif b_end <= a.first_seen_sec:
        earlier, later = b, a
    else:
        earlier, later = (a, b) if a.first_seen_sec <= b.first_seen_sec else (b, a)

    later_first = later.history[0]
    # Camera-motion-compensated: project the earlier track's last known box
    # through the SAME accumulated ego-motion transform used during live
    # tracking, to what it would look like at the moment the later track was
    # first (re)detected -- comparing raw, unprojected pixel coordinates
    # here would be wrong whenever the camera moved during the gap between
    # the two tracks (the common case, not an edge case, on panning
    # footage).
    projected = _project_polygon(
        earlier.last_polygon, earlier.history[-1].cumulative_transform, later_first.cumulative_transform,
    )
    emb_sim = _cosine_sim(earlier.last_embedding, later_first.verdict.embedding)

    # An embedding mismatch is disqualifying on its own, for EITHER merge
    # path below -- geometry alone (collinear or overlapping) was already
    # shown insufficient in isolation (both this project's merge_collinear
    # history, see module docstring, and the fact that many visually near-
    # identical adjacent weld spots exist on this footage).
    if emb_sim is not None and emb_sim < REID_MIN_EMBEDDING_SIM:
        return False

    # Path 1 -- re-identification: reappeared essentially in the same
    # (compensated) spot. Broader than "adjacent bead fragment": also
    # catches the exact same segment being lost for a few seconds (rough
    # detection patch, brief occlusion) and re-detected later.
    if _iou(projected, later_first.verdict.polygon) >= REID_IOU_THRESHOLD:
        return True

    # Path 2 -- adjacent fragment of the same continuous bead (not
    # overlapping even after compensation, but collinear with it).
    return _collinear_mergeable(projected, later_first.verdict.polygon)


def _as_bead(members: List[Track], bead_id: int) -> Track:
    all_frames: List[TrackedFrame] = []
    for t in members:
        all_frames.extend(t.history)
    all_frames.sort(key=lambda tf: tf.timestamp_sec)
    return Track(track_id=bead_id, status='finalized', history=all_frames)


def cluster_tracks_into_beads(
    tracks: List[Track], max_overlap_sec: float = DEFAULT_MAX_OVERLAP_SEC,
) -> Tuple[List[Track], Dict[int, int]]:
    """Groups finalized tracks using `_can_merge`'s temporal + geometric
    gate. `max_overlap_sec` should be the caller's actual lost-grace-period
    duration (see DEFAULT_MAX_OVERLAP_SEC's docstring) -- two tracks may
    overlap in presence by up to that much and still be considered the same
    physical weld recovering from a held/degraded stretch. Returns
    (merged_tracks, track_id_to_bead_id) -- merged tracks are
    plain `Track` objects (bead_id as their track_id, history = union of all
    member tracks' TrackedFrames sorted by time) so
    `track_aggregation.compute_track_verdict` works on them completely
    unchanged. The id map lets callers (e.g. overlay re-rendering) look up
    which bead an original per-frame track_id ended up in.

    Deliberately NOT a plain Union-Find over pairwise `_can_merge` results:
    confirmed on real data (2026-09-07) that transitive closure can chain
    together tracks whose DIRECT pairwise check fails -- e.g. A merges with
    B, B merges with C, so naive Union-Find puts A and C in one group even
    though A and C themselves overlapped in time for 2+ seconds (i.e. were
    two simultaneously-visible, genuinely distinct welds). That is exactly
    the "distinct adjacent welds silently fused together" failure this
    module's whole design is meant to prevent. So a candidate connected
    component is only actually merged if it's a full CLIQUE -- every pair
    inside it individually passes `_can_merge` -- otherwise none of its
    members are merged at all (erring toward under-merging, which only
    costs report tidiness, over over-merging, which risks masking a real
    defect)."""
    n = len(tracks)
    can_merge_matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _can_merge(tracks[i], tracks[j], max_overlap_sec):
                can_merge_matrix[i][j] = can_merge_matrix[j][i] = True

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_matrix[i][j]:
                union(i, j)

    candidate_groups: Dict[int, List[int]] = {}
    for i in range(n):
        candidate_groups.setdefault(find(i), []).append(i)

    merged_tracks: List[Track] = []
    track_id_to_bead_id: Dict[int, int] = {}
    next_bead_id = 0

    for members in candidate_groups.values():
        is_safe_clique = len(members) == 1 or all(
            can_merge_matrix[members[a]][members[b]]
            for a in range(len(members)) for b in range(a + 1, len(members))
        )
        if is_safe_clique:
            groups_to_emit = [members]
        else:
            # Not every pair in this candidate component is directly
            # mergeable -- fall back to reporting each track separately
            # rather than guessing which subset is the "real" clique.
            groups_to_emit = [[idx] for idx in members]

        for group in groups_to_emit:
            for idx in group:
                track_id_to_bead_id[tracks[idx].track_id] = next_bead_id
            merged_tracks.append(_as_bead([tracks[idx] for idx in group], next_bead_id))
            next_bead_id += 1

    return merged_tracks, track_id_to_bead_id
