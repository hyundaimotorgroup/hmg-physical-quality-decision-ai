"""Oriented-box object detector with tiled inference, backend-agnostic.

Tiled (sliced) oriented-box detection around a pluggable network backend: the
tiled WBF cascade, containment suppression, coarse-to-fine ROI re-detection,
``verify_conf`` gating, NMS de-duplication and collinear gap bridging -- so the
surrounding workflow behaves the same.  Only two things were replaced:

* ``model.predict(crop, conf=...)`` -> ``self._predict(crop)``, which calls a
  *backend* object (default :class:`pqdai.models.det_runner.DetRunner`) that maps
  a BGR ndarray to ``(polygons (N,4,2) float32, confidences (N,))``.
* ``sahi.get_sliced_prediction`` -> :meth:`TiledDetector._sliced_predict_candidates`,
  a re-implementation of SAHI's slicing grid, shifted per-tile predictions, the
  extra full-image pass and its plain-NMS merge (match threshold 0.99) -- the
  only SAHI behaviour the original code actually relied on.

Design notes carried over from the original implementation: WBF only fuses
axis-aligned boxes, so the fused boxes are used to decide which raw candidates
are duplicates and the *original* oriented polygon of the closest candidate is
kept; the tiled pass is treated as a location hint and each candidate is
re-detected on a generously padded full-resolution crop to recover weld extent
that no single tile saw (coarse-to-fine, cf. the "clipped object" failure mode
of sliced inference); ``merge_collinear`` is OFF by default because its
thresholds merged genuinely separate adjacent welds.

Detections are returned as pixel-space OBB polygons (4x2), ready for
:func:`pqdai.core.obb_crop.extract_obb_crop`.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

WBF_IOU_THRESHOLD = 0.03  # sweet spot from a threshold sweep on held-out folds
IOS_CONTAINMENT_THRESHOLD = 0.8  # see _suppress_contained_boxes docstring
ROI_REFINE_PADDING_FACTOR = 1.2  # half-crop = candidate size * this factor
ROI_REFINE_MIN_HALF_SIZE = 600  # px; floor so tiny candidates still get a generous crop
ROI_VERIFY_SAME_WELD_IOU = 0.05  # see _refine_with_roi_redetection's verify_conf branch
IOU_DEDUP_THRESHOLD = 0.3  # see _nms_dedup -- looser than IOS_CONTAINMENT_THRESHOLD's near-full-containment check, needed only in verify_conf mode

# _bridge_collinear_gaps thresholds -- see that method's docstring. Unlike
# _merge_collinear_fragments (which requires near-identical orientation and
# geometrically SPLICES the two polygons together, no re-detection involved),
# this targets fragments that visibly bend (e.g. a horizontal top-of-beam weld
# curving into a vertical corner weld) by asking the model itself, on a crop
# tightly centered on the midpoint between them, whether a bridging weld is
# really there -- confirmed on real data to recover a bead a much wider
# "spanning" crop of both fragments together completely missed (the object
# becomes too small/off-center in a crop that large).
GAP_BRIDGE_MAX_CENTER_DIST = 1200  # px; farther apart than this isn't one split bead
GAP_BRIDGE_MIN_HALF_SIZE = 400
GAP_BRIDGE_DIST_FACTOR = 0.55  # crop half-size = center distance * this factor
GAP_BRIDGE_PROJECTION_MARGIN = 0.15  # bridging box's center must project into the middle [margin, 1-margin] of the segment joining the two fragments' centers

# _merge_collinear_fragments thresholds -- see that method's docstring for
# why disjoint (non-overlapping) fragments need a separate pass from
# _suppress_contained_boxes (which only catches containment). Disabled by
# default (TiledDetector's merge_collinear=False) -- these thresholds proved too
# permissive in practice, merging genuinely-separate adjacent welds often
# enough to roughly double the pipeline's missed-weld count; see the
# `merge_collinear` ablation note in TiledDetector.__init__.
FRAGMENT_MERGE_ANGLE_TOL_DEG = 15.0    # max long-axis orientation difference to treat as the same bead
FRAGMENT_MERGE_WIDTH_RATIO = 2.0       # max (larger/smaller) short-side width ratio to still merge
FRAGMENT_MERGE_GAP_RATIO = 0.6         # max axis-aligned gap between fragments, relative to their average length
FRAGMENT_MERGE_PERP_OFFSET_RATIO = 0.75  # max perpendicular center offset, relative to their average width



@dataclass
class Detection:
    polygon: np.ndarray  # (4, 2) pixel coordinates
    confidence: float


SLICE_NMS_IOU = 0.99  # SAHI postprocess_match_threshold used by the original (effectively "dedupe exact copies only")
BATCH_TILES = 8       # tiles of identical shape are pushed through the backend in chunks of this size


def get_slice_bboxes(image_height: int, image_width: int, slice_height: int, slice_width: int,
                     overlap_height_ratio: float, overlap_width_ratio: float) -> List[List[int]]:
    """Slice grid, identical to ``sahi.slicing.get_slice_bboxes`` (fixed slice size).

    Returns ``[x_min, y_min, x_max, y_max]`` per slice; the last row/column is
    shifted back so every slice keeps ``slice_size`` pixels when the image allows it."""
    if overlap_height_ratio >= 1.0 or overlap_width_ratio >= 1.0:
        raise ValueError("Overlap ratio must be less than 1.0")
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    boxes: List[List[int]] = []
    y_max = y_min = 0
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)
                x_min = max(0, x_max - slice_width)
                y_min = max(0, y_max - slice_height)
                boxes.append([x_min, y_min, x_max, y_max])
            else:
                boxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return boxes


def _axis_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Plain greedy NMS on axis-aligned boxes; suppresses IoU > ``iou_thr``.
    Ties on score are broken like SAHI (by box coordinates) for determinism."""
    if len(boxes) == 0:
        return []
    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0], -scores))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: List[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        ix1 = np.maximum(boxes[i, 0], boxes[:, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[:, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = areas[i] + areas - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        suppressed |= iou > iou_thr
    return keep


def _shift_prediction(polygon: np.ndarray, shift_x: int, shift_y: int, full_h: int, full_w: int):
    """Mimic SAHI's OBB handling for one raw prediction, in slice coordinates.

    SAHI keeps the axis-aligned envelope of the polygon (clamped at 0 and at the
    *full-image* size), drops degenerate ones, then shifts both the envelope (used
    for NMS) and the polygon (clipped above at the full-image size) into
    full-image coordinates.  Returns ``(envelope_xyxy, polygon)`` or ``None``."""
    x1, y1 = float(polygon[:, 0].min()), float(polygon[:, 1].min())
    x2, y2 = float(polygon[:, 0].max()), float(polygon[:, 1].max())
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), max(0.0, x2), max(0.0, y2)
    x1, x2 = min(full_w, x1), min(full_w, x2)
    y1, y2 = min(full_h, y1), min(full_h, y2)
    if not (x1 < x2) or not (y1 < y2):
        return None
    env = np.array([x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y], dtype=np.float32)
    poly = np.empty((4, 2), dtype=np.float32)
    poly[:, 0] = np.minimum(polygon[:, 0] + shift_x, full_w)
    poly[:, 1] = np.minimum(polygon[:, 1] + shift_y, full_h)
    return env, poly



class TiledDetector:
    def __init__(self, weights_path: str = None, device: str = None, conf: float = 0.25,
                 tile_threshold: int = 1600, slice_size: int = 1024, overlap: float = 0.35,
                 tta_scales: Tuple[float, ...] = (1.0,), tta_flip: bool = False,
                 always_tile: bool = False, suppress_contained: bool = True,
                 merge_collinear: bool = False, roi_redetect: bool = True,
                 verify_conf: float = None, collect_rejected_candidates: bool = False,
                 backend=None):
        """``backend`` may be any object exposing ``predict(bgr, conf) -> (polygons, confs)``
        (and optionally ``predict_batch(list_of_bgr, conf)``); when omitted a
        :class:`~pqdai.models.det_runner.DetRunner` is built from ``weights_path``."""
        import torch
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conf = conf
        # Opt-in second threshold for _refine_with_roi_redetection: when set, a
        # candidate is DROPPED (not just left at its weak stage-1 box) if the
        # zoomed-in re-detection doesn't reconfirm it at >= verify_conf. Lets
        # `conf` be pushed low for stage-1 candidate recall (extract-all) while
        # still rejecting non-weld noise that a low conf lets through.
        self.verify_conf = verify_conf
        self.tile_threshold = tile_threshold
        self.slice_size = slice_size
        self.overlap = overlap
        self.weights_path = weights_path
        # Ablation switches. `merge_collinear` defaults OFF: its thresholds merged genuinely
        # separate adjacent welds, roughly doubling the missed-weld count on the audit set.
        # `always_tile` contributed ~nothing on its own and costs runtime, so it is off.
        self.always_tile = always_tile
        self.suppress_contained = suppress_contained
        self.merge_collinear = merge_collinear
        self.roi_redetect = roi_redetect
        # Multi-scale + flip TTA for the tiled path only. Off by default; each extra scale/flip
        # is a full additional tiled inference pass.
        self.tta_scales = tuple(tta_scales) if tta_scales else (1.0,)
        self.tta_flip = tta_flip
        # Opt-in hard-negative-mining hook: every stage-1 candidate dropped for failing to
        # reconfirm at >= verify_conf is recorded in last_rejected_candidates.
        self.collect_rejected_candidates = collect_rejected_candidates
        self.last_rejected_candidates: List[Detection] = []

        if backend is None:
            if weights_path is None:
                raise ValueError("either weights_path or backend must be given")
            from pqdai.models.det_runner import DetRunner
            backend = DetRunner(weights_path, device=self.device)
        self.backend = backend

    # ------------------------------------------------------------------ backend glue
    def _predict(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        polygons, confs = self.backend.predict(bgr, self.conf)
        return np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2), np.asarray(confs, dtype=np.float32).reshape(-1)

    def _predict_many(self, images: Sequence[np.ndarray]):
        """Backend inference for same-shaped images, chunked; falls back to a loop."""
        fn = getattr(self.backend, 'predict_batch', None)
        out = []
        for i in range(0, len(images), BATCH_TILES):
            chunk = list(images[i:i + BATCH_TILES])
            res = fn(chunk, self.conf) if fn is not None else [self.backend.predict(im, self.conf) for im in chunk]
            out.extend((np.asarray(p, dtype=np.float32).reshape(-1, 4, 2), np.asarray(c, dtype=np.float32).reshape(-1))
                       for p, c in res)
        return out

    def detect(self, image_path: str) -> List[Detection]:
        # Reset here (not just in _detect_tiled) so a small image that takes the
        # _detect_single_shot path -- which has no verify_conf/ROI-redetection rejection
        # step at all, and so never touches this list -- doesn't inherit stale entries
        # left over from whichever earlier image last went through _detect_tiled.
        self.last_rejected_candidates = []
        with Image.open(image_path) as im:
            w, h = im.size

        if self.always_tile or max(w, h) > self.tile_threshold:
            return self._detect_tiled(image_path)
        return self._detect_single_shot(image_path)

    def _detect_single_shot(self, image_path: str) -> List[Detection]:
        image = cv2.imread(image_path)  # cv2 applies EXIF rotation, like the rest of the pipeline
        if image is None:
            return []
        polygons, confs = self._predict(image)
        return [Detection(polygon=p, confidence=float(c)) for p, c in zip(polygons, confs)]

    def _sliced_predict_candidates(
        self, img, orig_w: int, orig_h: int, mirrored: bool,
    ) -> List[Tuple[np.ndarray, float]]:
        """One tiled pass over `img` (a possibly resized/mirrored PIL copy of the
        original image), with each resulting polygon mapped back into the ORIGINAL
        image's pixel space so multiple TTA passes can be pooled directly.

        Reproduces SAHI's ``get_sliced_prediction(..., postprocess_type="NMS",
        postprocess_match_metric="IOU", postprocess_match_threshold=0.99,
        force_postprocess_type=True)``: per-slice detections shifted to full-image
        coordinates, plus one extra full-image pass when there is more than one slice,
        then plain NMS at IoU 0.99 -- i.e. WBF later sees the full raw candidate pool."""
        rgb = np.asarray(img.convert('RGB'))
        full_h, full_w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        slices = get_slice_bboxes(full_h, full_w, self.slice_size, self.slice_size, self.overlap, self.overlap)

        jobs = [(np.ascontiguousarray(bgr[y1:y2, x1:x2]), x1, y1) for x1, y1, x2, y2 in slices]
        if len(slices) > 1:
            jobs.append((bgr, 0, 0))  # perform_standard_pred=True

        results = [None] * len(jobs)
        by_shape = {}
        for idx, (im, _, _) in enumerate(jobs):
            by_shape.setdefault(im.shape, []).append(idx)
        for idxs in by_shape.values():
            for idx, res in zip(idxs, self._predict_many([jobs[i][0] for i in idxs])):
                results[idx] = res

        envs: List[np.ndarray] = []
        polys: List[np.ndarray] = []
        scores: List[float] = []
        for (im, sx, sy), (p_arr, c_arr) in zip(jobs, results):
            for polygon, conf in zip(p_arr, c_arr):
                shifted = _shift_prediction(polygon, sx, sy, full_h, full_w)
                if shifted is None:
                    continue
                envs.append(shifted[0])
                polys.append(shifted[1])
                scores.append(float(conf))

        if len(envs) > 1:
            keep = _axis_nms(np.stack(envs), np.asarray(scores, dtype=np.float32), SLICE_NMS_IOU)
        else:
            keep = list(range(len(envs)))

        pass_w, pass_h = img.size
        candidates: List[Tuple[np.ndarray, float]] = []
        for k in keep:
            polygon = polys[k].copy()
            if mirrored:
                polygon[:, 0] = pass_w - polygon[:, 0]
            polygon[:, 0] *= orig_w / pass_w
            polygon[:, 1] *= orig_h / pass_h
            candidates.append((polygon, scores[k]))
        return candidates

    def _detect_tiled(self, image_path: str) -> List[Detection]:
        import PIL.ImageOps
        self.last_rejected_candidates = []
        # EXIF orientation fix (2026-08-28): phone photos routinely carry an
        # EXIF Orientation tag (e.g. 6 = rotate 90) that leaves the raw pixel
        # buffer in landscape while the photo displays as portrait. cv2.imread
        # (used everywhere else -- pipeline.py's crop extraction, this
        # project's own overlay drawing and the detector backend's image loading,
        # confirmed via the reported image shape) auto-
        # applies this rotation, but a plain Image.open().convert('RGB') does
        # not. Without exif_transpose here, every polygon this method returns
        # was computed in the WRONG (un-rotated) coordinate space -- looked
        # fine on paper (valid pixel coordinates) but named the wrong crop
        # when pipeline.py fed it to cv2.imread's rotated array, and drew in
        # the wrong place on any rotated overlay/viewer. Confirmed on real
        # data: a rotated photo's boxes lined up perfectly on the raw
        # (un-rotated) buffer but were scattered nonsensically once rendered
        # against the EXIF-corrected orientation everything else uses.
        with Image.open(image_path) as im:
            base_img = PIL.ImageOps.exif_transpose(im).convert('RGB')
        img_w, img_h = base_img.size

        candidates: List[Tuple[np.ndarray, float]] = []
        for scale in self.tta_scales:
            scaled_img = base_img if abs(scale - 1.0) < 1e-6 else base_img.resize(
                (max(1, round(img_w * scale)), max(1, round(img_h * scale)))
            )
            candidates += self._sliced_predict_candidates(scaled_img, img_w, img_h, mirrored=False)
        if self.tta_flip:
            candidates += self._sliced_predict_candidates(
                PIL.ImageOps.mirror(base_img), img_w, img_h, mirrored=True
            )

        if not candidates:
            return []
        merged = self._wbf_merge(candidates, img_w, img_h)
        if self.suppress_contained:
            merged = self._suppress_contained_boxes(merged)
        # Elongated welds split across a tile boundary into two DISJOINT
        # fragments (not nested, so _suppress_contained_boxes can't catch
        # them) need to be unified before refinement -- otherwise refinement
        # completes each fragment independently and classification later
        # judges the same physical weld as two separate verdicts. See
        # _merge_collinear_fragments docstring.
        if self.merge_collinear:
            merged = self._merge_collinear_fragments(merged)
        if not self.roi_redetect:
            return merged
        refined = self._refine_with_roi_redetection(image_path, merged, img_w, img_h)
        # Two independently-fragmented stage-1 candidates can converge onto
        # the same true weld once refined (each ends up with a near-identical
        # complete box) -- run containment suppression again to collapse them.
        if self.suppress_contained:
            refined = self._suppress_contained_boxes(refined)
        # Defensive re-run: refinement can occasionally leave a still-disjoint
        # pair (e.g. a fragment whose ROI redetection found nothing and fell
        # back to its stage-1 box) that the first merge pass, run before
        # refinement, never saw in its final form.
        if self.merge_collinear:
            refined = self._merge_collinear_fragments(refined)
        if self.verify_conf is not None:
            # verify_conf's "keep every qualifying redetection" mode
            # (_refine_with_roi_redetection) can leave partial-overlap
            # duplicates that IOS_CONTAINMENT_THRESHOLD's stricter near-full-
            # containment check misses -- clean those up before pairing
            # candidates for gap-bridging, or the O(n^2) pairwise scan there
            # multiplies duplicate-of-duplicate "bridges" across every
            # leftover near-duplicate.
            refined = self._nms_dedup(refined)
            refined = self._bridge_collinear_gaps(image_path, refined, img_w, img_h)
            refined = self._nms_dedup(refined)
        return refined

    @staticmethod
    def _polygon_to_axis_box(polygon: np.ndarray) -> np.ndarray:
        return np.array([
            float(polygon[:, 0].min()), float(polygon[:, 1].min()),
            float(polygon[:, 0].max()), float(polygon[:, 1].max()),
        ])

    @staticmethod
    def _axis_aligned_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _axis_aligned_ios(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Intersection over Smaller-area -- the usual metric for
        sliced-inference duplicate merging (get_sliced_prediction defaults to
        postprocess_match_metric="IOS"), specifically because IoU under-values
        a small tile-boundary fragment box that sits mostly *inside* a much
        larger full-object box from an adjacent tile: the union is dominated
        by the large box, so IoU stays low even when the fragment is 100%
        contained. IoS divides by the smaller box's area instead, so a fully
        contained fragment scores ~1.0 regardless of the size mismatch."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller = min(area_a, area_b)
        return inter / smaller if smaller > 0 else 0.0

    def _suppress_contained_boxes(self, detections: List[Detection]) -> List[Detection]:
        """Drop small boxes that sit almost entirely inside a larger one --
        a leftover fragment/duplicate WBF's IoU-based matching (iou_thr=0.03)
        can still miss when the size mismatch is large enough to keep IoU low
        even at full containment (see _axis_aligned_ios docstring). Slicing libraries can't
        do this step for us: it forces plain NMS for OBB models regardless of
        postprocess_type, and even that only ever matches on axis-aligned
        envelopes, never true rotated-polygon overlap (confirmed by reading
        the slicing library's postprocess code) -- so this
        gap exists in sliced-inference merging itself, not just in our WBF pass."""
        if len(detections) <= 1:
            return detections

        boxes = []
        for det in detections:
            x1, y1 = float(det.polygon[:, 0].min()), float(det.polygon[:, 1].min())
            x2, y2 = float(det.polygon[:, 0].max()), float(det.polygon[:, 1].max())
            boxes.append(np.array([x1, y1, x2, y2]))

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        suppressed = [False] * len(detections)
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if suppressed[i] or suppressed[j]:
                    continue
                if self._axis_aligned_ios(boxes[i], boxes[j]) < IOS_CONTAINMENT_THRESHOLD:
                    continue
                # (near-)nested pair -- drop the smaller box; on a rare exact
                # area tie, drop the lower-confidence one instead.
                if areas[i] != areas[j]:
                    suppressed[i if areas[i] < areas[j] else j] = True
                else:
                    suppressed[j if detections[i].confidence >= detections[j].confidence else i] = True

        return [det for det, sup in zip(detections, suppressed) if not sup]

    @staticmethod
    def _obb_axis_info(polygon: np.ndarray):
        """Return (center, unit_long_axis, length, width) for a 4-point OBB
        polygon, derived from the polygon's own edges (not cv2.minAreaRect's
        angle, whose convention/range differs across OpenCV versions) --
        edges 0 and 1 of a quadrilateral are always its two independent side
        directions, and the model/WBF always hand us the 4 corners in
        consistent perimeter order."""
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

    @classmethod
    def _fragments_mergeable(cls, polygon_a: np.ndarray, polygon_b: np.ndarray) -> bool:
        """True if the two OBBs look like disjoint fragments of the same
        elongated weld bead: near-identical orientation and width, small gap
        between them along their shared long axis, and small perpendicular
        offset (so two separate parallel welds sitting side by side are NOT
        merged -- only fragments that continue the same line are)."""
        center_a, dir_a, len_a, width_a = cls._obb_axis_info(polygon_a)
        center_b, dir_b, len_b, width_b = cls._obb_axis_info(polygon_b)

        # A rectangle's long axis has no inherent direction (dir and -dir are
        # the same axis), so compare orientation via the absolute dot product.
        cos_angle = min(1.0, abs(float(np.dot(dir_a, dir_b))))
        if np.degrees(np.arccos(cos_angle)) > FRAGMENT_MERGE_ANGLE_TOL_DEG:
            return False

        if max(width_a, width_b) / max(min(width_a, width_b), 1e-6) > FRAGMENT_MERGE_WIDTH_RATIO:
            return False

        # Align both directions to the same hemisphere, then average into one
        # shared axis to project both polygons' corners onto.
        if np.dot(dir_a, dir_b) < 0:
            dir_b = -dir_b
        axis = dir_a + dir_b
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        perp = np.array([-axis[1], axis[0]])

        proj_a = np.asarray(polygon_a, dtype=np.float64) @ axis
        proj_b = np.asarray(polygon_b, dtype=np.float64) @ axis
        gap = max(proj_b.min() - proj_a.max(), proj_a.min() - proj_b.max(), 0.0)
        avg_length = (len_a + len_b) / 2.0
        if gap > FRAGMENT_MERGE_GAP_RATIO * avg_length:
            return False

        perp_offset = abs(float(np.dot(center_a - center_b, perp)))
        avg_width = (width_a + width_b) / 2.0
        if perp_offset > FRAGMENT_MERGE_PERP_OFFSET_RATIO * avg_width:
            return False

        return True

    @staticmethod
    def _merge_obb_pair(polygon_a: np.ndarray, polygon_b: np.ndarray) -> np.ndarray:
        """Enclosing oriented rectangle over both fragments' corner points --
        the correct combined extent for two confirmed fragments of one bead
        (unlike WBF's axis-aligned fusion, this stays a valid rotated box)."""
        points = np.vstack([polygon_a, polygon_b]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        return cv2.boxPoints(rect)

    def _merge_collinear_fragments(self, detections: List[Detection]) -> List[Detection]:
        """Unify disjoint (non-overlapping) OBB fragments that are actually
        the same physical, elongated weld bead split apart by tile
        boundaries -- a distinct failure mode from the duplicate/contained
        fragments _suppress_contained_boxes handles, because two fragments
        of a long bead split end-to-end typically DON'T overlap at all (one
        covers the top half, the other the bottom half), so IoU/IoS-based
        dedup never fires for them. Without this, a single physical weld
        stays two separately-classified boxes, and it's common for only one
        of the two to be judged BAD -- i.e. exactly the "long weld split into
        several boxes, some GOOD some BAD" bug this fixes.

        Runs a fixed-point pairwise pass (merge, restart, repeat) so bead
        splits into 3+ fragments -- rare, but possible on a very long bead
        crossing two tile seams -- still collapse into one box."""
        if len(detections) <= 1:
            return detections

        dets = list(detections)
        merged_any = True
        while merged_any and len(dets) > 1:
            merged_any = False
            for i in range(len(dets)):
                if merged_any:
                    break
                for j in range(i + 1, len(dets)):
                    if not self._fragments_mergeable(dets[i].polygon, dets[j].polygon):
                        continue
                    new_polygon = self._merge_obb_pair(dets[i].polygon, dets[j].polygon)
                    new_confidence = max(dets[i].confidence, dets[j].confidence)
                    dets = [d for k, d in enumerate(dets) if k not in (i, j)]
                    dets.append(Detection(polygon=new_polygon, confidence=new_confidence))
                    merged_any = True
                    break
        return dets

    def _refine_with_roi_redetection(
        self, image_path: str, detections: List[Detection], img_w: int, img_h: int,
    ) -> List[Detection]:
        """Coarse-to-fine second pass: treat each stage-1 (tiled) detection as
        a location hint only, then re-run the detector once on a single
        generously-padded crop taken directly from the original full-res
        image, sized comfortably larger than the candidate so the true weld
        extent -- whatever it is -- fits entirely inside one clean crop with
        no tile seam anywhere near it. See module docstring for why this is
        needed (WBF/IoS cleanup can't recover extent no tile ever saw) and
        the literature this follows (coarse-to-fine / cascade re-detection).

        When `self.verify_conf` is set, this pass doubles as the "is this
        actually a weld" re-judgment stage of the extract-all-candidates ->
        verify -> classify cascade: instead of keeping only the single
        redetection closest to the original candidate's center (the old
        behavior below), EVERY redetection in the crop that clears
        verify_conf is kept. This matters because a padded ROI around one
        weld routinely contains a second, closely-packed weld too (confirmed
        on real data: two adjacent beads a few cm apart) -- the old
        nearest-to-center pick silently discarded the second one, and could
        even discard the RIGHT box in favor of a lower-confidence duplicate
        fragment that happened to sit closer to the target point. Any
        resulting duplicates (the same weld surfacing from two neighboring
        stage-1 candidates' overlapping crops) are collapsed by the
        containment-suppression pass _detect_tiled runs right after this
        method. Leave `verify_conf` unset to keep the old single-nearest-
        match-or-keep-stage1-box behavior exactly as before.
        """
        if not detections:
            return detections

        image = cv2.imread(image_path)
        if image is None:
            return detections

        refined = []
        for det in detections:
            x1, y1 = float(det.polygon[:, 0].min()), float(det.polygon[:, 1].min())
            x2, y2 = float(det.polygon[:, 0].max()), float(det.polygon[:, 1].max())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w = max((x2 - x1) * ROI_REFINE_PADDING_FACTOR, ROI_REFINE_MIN_HALF_SIZE)
            half_h = max((y2 - y1) * ROI_REFINE_PADDING_FACTOR, ROI_REFINE_MIN_HALF_SIZE)

            crop_x1 = max(0, int(cx - half_w))
            crop_y1 = max(0, int(cy - half_h))
            crop_x2 = min(img_w, int(cx + half_w))
            crop_y2 = min(img_h, int(cy + half_h))
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                if self.verify_conf is None:
                    refined.append(det)
                elif self.collect_rejected_candidates:
                    self.last_rejected_candidates.append(det)
                continue

            polygons, confs = self._predict(crop)
            if len(polygons) == 0:
                if self.verify_conf is None:
                    refined.append(det)  # nothing found in the crop -- keep stage-1 box
                elif self.collect_rejected_candidates:
                    self.last_rejected_candidates.append(det)
                continue

            if self.verify_conf is not None:
                qualifying = [i for i in range(len(polygons)) if confs[i] >= self.verify_conf]
                if not qualifying:
                    if self.collect_rejected_candidates:
                        self.last_rejected_candidates.append(det)
                    continue  # nothing reconfirmed -- drop this candidate

                # Primary: nearest-to-target among only the QUALIFYING
                # redetections (fixes a real bug -- picking nearest among ALL
                # redetections could land on a low-confidence duplicate
                # fragment that merely happened to sit closer to the target
                # point than the correct, higher-confidence box).
                target = np.array([cx - crop_x1, cy - crop_y1])
                primary_idx = min(
                    qualifying, key=lambda i: np.sum((polygons[i].mean(axis=0) - target) ** 2),
                )
                accepted = [primary_idx]

                # Extras: other qualifying redetections in this same crop are
                # only kept as a SEPARATE weld if they barely overlap the
                # primary box -- i.e. this padded crop happened to also frame
                # a second, closely-packed weld (see docstring). A qualifying
                # redetection that substantially overlaps the primary is
                # almost always the same physical weld re-detected as a
                # slightly different/partial box (confirmed on real data:
                # such pairs can land on opposite sides of the classifier's
                # decision boundary), so it's dropped here rather than kept
                # as a spurious duplicate verdict.
                primary_box = self._polygon_to_axis_box(polygons[primary_idx])
                for i in qualifying:
                    if i == primary_idx:
                        continue
                    if self._axis_aligned_iou(primary_box, self._polygon_to_axis_box(polygons[i])) < ROI_VERIFY_SAME_WELD_IOU:
                        accepted.append(i)

                for i in accepted:
                    global_polygon = polygons[i] + np.array([crop_x1, crop_y1], dtype=np.float32)
                    refined.append(Detection(polygon=global_polygon, confidence=float(confs[i])))
                continue

            # Old behavior: the padded crop may catch a neighboring weld too;
            # pick whichever single re-detection is closest to our original
            # candidate's center, not just the highest-confidence one in the crop.
            target = np.array([cx - crop_x1, cy - crop_y1])
            best_idx = min(
                range(len(polygons)),
                key=lambda i: np.sum((polygons[i].mean(axis=0) - target) ** 2),
            )
            polygon = polygons[best_idx] + np.array([crop_x1, crop_y1], dtype=np.float32)
            refined.append(Detection(polygon=polygon, confidence=float(confs[best_idx])))

        return refined

    def _nms_dedup(self, detections: List[Detection]) -> List[Detection]:
        """Greedy IoU-based NMS, LARGEST BOX wins (not highest confidence).
        Needed only in verify_conf mode: _refine_with_roi_redetection's "keep
        every qualifying redetection" branch can leave two partial/offset
        boxes of the SAME weld (confirmed on real data: IoU ~0.15-0.46) that
        _suppress_contained_boxes's near-full-containment IoS>=0.8 check
        doesn't catch -- left uncaught, the classifier can disagree between
        the two overlapping crops and flip a genuinely-GOOD weld's image
        verdict to BAD via the lower-quality duplicate.

        Tie-break is area, matching _suppress_contained_boxes's own
        rationale, and NOT confidence: confirmed on real data that a tight
        crop over only PART of a long weld can score a HIGHER confidence
        than the correct, full-extent redetection of the same weld (a
        cleaner sub-view apparently looks more confident to the model than
        the complete, more visually varied bead) -- picking by confidence
        there systematically under-sizes the box, which is worse than this
        pipeline's real goal of never missing/under-covering a real weld."""
        if len(detections) <= 1:
            return detections
        boxes = [self._polygon_to_axis_box(d.polygon) for d in detections]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = sorted(range(len(detections)), key=lambda i: areas[i], reverse=True)
        suppressed = [False] * len(detections)
        for a, i in enumerate(order):
            if suppressed[i]:
                continue
            for j in order[a + 1:]:
                if suppressed[j]:
                    continue
                if self._axis_aligned_iou(boxes[i], boxes[j]) > IOU_DEDUP_THRESHOLD:
                    suppressed[j] = True
        return [d for k, d in enumerate(detections) if not suppressed[k]]

    def _bridge_collinear_gaps(
        self, image_path: str, detections: List[Detection], img_w: int, img_h: int,
    ) -> List[Detection]:
        """Recover a real coverage hole between two verified-but-separate
        detections that are actually the same physical weld bead split by
        the tiled pass (confirmed on real data: a corner weld that bends from
        horizontal to vertical around a bolt hole -- the interruption is
        enough that neither fragment's own ROI-redetection crop, nor one
        large crop spanning both fragments together, ever finds the missing
        middle third; a fresh crop tightly centered on the MIDPOINT between
        the two fragments' centers does, at good confidence). Deliberately
        does not require matching orientation the way
        `_fragments_mergeable`/_merge_collinear_fragments does (a bend like
        this fails that check) -- instead it trusts the model's own
        redetection on a tightly-framed, well-centered crop as the judge of
        whether a bridging weld is really there, which is a much stronger
        signal against accidentally splicing two unrelated nearby welds
        together than geometry alone.

        Adds the bridging box as an independent new detection (with its own
        classification later) rather than merging/removing the original two
        -- the two fragments are still each individually valid, verified
        detections."""
        if len(detections) <= 1:
            return detections

        image = cv2.imread(image_path)
        if image is None:
            return detections

        additions = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                box_i = self._polygon_to_axis_box(detections[i].polygon)
                box_j = self._polygon_to_axis_box(detections[j].polygon)
                if self._axis_aligned_iou(box_i, box_j) > ROI_VERIFY_SAME_WELD_IOU:
                    continue  # already overlapping -- not a gap, handled elsewhere

                center_i = np.array([(box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2])
                center_j = np.array([(box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2])
                dist = float(np.linalg.norm(center_i - center_j))
                if dist > GAP_BRIDGE_MAX_CENTER_DIST:
                    continue

                mid = (center_i + center_j) / 2
                half = max(dist * GAP_BRIDGE_DIST_FACTOR, GAP_BRIDGE_MIN_HALF_SIZE)
                crop_x1, crop_y1 = max(0, int(mid[0] - half)), max(0, int(mid[1] - half))
                crop_x2, crop_y2 = min(img_w, int(mid[0] + half)), min(img_h, int(mid[1] + half))
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue

                polygons, confs = self._predict(crop)
                if len(polygons) == 0:
                    continue
                axis_vec = center_j - center_i
                axis_len_sq = float(np.dot(axis_vec, axis_vec))
                best = None
                for polygon, conf in zip(polygons, confs):
                    if conf < self.verify_conf:
                        continue
                    global_center = polygon.mean(axis=0) + np.array([crop_x1, crop_y1])
                    t = float(np.dot(global_center - center_i, axis_vec)) / axis_len_sq
                    if not (GAP_BRIDGE_PROJECTION_MARGIN < t < 1 - GAP_BRIDGE_PROJECTION_MARGIN):
                        continue  # re-detection is centered near one of the fragments, not the gap
                    if best is None or conf > best[1]:
                        best = (polygon, conf)

                if best is not None:
                    polygon, conf = best
                    global_polygon = polygon + np.array([crop_x1, crop_y1], dtype=np.float32)
                    additions.append(Detection(polygon=global_polygon, confidence=float(conf)))

        return detections + additions

    def _wbf_merge(self, candidates: List[Tuple[np.ndarray, float]], img_w: int, img_h: int) -> List[Detection]:
        from ensemble_boxes import weighted_boxes_fusion

        axis_boxes_norm = []
        axis_boxes_px = []
        for polygon, _ in candidates:
            x1, y1 = float(polygon[:, 0].min()), float(polygon[:, 1].min())
            x2, y2 = float(polygon[:, 0].max()), float(polygon[:, 1].max())
            axis_boxes_px.append(np.array([x1, y1, x2, y2]))
            axis_boxes_norm.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])

        confs = [conf for _, conf in candidates]
        labels = [0] * len(candidates)
        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            [axis_boxes_norm], [confs], [labels], weights=None,
            iou_thr=WBF_IOU_THRESHOLD, skip_box_thr=0.0,
        )

        detections = []
        for (fx1, fy1, fx2, fy2), fscore in zip(fused_boxes, fused_scores):
            fused_box_px = np.array([fx1 * img_w, fy1 * img_h, fx2 * img_w, fy2 * img_h])
            best_iou, best_idx = -1.0, 0
            for idx, box_px in enumerate(axis_boxes_px):
                iou = self._axis_aligned_iou(fused_box_px, box_px)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            # Borrow the nearest original candidate's oriented polygon -- WBF
            # fuses axis-aligned boxes/scores, but a rotated crop needs a real
            # oriented quadrilateral, not a geometric average of several.
            polygon = candidates[best_idx][0]
            detections.append(Detection(polygon=polygon, confidence=float(fscore)))
        return detections
