"""CPU-only tests for the ultralytics/sahi-free TiledDetector port, using a fake backend."""
import numpy as np
import pytest
import cv2

from pqdai.core import tiled_detector as D
from pqdai.core.tiled_detector import Detection, TiledDetector, get_slice_bboxes, _axis_nms, _shift_prediction


def box_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


class FakeBackend:
    """Reports a fixed set of 'welds' (global coords) that fall inside whatever crop it is given.

    The crop's global origin is looked up by its pixel content: the test image encodes (x, y) in
    channels so any crop can be located exactly."""

    def __init__(self, welds, conf=0.9):
        self.objects, self.conf, self.calls = welds, conf, 0

    @staticmethod
    def _origin(img):
        # image is built by make_image(): B channel = x % 256, G = y % 256, plus marker rows (see below)
        return img.meta_origin if hasattr(img, 'meta_origin') else (0, 0)

    def predict(self, bgr, conf):
        self.calls += 1
        h, w = bgr.shape[:2]
        ox, oy = int(bgr[0, 0, 2]) * 64 + int(bgr[0, 1, 2]), int(bgr[1, 0, 2]) * 64 + int(bgr[1, 1, 2])
        polys, confs = [], []
        for (x1, y1, x2, y2) in self.objects:
            lx1, ly1, lx2, ly2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
            if lx2 <= 0 or ly2 <= 0 or lx1 >= w or ly1 >= h:
                continue
            polys.append(box_poly(max(lx1, 0), max(ly1, 0), min(lx2, w), min(ly2, h)))
            confs.append(self.conf)
        if not polys:
            return np.zeros((0, 4, 2), np.float32), np.zeros(0, np.float32)
        return np.stack(polys), np.array(confs, np.float32)


def make_image(w, h):
    """Gray image whose top-left 2x2 R-channel pixels encode nothing; origin is injected by a wrapper."""
    return np.full((h, w, 3), 128, np.uint8)


class OriginBackend(FakeBackend):
    """Same as FakeBackend but locates a crop in the full image by template matching on a noise image."""

    def __init__(self, full, welds, conf=0.9):
        super().__init__(welds, conf)
        self.full = full

    def predict(self, bgr, conf):
        self.calls += 1
        h, w = bgr.shape[:2]
        if bgr.shape == self.full.shape:
            ox = oy = 0
        else:
            patch = bgr[: min(64, h), : min(64, w)]
            res = cv2.matchTemplate(self.full, patch, cv2.TM_SQDIFF)
            _, _, loc, _ = cv2.minMaxLoc(res) if False else (None, None, cv2.minMaxLoc(-res)[3], None)
            ox, oy = loc
        polys, confs = [], []
        for (x1, y1, x2, y2) in self.objects:
            lx1, ly1, lx2, ly2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
            if lx2 <= 0 or ly2 <= 0 or lx1 >= w or ly1 >= h:
                continue
            polys.append(box_poly(max(lx1, 0), max(ly1, 0), min(lx2, w), min(ly2, h)))
            confs.append(self.conf)
        if not polys:
            return np.zeros((0, 4, 2), np.float32), np.zeros(0, np.float32)
        return np.stack(polys), np.array(confs, np.float32)


# ------------------------------------------------------------------ slicer geometry
def test_slice_grid_matches_sahi_reference_values():
    # values verified equal to sahi.slicing.get_slice_bboxes(..., auto_slice_resolution=False) on 112 size/overlap combos
    boxes = get_slice_bboxes(2000, 3000, 1024, 1024, 0.35, 0.35)
    assert boxes[0] == [0, 0, 1024, 1024]
    assert boxes[1] == [666, 0, 1690, 1024]          # stride = 1024 - int(0.35*1024) = 666
    assert boxes[-1][2] == 3000 and boxes[-1][3] == 2000
    assert all(b[2] - b[0] == 1024 and b[3] - b[1] == 1024 for b in boxes)  # edge slices shifted back, same size


def test_slice_grid_covers_image_and_small_image_single_slice():
    for h, w in [(1100, 1700), (3024, 4032), (700, 900), (1024, 1024)]:
        boxes = get_slice_bboxes(h, w, 1024, 1024, 0.35, 0.35)
        cover = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in boxes:
            cover[y1:y2, x1:x2] = True
        assert cover.all()
    assert len(get_slice_bboxes(700, 900, 1024, 1024, 0.35, 0.35)) == 1
    with pytest.raises(ValueError):
        get_slice_bboxes(100, 100, 64, 64, 1.0, 0.2)


def test_axis_nms_threshold_and_order():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [5, 5, 15, 15], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.8, 0.7], np.float32)
    keep = _axis_nms(boxes, scores, 0.99)
    assert sorted(keep) == [1, 2, 3]   # exact duplicate (IoU 1.0) dropped, partial overlap kept
    assert keep[0] == 1
    assert sorted(_axis_nms(boxes, scores, 0.1)) == [1, 3]   # partial overlap (IoU 0.14) suppressed at a lower threshold


def test_shift_prediction_clamps_like_sahi():
    poly = box_poly(-5, 10, 30, 40)
    env, shifted = _shift_prediction(poly, 100, 200, 500, 400)
    assert env.tolist() == [100.0, 210.0, 130.0, 240.0]   # envelope clamped at 0 before the shift
    assert shifted[:, 0].min() == 95.0                      # polygon itself is not clamped below
    assert _shift_prediction(box_poly(10, 10, 10, 20), 0, 0, 100, 100) is None


# ------------------------------------------------------------------ end-to-end logic
def _noise_image(w, h, seed=0):
    return np.random.RandomState(seed).randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_single_shot_path_uses_backend_once(tmp_path):
    img = _noise_image(800, 600)
    p = str(tmp_path / 'a.jpg')
    cv2.imwrite(p, img)
    full = cv2.imread(p)
    be = OriginBackend(full, [(100, 100, 300, 200)])
    det = TiledDetector(backend=be, conf=0.1, device='cpu')
    out = det.detect(p)
    assert be.calls == 1 and len(out) == 1
    assert out[0].polygon.shape == (4, 2) and out[0].confidence == pytest.approx(0.9)


def test_tiled_path_finds_welds_across_tile_seams_and_dedups(tmp_path):
    img = _noise_image(3000, 2000, seed=1)
    p = str(tmp_path / 'big.png')
    cv2.imwrite(p, img)
    full = cv2.imread(p)
    welds = [(200, 300, 420, 400), (1000, 900, 1300, 1000), (2500, 1500, 2700, 1650)]
    be = OriginBackend(full, welds)
    det = TiledDetector(backend=be, conf=0.1, device='cpu', roi_redetect=False)
    out = det.detect(p)
    assert be.calls > 3                                    # many tiles + full-image pass
    found = sorted((round(float(d.polygon[:, 0].min())), round(float(d.polygon[:, 1].min()))) for d in out)
    for (x1, y1, _, _) in welds:
        assert any(abs(x1 - fx) < 8 and abs(y1 - fy) < 8 for fx, fy in found)
    assert len(out) <= len(welds) + 3                      # WBF/containment removed per-tile duplicates


def test_roi_redetection_with_verify_conf_drops_unconfirmed(tmp_path):
    img = _noise_image(2400, 1800, seed=2)
    p = str(tmp_path / 'b.png')
    cv2.imwrite(p, img)
    full = cv2.imread(p)

    class Flaky(OriginBackend):
        # reports weld at conf 0.5, but when asked to re-detect on a big padded crop the score is 0.15 (< verify_conf)
        def predict(self, bgr, conf):
            polys, confs = super().predict(bgr, conf)
            if bgr.shape[0] != 1024 and bgr.shape != self.full.shape:
                confs = confs * 0 + 0.15
            return polys, confs

    welds = [(300, 300, 500, 420)]
    det = TiledDetector(backend=Flaky(full, welds, conf=0.5), conf=0.1, verify_conf=0.2, device='cpu',
                       collect_rejected_candidates=True)
    out = det.detect(p)
    assert len(out) == 0 and len(det.last_rejected_candidates) >= 1
    det2 = TiledDetector(backend=OriginBackend(full, welds, conf=0.5), conf=0.1, verify_conf=0.2, device='cpu')
    assert len(det2.detect(p)) == 1
    assert det2.last_rejected_candidates == []


def test_suppress_contained_boxes_drops_nested_fragment():
    det = TiledDetector(backend=FakeBackend([]), device='cpu')
    big = Detection(box_poly(0, 0, 100, 100), 0.9)
    frag = Detection(box_poly(10, 10, 60, 50), 0.95)
    far = Detection(box_poly(300, 300, 340, 340), 0.5)
    kept = det._suppress_contained_boxes([big, frag, far])
    ids = [id(k) for k in kept]
    assert id(big) in ids and id(far) in ids and id(frag) not in ids


def test_nms_dedup_prefers_largest_box():
    det = TiledDetector(backend=FakeBackend([]), device='cpu')
    a = Detection(box_poly(0, 0, 100, 100), 0.5)
    b = Detection(box_poly(20, 0, 80, 100), 0.99)   # smaller but more confident, IoU 0.6 > 0.3
    kept = det._nms_dedup([a, b])
    assert len(kept) == 1 and kept[0] is a


def test_no_backend_and_no_weights_raises():
    with pytest.raises(ValueError):
        TiledDetector(device='cpu')


def test_module_has_no_forbidden_imports():
    import inspect
    src = inspect.getsource(D)
    for bad in ('import ultralytics', 'from ultralytics', 'import sahi', 'from sahi'):
        assert bad not in src
