import numpy as np
import pytest

from conftest import FakeClassifier, FakeDetector, QualityPipeline, box
from pqdai.core.obb_crop import extract_obb_crop, order_points
from pqdai.core.preprocessing import apply_brightness_correction, generate_photometric_tta_views
from pqdai.core.retrieval import retrieve_topk_neighbors
from pqdai.core.uncertainty import compute_entropy, compute_margin
from pqdai.core.voting import hard_vote


def test_entropy_bounds():
    assert compute_entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2), abs=1e-6)
    assert compute_entropy(np.array([1.0, 0.0])) < 1e-6
    assert compute_margin(np.array([0.7, 0.3])) == pytest.approx(0.4)


def test_hard_vote_tie_favours_bad():
    assert hard_vote(np.array([0, 1])) == 0
    lab, conf = hard_vote(np.array([1, 1, 0]), return_confidence=True)
    assert lab == 1 and conf == pytest.approx(2 / 3)


def test_retrieval_cosine_topk(ref_bank):
    emb, labels = ref_bank
    idx, scores = retrieve_topk_neighbors(np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32), emb, 3)
    assert set(labels[idx]) == {0} and scores[0] >= scores[-1]


def test_obb_crop_axis_aligned_and_rotated():
    img = np.zeros((200, 300, 3), np.uint8)
    img[50:90, 100:180] = 255
    crop = extract_obb_crop(img, box(140, 70, 80, 40))
    assert crop.shape[:2] == (40, 80) and crop.mean() > 240
    pts = order_points(box(0, 0)[[2, 0, 3, 1]])
    assert np.allclose(pts, box(0, 0))
    assert extract_obb_crop(img, np.zeros((4, 2), np.float32)) is None


def test_brightness_and_tta_views():
    dark = np.full((20, 20, 3), 40, np.uint8)
    assert apply_brightness_correction(dark).mean() > dark.mean()
    views = generate_photometric_tta_views(dark)
    assert len(views) == 5 and all(v.shape == dark.shape for v in views)


def test_pipeline_end_to_end(synthetic_image, ref_bank):
    path, _ = synthetic_image
    emb, labels = ref_bank
    pipe = QualityPipeline(FakeDetector([box(100, 200), box(500, 200)]), FakeClassifier(), emb, labels)
    v = pipe.judge_image(path)
    assert [w.class_name for w in v.objects] == ["GOOD", "BAD"]
    assert v.overall_class_name == "BAD" and v.overall_class == 0
    assert all(not w.used_retrieval for w in v.objects)


def test_pipeline_no_detections_is_good(synthetic_image, ref_bank):
    emb, labels = ref_bank
    v = QualityPipeline(FakeDetector([]), FakeClassifier(), emb, labels).judge_image(synthetic_image[0])
    assert v.objects == [] and v.overall_class_name == "GOOD"


def test_pipeline_uncertain_uses_retrieval(synthetic_image, ref_bank):
    # classifier is 50/50 -> entropy > alpha -> kNN vote against the bank decides
    emb, labels = ref_bank
    pipe = QualityPipeline(FakeDetector([box(100, 200), box(500, 200)]), FakeClassifier(uncertain=True), emb, labels, alpha=0.1, k=3)
    v = pipe.judge_image(synthetic_image[0])
    assert all(w.used_retrieval for w in v.objects)
    assert [w.class_name for w in v.objects] == ["GOOD", "BAD"]
    pipe.use_retrieval = False
    v2 = pipe.judge_image(synthetic_image[0])
    assert all(not w.used_retrieval for w in v2.objects)
