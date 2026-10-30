import numpy as np
import pytest

from pqdai import domains
from pqdai.core.pipeline import QualityPipeline


def test_builtin_domains_registered():
    names = domains.available()
    assert "weld" in names and "example_template" in names


def test_weld_spec_contract():
    s = domains.get("weld")
    assert list(s.class_names) == ["BAD", "GOOD"] and s.unsafe_class == "BAD"
    assert s.thresholds.alpha == 0.1 and s.thresholds.k == 3
    assert s.reference_bank_path.endswith(".npz")


def test_unknown_domain():
    with pytest.raises(KeyError):
        domains.get("nope")


def test_unsafe_class_must_be_first():
    with pytest.raises(ValueError):
        domains.DomainSpec(name="bad", description="", class_names=["OK", "DEFECT"], unsafe_class="DEFECT",
                           detector_factory=None, classifier_factory=None, preprocess_fn=lambda x: x)


def test_template_plugin_runs_with_stubs(tmp_path):
    import cv2
    s = domains.get("example_template")
    assert list(s.class_names) == ["DEFECT", "OK"]
    det = s.detector_factory(None, None, s.thresholds)
    clf = s.classifier_factory(None, None)
    emb = np.zeros((2, 8), np.float32)
    pipe = QualityPipeline(det, clf, emb, np.array([0, 1]), class_names=list(s.class_names))
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), np.full((64, 64, 3), 100, np.uint8))
    v = pipe.judge_image(str(p))
    assert v.objects == [] and v.overall_class_name == "OK" and v.unsafe_class == "DEFECT"


def test_custom_unsafe_class_propagates(fakes, ref_bank, synthetic_image):
    from conftest import box
    FakeDetector, FakeClassifier = fakes
    emb, labels = ref_bank
    pipe = QualityPipeline(FakeDetector([box(100, 200), box(500, 200)]), FakeClassifier(), emb, labels,
                           class_names=["DEFECT", "OK"])
    v = pipe.judge_image(synthetic_image[0])
    assert [o.class_name for o in v.objects] == ["OK", "DEFECT"]
    assert v.overall_class_name == "DEFECT"
