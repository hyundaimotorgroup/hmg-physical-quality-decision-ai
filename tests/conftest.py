"""Test helpers: CPU-only fakes so pipeline/video tests do not need weights."""
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class _Det:
    polygon: np.ndarray
    confidence: float


from pqdai.core.pipeline import QualityPipeline  # noqa: E402


def box(cx, cy, w=60, h=40):
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                     [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], dtype=np.float32)


class FakeDetector:
    """Returns fixed polygons for every image."""

    def __init__(self, polygons):
        self.polygons = polygons
        self.last_rejected_candidates = []

    def detect(self, image_path):
        return [_Det(polygon=p.copy(), confidence=0.9) for p in self.polygons]


class FakeClassifier:
    """Classifies by mean brightness: dark crop -> BAD, bright crop -> GOOD.
    Embedding is a deterministic 8-d function of the crop so retrieval can be tested."""

    def __init__(self, p_bad_for_dark=0.999, uncertain=False):
        self.uncertain = uncertain
        self.p = p_bad_for_dark

    def _one(self, img):
        dark = float(img.mean()) < 100
        pb = 0.5 if self.uncertain else (self.p if dark else 1 - self.p)
        probs = np.array([pb, 1 - pb], dtype=np.float32)
        emb = np.zeros(8, dtype=np.float32)
        emb[0 if dark else 1] = 1.0
        return {"predicted_class": int(probs.argmax()), "class_name": ["BAD", "GOOD"][int(probs.argmax())],
                "probabilities": probs, "uncertainty": {}, "embedding": emb}

    def predict_with_uncertainty(self, image):
        return self._one(image)

    def predict_batch_with_uncertainty(self, images):
        return [self._one(i) for i in images]


@pytest.fixture
def fakes():
    return FakeDetector, FakeClassifier


@pytest.fixture
def ref_bank():
    emb = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, .1, 0, 0, 0, 0, 0, 0], [1, 0, .1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, .1, 0, 0, 0, 0, 0], [.1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])
    return emb, labels


@pytest.fixture
def synthetic_image(tmp_path):
    """Image with a bright 'weld' at left and a dark 'weld' at right."""
    import cv2
    img = np.full((400, 600, 3), 130, np.uint8)
    img[160:240, 70:130] = 235   # bright patch -> GOOD
    img[160:240, 470:530] = 25   # dark patch   -> BAD
    p = tmp_path / "img.jpg"
    cv2.imwrite(str(p), img)
    return str(p), img


__all__ = ["QualityPipeline", "box", "FakeDetector", "FakeClassifier"]
