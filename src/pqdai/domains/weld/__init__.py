"""Weld domain (first plugin): spot-weld / bead detection and GOOD/BAD judgment."""
import os

from .. import DomainSpec, Thresholds, register
from ...core.preprocessing import apply_brightness_correction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def _weights_dir() -> str:
    env = os.environ.get("PQDAI_WELD_WEIGHTS")
    if env:
        return env
    repo = os.path.join(ROOT, "weights", "weld")            # source checkout
    return repo if os.path.isdir(repo) else os.path.join(HERE, "weights")


def _detector(weights_path, device, thr):
    from ...core.tiled_detector import TiledDetector
    return TiledDetector(weights_path, device=device, conf=thr.conf, verify_conf=thr.verify_conf)


def _classifier(weights_path, device):
    from ...models.crop_classifier import CropClassifier
    return CropClassifier(weights_path, device=device, class_names=["BAD", "GOOD"])


SPEC = register(DomainSpec(
    name="weld",
    description="Weld detection (oriented boxes) and GOOD/BAD quality judgment from photos and video.",
    class_names=["BAD", "GOOD"],
    unsafe_class="BAD",
    detector_factory=_detector,
    classifier_factory=_classifier,
    preprocess_fn=apply_brightness_correction,
    thresholds=Thresholds(conf=0.1, verify_conf=0.2, alpha=0.1, k=3, crop_pad_ratio=0.1),
    weights_dir=_weights_dir(),
    reference_bank_path=os.path.join(HERE, "reference_bank", "reference_embeddings_v4_494.npz"),
    docs="docs/MODEL_CARD.md",
))
