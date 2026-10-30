"""Template for a new domain -- copy this folder to ``pqdai/domains/<your_domain>/``.

It shows the full interface with *stub* models (it ships no weights, so it is not functional
on real images).  Steps for a real domain:

1. Train (or bring) an oriented-box detector and a crop classifier whose weights follow the
   release format (``*.safetensors`` + ``*.manifest.json``, see docs/DOMAIN_PLUGINS.md), or write
   factories that wrap your own models -- the core only needs ``detector.detect(path)`` and
   ``classifier.predict_batch_with_uncertainty(crops)``.
2. Fill in :class:`DomainSpec` below (class names, unsafe class, thresholds, preprocessing).
3. Build a reference bank with ``pqdai bank build --domain <name> --crops DIR --out bank.npz``.
4. Add ``<name>`` to ``BUILTIN`` in ``pqdai/domains/__init__.py`` or expose an entry point in the
   ``pqdai.domains`` group.
"""
import numpy as np

from .. import DomainSpec, Thresholds, register


class StubDetector:
    last_rejected_candidates: list = []

    def detect(self, image_path):
        return []


class StubClassifier:
    def predict_batch_with_uncertainty(self, images):
        return [{"predicted_class": 1, "class_name": "OK", "probabilities": np.array([0.01, 0.99], np.float32),
                 "uncertainty": {}, "embedding": np.zeros(8, np.float32)} for _ in images]

    def predict_with_uncertainty(self, image):
        return self.predict_batch_with_uncertainty([image])[0]


SPEC = register(DomainSpec(
    name="example_template",
    description="Documentation-only template domain with stub models (no weights).",
    class_names=["DEFECT", "OK"],
    unsafe_class="DEFECT",
    detector_factory=lambda weights_path, device, thr: StubDetector(),
    classifier_factory=lambda weights_path, device: StubClassifier(),
    preprocess_fn=lambda bgr: bgr,
    thresholds=Thresholds(),
    weights_dir=None,
    reference_bank_path=None,
))
