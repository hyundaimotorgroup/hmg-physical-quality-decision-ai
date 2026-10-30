"""Build a ready-to-use :class:`QualityPipeline` for a domain."""
from __future__ import annotations

import os
from typing import Optional

from .. import domains
from .pipeline import QualityPipeline
from .reference_bank import ReferenceBank
from .weights import read_manifest, verify_weights


def build_pipeline(domain: str = "weld", device: Optional[str] = None, detector_weights: Optional[str] = None,
                   classifier_weights: Optional[str] = None, reference_bank: Optional[str] = None, conf: Optional[float] = None,
                   verify_conf: Optional[float] = None, alpha: Optional[float] = None, k: Optional[int] = None,
                   verify: bool = True, use_retrieval: bool = True) -> QualityPipeline:
    spec = domains.get(domain)
    thr = type(spec.thresholds)(**vars(spec.thresholds))
    for name, val in (("conf", conf), ("verify_conf", verify_conf), ("alpha", alpha), ("k", k)):
        if val is not None:
            setattr(thr, name, val)
    dpath, cpath = spec.detector_path(detector_weights), spec.classifier_path(classifier_weights)
    model_id = None
    if verify:
        for path in (dpath, cpath):
            if path:
                m = verify_weights(path)
                if path == cpath:
                    model_id = m["sha256"]
    elif cpath and os.path.exists(cpath):
        model_id = read_manifest(cpath).get("sha256")
    detector = spec.detector_factory(dpath, device, thr)
    classifier = spec.classifier_factory(cpath, device)
    bank_path = reference_bank or spec.reference_bank_path
    if bank_path is None:
        raise FileNotFoundError(f"domain {domain!r} has no reference bank; build one with `pqdai bank build`")
    bank = ReferenceBank.load(bank_path)
    bank.validate(expected_classes=list(spec.class_names))
    bank.check_compatible(model_id)
    pipe = QualityPipeline(detector, classifier, bank.embeddings, bank.labels, alpha=thr.alpha, k=thr.k,
                           crop_pad_ratio=thr.crop_pad_ratio, preprocess_fn=spec.preprocess_fn,
                           use_retrieval=use_retrieval, class_names=list(spec.class_names))
    pipe.reference_bank = bank
    pipe.domain = spec
    return pipe
