"""Domain plugins.

A *domain* (``weld`` is the first) supplies everything that is specific to one kind of physical
object -- class names, detector/classifier weights, preprocessing, thresholds and a reference
bank.  Everything else (tiling, retrieval gate, tracking, reports, web UI) is the shared core.

Register a domain by calling :func:`register` at import time of ``pqdai.domains.<name>`` or by
exposing an entry point in the ``pqdai.domains`` group that returns a :class:`DomainSpec`.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

_REGISTRY: Dict[str, "DomainSpec"] = {}
BUILTIN = ("weld", "example_template")


@dataclass
class Thresholds:
    conf: float = 0.1            # stage-1 detector confidence (extract-all)
    verify_conf: float = 0.2     # ROI re-detection confirmation
    alpha: float = 0.1           # entropy gate for retrieval
    k: int = 3                   # neighbours voting
    crop_pad_ratio: float = 0.1


@dataclass
class DomainSpec:
    name: str
    description: str
    class_names: Sequence[str]                              # index order = classifier output order
    unsafe_class: str                                       # any object with this class makes the whole item fail
    detector_factory: Callable[..., object]                 # (weights_path|None, device|None, thresholds) -> detector
    classifier_factory: Callable[..., object]               # (weights_path|None, device|None) -> classifier
    preprocess_fn: Callable                                 # BGR crop -> BGR crop
    thresholds: Thresholds = field(default_factory=Thresholds)
    weights_dir: Optional[str] = None                       # contains manifest.json + *.safetensors
    detector_file: str = "detector.safetensors"
    classifier_file: str = "classifier.safetensors"
    reference_bank_path: Optional[str] = None
    docs: str = ""

    def __post_init__(self):
        if list(self.class_names)[0] != self.unsafe_class:
            raise ValueError("class_names[0] must be the unsafe class (hard-vote ties and the any-unsafe rule rely on it)")

    # ---- convenience --------------------------------------------------------------
    def detector_path(self, override: Optional[str] = None) -> Optional[str]:
        return override or (os.path.join(self.weights_dir, self.detector_file) if self.weights_dir else None)

    def classifier_path(self, override: Optional[str] = None) -> Optional[str]:
        return override or (os.path.join(self.weights_dir, self.classifier_file) if self.weights_dir else None)


def register(spec: DomainSpec) -> DomainSpec:
    _REGISTRY[spec.name] = spec
    return spec


def _load_entry_points() -> None:
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="pqdai.domains") if hasattr(eps, "select") else eps.get("pqdai.domains", [])
        for ep in group:
            obj = ep.load()
            spec = obj() if callable(obj) else obj
            if isinstance(spec, DomainSpec):
                register(spec)
    except Exception:  # noqa: BLE001 - third-party plugins must never break the core
        pass


def available() -> List[str]:
    for n in BUILTIN:
        if n not in _REGISTRY:
            importlib.import_module(f"pqdai.domains.{n}")
    _load_entry_points()
    return sorted(_REGISTRY)


def get(name: str) -> DomainSpec:
    available()
    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
