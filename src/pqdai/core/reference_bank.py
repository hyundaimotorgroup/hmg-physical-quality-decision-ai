"""Reference bank: labelled embeddings used by the uncertainty-gated retrieval harness.

The pipeline trusts the classifier while it is confident.  When the entropy of its
(TTA-averaged) probabilities exceeds ``alpha`` the crop is routed to retrieval: its
embedding is compared (cosine) with the bank and the top-``k`` neighbours vote.  The bank
therefore has to live in the *same embedding space* as the classifier that is deployed.

File format (``.npz``, schema 2)::

    embeddings   float32 (N, D)
    labels       int64   (N,)         index into class_names
    class_names  str     (C,)
    schema       int                  2
    meta         json str             provenance / compatibility info

Legacy files (schema 1: ``embeddings``, ``labels``, ``image_paths``, ``metadata``) still load.
No images or absolute paths are stored; ``names`` keeps relative basenames only.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .retrieval import retrieve_topk_neighbors

SCHEMA = 2
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class BankError(ValueError):
    pass


@dataclass
class ReferenceBank:
    embeddings: np.ndarray
    labels: np.ndarray
    class_names: List[str] = field(default_factory=lambda: ["BAD", "GOOD"])
    names: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def counts(self) -> Dict[str, int]:
        return {c: int((self.labels == i).sum()) for i, c in enumerate(self.class_names)}

    def validate(self, expected_dim: Optional[int] = None, expected_classes: Optional[Sequence[str]] = None) -> "ReferenceBank":
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.labels):
            raise BankError(f"embeddings {self.embeddings.shape} and labels {self.labels.shape} are inconsistent")
        if len(self) == 0:
            raise BankError("reference bank is empty")
        if not np.isfinite(self.embeddings).all():
            raise BankError("reference bank contains non-finite embeddings")
        if self.labels.min() < 0 or self.labels.max() >= len(self.class_names):
            raise BankError(f"labels outside [0, {len(self.class_names)}) for classes {self.class_names}")
        if expected_dim is not None and self.dim != expected_dim:
            raise BankError(f"bank embedding dim {self.dim} != classifier embedding dim {expected_dim}")
        if expected_classes is not None and list(expected_classes) != list(self.class_names):
            raise BankError(f"bank classes {self.class_names} != domain classes {list(expected_classes)}")
        missing = [c for i, c in enumerate(self.class_names) if not (self.labels == i).any()]
        if missing:
            warnings.warn(f"reference bank has no examples for class(es) {missing}; k-NN voting will never choose them")
        return self

    def check_compatible(self, model_id: Optional[str]) -> None:
        """Warn when the bank was built for a different classifier build than ``model_id``.

        A mismatch is not fatal (a bank built in an aligned embedding space keeps working,
        as the bundled weld bank does) but verdicts routed through retrieval may be unreliable.
        """
        built_for = self.meta.get("built_for_model")
        aligned = self.meta.get("aligned_to")
        if built_for and model_id and built_for != model_id and aligned != model_id:
            warnings.warn(
                f"reference bank was built for classifier {built_for[:12]}..., deployed classifier is {model_id[:12]}...; "
                "retrieval verdicts may be unreliable -- rebuild the bank with `pqdai bank build` "
                "or check it with `pqdai bank eval`.", stacklevel=2)

    # ------------------------------------------------------------------ query
    def query(self, embedding: np.ndarray, k: int = 3, metric: str = "cosine"):
        idx, scores = retrieve_topk_neighbors(embedding, self.embeddings, k, metric=metric)
        return idx, scores, self.labels[idx]

    def evaluate(self, k: int = 3) -> Dict:
        """Leave-one-out k-NN accuracy (hard vote, ties favour class 0 = the unsafe class)."""
        from .voting import hard_vote

        n = len(self)
        e = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        sim = e @ e.T
        np.fill_diagonal(sim, -np.inf)
        kk = min(k, n - 1)
        top = np.argsort(-sim, axis=1)[:, :kk]
        pred = np.array([hard_vote(self.labels[t]) for t in top])
        per = {c: float((pred[self.labels == i] == i).mean()) if (self.labels == i).any() else None
               for i, c in enumerate(self.class_names)}
        return {"n": n, "k": kk, "loo_accuracy": float((pred == self.labels).mean()), "per_class_recall": per, "counts": self.counts()}

    # ------------------------------------------------------------------ editing
    def add(self, embeddings: np.ndarray, labels: np.ndarray, names: Optional[Sequence[str]] = None) -> "ReferenceBank":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[1] != self.dim:
            raise BankError(f"cannot add embeddings of dim {embeddings.shape[1]} to a bank of dim {self.dim}")
        out = ReferenceBank(np.concatenate([self.embeddings, embeddings]), np.concatenate([self.labels, np.asarray(labels, np.int64)]),
                            list(self.class_names), None, dict(self.meta))
        if self.names is not None or names is not None:
            out.names = list(self.names or [""] * len(self)) + list(names or [""] * len(embeddings))
        return out.validate()

    @staticmethod
    def merge(banks: Sequence["ReferenceBank"]) -> "ReferenceBank":
        first = banks[0]
        for b in banks[1:]:
            if b.dim != first.dim or b.class_names != first.class_names:
                raise BankError("cannot merge banks with different embedding dim or class names")
        names = None
        if any(b.names for b in banks):
            names = sum([list(b.names or [""] * len(b)) for b in banks], [])
        return ReferenceBank(np.concatenate([b.embeddings for b in banks]), np.concatenate([b.labels for b in banks]),
                             list(first.class_names), names, dict(first.meta, merged_from=len(banks))).validate()

    def balanced(self, per_class: Optional[int] = None, seed: int = 0) -> "ReferenceBank":
        rng = np.random.default_rng(seed)
        per_class = per_class or int(min(c for c in self.counts().values() if c > 0))
        keep = []
        for i in range(len(self.class_names)):
            ix = np.flatnonzero(self.labels == i)
            if len(ix):
                keep.append(rng.choice(ix, size=min(per_class, len(ix)), replace=False))
        keep = np.sort(np.concatenate(keep))
        return ReferenceBank(self.embeddings[keep], self.labels[keep], list(self.class_names),
                             [self.names[i] for i in keep] if self.names else None, dict(self.meta, balanced=True))

    def dedup(self, threshold: float = 0.995) -> "ReferenceBank":
        """Drop near-duplicate embeddings (cosine >= threshold) within the same class, keeping the first."""
        e = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        keep: List[int] = []
        for i in range(len(self)):
            if any(self.labels[j] == self.labels[i] and float(e[i] @ e[j]) >= threshold for j in keep):
                continue
            keep.append(i)
        return ReferenceBank(self.embeddings[keep], self.labels[keep], list(self.class_names),
                             [self.names[i] for i in keep] if self.names else None, dict(self.meta, dedup_threshold=threshold))

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrs = dict(embeddings=self.embeddings, labels=self.labels, class_names=np.array(self.class_names),
                    schema=np.array(SCHEMA), meta=np.array(json.dumps(self.meta, sort_keys=True)))
        if self.names is not None:
            arrs["names"] = np.array([os.path.basename(str(n).replace("\\", "/")) for n in self.names])
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path: str) -> "ReferenceBank":
        z = np.load(path, allow_pickle=False if _is_schema2(path) else True)
        if "schema" in z.files:
            meta = json.loads(str(z["meta"]))
            names = [str(x) for x in z["names"]] if "names" in z.files else None
            return ReferenceBank(z["embeddings"], z["labels"], [str(c) for c in z["class_names"]], names, meta).validate()
        # legacy schema 1
        meta = {}
        if "metadata" in z.files:
            try:
                m = z["metadata"][0]
                meta = dict(m) if isinstance(m, dict) else {}
            except Exception:  # noqa: BLE001
                meta = {}
        names = [os.path.basename(str(str(p).replace("\\", "/"))) for p in z["image_paths"]] if "image_paths" in z.files else None
        meta.setdefault("legacy_schema", 1)
        return ReferenceBank(z["embeddings"], z["labels"], ["BAD", "GOOD"], names, meta).validate()


def _is_schema2(path: str) -> bool:
    with np.load(path, allow_pickle=False) as z:
        return "schema" in z.files


def build_from_crops(crops_dir: str, classifier, class_names: Sequence[str], model_id: Optional[str] = None,
                     dedup: Optional[float] = 0.995, balance: bool = False, batch_size: int = 32, extra_meta: Optional[Dict] = None,
                     preprocess=None) -> ReferenceBank:
    """Embed ``crops_dir/<CLASS>/*.jpg`` with ``classifier`` (a CropClassifier-like object).

    ``preprocess`` (e.g. the domain's brightness correction) is applied to each crop before embedding.
    """
    import cv2

    embs, labels, names = [], [], []
    for ci, cname in enumerate(class_names):
        d = os.path.join(crops_dir, cname)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        for s in range(0, len(files), batch_size):
            imgs = []
            for f in files[s:s + batch_size]:
                im = cv2.imread(os.path.join(d, f))
                if im is None:
                    continue
                imgs.append(preprocess(im) if preprocess else im)
                names.append(f)
                labels.append(ci)
            if imgs:
                embs.extend(r["embedding"] for r in classifier.predict_batch_with_uncertainty(imgs))
    if not embs:
        raise BankError(f"no images found under {crops_dir}/<{'|'.join(class_names)}>/")
    bank = ReferenceBank(np.stack(embs), np.array(labels), list(class_names), names,
                         dict(extra_meta or {}, built_for_model=model_id, source="pqdai bank build", n_input=len(embs)))
    if dedup:
        bank = bank.dedup(dedup)
    if balance:
        bank = bank.balanced()
    return bank.validate()
