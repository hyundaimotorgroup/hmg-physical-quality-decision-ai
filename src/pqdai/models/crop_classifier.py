"""Crop classifier wrapper.

Public API: ``predict_with_uncertainty(image)`` / ``predict_batch_with_uncertainty(images)``
returning dicts with predicted_class, class_name, probabilities, uncertainty{entropy,margin,max_prob}, embedding (1280-d).
Input images are BGR numpy arrays (or a path); preprocessing = shorter-side resize to 224 (bilinear) + center crop.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from .classifier import load_classifier

CLASS_NAMES = ["BAD", "GOOD"]


def compute_all_uncertainties(probs: np.ndarray, eps: float = 1e-10) -> Dict[str, float]:
    p = np.clip(probs, eps, 1.0)
    s = np.sort(probs)[::-1]
    return {"entropy": float(-(p * np.log(p)).sum()), "margin": float(s[0] - s[1]) if len(s) > 1 else 0.0, "max_prob": float(s.max())}


def resize_center_crop(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Shorter side -> ``size`` (PIL bilinear), then center crop. Returns HxWx3 uint8 RGB."""
    from PIL import Image

    h, w = rgb.shape[:2]
    if h <= w:
        nh, nw = size, max(size, int(size * w / h))
    else:
        nw, nh = size, max(size, int(size * h / w))
    im = Image.fromarray(rgb)
    if (nh, nw) != (h, w):
        im = im.resize((nw, nh), Image.BILINEAR)
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    return np.asarray(im.crop((left, top, left + size, top + size)))


class CropClassifier:
    def __init__(self, model_path: str, device: str = None, class_names: List[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or CLASS_NAMES
        self.model = load_classifier(model_path, self.device)
        self.embedding = None

    @staticmethod
    def _prep(image) -> np.ndarray:
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            image = cv2.imread(str(image))
        return resize_center_crop(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def _forward(self, images: List) -> List[Dict]:
        x = torch.from_numpy(np.stack([self._prep(i) for i in images])).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        logits, emb = self.model(x)
        probs = torch.softmax(logits.float(), 1).cpu().numpy()
        emb = emb.float().cpu().numpy()
        self.embedding = emb
        out = []
        for p, e in zip(probs, emb):
            c = int(np.argmax(p))
            out.append({"predicted_class": c, "class_name": self.class_names[c], "probabilities": p,
                        "uncertainty": compute_all_uncertainties(p), "embedding": e})
        return out

    def predict_with_uncertainty(self, image) -> Dict:
        return self._forward([image])[0]

    def predict_batch_with_uncertainty(self, images: List) -> List[Dict]:
        return self._forward(list(images))
