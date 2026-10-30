"""Inference backend for the student oriented-box detector.

Contract (used by :class:`pqdai.core.tiled_detector.TiledDetector`):

    runner = DetRunner(weights_path, device="cuda", imgsz=640, nms_iou=0.7, max_det=300)
    polys, confs = runner.predict(bgr, conf)          # (N,4,2) float32 px, (N,) float32
    results      = runner.predict_batch([bgr, ...], conf)

Polygon corner order matches the original pipeline (xywhr -> 4 corners).  The score
filter is strict ``>``.  Images are letterboxed to a centred square by default, which
what the detector expects; ``rect=True`` switches to a stride-32 rectangular letterbox.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .detector import STRIDES, PRESETS, OBBDetectorNet, decode_rboxes, make_anchors
from ..core.rbox import rotated_nms


def _letterbox(img: np.ndarray, size: int, rect: bool, stride: int = 32, pad_val: int = 114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    if rect:
        dw, dh = (size - nw) % stride / 2, (size - nh) % stride / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
    else:
        dw, dh = (size - nw) / 2, (size - nh) / 2
        top, bottom = int(dh), size - nh - int(dh)
        left, right = int(dw), size - nw - int(dw)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_val,) * 3)
    return img, r, left, top


def _xywhr_to_corners(x: np.ndarray) -> np.ndarray:
    ctr, w, h, a = x[..., :2], x[..., 2:3], x[..., 3:4], x[..., 4:5]
    c, s = np.cos(a), np.sin(a)
    v1 = np.concatenate([w / 2 * c, w / 2 * s], -1)
    v2 = np.concatenate([-h / 2 * s, h / 2 * c], -1)
    return np.stack([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2], -2)


def load_detector(weights_path: str, device: str = "cpu") -> OBBDetectorNet:
    """Load ``detector.safetensors``; the architecture preset comes from the sibling manifest."""
    import json
    import os

    from safetensors.torch import load_file

    mf = os.path.splitext(weights_path)[0] + ".manifest.json"
    arch = json.load(open(mf, encoding="utf-8"))["arch"] if os.path.exists(mf) else "small"
    if arch not in PRESETS:
        raise ValueError(f"unknown detector architecture {arch!r}; known: {sorted(PRESETS)}")
    net = OBBDetectorNet(arch)
    net.load_state_dict(load_file(weights_path))
    return net.to(device).eval()


class DetRunner:
    def __init__(self, weights_path: str, device: str = "cuda", imgsz: int = 640, nms_iou: Optional[float] = 0.7,
                 max_det: int = 300, rect: bool = False, half: bool = False):
        self.device, self.imgsz, self.nms_iou, self.max_det, self.rect = device, imgsz, nms_iou, max_det, rect
        self.half = half and str(device).startswith("cuda")
        self.model = load_detector(weights_path, device)
        if self.half:
            self.model.half()
        self._anchors = {}

    def _get_anchors(self, shapes):
        key = tuple(map(tuple, shapes))
        if key not in self._anchors:
            self._anchors[key] = make_anchors(shapes, STRIDES, device=self.device)
        return self._anchors[key]

    @torch.no_grad()
    def _run(self, images: Sequence[np.ndarray], conf: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        lbs = [_letterbox(im, self.imgsz, self.rect) for im in images]
        x = np.stack([lb[0][:, :, ::-1] for lb in lbs])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = x.half() if self.half else x
        o = self.model(x)
        anc, st = self._get_anchors([f.shape[2:] for f in o["feats"]])
        br = o["one2one"]
        boxes = decode_rboxes(br["boxes"].float(), br["angle"].float(), anc, st)       # (B,N,5)
        sc = br["scores"].float().sigmoid()[:, 0]
        k = min(self.max_det, sc.shape[1])
        top_sc, idx = sc.topk(k, dim=1)
        boxes = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
        out = []
        for b, im in enumerate(images):
            keep = top_sc[b] > conf
            r, cf = boxes[b][keep], top_sc[b][keep]
            if self.nms_iou is not None and self.nms_iou < 1.0 and len(r) > 1:
                kept = rotated_nms(r, cf, self.nms_iou)
                r, cf = r[kept], cf[kept]
            r = r.cpu().numpy().astype(np.float64)
            cf = cf.cpu().numpy().astype(np.float32)
            _, ratio, padx, pady = lbs[b]
            r[:, 0] = (r[:, 0] - padx) / ratio
            r[:, 1] = (r[:, 1] - pady) / ratio
            r[:, 2:4] /= ratio
            polys = _xywhr_to_corners(r).astype(np.float32) if len(r) else np.zeros((0, 4, 2), np.float32)
            out.append((polys, cf))
        return out

    def predict_batch(self, images: Sequence[np.ndarray], conf: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Runs same-shaped images together; mixed shapes are grouped, order is preserved."""
        groups = defaultdict(list)
        for i, im in enumerate(images):
            groups[im.shape].append(i)
        res = [None] * len(images)
        for idxs in groups.values():
            for s in range(0, len(idxs), 16):
                chunk = idxs[s:s + 16]
                for i, r in zip(chunk, self._run([images[j] for j in chunk], conf)):
                    res[i] = r
        return res

    def predict(self, bgr: np.ndarray, conf: float) -> Tuple[np.ndarray, np.ndarray]:
        return self._run([bgr], conf)[0]
