"""Rotated-box utilities (pure PyTorch / NumPy).

Convention: an oriented box is ``(cx, cy, w, h, theta)`` with ``theta`` in
radians, measured clockwise in image coordinates (x right, y down); ``w`` is
the extent along the rotated x-axis.
"""
from __future__ import annotations

import math

import numpy as np
import torch

EPS = 1e-7



def poly_to_rbox_np(poly: np.ndarray) -> np.ndarray:
    """(4,2) polygon -> (cx, cy, w, h, theta) with theta in [-pi/4, 3pi/4)."""
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    c = p.mean(0)
    e0, e1 = p[1] - p[0], p[2] - p[1]
    w, h = np.linalg.norm(e0), np.linalg.norm(e1)
    t = math.atan2(e0[1], e0[0])
    # canonicalise to [-pi/4, 3pi/4)
    while t >= 3 * math.pi / 4:
        t -= math.pi
    while t < -math.pi / 4:
        t += math.pi
    if t >= math.pi / 4 and False:
        pass
    return np.array([c[0], c[1], w, h, t], dtype=np.float32)


def rbox_to_poly_np(rbox: np.ndarray) -> np.ndarray:
    cx, cy, w, h, t = [float(v) for v in rbox]
    cos, sin = math.cos(t), math.sin(t)
    d = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    r = np.array([[cos, -sin], [sin, cos]])
    return (d @ r.T + [cx, cy]).astype(np.float32)



def rotated_nms(rboxes: torch.Tensor, scores: torch.Tensor, iou_thr: float, max_keep: int = 300) -> torch.Tensor:
    """Greedy rotated NMS: one batched pairwise-IoU matrix, then a CPU greedy pass.

    Returns indices (into ``rboxes``) of the kept boxes, highest score first.
    """
    n = rboxes.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=rboxes.device)
    order = scores.argsort(descending=True)[:1000]
    rb = rboxes[order]
    if len(order) == 1:
        return order
    iou = approx_riou(rb, rb).cpu().numpy()
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_keep:
            break
        suppressed |= iou[i] > iou_thr
    return order[torch.as_tensor(keep, device=order.device)]


def approx_riou(a: torch.Tensor, b: torch.Tensor, n: int = 24) -> torch.Tensor:
    """Pairwise IoU (N,M) of rotated boxes via rasterising each box on a local
    grid (accurate to ~1/n of the box size; adequate for NMS)."""
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return a.new_zeros(N, M)
    ca, cb = a[:, None, :], b[None, :, :]
    # sample points inside box a (grid), test membership in b
    g = (torch.arange(n, device=a.device, dtype=a.dtype) + 0.5) / n - 0.5
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)  # (n*n)

    def inside(src, dst):
        # points from src box grid -> in dst box? src:(N,1,5) dst:(1,M,5)
        cos, sin = torch.cos(src[..., 4:5]), torch.sin(src[..., 4:5])
        px = src[..., 0:1] + (gx * src[..., 2:3]) * cos - (gy * src[..., 3:4]) * sin
        py = src[..., 1:2] + (gx * src[..., 2:3]) * sin + (gy * src[..., 3:4]) * cos
        dx, dy = px - dst[..., 0:1], py - dst[..., 1:2]
        cd, sd = torch.cos(dst[..., 4:5]), torch.sin(dst[..., 4:5])
        u = dx * cd + dy * sd
        v = -dx * sd + dy * cd
        return ((u.abs() <= dst[..., 2:3] / 2) & (v.abs() <= dst[..., 3:4] / 2)).float().mean(-1)

    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    inter_a = inside(ca, cb) * area_a[:, None]
    inter_b = inside(cb, ca) * area_b[None, :]
    inter = 0.5 * (inter_a + inter_b)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=EPS)


def canonical_rbox_np(rbox: np.ndarray) -> np.ndarray:
    """Return an equivalent (cx, cy, w, h, theta) with |theta| <= pi/4 (swap w/h and
    shift by +-pi/2 as needed) so that axis-aligned boxes always have theta ~ 0."""
    cx, cy, w, h, t = [float(v) for v in rbox]
    t = (t + math.pi / 2) % math.pi - math.pi / 2          # [-pi/2, pi/2)
    if t >= math.pi / 4:
        t -= math.pi / 2
        w, h = h, w
    elif t < -math.pi / 4:
        t += math.pi / 2
        w, h = h, w
    return np.array([cx, cy, w, h, t], dtype=np.float32)


def poly_to_canonical_rbox_np(poly: np.ndarray) -> np.ndarray:
    return canonical_rbox_np(poly_to_rbox_np(poly))
