"""Anchor-free oriented-box detector, written from scratch in plain PyTorch.

Backbone (CSP-style, own implementation) -> PAN-FPN neck -> decoupled head with
a *one-to-many* branch (dense assignment, auxiliary) and a *one-to-one* branch (NMS-free inference).
Outputs:

    feats   : strides 8 / 16 / 32 neck features
    boxes   : (B, 4, N)  distances l, t, r, b (in stride units) from the cell centre
    scores  : (B, 1, N)  objectness/class logits
    angle   : (B, 1, N)  rotation in radians (raw regression, no squashing)

Decoded oriented box: centre = anchor + R(theta) @ ((r-l)/2, (b-t)/2),
size = (l + r, t + b), in pixels after multiplying by the stride.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = (8, 16, 32)


def autopad(k: int, p=None):
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True, e: float = 1.0, k: int = 3):
        super().__init__()
        h = int(c * e)
        self.cv1 = ConvBNAct(c, h, k)
        self.cv2 = ConvBNAct(h, c, k)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CSPBlock(nn.Module):
    """Cross-stage-partial block: split, run n bottlenecks on one half, concat all stages."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c = c1 // 2
        self.cv1 = ConvBNAct(c1, c, 1)
        self.cv2 = ConvBNAct(c * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SelfAttn2d(nn.Module):
    """Light multi-head self-attention on the stride-32 map (global context for thin welds)."""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(c, 3 * c, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.pe = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.norm = nn.BatchNorm2d(c)
        self.ffn = nn.Sequential(ConvBNAct(c, 2 * c, 1), ConvBNAct(2 * c, c, 1, act=False))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, self.heads, 3 * c // self.heads, h * w).split(c // self.heads, 2)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        o = o.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        x = x + self.proj(o)
        return x + self.ffn(x)


@dataclass
class DetConfig:
    width: Sequence[int] = (32, 64, 128, 256, 512)   # stem, s4, s8, s16, s32 channels
    depth: Sequence[int] = (1, 2, 2, 1)              # CSP block repeats for s4..s32
    neck: Sequence[int] = (128, 256, 512)            # head input channels at strides 8/16/32
    head_c: int = 64
    attn: bool = True
    nc: int = 1
    reg_max: int = 1


PRESETS: Dict[str, DetConfig] = {
    "nano": DetConfig(width=(16, 32, 64, 128, 256), depth=(1, 1, 1, 1), neck=(64, 128, 256), head_c=48),
    "small": DetConfig(width=(32, 64, 128, 256, 384), depth=(1, 2, 2, 1), neck=(96, 192, 384), head_c=64),
    "medium": DetConfig(width=(48, 96, 192, 384, 512), depth=(2, 3, 3, 2), neck=(256, 512, 512), head_c=96),
    "large": DetConfig(width=(64, 128, 256, 512, 640), depth=(2, 4, 4, 2), neck=(256, 512, 512), head_c=128),
}


class Backbone(nn.Module):
    def __init__(self, cfg: DetConfig):
        super().__init__()
        w, d = cfg.width, cfg.depth
        self.stem = nn.Sequential(ConvBNAct(3, w[0], 3, 2), ConvBNAct(w[0], w[1], 3, 2))     # s4
        self.s4 = CSPBlock(w[1], w[1], d[0])
        self.d8 = ConvBNAct(w[1], w[2], 3, 2)
        self.s8 = CSPBlock(w[2], w[2], d[1])
        self.d16 = ConvBNAct(w[2], w[3], 3, 2)
        self.s16 = CSPBlock(w[3], w[3], d[2])
        self.d32 = ConvBNAct(w[3], w[4], 3, 2)
        self.s32 = CSPBlock(w[4], w[4], d[3])
        self.sppf = SPPF(w[4], w[4])
        self.attn = SelfAttn2d(w[4]) if cfg.attn else nn.Identity()

    def forward(self, x):
        x = self.s4(self.stem(x))
        p3 = self.s8(self.d8(x))
        p4 = self.s16(self.d16(p3))
        p5 = self.attn(self.sppf(self.s32(self.d32(p4))))
        return p3, p4, p5


class PAN(nn.Module):
    def __init__(self, cfg: DetConfig):
        super().__init__()
        c3, c4, c5 = cfg.width[2], cfg.width[3], cfg.width[4]
        n3, n4, n5 = cfg.neck
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td4 = CSPBlock(c5 + c4, n4, 1, False)
        self.td3 = CSPBlock(n4 + c3, n3, 1, False)
        self.dn3 = ConvBNAct(n3, n3, 3, 2)
        self.bu4 = CSPBlock(n3 + n4, n4, 1, False)
        self.dn4 = ConvBNAct(n4, n4, 3, 2)
        self.bu5 = CSPBlock(n4 + c5, n5, 1, False)

    def forward(self, p3, p4, p5):
        t4 = self.td4(torch.cat([self.up(p5), p4], 1))
        o3 = self.td3(torch.cat([self.up(t4), p3], 1))
        o4 = self.bu4(torch.cat([self.dn3(o3), t4], 1))
        o5 = self.bu5(torch.cat([self.dn4(o4), p5], 1))
        return o3, o4, o5


def _branch(cin: int, c: int, cout: int) -> nn.Sequential:
    return nn.Sequential(ConvBNAct(cin, c, 3), ConvBNAct(c, c, 3), nn.Conv2d(c, cout, 1))


def _dw_branch(cin: int, c: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Sequential(ConvBNAct(cin, cin, 3, g=cin), ConvBNAct(cin, c, 1)),
        nn.Sequential(ConvBNAct(c, c, 3, g=c), ConvBNAct(c, c, 1)),
        nn.Conv2d(c, cout, 1),
    )


class OBBHead(nn.Module):
    """Dual (one-to-many / one-to-one) decoupled oriented-box head."""

    def __init__(self, cfg: DetConfig):
        super().__init__()
        self.nc = cfg.nc
        ch = cfg.neck
        c = max(cfg.head_c, 16)
        self.reg = nn.ModuleList(_branch(x, c, 4) for x in ch)
        self.cls = nn.ModuleList(_dw_branch(x, c, cfg.nc) for x in ch)
        self.ang = nn.ModuleList(_branch(x, max(c // 2, 16), 1) for x in ch)
        self.reg1 = nn.ModuleList(_branch(x, c, 4) for x in ch)
        self.cls1 = nn.ModuleList(_dw_branch(x, c, cfg.nc) for x in ch)
        self.ang1 = nn.ModuleList(_branch(x, max(c // 2, 16), 1) for x in ch)
        self._init_bias()

    def _init_bias(self):
        for regs, clss in ((self.reg, self.cls), (self.reg1, self.cls1)):
            for i, (r, c) in enumerate(zip(regs, clss)):
                r[-1].bias.data[:] = 2.0
                c[-1].bias.data[:] = math.log(5 / self.nc / (640 / STRIDES[i]) ** 2)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        b = feats[0].shape[0]
        out = {}
        for name, (R, C, A) in {"one2many": (self.reg, self.cls, self.ang), "one2one": (self.reg1, self.cls1, self.ang1)}.items():
            f = feats if name == "one2many" else [x.detach() for x in feats]
            out[name] = {
                "boxes": torch.cat([R[i](f[i]).view(b, 4, -1) for i in range(3)], 2),
                "scores": torch.cat([C[i](f[i]).view(b, self.nc, -1) for i in range(3)], 2),
                "angle": torch.cat([A[i](f[i]).view(b, 1, -1) for i in range(3)], 2),
            }
        return out


class OBBDetectorNet(nn.Module):
    def __init__(self, cfg: DetConfig | str = "small"):
        super().__init__()
        self.cfg = PRESETS[cfg] if isinstance(cfg, str) else cfg
        self.backbone = Backbone(self.cfg)
        self.neck = PAN(self.cfg)
        self.head = OBBHead(self.cfg)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        feats = list(self.neck(*self.backbone(x)))
        out = self.head(feats)
        out["feats"] = feats
        return out


def make_anchors(shapes: Sequence[Tuple[int, int]], strides: Sequence[int] = STRIDES, device=None, dtype=torch.float32):
    ap, st = [], []
    for (h, w), s in zip(shapes, strides):
        sy, sx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) + 0.5, torch.arange(w, device=device, dtype=dtype) + 0.5, indexing="ij"
        )
        ap.append(torch.stack((sx, sy), -1).view(-1, 2))
        st.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(ap), torch.cat(st)


def decode_rboxes(boxes: torch.Tensor, angle: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """boxes (B,4,N) ltrb, angle (B,1,N) rad -> (B,N,5) = cx, cy, w, h, theta in input pixels."""
    d = boxes.permute(0, 2, 1)
    th = angle.permute(0, 2, 1)
    lt, rb = d[..., :2], d[..., 2:]
    cos, sin = torch.cos(th), torch.sin(th)
    xf, yf = ((rb - lt) / 2).split(1, -1)
    xy = torch.cat([xf * cos - yf * sin, xf * sin + yf * cos], -1) + anchors
    wh = lt + rb
    return torch.cat([xy * strides, wh * strides, th], -1)


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


if __name__ == "__main__":
    for k in PRESETS:
        m = OBBDetectorNet(k)
        o = m(torch.randn(1, 3, 640, 640))
        print(k, "%.2fM" % count_params(m), [tuple(f.shape) for f in o["feats"]], tuple(o["one2one"]["boxes"].shape))
