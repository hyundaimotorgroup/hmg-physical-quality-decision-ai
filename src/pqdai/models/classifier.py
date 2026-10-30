"""Pure-PyTorch GOOD/BAD crop classifier.

Backbone: small residual CNN (stem + 4 stages) written from scratch, followed by a
embedding head: 1x1 conv -> BN -> SiLU -> global avg pool (=1280-d embedding,
the space the reference bank lives in) -> dropout -> Linear(1280, 2).
Class order is [BAD, GOOD].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, max(8, c // r), 1), nn.Conv2d(max(8, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class ResBlock(nn.Module):
    """Basic residual block (3x3,3x3); optional squeeze-excite; stride via first conv."""

    def __init__(self, cin, cout, stride=1, se=False):
        super().__init__()
        self.c1 = conv_bn(cin, cout, 3, stride)
        self.c2 = conv_bn(cout, cout, 3, 1, act=False)
        self.se = SE(cout) if se else nn.Identity()
        self.short = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.silu(self.se(self.c2(self.c1(x))) + self.short(x))


@dataclass
class ClsConfig:
    widths: Tuple[int, ...] = (32, 64, 128, 256)
    depths: Tuple[int, ...] = (1, 2, 3, 2)
    stem: int = 24
    se: bool = True
    embed_dim: int = 1280
    drop: float = 0.2
    num_classes: int = 2

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["widths"], d["depths"] = list(self.widths), list(self.depths)
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ClsConfig":
        d = dict(d)
        d["widths"], d["depths"] = tuple(d["widths"]), tuple(d["depths"])
        return ClsConfig(**{k: v for k, v in d.items() if k in ClsConfig.__dataclass_fields__})


class CropClassifierNet(nn.Module):
    """forward(x in [0,1], RGB, NCHW) -> (logits[B,2], embedding[B,embed_dim])."""

    def __init__(self, cfg: ClsConfig = ClsConfig()):
        super().__init__()
        self.cfg = cfg
        w = cfg.widths
        self.stem = nn.Sequential(conv_bn(3, cfg.stem, 3, 2), conv_bn(cfg.stem, w[0], 3, 2))   # /4
        stages, cin = [], w[0]
        for i, (c, d) in enumerate(zip(w, cfg.depths)):
            blocks = []
            for j in range(d):
                blocks.append(ResBlock(cin, c, stride=2 if (j == 0 and i > 0) else 1, se=cfg.se))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)                                                    # /32
        self.neck = conv_bn(cin, cfg.embed_dim, 1)
        self.drop = nn.Dropout(cfg.drop)
        self.fc = nn.Linear(cfg.embed_dim, cfg.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():  # zero-init last BN of each residual branch
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.c2[1].weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.neck(self.stages(self.stem(x)))
        emb = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.fc(self.drop(emb)), emb


def build_model(cfg) -> CropClassifierNet:
    return CropClassifierNet(cfg if isinstance(cfg, ClsConfig) else ClsConfig.from_dict(cfg))


def load_classifier(weights_path: str, device="cpu", config=None) -> CropClassifierNet:
    """Load ``classifier.safetensors`` (config from the sibling ``classifier.manifest.json``
    unless ``config`` is given)."""
    import json
    import os

    from safetensors.torch import load_file

    if config is None:
        mf = os.path.splitext(weights_path)[0] + ".manifest.json"
        config = json.load(open(mf, encoding="utf-8"))["config"]
    model = build_model(config)
    model.load_state_dict(load_file(weights_path))
    return model.to(device).eval()
