"""Weights manifests: locate, verify (sha256) and describe released model files."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List


class WeightsError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(weights_file: str) -> Dict:
    mf = os.path.splitext(weights_file)[0] + ".manifest.json"
    if not os.path.exists(mf):
        raise WeightsError(f"manifest not found next to {weights_file} (expected {mf})")
    return json.load(open(mf, encoding="utf-8"))


def verify_weights(weights_file: str) -> Dict:
    """Raise :class:`WeightsError` unless the file exists and matches the sha256 in its manifest."""
    if not os.path.exists(weights_file):
        raise WeightsError(f"weights file missing: {weights_file} (see weights/README.md; files are tracked with git-lfs)")
    if os.path.getsize(weights_file) < 1024:
        raise WeightsError(f"{weights_file} is tiny -- looks like a git-lfs pointer; run `git lfs pull`")
    m = read_manifest(weights_file)
    actual = sha256_file(weights_file)
    if actual != m["sha256"]:
        raise WeightsError(f"sha256 mismatch for {weights_file}: manifest {m['sha256'][:16]}..., file {actual[:16]}...")
    return m


def verify_dir(weights_dir: str) -> List[Dict]:
    index = os.path.join(weights_dir, "manifest.json")
    if not os.path.exists(index):
        raise WeightsError(
            f"no manifest.json in {weights_dir}. The weights ship in weights/<domain>/ and are tracked with git-lfs: "
            "install git-lfs (`git lfs install`) and run `git lfs pull`, or point --dir to a folder that contains the "
            "released *.safetensors files with their manifests.")
    with open(index, encoding="utf-8") as f:
        idx = json.load(f)
    out = []
    for name, info in idx["models"].items():
        m = verify_weights(os.path.join(weights_dir, info["file"]))
        out.append(dict(name=name, status=idx.get("status"), version=idx.get("version"), sha256=m["sha256"], params=m.get("params")))
    return out
