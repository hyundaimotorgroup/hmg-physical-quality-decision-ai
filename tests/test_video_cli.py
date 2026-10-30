"""End-to-end video CLI run with stub detector/classifier (CPU, tiny synthetic video)."""
import json
import sys

import cv2
import numpy as np

from conftest import FakeClassifier, FakeDetector, box


def _make_video(path, n=40, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    rng = np.random.default_rng(1)
    bg = (rng.random((240, 320, 3)) * 80 + 90).astype(np.uint8)
    for _ in range(n):
        f = bg.copy()
        f[100:140, 40:100] = 235
        f[100:140, 220:280] = 25
        vw.write(f)
    vw.release()


def test_video_cli_end_to_end(tmp_path, monkeypatch):
    from pqdai.video import run

    vid = tmp_path / "v.mp4"
    _make_video(vid)
    bank = tmp_path / "bank.npz"
    emb = np.zeros((4, 8), np.float32)
    emb[:2, 0] = 1
    emb[2:, 1] = 1
    np.savez(bank, embeddings=emb, labels=np.array([0, 0, 1, 1]))
    polys = [box(70, 120, 60, 40), box(250, 120, 60, 40)]
    from pqdai.core.pipeline import QualityPipeline

    def fake_build(domain, device, dw, cw, bank_path, **kw):
        b = np.load(bank_path)
        return QualityPipeline(FakeDetector(polys), FakeClassifier(), b["embeddings"], b["labels"], alpha=kw["alpha"], k=kw["k"])

    monkeypatch.setattr(run, "build_pipeline", fake_build)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["pqdai-video", "--video", str(vid), "--out", str(out), "--sample-fps", "5",
                                      "--reference-bank", str(bank), "--no-overlays", "--progress-every", "5"])
    run.main()
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    text = json.dumps(res)
    assert "BAD" in text and "GOOD" in text
    assert (out / "report.html").exists()
