import json
import os
import shutil

import cv2
import numpy as np
import pytest
import torch

from pqdai import domains
from pqdai.core.weights import WeightsError, sha256_file, verify_dir, verify_weights

WDIR = domains.get("weld").weights_dir
needs_weights = pytest.mark.skipif(not os.path.exists(os.path.join(WDIR, "classifier.safetensors")), reason="weld weights not present (git lfs pull)")


@needs_weights
def test_manifests_and_sha256():
    rows = verify_dir(WDIR)
    assert {r["name"] for r in rows} == {"classifier", "detector"}
    for r in rows:
        m = json.load(open(os.path.join(WDIR, f"{r['name']}.manifest.json")))
        assert m["license"] == "Apache-2.0" and m["sha256"] == r["sha256"]


@needs_weights
def test_tampered_weights_rejected(tmp_path):
    for f in os.listdir(WDIR):
        shutil.copy(os.path.join(WDIR, f), tmp_path / f)
    with open(tmp_path / "classifier.safetensors", "r+b") as fh:
        fh.seek(2000)
        fh.write(b"\x00\x01\x02\x03")
    with pytest.raises(WeightsError, match="sha256 mismatch"):
        verify_weights(str(tmp_path / "classifier.safetensors"))


def test_missing_and_lfs_pointer(tmp_path):
    with pytest.raises(WeightsError, match="missing"):
        verify_weights(str(tmp_path / "nope.safetensors"))
    (tmp_path / "x.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(WeightsError, match="git-lfs"):
        verify_weights(str(tmp_path / "x.safetensors"))


@needs_weights
def test_classifier_forward():
    from pqdai.models.classifier import load_classifier
    m = load_classifier(os.path.join(WDIR, "classifier.safetensors"))
    with torch.no_grad():
        logits, emb = m(torch.rand(3, 3, 224, 224))
    assert logits.shape == (3, 2) and emb.shape == (3, 1280)


@needs_weights
def test_detector_forward_and_runner():
    from pqdai.models.det_runner import DetRunner, load_detector
    m = load_detector(os.path.join(WDIR, "detector.safetensors"))
    with torch.no_grad():
        o = m(torch.rand(1, 3, 320, 320))
    assert o["one2one"]["boxes"].shape == (1, 4, 40 * 40 + 20 * 20 + 10 * 10)
    r = DetRunner(os.path.join(WDIR, "detector.safetensors"), device="cpu")
    polys, confs = r.predict(np.full((200, 300, 3), 128, np.uint8), 0.25)
    assert polys.ndim == 3 and polys.shape[1:] == (4, 2) and len(polys) == len(confs)


@needs_weights
def test_cli_image_end_to_end(tmp_path, capsys):
    from pqdai.cli import main
    img = np.full((480, 640, 3), 120, np.uint8)
    cv2.rectangle(img, (200, 200), (420, 280), (210, 210, 210), -1)
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), img)
    rc = main(["image", "--image", str(p), "--out", str(tmp_path / "o.jpg"), "--json", str(tmp_path / "o.json"), "--device", "cpu"])
    assert rc == 0
    data = json.load(open(tmp_path / "o.json"))
    assert data["domain"] == "weld" and data["overall"] in ("GOOD", "BAD") and "objects" in data
    assert (tmp_path / "o.jpg").exists()


@needs_weights
def test_cli_verify_and_domains(capsys):
    from pqdai.cli import main
    assert main(["verify-weights"]) == 0
    assert main(["domains"]) == 0
    out = capsys.readouterr().out
    assert "weld" in out and "example_template" in out and "OK" in out


@needs_weights
def test_student_embeddings_align_with_bank():
    """The bundled bank lives in the 1280-d space of the released classifier."""
    from pqdai.core.reference_bank import ReferenceBank
    b = ReferenceBank.load(domains.get("weld").reference_bank_path)
    from pqdai.models.classifier import load_classifier
    m = load_classifier(os.path.join(WDIR, "classifier.safetensors"))
    assert b.dim == m.cfg.embed_dim
    assert sha256_file(os.path.join(WDIR, "classifier.safetensors"))


def test_verify_dir_without_manifest_is_actionable(tmp_path):
    import pytest
    from pqdai.core.weights import WeightsError, verify_dir
    with pytest.raises(WeightsError, match="git lfs"):
        verify_dir(str(tmp_path))


def test_cli_verify_weights_reports_failure(tmp_path, capsys):
    from pqdai.cli import main
    rc = main(["verify-weights", "--domain", "weld", "--dir", str(tmp_path)])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err
