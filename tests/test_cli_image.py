"""`pqdai image --out` must report a result image that could not be saved (CPU-only, stub detector/classifier)."""
import cv2
import pytest

from conftest import FakeClassifier, FakeDetector, box
from pqdai import cli
from pqdai.core.pipeline import QualityPipeline


@pytest.fixture
def stub_pipeline(monkeypatch, ref_bank):
    emb, labels = ref_bank
    polys = [box(100, 200, 60, 80), box(500, 200, 60, 80)]
    monkeypatch.setattr("pqdai.core.runtime.build_pipeline",
                        lambda *a, **kw: QualityPipeline(FakeDetector(polys), FakeClassifier(), emb, labels))


def test_image_out_is_written_and_exits_zero(stub_pipeline, synthetic_image, tmp_path, capsys):
    path, img = synthetic_image
    out = tmp_path / "result.jpg"
    assert cli.main(["image", "--image", path, "--out", str(out)]) == 0
    assert cv2.imread(str(out)).shape == img.shape
    assert "overall:" in capsys.readouterr().out


def test_image_out_in_missing_directory_fails(stub_pipeline, synthetic_image, tmp_path, capsys):
    """cv2.imwrite returns False (no exception) when the parent directory does not exist."""
    path, _ = synthetic_image
    out = tmp_path / "results" / "first-run" / "result.jpg"
    assert cli.main(["image", "--image", path, "--out", str(out)]) == 1
    assert not out.exists()
    assert str(out) in capsys.readouterr().err


def test_image_out_failure_skips_json_export(stub_pipeline, synthetic_image, tmp_path):
    """With --out and --json, a failed image save stops the command before the JSON export step."""
    path, _ = synthetic_image
    out = tmp_path / "missing-dir" / "result.jpg"
    js = tmp_path / "result.json"  # its folder exists, so a missing file means the export did not run
    assert cli.main(["image", "--image", path, "--out", str(out), "--json", str(js)]) == 1
    assert not js.exists()


def test_image_out_imwrite_false_fails(stub_pipeline, synthetic_image, tmp_path, monkeypatch, capsys):
    path, _ = synthetic_image
    out = tmp_path / "result.jpg"
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **kw: False)
    assert cli.main(["image", "--image", path, "--out", str(out)]) == 1
    assert str(out) in capsys.readouterr().err
