"""Guards what the public repo may contain: no training code, no third-party detector stacks, no private strings."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = ["src", "webapp", "scripts", "tests", ".github"]
SKIP_FILES = {"test_public_surface.py"}
TEXT_EXT = {".py", ".js", ".html", ".css", ".sh", ".bat", ".command", ".yml", ".yaml", ".toml", ".md", ".json", ".cff", ".txt", ".cfg"}

FORBIDDEN_IMPORTS = [r"^\s*(from|import)\s+ultralytics\b", r"^\s*(from|import)\s+sahi\b"]
FORBIDDEN_STRINGS = [
    "C:\\Users", "X:\\work_space", "D:\\weld", "6309806", "cvas_legal_service", "welding-detection-system",
    "welding-quality-assessment", "weld_selective_paper", "weld-paper", "teacher_det", "teacher_cls", "det_rsi", "cls_runs",
    "rsi_detector", "rsi_classifier", "train_det", "train_cls", "DetKDLoss",
]
FORBIDDEN_PATHS = ["distill", "losses", "rsi_", "train_", "_kd.py", "det_data.py", "cls_data.py", "export_release.py"]


def _files():
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in TEXT_EXT and p.name not in SKIP_FILES:
                yield p


def test_no_ultralytics_or_sahi_imports():
    for p in _files():
        if p.suffix == ".py":
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in FORBIDDEN_IMPORTS:
                assert not re.search(pat, text, re.M), f"{p} imports a forbidden package"


def test_dependencies_are_clean():
    toml = (ROOT / "pyproject.toml").read_text()
    assert "ultralytics" not in toml and "sahi" not in toml


def test_no_training_modules_shipped():
    for p in list((ROOT / "src").rglob("*")) + list((ROOT / "scripts").rglob("*")):
        rel = str(p.relative_to(ROOT))
        if "__pycache__" in rel:
            continue
        for bad in FORBIDDEN_PATHS:
            assert bad not in rel, f"training-related path in public tree: {rel}"


def test_no_private_strings():
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for s in FORBIDDEN_STRINGS:
            assert s not in text, f"{p}: forbidden string {s!r}"


def test_no_korean_in_ui_and_generated_reports():
    """The web UI, the generated HTML reports and the user guide are English-only."""
    root = pathlib.Path(__file__).resolve().parents[1]
    hangul = re.compile("[가-힣]")
    files = list((root / "webapp").rglob("*.*")) + [root / "docs" / "user_guide.html"]
    files += list((root / "src" / "pqdai" / "video").glob("*.py"))
    bad = [str(f.relative_to(root)) for f in files if f.suffix in {".html", ".js", ".css", ".py"} and hangul.search(f.read_text(encoding="utf-8"))]
    assert not bad, f"Korean text left in: {bad}"


def test_preview_notice_present():
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("README.md", "README.ko.md", "docs/MODEL_CARD.md"):
        assert "v0.1" in (root / name).read_text(encoding="utf-8"), name
    card = (root / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "not safety-certified" in card or "not** safety-certified" in card
