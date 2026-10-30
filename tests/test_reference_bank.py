import json
import warnings

import numpy as np
import pytest

from pqdai.core.reference_bank import BankError, ReferenceBank, build_from_crops


def _bank(n=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, (n, d)).astype(np.float32)
    a[:, 0] += 1
    b = rng.normal(0, 0.05, (n, d)).astype(np.float32)
    b[:, 1] += 1
    return ReferenceBank(np.concatenate([a, b]), np.array([0] * n + [1] * n), ["BAD", "GOOD"], [f"{i}.jpg" for i in range(2 * n)], {"built_for_model": "abc" * 20})


def test_roundtrip_schema2(tmp_path):
    b = _bank()
    p = tmp_path / "b.npz"
    b.save(str(p))
    b2 = ReferenceBank.load(str(p))
    assert len(b2) == 80 and b2.dim == 16 and b2.class_names == ["BAD", "GOOD"] and b2.meta["built_for_model"] == "abc" * 20
    assert b2.names[0] == "0.jpg"
    np.testing.assert_allclose(b2.embeddings, b.embeddings)


def test_no_absolute_paths_stored(tmp_path):
    b = _bank()
    b.names = [f"C:\\secret\\dir\\{i}.jpg" for i in range(len(b))]
    b.save(str(tmp_path / "b.npz"))
    z = np.load(tmp_path / "b.npz")
    assert all("\\" not in str(n) and "/" not in str(n) for n in z["names"])


def test_legacy_schema1_loads(tmp_path):
    p = tmp_path / "legacy.npz"
    np.savez(p, embeddings=np.eye(4, dtype=np.float32), labels=np.array([0, 0, 1, 1]),
             image_paths=np.array(["C:\\x\\BAD\\1.jpg", "C:\\x\\BAD\\2.jpg", "C:\\x\\GOOD\\3.jpg", "C:\\x\\GOOD\\4.jpg"]),
             metadata=np.array([{"embedding_dim": 4}], dtype=object))
    b = ReferenceBank.load(str(p))
    assert b.names == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"] and b.meta["legacy_schema"] == 1


def test_validation_errors():
    with pytest.raises(BankError):
        ReferenceBank(np.zeros((3, 4)), np.array([0, 1])).validate()
    with pytest.raises(BankError):
        ReferenceBank(np.zeros((2, 4)), np.array([0, 5])).validate()
    with pytest.raises(BankError):
        _bank().validate(expected_dim=99)
    with pytest.raises(BankError):
        _bank().validate(expected_classes=["A", "B"])
    with pytest.raises(BankError):
        ReferenceBank(np.full((2, 4), np.nan), np.array([0, 1])).validate()


def test_compat_warning():
    b = _bank()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b.check_compatible("different" * 8)
    assert any("rebuild" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b.check_compatible("abc" * 20)
    assert not w


def test_evaluate_query_and_ops():
    b = _bank()
    r = b.evaluate(3)
    assert r["loo_accuracy"] == 1.0 and r["per_class_recall"]["BAD"] == 1.0
    idx, sc, lab = b.query(b.embeddings[0] , 3)
    assert set(lab.tolist()) == {0}
    assert len(b.balanced(10)) == 20
    assert len(ReferenceBank.merge([b, b])) == 160
    dd = ReferenceBank.merge([b, b]).dedup(0.9999)
    assert len(dd) == len(b)
    assert len(b.add(b.embeddings[:2], [0, 1], ["x.jpg", "y.jpg"])) == 82
    with pytest.raises(BankError):
        b.add(np.zeros((1, 3)), [0])


def test_build_from_crops(tmp_path):
    import cv2

    class Clf:
        def predict_batch_with_uncertainty(self, imgs):
            return [{"embedding": np.array([float(i.mean()), 1.0, 0.0, 0.0], np.float32)} for i in imgs]

    for cls, val in (("BAD", 30), ("GOOD", 220)):
        d = tmp_path / "crops" / cls
        d.mkdir(parents=True)
        for i in range(5):
            cv2.imwrite(str(d / f"{i}.png"), np.full((20, 20, 3), val + i, np.uint8))
    bank = build_from_crops(str(tmp_path / "crops"), Clf(), ["BAD", "GOOD"], model_id="m1", dedup=None)
    assert len(bank) == 10 and bank.counts() == {"BAD": 5, "GOOD": 5} and bank.meta["built_for_model"] == "m1"
    assert bank.names[0] == "0.png"
    with pytest.raises(BankError):
        build_from_crops(str(tmp_path / "empty"), Clf(), ["BAD", "GOOD"])


def test_bundled_weld_bank():
    from pqdai import domains
    b = ReferenceBank.load(domains.get("weld").reference_bank_path)
    assert b.dim == 1280 and len(b) == 494 and b.counts() == {"BAD": 247, "GOOD": 247}
    assert all("\\" not in n and ":" not in n for n in b.names)
    assert json.dumps(b.meta, default=str)
