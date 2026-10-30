"""``pqdai`` command line: image / video / serve / bank / domains / verify-weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _add_model_args(p):
    p.add_argument("--domain", default="weld")
    p.add_argument("--device", default=None)
    p.add_argument("--detector-weights", default=None)
    p.add_argument("--classifier-weights", default=None)
    p.add_argument("--reference-bank", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--verify-conf", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--k", type=int, default=None)


COLORS = {"GOOD": (0, 200, 0), "BAD": (0, 0, 255)}


def draw_verdict(image, verdict, spec=None):
    import cv2
    canvas = image.copy()
    unsafe = verdict.unsafe_class
    for obj in verdict.objects:
        color = (0, 0, 255) if obj.class_name == unsafe else (0, 200, 0)
        pts = obj.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 3)
        p = float(obj.cnn_probabilities[0 if obj.class_name == unsafe else -1])
        cv2.putText(canvas, f"{obj.class_name} {p:.2f}", (int(pts[0, 0, 0]), max(int(pts[0, 0, 1]) - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    color = (0, 0, 255) if verdict.overall_class_name == unsafe else (0, 200, 0)
    cv2.putText(canvas, f"OVERALL: {verdict.overall_class_name} ({len(verdict.objects)} object(s))", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return canvas


def cmd_image(a):
    import cv2

    from .core.runtime import build_pipeline
    pipe = build_pipeline(a.domain, a.device, a.detector_weights, a.classifier_weights, a.reference_bank, a.conf, a.verify_conf, a.alpha, a.k,
                          use_retrieval=not a.no_retrieval)
    verdict = pipe.judge_image(a.image)
    print(f"overall: {verdict.overall_class_name}  ({len(verdict.objects)} object(s))")
    for i, o in enumerate(verdict.objects):
        print(f"  #{i}: {o.class_name}  det_conf={o.detection_confidence:.2f}  entropy={o.entropy:.3f}  retrieval={o.used_retrieval}")
    if a.out:
        cv2.imwrite(a.out, draw_verdict(cv2.imread(a.image), verdict))
    if a.json:
        payload = {"image": a.image, "domain": a.domain, "overall": verdict.overall_class_name, "objects": [
            {"polygon": o.polygon.tolist(), "class": o.class_name, "detection_confidence": o.detection_confidence,
             "entropy": o.entropy, "used_retrieval": o.used_retrieval,
             "probabilities": [float(x) for x in o.cnn_probabilities]} for o in verdict.objects]}
        Path(a.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def cmd_video(a, rest):
    from .video import run
    sys.argv = ["pqdai video"] + rest
    run.main()
    return 0


def cmd_serve(a):
    try:
        import uvicorn
    except ImportError:
        print("install the web extra: pip install 'physical-quality-decision-ai[web]'", file=sys.stderr)
        return 2
    import os
    import sys as _s
    webapp = Path(__file__).resolve().parents[2] / "webapp" / "backend"
    if not webapp.exists():
        print(f"webapp not found at {webapp} (run from a source checkout)", file=sys.stderr)
        return 2
    os.environ["PQDAI_DOMAIN"] = a.domain
    _s.path.insert(0, str(webapp))
    uvicorn.run("main:app", host=a.host, port=a.port, app_dir=str(webapp))
    return 0


def cmd_bank(a):
    from . import domains
    from .core.reference_bank import ReferenceBank, build_from_crops
    from .core.weights import read_manifest, verify_weights
    spec = domains.get(a.domain)
    if a.bank_cmd == "build":
        from .core.runtime import build_pipeline  # noqa: F401 - validates the domain loads
        cpath = spec.classifier_path(a.classifier_weights)
        m = verify_weights(cpath)
        clf = spec.classifier_factory(cpath, a.device)
        bank = build_from_crops(a.crops, clf, spec.class_names, model_id=m["sha256"], dedup=None if a.no_dedup else a.dedup,
                                balance=a.balance, preprocess=None if a.raw else spec.preprocess_fn,
                                extra_meta={"domain": a.domain, "classifier_version": m.get("version")})
        bank.save(a.out)
        print(f"wrote {a.out}: {len(bank)} embeddings, dim {bank.dim}, {bank.counts()}")
        print(json.dumps(bank.evaluate(a.k), indent=2))
    elif a.bank_cmd == "eval":
        bank = ReferenceBank.load(a.bank or spec.reference_bank_path)
        print(json.dumps(dict(bank.evaluate(a.k), meta=bank.meta), indent=2, default=str))
    elif a.bank_cmd == "info":
        bank = ReferenceBank.load(a.bank or spec.reference_bank_path)
        print(json.dumps(dict(n=len(bank), dim=bank.dim, classes=bank.class_names, counts=bank.counts(), meta=bank.meta), indent=2, default=str))
    elif a.bank_cmd == "merge":
        merged = ReferenceBank.merge([ReferenceBank.load(p) for p in a.inputs])
        merged.save(a.out)
        print(f"wrote {a.out}: {len(merged)} embeddings")
    _ = read_manifest
    return 0


def cmd_domains(a):
    from . import domains
    for n in domains.available():
        s = domains.get(n)
        print(f"{n:18s} classes={list(s.class_names)} unsafe={s.unsafe_class}  {s.description}")
    return 0


def cmd_verify(a):
    from . import domains
    from .core.weights import verify_dir
    spec = domains.get(a.domain)
    d = a.dir or spec.weights_dir
    from .core.weights import WeightsError
    try:
        results = verify_dir(d)
    except WeightsError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        print("How to get the weights: they are bundled in weights/<domain>/ via git-lfs "
              "(`git lfs install && git lfs pull`); see weights/README.md.", file=sys.stderr)
        return 1
    for r in results:
        print(f"OK  {r['name']:12s} sha256={r['sha256'][:16]}... params={r['params']} status={r['status']} version={r['version']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pqdai", description="Physical Quality Decision AI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image", help="judge one photo")
    p.add_argument("--image", required=True)
    p.add_argument("--out")
    p.add_argument("--json")
    p.add_argument("--no-retrieval", action="store_true")
    _add_model_args(p)
    p.set_defaults(fn=cmd_image)
    sub.add_parser("video", help="judge a video (all arguments are forwarded to pqdai.video.run)")
    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--domain", default="weld")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("bank", help="reference bank tools")
    bs = p.add_subparsers(dest="bank_cmd", required=True)
    b = bs.add_parser("build")
    b.add_argument("--domain", default="weld")
    b.add_argument("--crops", required=True, help="folder with one sub-folder per class name")
    b.add_argument("--out", required=True)
    b.add_argument("--classifier-weights")
    b.add_argument("--device")
    b.add_argument("--dedup", type=float, default=0.995)
    b.add_argument("--no-dedup", action="store_true")
    b.add_argument("--balance", action="store_true")
    b.add_argument("--raw", action="store_true", help="skip the domain preprocessing")
    b.add_argument("--k", type=int, default=3)
    for nm in ("eval", "info"):
        e = bs.add_parser(nm)
        e.add_argument("--domain", default="weld")
        e.add_argument("--bank")
        e.add_argument("--k", type=int, default=3)
    mg = bs.add_parser("merge")
    mg.add_argument("inputs", nargs="+")
    mg.add_argument("--out", required=True)
    mg.add_argument("--domain", default="weld")
    p.set_defaults(fn=cmd_bank)
    p = sub.add_parser("domains", help="list domains")
    p.set_defaults(fn=cmd_domains)
    p = sub.add_parser("verify-weights", help="check sha256 of released weights")
    p.add_argument("--domain", default="weld")
    p.add_argument("--dir")
    p.set_defaults(fn=cmd_verify)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "video":
        return cmd_video(None, argv[1:])
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
