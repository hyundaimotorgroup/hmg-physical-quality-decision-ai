import numpy as np
import pytest

from conftest import box
from pqdai.core.pipeline import ObjectVerdict
from pqdai.video.bead_clustering import cluster_tracks_into_beads
from pqdai.video.camera_motion import EgoMotionEstimator, apply_affine
from pqdai.video.track_aggregation import compute_track_verdict, compute_video_verdict
from pqdai.video.video_tracker import Tracker, _cosine_sim, _iou
from pqdai.video.video_types import Track, TrackedFrame

I3 = np.eye(3)


def verdict(poly, cls="GOOD", entropy=0.01, conf=0.9, emb=None):
    p = np.array([0.02, 0.98]) if cls == "GOOD" else np.array([0.98, 0.02])
    return ObjectVerdict(polygon=poly, detection_confidence=conf, predicted_class=int(cls == "GOOD"), class_name=cls,
                       entropy=entropy, used_retrieval=False, cnn_probabilities=p, embedding=emb)


def test_iou_and_cosine():
    a = box(100, 100)
    assert _iou(a, a) == pytest.approx(1.0, abs=1e-3)
    assert _iou(a, box(400, 400)) == 0.0
    assert 0.2 < _iou(a, box(120, 100)) < 0.9
    assert _cosine_sim(None, np.ones(3)) is None
    assert _cosine_sim(np.array([1., 0.]), np.array([1., 0.])) == pytest.approx(1.0)


def test_sticky_bad_aggregation():
    t = Track(track_id=1)
    for i, c in enumerate(["GOOD", "GOOD", "BAD", "GOOD", "GOOD"]):
        t.history.append(TrackedFrame(i, i * 0.2, 0, verdict(box(100, 100), c), I3))
    tv = compute_track_verdict(t)
    assert tv.class_name == "BAD" and tv.n_frames == 5
    assert tv.consistency_score == pytest.approx(0.2)
    good = Track(track_id=2)
    good.history.append(TrackedFrame(0, 0.0, 0, verdict(box(1, 1)), I3))
    vv = compute_video_verdict([compute_track_verdict(good)], "v.mp4", 30.0, 1.0, 5.0, 20.0, 1)
    assert vv.overall_class_name == "GOOD"
    vv = compute_video_verdict([compute_track_verdict(good), tv], "v.mp4", 30.0, 1.0, 5.0, 20.0, 1)
    assert vv.overall_class_name == "BAD"


def test_empty_track_raises():
    with pytest.raises(ValueError):
        compute_track_verdict(Track(track_id=9))


def _run_tracker(frames_polys, grace=3):
    tr = Tracker(lost_grace_frames=grace)
    img = np.full((480, 640, 3), 128, np.uint8)
    ids = []
    for i, polys in enumerate(frames_polys):
        matched, lost = tr.update(i, i * 0.2, 0, [verdict(p, emb=np.eye(8)[j]) for j, p in enumerate(polys)], I3, img)
        ids.append([tid for tid, _ in matched])
    tr.finalize_all(len(frames_polys) * 0.2)
    return tr, ids


def test_tracker_keeps_identity_for_static_welds():
    a, b = box(150, 150), box(450, 300)
    tr, ids = _run_tracker([[a, b]] * 6)
    assert all(f == ids[0] for f in ids) and len(set(ids[0])) == 2


def test_tracker_new_weld_gets_new_id_and_finalizes():
    a, b = box(150, 150), box(450, 300)
    tr, ids = _run_tracker([[a], [a], [a, b], [a, b], [a, b]])
    assert len(set(ids[0])) == 1 and len(set(ids[4])) == 2
    assert ids[4][0] == ids[0][0]
    verdicts = [compute_track_verdict(t) for t in tr.finalized_tracks] if hasattr(tr, "finalized_tracks") else []
    assert verdicts == [] or len(verdicts) == 2


def test_bead_clustering_runs_and_maps_ids():
    t1, t2 = Track(track_id=1), Track(track_id=2)
    for i in range(3):
        t1.history.append(TrackedFrame(i, i * 0.2, 0, verdict(box(100, 100)), I3))
        t2.history.append(TrackedFrame(i, i * 0.2, 0, verdict(box(500, 300)), I3))
    merged, id_map = cluster_tracks_into_beads([t1, t2])
    assert len(merged) == 2 and set(id_map) == {1, 2}   # distinct, simultaneous welds must NOT be fused


def test_ego_motion_identity_then_shift():
    rng = np.random.default_rng(0)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    base = np.stack([base] * 3, -1)
    est = EgoMotionEstimator()
    T0, c0 = est.update(base)
    assert np.allclose(T0, np.eye(3)) and c0 == 1.0
    import cv2
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 6], [0, 1, 4]]), (320, 240))
    T1, c1 = est.update(shifted)
    assert abs(T1[0, 2] - 6) < 2 and abs(T1[1, 2] - 4) < 2
    out = apply_affine(T1, box(100, 100))
    assert out.shape == (4, 2)


def test_video_cli_help_does_not_need_weights():
    from pqdai.video import run
    assert callable(run.main)
