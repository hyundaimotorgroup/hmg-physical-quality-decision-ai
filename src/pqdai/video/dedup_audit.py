"""QA visualization: per-weld (bead) identity/count audit.

Built to answer one question a human reviewer can't get from report.html or
the plain overlay video: "as I page through the sampled frames one at a
time, is every weld counted exactly once into the final GOOD/BAD tally --
not zero times (missed), and not twice (a duplicate)? And for any one
weld, is EVERY crop labeled with its number really the same physical
weld, or did the tracker drift onto something else along the way?"

Numbering (2026-09-11, fixed after user confusion): every weld is
identified by `weld_no` -- the order it was FIRST counted (1, 2, 3, ...) --
and this SAME number is used everywhere: burned into the "id=" label on
every box in every frame image, in the "[NEW #k]"/"[counted]" tag, and in
the per-weld index below. Earlier versions used bead_clustering.py's
internal bead_id for the on-image label while the index used something
else, which meant the numbers in the index never matched what was on
screen -- there is now exactly one numbering scheme.

Single interactive view (2026-09-14, rebuilt to match the sidebar+canvas+
detail-panel interaction model of view_user_photos_conf01_verified.html
instead of a read-only slider + separate gallery modal):
- Left sidebar lists every sampled frame, expandable into that frame's
  weld rows (id, GOOD/BAD, `[NEW]`/`[counted]`, dup-suspect marker, deleted
  strikethrough). Filters for BAD-containing / has-new / dup-suspect /
  has-a-deletion frames.
- Canvas loads the frame's CLEAN source image (not a pre-burned overlay)
  and draws every box live from `frames[i].entries[].polygon`, so a click
  actually hits a specific box: NEW boxes get a thicker amber-highlighted
  outline, already-counted boxes a plain one, held (interpolated) boxes a
  dashed one. Clicking a box or a sidebar row selects it (yellow ring) and
  opens the detail panel with a client-side crop thumbnail (same
  bounding-box-crop technique as the photo viewer's `updateCropThumb`) and
  GOOD/BAD/UNCERTAIN relabel buttons and a "Delete (not an object)" toggle for a
  detection that isn't actually a weld. Both are review-only annotations,
  persisted to localStorage only -- they never rewrite `weld_no`, the
  GOOD/BAD tally, or any file this module already produces. "Export report + edits" (`exportAll`) downloads both a raw edit log
  (`exportEdits`/`dedup_audit_edits.json`) and a summary report
  (`buildReportHtml`/`dedup_audit_final_report.html`) -- overall GOOD/BAD
  tally with those edits applied (see `computeFinalWeldResults`'s docstring
  for the exact exclusion/aggregation rule) plus a per-weld card grid (id,
  GOOD/BAD/UNCERTAIN, confirming frame, crop), the hand-off/slide-ready
  "here's the final result" artifact meant to be produced once review is
  done. Crop `<img>`s use an absolute `file://` URL back into this results
  folder's `dedup_gallery/` (see `absoluteFileUrl`'s docstring for why NOT
  a self-contained base64 embed) -- keep the report next to that folder,
  don't move it alone.
- The old per-weld crop gallery (`openWeld`) and duplicate-candidate list
  are kept, now reachable from the toolbar/detail panel instead of being
  laid out as page sections, since the rest of the page is now a
  single-screen app rather than a scrollable report.

This requires the FULL per-frame box list (not just a NEW-id summary and a
handful of sampled gallery crops), so `generate_dedup_audit` now serializes
`frames_meta[i]['entries']` -- every box's weld_no/class/kind/is_new/polygon
for that exact frame -- into `dedup_trace.json`, which earlier versions of
this module didn't persist (that data only existed transiently in
`pqdai/video/run.py`'s `frame_overlay_data`/`frame_lost_data`).

Secondary: a duplicate-candidate cross-check. Deliberately NOT a reuse of
bead_clustering.py's `_can_merge` (tried first, reverted 2026-09-11): that
function is validated for checking ONE candidate pair at a time (a track
that just went quiet against a track that just appeared), where its
REID_MIN_EMBEDDING_SIM=0.6 and "collinear fragment" path are calibrated
tradeoffs for a narrow comparison. Run EXHAUSTIVELY across every pair of
final welds in a whole video (steps that _can_merge was never exercised
against), that same 0.6 threshold plus the collinear path -- which merges
on "same orientation/width, small gap, small perpendicular offset" without
requiring the SAME location -- produces mass false positives: a run with
258 welds flagged 64 "duplicate" pairs, and manual crop comparison showed
they were a sequence of genuinely different, individually-correct welds
along the same seam that all happen to be visually similar and
collinear-adjacent to their neighbors (measured 2026-09-11). `_strict_duplicate_check`
below only uses `_can_merge`'s Path 1 (re-identification: SAME
camera-motion-compensated location, not just "collinear-looking"), at the
embedding bar live tracking itself uses (0.85, not 0.6) -- a real
duplicate should look like the exact same weld reappearing in the exact
same place, not merely "a similar-looking weld somewhere nearby."
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .video_types import Track
from .video_tracker import _iou, _cosine_sim
from .bead_clustering import _project_polygon, REID_IOU_THRESHOLD, MAX_TIME_GAP_SEC

STRICT_DUPLICATE_MIN_EMBEDDING_SIM = 0.85  # matches MIN_EMBEDDING_SIM_FOR_PROXIMITY_MATCH in video_tracker.py, not bead_clustering's lenient 0.6


def _strict_duplicate_check(a: Track, b: Track) -> bool:
    """Stricter, exhaustive-search-safe version of bead_clustering.py's
    `_can_merge` -- see module docstring for why the original isn't reused
    as-is here. Only the position-based re-identification path, at the
    live-tracking embedding bar."""
    a_end, b_end = a.presence_end_sec, b.presence_end_sec
    overlap = min(a_end, b_end) - max(a.first_seen_sec, b.first_seen_sec)
    if overlap > 0:
        return False  # genuinely co-existed on screen -- definitely different welds
    if -overlap > MAX_TIME_GAP_SEC:
        return False
    if a_end <= b.first_seen_sec:
        earlier, later = a, b
    elif b_end <= a.first_seen_sec:
        earlier, later = b, a
    else:
        return False
    later_first = later.history[0]
    projected = _project_polygon(
        earlier.last_polygon, earlier.history[-1].cumulative_transform, later_first.cumulative_transform,
    )
    emb_sim = _cosine_sim(earlier.anchor_embedding, later_first.verdict.embedding)
    if emb_sim is None or emb_sim < STRICT_DUPLICATE_MIN_EMBEDDING_SIM:
        return False
    return _iou(projected, later_first.verdict.polygon) >= REID_IOU_THRESHOLD

NEW_HIGHLIGHT_COLOR = (0, 220, 255)  # BGR amber
COLOR_GOOD = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
CROP_PADDING_PX = 15
MAX_GALLERY_THUMBS = 12  # evenly sampled across the weld's lifetime if it has more appearances than this


def _draw_dashed_polygon(canvas: np.ndarray, pts: np.ndarray, color, thickness: int, dash_len: int = 10) -> None:
    pts = pts.reshape(-1, 2)
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        seg_len = float(np.linalg.norm(p2 - p1))
        n_dashes = max(1, int(seg_len // dash_len))
        for d in range(0, n_dashes, 2):
            a = p1 + (p2 - p1) * (d / n_dashes)
            b = p1 + (p2 - p1) * (min(d + 1, n_dashes) / n_dashes)
            cv2.line(canvas, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def _crop_polygon(frame_bgr: np.ndarray, polygon: np.ndarray, pad: int = CROP_PADDING_PX) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = max(0, int(polygon[:, 0].min()) - pad)
    x1 = min(w, int(polygon[:, 0].max()) + pad)
    y0 = max(0, int(polygon[:, 1].min()) - pad)
    y1 = min(h, int(polygon[:, 1].max()) + pad)
    if x1 <= x0 or y1 <= y0:
        return frame_bgr[0:1, 0:1]
    return frame_bgr[y0:y1, x0:x1]


def _evenly_sample(items: List, n: int) -> List:
    if len(items) <= n:
        return items
    idxs = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idxs.tolist()))]


def generate_dedup_audit(
    frame_overlay_data: Dict[int, List[Tuple[int, object]]],
    frame_lost_data: Dict[int, List[Tuple[int, np.ndarray, object, float]]],
    track_id_to_bead_id: Dict[int, int],
    finalized_tracks: List[Track],
    frames_dir: Path,
    out_dir: Path,
    rerun_note: str = None,
    video_path: str = None,
) -> None:
    audit_frames_dir = out_dir / 'dedup_frames'
    gallery_dir = out_dir / 'dedup_gallery'
    audit_frames_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    bead_to_raw_tracks: Dict[int, List[int]] = {}
    for tid, bid in track_id_to_bead_id.items():
        bead_to_raw_tracks.setdefault(bid, []).append(tid)
    tracks_by_id = {t.track_id: t for t in finalized_tracks}

    all_frame_indices = sorted(set(frame_overlay_data) | set(frame_lost_data))

    # ---- per-frame entries, per bead_id, in a uniform shape ----
    per_frame_entries: Dict[int, List[Tuple[int, np.ndarray, str, str]]] = {}
    for frame_idx in all_frame_indices:
        entries = []
        for bead_id, verdict in frame_overlay_data.get(frame_idx, []):
            entries.append((bead_id, verdict.polygon, verdict.class_name, 'real'))
        for bead_id, poly, verdict, _std in frame_lost_data.get(frame_idx, []):
            entries.append((bead_id, poly, verdict.class_name, 'held'))
        per_frame_entries[frame_idx] = entries

    # ---- discovery order -> weld_no, THE single numbering used everywhere ----
    weld_no_by_bead: Dict[int, int] = {}
    for frame_idx in all_frame_indices:
        for bead_id, _poly, _cls, _kind in per_frame_entries[frame_idx]:
            if bead_id not in weld_no_by_bead:
                weld_no_by_bead[bead_id] = len(weld_no_by_bead) + 1

    # ---- per-weld (by weld_no) chronological entries, for the gallery ----
    weld_entries: Dict[int, List[Tuple[int, np.ndarray, str, str]]] = {}
    for frame_idx in all_frame_indices:
        for bead_id, poly, cls, kind in per_frame_entries[frame_idx]:
            weld_no = weld_no_by_bead[bead_id]
            weld_entries.setdefault(weld_no, []).append((frame_idx, poly, cls, kind))

    # ---- duplicate-candidate check (see _strict_duplicate_check's docstring for why
    # this is NOT bead_clustering's own _can_merge run exhaustively) ----
    bead_ids_sorted = sorted(bead_to_raw_tracks.keys())
    duplicate_candidates = []
    for i in range(len(bead_ids_sorted)):
        for j in range(i + 1, len(bead_ids_sorted)):
            bid_a, bid_b = bead_ids_sorted[i], bead_ids_sorted[j]
            tracks_a = [tracks_by_id[t] for t in bead_to_raw_tracks.get(bid_a, []) if t in tracks_by_id]
            tracks_b = [tracks_by_id[t] for t in bead_to_raw_tracks.get(bid_b, []) if t in tracks_by_id]
            found = False
            for ta in tracks_a:
                for tb in tracks_b:
                    try:
                        if _strict_duplicate_check(ta, tb):
                            duplicate_candidates.append((weld_no_by_bead[bid_a], weld_no_by_bead[bid_b]))
                            found = True
                            break
                    except Exception:
                        continue
                if found:
                    break

    # ---- render full frames: every box labeled by weld_no, NEW vs counted ----
    seen_welds = set()
    good_count = 0
    bad_count = 0
    frames_meta = []

    for frame_idx in all_frame_indices:
        frame_bgr = cv2.imread(str(frames_dir / f"frame_{frame_idx:06d}.jpg"))
        if frame_bgr is None:
            continue
        canvas = frame_bgr.copy()
        new_this_frame = []
        entries_meta = []

        for bead_id, polygon, class_name, kind in per_frame_entries[frame_idx]:
            weld_no = weld_no_by_bead[bead_id]
            is_new = weld_no not in seen_welds
            if is_new:
                seen_welds.add(weld_no)
                if class_name == 'BAD':
                    bad_count += 1
                else:
                    good_count += 1
                new_this_frame.append(weld_no)

            entries_meta.append({
                "weld_no": weld_no,
                "class_name": class_name,
                "kind": kind,
                "is_new": is_new,
                "polygon": polygon.tolist(),
            })

            color = COLOR_BAD if class_name == 'BAD' else COLOR_GOOD
            pts = polygon.astype(np.int32).reshape(-1, 2)
            thickness = 4 if is_new else 2
            if kind == 'real':
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)
            else:
                _draw_dashed_polygon(canvas, pts, color, thickness)
            if is_new:
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=NEW_HIGHLIGHT_COLOR, thickness=1)

            label = f"id={weld_no} {class_name} [NEW #{weld_no}]" if is_new else f"id={weld_no} {class_name} [counted]"
            origin = tuple(pts[0])
            label_color = NEW_HIGHLIGHT_COLOR if is_new else color
            cv2.putText(canvas, label, (origin[0], max(origin[1] - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

        banner1 = f"frame_{frame_idx:06d}  |  detections this frame: {len(per_frame_entries[frame_idx])}  |  newly counted: +{len(new_this_frame)}"
        banner2 = f"cumulative unique welds so far: {len(seen_welds)}  (GOOD {good_count} / BAD {bad_count})"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (20, 20, 20), -1)
        cv2.putText(canvas, banner1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, banner2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        out_name = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(audit_frames_dir / out_name), canvas)
        frames_meta.append({
            "frame_idx": frame_idx,
            "file": f"dedup_frames/{out_name}",
            "clean_file": f"frames/{out_name}",  # frames_dir already holds the pre-overlay source JPGs
            "n_detections": len(per_frame_entries[frame_idx]),
            "n_new": len(new_this_frame),
            "new_weld_nos": new_this_frame,
            "cumulative_total": len(seen_welds),
            "entries": entries_meta,
        })

    # ---- render per-weld galleries ----
    frame_cache: Dict[int, np.ndarray] = {}

    def get_frame(fid: int):
        if fid not in frame_cache:
            frame_cache[fid] = cv2.imread(str(frames_dir / f"frame_{fid:06d}.jpg"))
        return frame_cache[fid]

    gallery_meta = {}
    for weld_no, entries in weld_entries.items():
        sampled = _evenly_sample(entries, MAX_GALLERY_THUMBS)
        thumbs = []
        for frame_idx, polygon, class_name, kind in sampled:
            frame_bgr = get_frame(frame_idx)
            if frame_bgr is None:
                continue
            crop = _crop_polygon(frame_bgr, polygon)
            thumb_name = f"weld{weld_no:04d}_fid{frame_idx:06d}.jpg"
            cv2.imwrite(str(gallery_dir / thumb_name), crop)
            thumbs.append({"frame_idx": frame_idx, "kind": kind, "class_name": class_name,
                            "file": f"dedup_gallery/{thumb_name}"})
        final_class = 'BAD' if any(e[2] == 'BAD' for e in entries) else 'GOOD'
        bead_id = next(bid for bid, wn in weld_no_by_bead.items() if wn == weld_no)
        gallery_meta[weld_no] = {
            "final_class": final_class,
            "n_frames": len(entries),
            "first_frame_idx": entries[0][0],
            "last_frame_idx": entries[-1][0],
            "thumbs": thumbs,
            "bead_id": bead_id,
            "member_track_ids": bead_to_raw_tracks.get(bead_id, []),
        }

    # ---- JSON trace ----
    trace = {
        "frames": frames_meta,
        "final_unique_count": len(seen_welds),
        "final_good": good_count,
        "final_bad": bad_count,
        "objects": gallery_meta,
        "duplicate_candidates": [{"weld_a": a, "weld_b": b} for a, b in duplicate_candidates],
        "video_path": video_path,
    }
    with open(out_dir / 'dedup_trace.json', 'w', encoding='utf-8') as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    _write_html(out_dir, frames_meta, gallery_meta, duplicate_candidates, good_count, bad_count, rerun_note, video_path)


def _write_html(out_dir, frames_meta, gallery_meta, duplicate_candidates, good_count, bad_count, rerun_note=None, video_path=None):
    frames_json = json.dumps(frames_meta)
    gallery_json = json.dumps(gallery_meta)
    dup_json = json.dumps([{"a": a, "b": b} for a, b in duplicate_candidates])
    n_welds = len(gallery_meta)

    # Plain string + placeholder substitution (not an f-string): the template body
    # below is mostly literal CSS/JS braces, which would need constant {{ }} escaping
    # in an f-string -- see view_user_photos_conf01_verified.html for the sibling
    # photo-review tool this page's interaction model (sidebar + canvas + detail
    # panel) is deliberately modeled on.
    template = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>Object count audit (Dedup Audit)</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',Arial,sans-serif; background:#1e1e1e; color:#d4d4d4; display:flex; height:100vh; width:100vw; overflow:hidden; }
  #sidebar { width:340px; min-width:340px; background:#252526; border-right:1px solid #3e3e42; display:flex; flex-direction:column; overflow:hidden; }
  #sidebar-header { padding:14px 16px; border-bottom:1px solid #3e3e42; }
  #sidebar-header h1 { font-size:14px; margin-bottom:6px; }
  .video-path { font-size:11px; color:#9d9d9d; word-break:break-all; margin-top:2px; }
  #summary { font-size:11px; color:#c9c9c9; margin-top:8px; line-height:1.7; }
  #summary b.bad { color:#f14c4c; } #summary b.good { color:#4ec9b0; }
  #filters { display:flex; gap:6px; flex-wrap:wrap; padding:10px 16px; border-bottom:1px solid #3e3e42; }
  #filters button { background:#3e3e42; color:#d4d4d4; border:none; padding:5px 10px; cursor:pointer; border-radius:3px; font-size:11px; }
  #filters button.active { background:#0e639c; color:#fff; }
  #list { flex:1; overflow-y:auto; }
  .list-item { padding:8px 16px; border-bottom:1px solid #2d2d30; cursor:pointer; font-size:12px; }
  .list-item:hover { background:#2a2d2e; }
  .list-item.active { background:#04395e; }
  .list-item .fname { display:flex; justify-content:space-between; gap:6px; font-family:monospace; }
  .list-item .meta { color:#9d9d9d; font-size:11px; margin-top:2px; }
  .badge { font-size:10px; font-weight:bold; padding:1px 6px; border-radius:3px; flex-shrink:0; }
  .badge.BAD { background:#5a1d1d; color:#ff8080; }
  .badge.GOOD { background:#1d4a3a; color:#7cffc4; }
  .badge.UNCERTAIN { background:#5a4412; color:#ffe6a8; }
  .weld-item { padding:4px 16px 4px 30px; border-bottom:1px solid #262626; cursor:pointer; font-size:11px; display:flex; align-items:center; gap:6px; color:#bbb; }
  .weld-item:hover { background:#2a2d2e; }
  .weld-item.selected { background:#5a4a12; box-shadow:inset 0 0 0 1px #ffe600; }
  .weld-item.deleted { opacity:0.5; }
  .weld-item.deleted .widx { text-decoration:line-through; }
  .weld-item.dimmed { opacity:0.35; filter:grayscale(1) blur(0.4px); }
  .weld-item .widx { color:#888; font-family:monospace; min-width:46px; }
  .weld-item .new-mark { color:#ffd23f; font-size:10px; }
  .weld-item .counted-mark { color:#888; font-size:10px; }
  .weld-item .del-mark { color:#999; font-size:10px; }
  .weld-item .dup-mark { color:#ff8080; font-size:10px; }
  .weld-item .relabel-mark { color:#4ea8de; font-size:10px; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #toolbar { background:#252526; border-bottom:1px solid #3e3e42; padding:10px 16px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; flex-shrink:0; }
  .fname-display { font-size:13px; margin-right:auto; font-family:monospace; }
  button { background:#0e639c; color:#fff; border:none; padding:6px 12px; cursor:pointer; border-radius:3px; font-size:12px; }
  button:hover { background:#1177bb; }
  button.toggle.off { background:#3e3e42; color:#9d9d9d; }
  button.danger { background:#8a3a3a; }
  #canvas-area { flex:1; min-height:0; overflow:auto; background:#141414; cursor:grab; position:relative; }
  #canvas-area.panning { cursor:grabbing; }
  canvas { display:block; }
  #legend { background:#252526; border-top:1px solid #3e3e42; padding:8px 16px; font-size:11px; color:#c9c9c9; display:flex; gap:16px; flex-wrap:wrap; flex-shrink:0; align-items:center; }
  .swatch { display:inline-block; width:10px; height:10px; margin-right:5px; vertical-align:middle; }
  #detail-panel { position:absolute; top:12px; right:12px; background:#252526ee; border:1px solid #3e3e42; border-radius:4px; padding:10px 12px; font-size:12px; min-width:280px; max-width:340px; display:none; max-height:calc(100% - 24px); overflow-y:auto; box-shadow:0 4px 12px rgba(0,0,0,0.5); }
  #cropThumbWrap { display:none; width:100%; height:220px; overflow:hidden; position:relative; background:#000; border:1px solid #3e3e42; border-radius:3px; margin-bottom:8px; }
  #cropThumb { position:absolute; left:0; top:0; }
  #detail-panel h3 { font-size:12px; margin-bottom:6px; }
  #detail-panel .row { display:flex; justify-content:space-between; gap:12px; margin:2px 0; color:#c9c9c9; }
  #detail-panel .row b { color:#fff; }
  #detail-panel .close { position:absolute; top:6px; right:8px; cursor:pointer; color:#9d9d9d; }
  .detail-actions { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
  .detail-actions button { font-size:11px; padding:4px 8px; }
  .detail-actions button.active-label { outline:2px solid #fff; }
  .detail-hint { margin-top:8px; color:#9d9d9d; font-size:11px; line-height:1.5; }
  .dup-warn { background:#3a1d1d; border:1px solid #5a2f2f; border-radius:6px; padding:8px 10px; margin-top:8px; color:#ffd7d7; font-size:11px; line-height:1.5; }
  a { color:#4fc3f7; }
  #galleryModal, #dupModal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:50; overflow-y:auto; }
  #galleryModal .modal-inner, #dupModal .modal-inner { max-width:1000px; margin:40px auto; background:#1e1e1e; border:1px solid #3e3e42; border-radius:10px; padding:24px; }
  .modal-close { float:right; background:#2d2d30; color:#fff; border:1px solid #3e3e42; border-radius:6px; padding:6px 14px; cursor:pointer; }
  .gthumbs { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
  .gthumb { text-align:center; font-size:11px; color:#9d9d9d; }
  .gthumb img { display:block; width:160px; height:160px; object-fit:contain; background:#000; border:2px solid #3e3e42; border-radius:4px; }
  .gthumb.bad img { border-color:#b23b3b; } .gthumb.good img { border-color:#2f9e5f; }
  .gthumb.held { opacity:0.75; }
  #dupModal table { width:100%; border-collapse:collapse; font-size:12px; }
  #dupModal th, #dupModal td { border-bottom:1px solid #3e3e42; padding:6px; text-align:left; }
  #rerunNote { background:#5a4412; border:1px solid #7a5a12; border-radius:6px; padding:8px 10px; margin-top:8px; color:#ffe6a8; font-size:11px; line-height:1.5; }
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Object count audit (frame viewer)</h1>
    <div class="video-path">video: __VIDEO_PATH__</div>
    __RERUN_NOTE_HTML__
    <div id="summary"></div>
  </div>
  <div id="filters">
    <button id="filterAll" class="active">All</button>
    <button id="filterBad">BAD only</button>
    <button id="filterGood">GOOD only</button>
    <button id="filterNew">New only</button>
    <button id="filterDup">Suspected duplicates</button>
    <button id="filterEdited">Edited</button>
  </div>
  <div id="list"></div>
</div>

<div id="main">
  <div id="toolbar">
    <button id="prevFrameBtn">◀◀ Previous frame</button>
    <button id="prevBtn">◀ Previous object (A)</button>
    <button id="nextBtn">Next object (D) ▶</button>
    <button id="nextFrameBtn">Next frame ▶▶</button>
    <button id="zoomOutBtn">－</button>
    <span id="zoomLabel" style="font-size:12px;color:#9d9d9d;">100%</span>
    <button id="zoomInBtn">＋</button>
    <button id="zoomResetBtn">Fit to screen</button>
    <button id="toggleLabels" class="toggle">Label text (L)</button>
    <button id="dupListBtn">Duplicate candidates (__N_DUP__)</button>
    <button id="exportBtn">Export report + edits (E)</button>
    <button id="resetEditsBtn" class="danger">Reset all edits</button>
    <span class="fname-display" id="fnameDisplay"></span>
  </div>
  <div id="canvas-area">
    <div id="canvas-wrapper" style="display:inline-block;">
      <canvas id="canvas"></canvas>
    </div>
    <div id="detail-panel">
      <span class="close" id="detailClose">✕</span>
      <h3>Object details</h3>
      <div id="cropThumbWrap"><img id="cropThumb" alt=""></div>
      <div id="detail-body"></div>
    </div>
  </div>
  <div id="legend">
    <span><span class="swatch" style="background:#4ec9b0"></span>GOOD</span>
    <span><span class="swatch" style="background:#f14c4c"></span>BAD</span>
    <span><span class="swatch" style="background:#e0a72a"></span>UNCERTAIN (relabeled by a reviewer as ambiguous)</span>
    <span><span class="swatch" style="background:#ffd23f"></span>yellow highlight + [NEW] = counted for the first time in this frame</span>
    <span><span class="swatch" style="background:#888;border:1px dashed #999"></span>dashed = held (re-detection failed, estimated position)</span>
    <span><span class="swatch" style="background:#888;border:1px dashed #ccc"></span>thick gray dashed = marked deleted (not an object; affects only this view / the export)</span>
    <span style="color:#9d9d9d;">Click a box or a row on the left = select · A/D (←/→) = previous/next object (crosses frame boundaries) · relabel as GOOD/BAD/UNCERTAIN in the details panel · wheel-drag = pan</span>
  </div>
</div>

<div id="galleryModal">
  <div class="modal-inner">
    <button class="modal-close" onclick="closeGallery()">Close ✕</button>
    <h2 id="galleryTitle" style="margin-top:0"></h2>
    <div class="gthumbs" id="galleryThumbs"></div>
  </div>
</div>

<div id="dupModal">
  <div class="modal-inner" style="max-width:600px;">
    <button class="modal-close" onclick="document.getElementById('dupModal').style.display='none'">Close ✕</button>
    <h2 style="margin-top:0">Potential duplicate candidates (manual check required)</h2>
    <p style="color:#9d9d9d;font-size:12px;">The pairs below were counted under different numbers, but re-checking them with the merge criteria of bead_clustering.py suggests they "may be the same physical object". Open the gallery to compare them yourself.</p>
    <div id="dupList"></div>
  </div>
</div>

<script>
const frames = __FRAMES_JSON__;
const welds = __GALLERY_JSON__;
const duplicateCandidates = __DUP_JSON__;
const GOOD_COUNT = __GOOD_COUNT__;
const BAD_COUNT = __BAD_COUNT__;
const N_WELDS = __N_WELDS__;

const EDIT_STORAGE_KEY = "weld_dedup_audit_edit_state_v1";

let currentIndex = 0;
let zoom = 1, fitZoom = 1;
let showLabels = true;
let selected = null; // index into filteredData[currentIndex].entries
const img = new Image();

let filteredData = frames;
// editState: { [frame_idx]: { deleted: { [weld_no]: true } } } -- review-only
// annotations ("this box isn't really a weld"); never mutates frames/welds
// themselves, only what's shown on screen and exported.
let editState = JSON.parse(localStorage.getItem(EDIT_STORAGE_KEY) || "{}");

const dupIds = new Set();
duplicateCandidates.forEach(d => { dupIds.add(d.a); dupIds.add(d.b); });
function dupPartnersOf(weldNo) {
  return duplicateCandidates.filter(d => d.a === weldNo || d.b === weldNo).map(d => d.a === weldNo ? d.b : d.a);
}

function getEditState(frameIdx) {
  if (!editState[frameIdx]) editState[frameIdx] = { deleted: {}, relabels: {} };
  if (!editState[frameIdx].deleted) editState[frameIdx].deleted = {};
  if (!editState[frameIdx].relabels) editState[frameIdx].relabels = {};
  return editState[frameIdx];
}
function isEntryDeleted(frame, weldNo) {
  const est = editState[frame.frame_idx];
  return !!(est && est.deleted && est.deleted[weldNo]);
}
function isEntryRelabeled(frame, weldNo) {
  const est = editState[frame.frame_idx];
  return !!(est && est.relabels && est.relabels[weldNo]);
}
function getEffectiveClass(frame, entry) {
  const est = editState[frame.frame_idx];
  return (est && est.relabels && est.relabels[entry.weld_no]) || entry.class_name;
}
function saveEditState() {
  localStorage.setItem(EDIT_STORAGE_KEY, JSON.stringify(editState));
  computeSummary();
  renderList();
}

function classColor(cls) { return cls === "BAD" ? "#f14c4c" : cls === "UNCERTAIN" ? "#e0a72a" : "#4ec9b0"; }

function computeSummary() {
  let nDeleted = 0, nRelabel = 0;
  Object.values(editState).forEach(est => {
    nDeleted += Object.keys(est.deleted || {}).length;
    nRelabel += Object.keys(est.relabels || {}).length;
  });
  document.getElementById("summary").innerHTML =
    `${frames.length} frames total<br>` +
    `<u>Unique objects</u><br>` +
    `　${N_WELDS} (<b class="good">GOOD ${GOOD_COUNT}</b> / <b class="bad">BAD ${BAD_COUNT}</b>) · suspected duplicate pairs ${duplicateCandidates.length}<br>` +
    `<u>Review marks</u><br>` +
    `　relabeled ${nRelabel} · marked deleted ${nDeleted}`;
}

// "All"/"BAD only"/"GOOD only" are one mutually-exclusive radio group;
// "New only"/"Suspected duplicates"/"Edited" are independently toggle-able and combine
// with the radio group and each other by AND (all active criteria must hold
// -- e.g. radio=BAD + toggle=dup shows only BAD welds that are also
// dup-suspects). A frame stays in the sidebar list if ANY of its entries
// matches; entries within a shown frame that DON'T individually match are
// kept visible but dimmed (renderList), not hidden, so filtering narrows
// focus without losing the surrounding frame context.
let filterMode = "all"; // "all" | "good" | "bad"
let filterToggles = { new: false, dup: false, edited: false };

function entryMatchesFilter(frame, entry) {
  const effClass = getEffectiveClass(frame, entry);
  if (filterMode === "good" && effClass !== "GOOD") return false;
  if (filterMode === "bad" && effClass !== "BAD") return false;
  if (filterToggles.new && !entry.is_new) return false;
  if (filterToggles.dup && !dupIds.has(entry.weld_no)) return false;
  if (filterToggles.edited && !(isEntryDeleted(frame, entry.weld_no) || isEntryRelabeled(frame, entry.weld_no))) return false;
  return true;
}

function updateFilterButtonStates() {
  document.getElementById("filterAll").classList.toggle("active", filterMode === "all");
  document.getElementById("filterBad").classList.toggle("active", filterMode === "bad");
  document.getElementById("filterGood").classList.toggle("active", filterMode === "good");
  document.getElementById("filterNew").classList.toggle("active", filterToggles.new);
  document.getElementById("filterDup").classList.toggle("active", filterToggles.dup);
  document.getElementById("filterEdited").classList.toggle("active", filterToggles.edited);
}

function setFilterMode(mode) {
  filterMode = mode;
  applyFilter();
}
function toggleFilter(key) {
  filterToggles[key] = !filterToggles[key];
  applyFilter();
}

function applyFilter() {
  updateFilterButtonStates();
  const isDefault = filterMode === "all" && !filterToggles.new && !filterToggles.dup && !filterToggles.edited;
  filteredData = isDefault ? frames : frames.filter(f => f.entries.some(e => entryMatchesFilter(f, e)));

  renderList();
  if (filteredData.length > 0) loadIndex(0);
}

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  filteredData.forEach((f, idx) => {
    const div = document.createElement("div");
    div.className = "list-item" + (idx === currentIndex ? " active" : "");
    const nBad = f.entries.filter(e => !isEntryDeleted(f, e.weld_no) && getEffectiveClass(f, e) === "BAD").length;
    div.innerHTML = `
      <div class="fname"><span>frame_${String(f.frame_idx).padStart(6,"0")}</span><span>${f.entries.length} obj${nBad ? ` · BAD ${nBad}` : ""}</span></div>
      <div class="meta">new +${f.n_new} · cumulative ${f.cumulative_total}</div>
    `;
    div.addEventListener("click", () => loadIndex(idx));
    list.appendChild(div);

    if (idx === currentIndex) {
      f.entries.forEach((e, wi) => {
        const del = isEntryDeleted(f, e.weld_no);
        const relabeled = isEntryRelabeled(f, e.weld_no);
        const effClass = getEffectiveClass(f, e);
        const isSel = selected === wi;
        const dimmed = !entryMatchesFilter(f, e);
        const wdiv = document.createElement("div");
        wdiv.className = "weld-item" + (isSel ? " selected" : "") + (del ? " deleted" : "") + (dimmed ? " dimmed" : "");
        wdiv.innerHTML =
          `<span class="widx">id=${e.weld_no}</span>` +
          `<span class="badge ${effClass}">${effClass}</span>` +
          (relabeled ? `<span class="relabel-mark">(reviewer edit, ML:${e.class_name})</span>` : "") +
          (e.is_new ? `<span class="new-mark">[NEW]</span>` : `<span class="counted-mark">[counted]</span>`) +
          (e.kind === "held" ? `<span class="counted-mark">(hold)</span>` : "") +
          (dupIds.has(e.weld_no) ? `<span class="dup-mark">⚠suspected duplicate</span>` : "") +
          (del ? `<span class="del-mark">deleted</span>` : "");
        wdiv.addEventListener("click", (ev) => { ev.stopPropagation(); selectEntry(wi); });
        list.appendChild(wdiv);
      });
    }
  });
}

function drawPoly(ctx, polyPx, color, lineWidth, dash) {
  ctx.beginPath();
  ctx.moveTo(polyPx[0][0] * zoom, polyPx[0][1] * zoom);
  for (let i = 1; i < polyPx.length; i++) ctx.lineTo(polyPx[i][0] * zoom, polyPx[i][1] * zoom);
  ctx.closePath();
  ctx.fillStyle = color + "22";
  ctx.fill();
  ctx.setLineDash((dash || []).map(d => d * zoom));
  ctx.lineWidth = lineWidth * zoom;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.setLineDash([]);
}

function updateCropThumb(poly) {
  const wrap = document.getElementById("cropThumbWrap");
  const thumb = document.getElementById("cropThumb");
  if (!poly || !img.naturalWidth) { wrap.style.display = "none"; return; }
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  const x1 = Math.max(0, Math.min(...xs)), x2 = Math.min(img.naturalWidth, Math.max(...xs));
  const y1 = Math.max(0, Math.min(...ys)), y2 = Math.min(img.naturalHeight, Math.max(...ys));
  const cw = Math.max(1, x2 - x1), ch = Math.max(1, y2 - y1);
  const wrapW = wrap.clientWidth || 300, wrapH = wrap.clientHeight || 220;
  const scale = Math.min(wrapW / cw, wrapH / ch);
  if (thumb.src !== img.src) thumb.src = img.src;
  thumb.style.width = (img.naturalWidth * scale) + "px";
  thumb.style.height = (img.naturalHeight * scale) + "px";
  thumb.style.left = (-x1 * scale) + "px";
  thumb.style.top = (-y1 * scale) + "px";
  wrap.style.display = "block";
}

function drawCanvas() {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const w = img.naturalWidth * zoom, h = img.naturalHeight * zoom;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);

  const frame = filteredData[currentIndex];
  if (!frame) return;

  frame.entries.forEach((e, i) => {
    const del = isEntryDeleted(frame, e.weld_no);
    const relabeled = isEntryRelabeled(frame, e.weld_no);
    const effClass = getEffectiveClass(frame, e);
    const color = del ? "#888888" : classColor(effClass);
    const dash = del ? [4, 4] : (e.kind === "held" ? [10, 6] : []);
    const thickness = del ? 3 : (e.is_new ? 4 : 2);
    drawPoly(ctx, e.polygon, color, thickness, dash);
    if (e.is_new && !del) drawPoly(ctx, e.polygon, "#ffd23f", 1, []);

    if (showLabels) {
      ctx.fillStyle = del ? "#999" : (e.is_new ? "#ffd23f" : color);
      ctx.font = `bold ${Math.round(15 * zoom)}px Arial`;
      let label = del ? `id=${e.weld_no} deleted (not an object)` : `id=${e.weld_no} ${effClass} ${e.is_new ? "[NEW]" : "[counted]"}`;
      if (relabeled && !del) label += ` (reviewer edit, ML:${e.class_name})`;
      if (e.kind === "held") label += " (hold)";
      if (dupIds.has(e.weld_no) && !del) label += " ⚠suspected duplicate";
      ctx.fillText(label, e.polygon[0][0] * zoom + 3, e.polygon[0][1] * zoom - 5);
    }

    if (selected === i) {
      ctx.save();
      ctx.setLineDash([]);
      ctx.lineWidth = 3 * zoom;
      ctx.strokeStyle = "#ffe600";
      ctx.beginPath();
      ctx.moveTo(e.polygon[0][0] * zoom, e.polygon[0][1] * zoom);
      for (let k = 1; k < e.polygon.length; k++) ctx.lineTo(e.polygon[k][0] * zoom, e.polygon[k][1] * zoom);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }
  });
}

function pointInPolygon(px, py, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1];
    const xj = pts[j][0], yj = pts[j][1];
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function canvasPointFromEvent(e) {
  const canvas = document.getElementById("canvas");
  const rect = canvas.getBoundingClientRect();
  return [
    (e.clientX - rect.left) * (canvas.width / rect.width) / zoom,
    (e.clientY - rect.top) * (canvas.height / rect.height) / zoom,
  ];
}

function showDetail(entryIdx) {
  selected = entryIdx;
  const frame = filteredData[currentIndex];
  const e = frame.entries[entryIdx];
  const info = welds[e.weld_no];
  const del = isEntryDeleted(frame, e.weld_no);
  const relabeled = isEntryRelabeled(frame, e.weld_no);
  const effClass = getEffectiveClass(frame, e);
  const panel = document.getElementById("detail-panel");
  const body = document.getElementById("detail-body");
  updateCropThumb(e.polygon);

  const dupPartners = dupPartnersOf(e.weld_no);

  body.innerHTML = `
    <div class="row"><span>Object id</span><b>id=${e.weld_no}</b></div>
    <div class="row"><span>ML verdict</span><b style="color:${classColor(e.class_name)}">${e.class_name}</b></div>
    ${relabeled ? `<div class="row"><span>Reviewer final verdict</span><b style="color:${classColor(effClass)}">${effClass} (edited)</b></div>` : `<div class="row"><span>Reviewer final verdict</span><b style="color:#9d9d9d">same as ML (unedited)</b></div>`}
    <div class="row"><span>State in this frame</span><b style="color:${e.is_new ? "#ffd23f" : "#9d9d9d"}">${e.is_new ? "new count (counted here for the first time)" : "counted (re-appearance of an already counted object)"}</b></div>
    <div class="row"><span>Detection type</span><b>${e.kind === "real" ? "real detection" : "held (estimated position, re-detection failed)"}</b></div>
    ${del ? `<div class="row"><span>Review state</span><b style="color:#999">marked deleted (not an object)</b></div>` : ""}
    ${info ? `<div class="row"><span>All appearances of this object</span><b>fid ${info.first_frame_idx}~${info.last_frame_idx} (${info.n_frames} frames)</b></div>` : ""}
    ${dupPartners.length ? `<div class="dup-warn">⚠ Suspected duplicate: may be the same physical object as id=${dupPartners.join(", id=")}. Compare them in the gallery.</div>` : ""}
    <div class="detail-actions">
      <button class="relabel-btn${effClass === "BAD" ? " active-label" : ""}" data-label="BAD" style="background:#8a2e2e;" ${del ? "disabled" : ""}>Relabel as BAD</button>
      <button class="relabel-btn${effClass === "GOOD" ? " active-label" : ""}" data-label="GOOD" style="background:#1f6b4f;" ${del ? "disabled" : ""}>Relabel as GOOD</button>
      <button class="relabel-btn${effClass === "UNCERTAIN" ? " active-label" : ""}" data-label="UNCERTAIN" style="background:#7a5a12;" ${del ? "disabled" : ""}>Relabel as UNCERTAIN</button>
    </div>
    <div class="detail-actions">
      <button id="toggleDeleteBtn" style="background:${del ? "#555" : "#6a4a2a"};">${del ? "Undo delete" : "Delete (not an object)"}</button>
      <button id="openGalleryBtn" style="background:#2f5f7a;">Show all crops of this object</button>
    </div>
    <div class="detail-hint">This verdict is the pipeline result at capture time. "Relabel" and "Delete" are review marks for this view / the export only and do not change the original tally.</div>
  `;
  body.querySelectorAll(".relabel-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const label = btn.dataset.label;
      const est = getEditState(frame.frame_idx);
      if (e.class_name === label) delete est.relabels[e.weld_no];
      else est.relabels[e.weld_no] = label;
      saveEditState();
      showDetail(entryIdx);
      drawCanvas();
    });
  });
  document.getElementById("toggleDeleteBtn").addEventListener("click", () => {
    const est = getEditState(frame.frame_idx);
    if (est.deleted[e.weld_no]) delete est.deleted[e.weld_no];
    else est.deleted[e.weld_no] = true;
    saveEditState();
    showDetail(entryIdx);
    drawCanvas();
  });
  document.getElementById("openGalleryBtn").addEventListener("click", () => openWeld(e.weld_no));
  panel.style.display = "block";
}

document.getElementById("detailClose").addEventListener("click", (ev) => {
  ev.stopPropagation();
  selected = null;
  document.getElementById("detail-panel").style.display = "none";
  drawCanvas();
});

function findFrameByIdx(frameIdx) {
  return frames.find(f => f.frame_idx === Number(frameIdx));
}

function exportEdits() {
  const result = [];
  Object.entries(editState).forEach(([frameIdx, est]) => {
    const frame = findFrameByIdx(frameIdx);
    Object.entries(est.relabels || {}).forEach(([weldNo, label]) => {
      const entry = frame && frame.entries.find(e => e.weld_no === Number(weldNo));
      result.push({
        frame_idx: Number(frameIdx), weld_no: Number(weldNo), type: "relabel",
        original_class: entry ? entry.class_name : null, corrected_class: label,
      });
    });
    Object.keys(est.deleted || {}).forEach(weldNo => {
      result.push({ frame_idx: Number(frameIdx), weld_no: Number(weldNo), type: "delete", reason: "not_a_weld" });
    });
  });
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dedup_audit_edits.json";
  a.click();
  URL.revokeObjectURL(url);
}

// Both exclusion and final class are decided ENTIRELY by the OCCURRENCE
// THAT ORIGINALLY TRIGGERED THIS weld_no's COUNT (is_new === true) -- that's
// the exact moment and exact class_name the generator's good_count/bad_count
// tally (GOOD_COUNT/BAD_COUNT consts) actually incremented on, so mirroring
// it here is what guarantees a report exported with ZERO edits reproduces
// that same baseline exactly (an earlier version of this function aggregated
// "any BAD occurrence anywhere -> BAD" across every frame the weld appeared
// in, which silently reclassified dozens of welds even with no edits at all
// -- confirmed concretely: 280/198/82 baseline came out as 280/167/113 with
// nothing touched). Deleting/relabeling some OTHER re-detection of the same
// weld_no elsewhere only changes that one frame's displayed box/label for
// audit purposes; it does not move the final tally, since that weld was
// already counted (or not) at its real first appearance.
function computeFinalWeldResults() {
  const byWeld = {};
  frames.forEach(f => {
    f.entries.forEach(e => {
      (byWeld[e.weld_no] = byWeld[e.weld_no] || []).push({ frame: f, entry: e });
    });
  });
  const results = {};
  Object.entries(byWeld).forEach(([weldNoStr, occs]) => {
    const weldNo = Number(weldNoStr);
    const firstOcc = occs.find(o => o.entry.is_new) || occs[0];
    const excluded = isEntryDeleted(firstOcc.frame, weldNo);
    const finalClass = excluded ? null : getEffectiveClass(firstOcc.frame, firstOcc.entry);
    results[weldNo] = { excluded, finalClass };
  });
  return results;
}

// Re-encodes a (possibly large) crop file as a small embedded JPEG data URL
// so the exported report keeps showing real crops. Deliberately NOT a
// canvas-drawn/toDataURL() base64 re-encode (tried first, reverted
// 2026-09-14): on a file:// page, Chrome allows loading a local image into
// an <img> fine, but drawing that same image onto a <canvas> and calling
// toDataURL() throws a SecurityError ("tainted canvases may not be
// exported") -- confirmed concretely: every card rendered as a broken image
// icon in the exported report because the catch block silently resolved
// null. An absolute file:// URL has no such restriction (it's the same
// unrestricted load an <img src="dedup_gallery/...\"> already does inside
// this app) and needs no canvas, no async wait, and no per-image failure
// path at all -- the tradeoff is the report is no longer a fully portable
// single file: it still points at this results folder's dedup_gallery/, so
// keep them together (see the report's own footnote, which says this).
function absoluteFileUrl(relativePath) {
  return new URL(relativePath, document.baseURI).href;
}

function buildReportHtml() {
  const results = computeFinalWeldResults();
  const keptEntries = Object.entries(results).filter(([, r]) => !r.excluded);
  const finalGood = keptEntries.filter(([, r]) => r.finalClass === "GOOD").length;
  const finalBad = keptEntries.filter(([, r]) => r.finalClass === "BAD").length;
  const finalUncertain = keptEntries.filter(([, r]) => r.finalClass === "UNCERTAIN").length;
  const finalTotal = keptEntries.length;
  const nExcluded = Object.values(results).filter(r => r.excluded).length;
  let nRelabel = 0;
  Object.values(editState).forEach(est => { nRelabel += Object.keys(est.relabels || {}).length; });
  const ts = new Date().toISOString().replace("T", " ").slice(0, 19);

  // One representative crop per weld: thumbs[0] is always the weld's FIRST
  // chronological appearance (see _evenly_sample -- index 0 is always kept),
  // i.e. the exact same occurrence computeFinalWeldResults counted/classified.
  const cards = keptEntries.map(([weldNoStr, r]) => {
    const weldNo = Number(weldNoStr);
    const info = welds[weldNo];
    const thumb = info && info.thumbs && info.thumbs[0];
    const imgUrl = thumb ? absoluteFileUrl(thumb.file) : null;
    const frameIdx = thumb ? thumb.frame_idx : (info ? info.first_frame_idx : "?");
    return { weldNo, cls: r.finalClass, imgUrl, frameIdx };
  });
  cards.sort((a, b) => a.weldNo - b.weldNo);

  function section(cls, label) {
    const items = cards.filter(c => c.cls === cls);
    if (!items.length) return "";
    const grid = items.map(c => `<div class="weld-card ${cls.toLowerCase()}">
      <img src="${c.imgUrl || ""}" alt="id=${c.weldNo}">
      <div class="wc-body"><div class="wc-id">id=${c.weldNo}</div><div class="wc-frame">fid=${c.frameIdx}</div></div>
    </div>`).join("");
    return `<h2 class="section-title ${cls.toLowerCase()}">${label} (${items.length})</h2><div class="weld-grid">${grid}</div>`;
  }

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Final quality review result</title>
<style>
  * { box-sizing:border-box; }
  body { font-family:'Segoe UI',Arial,sans-serif; background:#1e1e1e; color:#d4d4d4; margin:0; padding:28px 32px 60px; }
  .summary-card { background:#252526; border:1px solid #3e3e42; border-radius:12px; padding:28px 36px; max-width:640px; margin:0 auto 32px; }
  h1 { font-size:20px; color:#fff; margin:0 0 4px; }
  .sub { font-size:12px; color:#9d9d9d; margin-bottom:20px; }
  .stat-row { display:flex; justify-content:space-between; align-items:baseline; padding:10px 0; border-bottom:1px solid #333; }
  .stat-row .label { font-size:13px; color:#c9c9c9; }
  .stat-row .value { font-size:22px; font-weight:bold; }
  .value.good { color:#4ec9b0; } .value.bad { color:#f14c4c; } .value.uncertain { color:#e0a72a; } .value.total { color:#fff; }
  .footnote { margin-top:20px; font-size:11px; color:#888; line-height:1.7; }
  .section-title { font-size:15px; color:#fff; margin:32px auto 12px; padding-bottom:6px; border-bottom:1px solid #3e3e42; max-width:1100px; }
  .section-title.bad { color:#f14c4c; } .section-title.good { color:#4ec9b0; } .section-title.uncertain { color:#e0a72a; }
  .weld-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:12px; max-width:1100px; margin:0 auto; }
  .weld-card { background:#252526; border:2px solid #3e3e42; border-radius:8px; overflow:hidden; font-size:11px; }
  .weld-card.bad { border-color:#b23b3b; } .weld-card.good { border-color:#2f9e5f; } .weld-card.uncertain { border-color:#a8842f; }
  .weld-card img { display:block; width:100%; height:120px; object-fit:contain; background:#000; }
  .weld-card .wc-body { padding:6px 8px; text-align:center; }
  .weld-card .wc-id { color:#fff; font-weight:bold; }
  .weld-card .wc-frame { color:#9d9d9d; }
</style></head><body>
<div class="summary-card">
  <h1>Final quality review result</h1>
  <div class="sub">Generated: ${ts} · based on ${frames.length} frames</div>
  <div class="stat-row"><span class="label">Final unique objects</span><span class="value total">${finalTotal}</span></div>
  <div class="stat-row"><span class="label">GOOD</span><span class="value good">${finalGood}</span></div>
  <div class="stat-row"><span class="label">BAD</span><span class="value bad">${finalBad}</span></div>
  ${finalUncertain ? `<div class="stat-row"><span class="label">UNCERTAIN</span><span class="value uncertain">${finalUncertain}</span></div>` : ""}
  <div class="footnote">
    Original ML tally (before review): ${N_WELDS} (GOOD ${GOOD_COUNT} / BAD ${BAD_COUNT})<br>
    Reviewer edits applied: relabeled ${nRelabel} · deleted as not-an-object and excluded from the final tally ${nExcluded}<br>
    Potential duplicate candidates (manual check required, not reflected in these numbers): ${duplicateCandidates.length} pair(s)<br>
    Each card photo is the crop from the frame (fid) in which the object was first counted.<br>
    ⚠ The photos reference this result folder (dedup_gallery/) directly -- moving only this report file breaks the images, so move the whole folder or keep the report in place.
  </div>
</div>
${section("BAD", "BAD")}
${section("GOOD", "GOOD")}
${section("UNCERTAIN", "UNCERTAIN")}
</body></html>`;
}

function exportReport() {
  const htmlContent = buildReportHtml();
  const blob = new Blob([htmlContent], { type: "text/html;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dedup_audit_final_report.html";
  a.click();
  URL.revokeObjectURL(url);
  uploadExportCopyToServer(htmlContent, "dedup_audit_final_report.html");
}

// Best-effort only: if this page is being served over http(s) through the
// pqdai webapp (not opened directly from a local
// results folder as file://), also send a copy to the backend so it shows
// up in that user's analysis history. No-ops over file:// and swallows any
// request failure -- must never block or affect the browser download above,
// which is the only behavior when this file is used outside the webapp
// (the original desktop/file:// workflow this tool was built for).
function uploadExportCopyToServer(htmlContent, filename) {
  if (!location.protocol.startsWith("http")) return;
  const match = location.pathname.match(/\/results\/([^/]+)\//);
  if (!match) return;
  const jobId = match[1];
  fetch(`/api/jobs/${jobId}/exports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ filename, html: htmlContent }),
  }).catch(() => {});
}

function exportAll() {
  exportReport();
  setTimeout(exportEdits, 300);
}

function resetAllEdits() {
  const n = Object.keys(editState).length;
  if (!confirm(`Reset the relabel/delete marks of ${n} frame(s)?`)) return;
  editState = {};
  localStorage.removeItem(EDIT_STORAGE_KEY);
  selected = null;
  document.getElementById("detail-panel").style.display = "none";
  computeSummary();
  renderList();
  drawCanvas();
}

function fitToScreen() {
  const area = document.getElementById("canvas-area");
  const maxW = area.clientWidth - 40;
  const maxH = area.clientHeight - 40;
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1.5);
  fitZoom = scale;
  zoom = scale;
  document.getElementById("zoomLabel").textContent = Math.round(zoom * 100) + "%";
}

function setZoom(z) {
  zoom = Math.max(0.05, Math.min(z, 8));
  document.getElementById("zoomLabel").textContent = Math.round(zoom * 100) + "%";
  drawCanvas();
}

function loadIndex(idx, pendingSelectIdx) {
  currentIndex = Math.max(0, Math.min(filteredData.length - 1, idx));
  const f = filteredData[currentIndex];
  if (!f) return;
  selected = null;
  document.getElementById("detail-panel").style.display = "none";
  const nBad = f.entries.filter(e => !isEntryDeleted(f, e.weld_no) && e.class_name === "BAD").length;
  document.getElementById("fnameDisplay").textContent =
    `[${currentIndex + 1}/${filteredData.length}] frame_${String(f.frame_idx).padStart(6, "0")}  objects=${f.entries.length}  BAD=${nBad}  new=+${f.n_new}  cumulative=${f.cumulative_total}`;
  img.onload = () => {
    fitToScreen();
    if (pendingSelectIdx != null && pendingSelectIdx >= 0) showDetail(pendingSelectIdx);
    drawCanvas();
    renderList();
  };
  img.src = f.clean_file;
  renderList();
}

function stepIndex(delta) { loadIndex(currentIndex + delta); }
function selectEntry(entryIdx) { showDetail(entryIdx); drawCanvas(); renderList(); }

// ---------------- Weld-by-weld (A/D) navigation, across frame boundaries ----------------

function buildFlatWeldSequence() {
  const seq = [];
  filteredData.forEach((f, imgIdx) => {
    f.entries.forEach((e, entryIdx) => seq.push({ imgIdx, entryIdx }));
  });
  return seq;
}

function currentFlatPos(seq) {
  if (selected !== null) {
    const exact = seq.findIndex(s => s.imgIdx === currentIndex && s.entryIdx === selected);
    if (exact !== -1) return exact;
  }
  let pos = -1;
  for (let i = 0; i < seq.length; i++) {
    if (seq[i].imgIdx >= currentIndex) { pos = i - 1; break; }
    pos = i;
  }
  return pos;
}

function stepWeld(delta) {
  const seq = buildFlatWeldSequence();
  if (!seq.length) { stepIndex(delta); return; }
  let pos = currentFlatPos(seq);
  pos = Math.max(-1, Math.min(seq.length, pos)) + delta;
  pos = Math.max(0, Math.min(seq.length - 1, pos));
  const target = seq[pos];
  if (target.imgIdx !== currentIndex) loadIndex(target.imgIdx, target.entryIdx);
  else selectEntry(target.entryIdx);
}

function gotoFrameWeld(frameIdx, weldNo) {
  closeGallery();
  const fi = filteredData.findIndex(f => f.frame_idx === frameIdx);
  if (fi === -1) { alert('This frame is not visible under the current filter. Switch to the "All" filter and try again.'); return; }
  const entryIdx = filteredData[fi].entries.findIndex(e => e.weld_no === weldNo);
  loadIndex(fi, entryIdx);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openWeld(weldNo) {
  const info = welds[weldNo];
  if (!info) return;
  document.getElementById("galleryTitle").textContent =
    `id=${weldNo} (${info.final_class}) -- fid ${info.first_frame_idx}~${info.last_frame_idx}, showing ${info.thumbs.length} of ${info.n_frames} frames` +
    ` [bead_id=${info.bead_id}, raw track_id=${JSON.stringify(info.member_track_ids)}]`;
  const container = document.getElementById("galleryThumbs");
  container.innerHTML = "";
  info.thumbs.forEach(t => {
    const div = document.createElement("div");
    div.className = "gthumb " + (t.class_name === "BAD" ? "bad" : "good") + (t.kind === "held" ? " held" : "");
    div.innerHTML = `<img src="${t.file}"><div>fid=${t.frame_idx} ${t.class_name}${t.kind === "held" ? " (held)" : ""}</div>
      <a href="#" onclick="gotoFrameWeld(${t.frame_idx}, ${weldNo});return false;">View in this frame</a>`;
    container.appendChild(div);
  });
  document.getElementById("galleryModal").style.display = "block";
}
function closeGallery() { document.getElementById("galleryModal").style.display = "none"; }

function renderDupList() {
  const el = document.getElementById("dupList");
  if (!duplicateCandidates.length) { el.innerHTML = '<p style="color:#9d9d9d;">None</p>'; return; }
  el.innerHTML = "<table><tr><th>object A</th><th>object B</th><th>Go to</th></tr>" +
    duplicateCandidates.map(d =>
      `<tr><td>id=${d.a}</td><td>id=${d.b}</td><td>` +
      `<a href="#" onclick="document.getElementById('dupModal').style.display='none';openWeld(${d.a});return false;">id=${d.a} gallery</a> / ` +
      `<a href="#" onclick="document.getElementById('dupModal').style.display='none';openWeld(${d.b});return false;">id=${d.b} gallery</a></td></tr>`
    ).join("") + "</table>";
}

document.getElementById("prevFrameBtn").addEventListener("click", () => stepIndex(-1));
document.getElementById("nextFrameBtn").addEventListener("click", () => stepIndex(1));
document.getElementById("prevBtn").addEventListener("click", () => stepWeld(-1));
document.getElementById("nextBtn").addEventListener("click", () => stepWeld(1));
document.getElementById("zoomInBtn").addEventListener("click", () => setZoom(zoom * 1.2));
document.getElementById("zoomOutBtn").addEventListener("click", () => setZoom(zoom / 1.2));
document.getElementById("zoomResetBtn").addEventListener("click", () => setZoom(fitZoom));
document.getElementById("toggleLabels").addEventListener("click", () => {
  showLabels = !showLabels;
  document.getElementById("toggleLabels").classList.toggle("off", !showLabels);
  drawCanvas();
});
document.getElementById("filterAll").addEventListener("click", () => setFilterMode("all"));
document.getElementById("filterBad").addEventListener("click", () => setFilterMode("bad"));
document.getElementById("filterGood").addEventListener("click", () => setFilterMode("good"));
document.getElementById("filterNew").addEventListener("click", () => toggleFilter("new"));
document.getElementById("filterDup").addEventListener("click", () => toggleFilter("dup"));
document.getElementById("filterEdited").addEventListener("click", () => toggleFilter("edited"));
document.getElementById("exportBtn").addEventListener("click", exportAll);
document.getElementById("resetEditsBtn").addEventListener("click", resetAllEdits);
document.getElementById("dupListBtn").addEventListener("click", () => {
  renderDupList();
  document.getElementById("dupModal").style.display = "block";
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const galleryOpen = document.getElementById("galleryModal").style.display === "block";
  const dupOpen = document.getElementById("dupModal").style.display === "block";
  if (galleryOpen || dupOpen) {
    if (e.key === "Escape") { closeGallery(); document.getElementById("dupModal").style.display = "none"; }
    return;
  }
  if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") stepWeld(-1);
  else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") stepWeld(1);
  else if (e.key === "l" || e.key === "L") document.getElementById("toggleLabels").click();
  else if (e.key === "e" || e.key === "E") exportAll();
});

const canvasArea = document.getElementById("canvas-area");
canvasArea.addEventListener("wheel", (e) => {
  e.preventDefault();
  setZoom(zoom * (e.deltaY < 0 ? 1.1 : 0.9));
}, { passive: false });

let isPanning = false, panStart = null, scrollStart = null, dragMoved = 0;
canvasArea.addEventListener("mousedown", (e) => {
  isPanning = true;
  dragMoved = 0;
  panStart = [e.clientX, e.clientY];
  scrollStart = [canvasArea.scrollLeft, canvasArea.scrollTop];
  canvasArea.classList.add("panning");
});
window.addEventListener("mousemove", (e) => {
  if (!isPanning) return;
  const dx = e.clientX - panStart[0];
  const dy = e.clientY - panStart[1];
  dragMoved = Math.max(dragMoved, Math.abs(dx), Math.abs(dy));
  canvasArea.scrollLeft = scrollStart[0] - dx;
  canvasArea.scrollTop = scrollStart[1] - dy;
});
window.addEventListener("mouseup", (e) => {
  if (!isPanning) return;
  isPanning = false;
  canvasArea.classList.remove("panning");
  if (dragMoved < 4) {
    const [ix, iy] = canvasPointFromEvent(e);
    const frame = filteredData[currentIndex];
    if (frame) {
      let hit = null;
      for (let i = frame.entries.length - 1; i >= 0; i--) {
        if (pointInPolygon(ix, iy, frame.entries[i].polygon)) { hit = i; break; }
      }
      if (hit !== null) {
        selectEntry(hit);
      } else {
        selected = null;
        document.getElementById("detail-panel").style.display = "none";
        drawCanvas();
      }
    }
  }
});
window.addEventListener("resize", () => { fitToScreen(); drawCanvas(); });

computeSummary();
renderList();
if (filteredData.length > 0) loadIndex(0);
</script>
</body>
</html>"""

    rerun_note_html = f'<div id="rerunNote">{rerun_note}</div>' if rerun_note else ""
    video_path_html = (video_path or "(unknown)").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html_doc = (template
        .replace("__FRAMES_JSON__", frames_json)
        .replace("__GALLERY_JSON__", gallery_json)
        .replace("__DUP_JSON__", dup_json)
        .replace("__GOOD_COUNT__", str(good_count))
        .replace("__BAD_COUNT__", str(bad_count))
        .replace("__N_WELDS__", str(n_welds))
        .replace("__N_DUP__", str(len(duplicate_candidates)))
        .replace("__RERUN_NOTE_HTML__", rerun_note_html)
        .replace("__VIDEO_PATH__", video_path_html)
    )
    (out_dir / 'dedup_audit.html').write_text(html_doc, encoding='utf-8')
