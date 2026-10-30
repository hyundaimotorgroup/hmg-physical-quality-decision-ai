"""Web backend for the weld video pipeline.

Serves the frontend (static files), accepts a video/image upload, runs the
pqdai.video.run CLI as a subprocess (video) or the
photo-only QualityPipeline in-process (image), and exposes results over
HTTP so the frontend can play/embed them. Also handles password-less login
(auth.py) and per-user analysis history (history.py).

Run via ../start_linux.sh, ../start_mac.command or ../start_windows.bat, or
manually: `cd webapp/backend && uvicorn main:app`.

SECURITY NOTE: login is a name-picker without passwords (see auth.py) and
uploads are not size/type-limited. It is NOT access control -- run it only on
a trusted network.
"""

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import history
import image_pipeline
import image_visualize

# job_id is always server-generated as uuid.uuid4().hex[:12] (see `analyze`
# and `analyze_image` below) -- never accept one from a client that doesn't
# match this exact shape. Without this check, a job_id containing "../" can
# walk RESULTS_DIR/job_id out of RESULTS_DIR entirely (path traversal), since
# job_id is concatenated into filesystem paths in several endpoints below.
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Matches pqdai.video.run's startup estimate line, e.g.:
#   "sample_fps:       5.0 (~150 frames will be processed)"
TOTAL_FRAMES_RE = re.compile(r"\(~(\d+) frames will be processed\)")
# Matches its periodic per-frame progress line, e.g.:
#   "  [20/150] t=  4.0s chunk=0 active_tracks=3 finalized=5 | 45.2s elapsed, ~293s remaining (0.44 frame/s)"
PROGRESS_RE = re.compile(
    r"\[(\d+)/(\d+)\].*?\|\s*([\d.]+)s elapsed,\s*~?([\d.]+)s remaining\s*\(([\d.]+) frame/s\)"
)

WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
FRONTEND_DIR = WEBAPP_DIR / "frontend"
UPLOAD_DIR = WEBAPP_DIR / "uploads"
RESULTS_DIR = WEBAPP_DIR / "results"
# Deliberately NOT under RESULTS_DIR: everything in RESULTS_DIR is served
# inline by the `/results` StaticFiles mount below, but an export's `html`
# is arbitrary user-supplied content (see add_export) -- if it lived under
# RESULTS_DIR, a browser would render it inline (Stored XSS) regardless of
# any Content-Disposition header the dedicated download route below sets.
# Keeping exports in their own directory means the ONLY way to reach one is
# through that route, which always forces a download.
EXPORTS_DIR = WEBAPP_DIR / "exports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Weld Video Pipeline")

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Video jobs run ONE AT A TIME through this queue+worker instead of a thread
# per upload. Multiple videos uploaded together used to all launch their own
# video pipeline subprocess immediately, fighting over the same
# CPU/GPU and slowing every one of them down together -- a single worker
# means upload #2 just waits for #1 to finish, at full speed, instead.
# Image analysis stays synchronous/per-request (it's already seconds, not
# minutes) but is serialized too (see image_pipeline_lock below), since the
# one shared model instance isn't safe for concurrent inference calls.
video_queue: "queue.Queue[tuple[str, Path, Path]]" = queue.Queue()
queued_job_ids: list[str] = []  # order-preserving, for reporting queue position
image_pipeline_lock = threading.Lock()


def _video_worker() -> None:
    while True:
        job_id, video_path, out_dir = video_queue.get()
        with jobs_lock:
            if job_id in queued_job_ids:
                queued_job_ids.remove(job_id)
        try:
            run_pipeline(job_id, video_path, out_dir)
        finally:
            video_queue.task_done()


threading.Thread(target=_video_worker, daemon=True).start()


# ── Auth ────────────────────────────────────────────────────────────────

class LoginPayload(BaseModel):
    empNo: str


@app.get("/api/users")
async def list_users():
    return {"users": auth.load_users()}


@app.post("/api/login")
async def login(payload: LoginPayload, response: Response):
    user = auth.find_user(payload.empNo)
    if user is None:
        raise HTTPException(status_code=401, detail="unknown emp_no")
    token = auth.create_session(user)
    response.set_cookie(
        key=auth.SESSION_COOKIE, value=token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30,
    )
    return {"user": user}


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        auth.destroy_session(token)
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me")
async def me(user: dict = Depends(auth.get_current_user)):
    return {"user": user}


# ── History ─────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(user: dict = Depends(auth.get_current_user)):
    return {"records": history.list_for_user(user["empNo"])}


class ExportPayload(BaseModel):
    filename: str
    html: str


def get_owned_job(job_id: str, user: dict = Depends(auth.get_current_user)) -> dict:
    """Shared dependency for every job_id-keyed endpoint below.

    `auth.get_current_user` only proves "someone is logged in" -- it says
    nothing about whether THIS user owns THIS job_id, so without this check
    any logged-in user could read or export any other user's job by guessing
    or enumerating job_ids (IDOR). This also validates job_id's shape against
    JOB_ID_RE: job_id is server-generated (uuid.uuid4().hex[:12]) and gets
    concatenated into filesystem paths below, so a non-matching value is
    rejected before it ever reaches a path operation (path traversal).

    404 (not 403) in both the "no such job" and "not your job" cases, so a
    caller can't distinguish "doesn't exist" from "exists but isn't mine" --
    the distinction itself would leak which job_ids are valid."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="unknown job_id")
    record = history.get_record(job_id)
    if record is None or record["emp_no"] != user["empNo"]:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return record


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, record: dict = Depends(get_owned_job)):
    """Reconstructs a past job's viewable result paths straight from disk +
    history.json -- unlike /api/status/{job_id}, this works even after a
    server restart wipes the in-memory `jobs` dict, since history.html's
    "view again" links need to keep working across restarts (that's the
    whole point of persisting history to a file instead of memory)."""
    out_dir = RESULTS_DIR / job_id
    result: dict = {}
    if record["kind"] == "video":
        if (out_dir / "overlay_web.mp4").exists():
            result["overlay_video"] = f"/results/{job_id}/overlay_web.mp4"
        elif (out_dir / "overlay.mp4").exists():
            result["overlay_video"] = f"/results/{job_id}/overlay.mp4"
        else:
            result["overlay_video"] = None
        result["dedup_audit"] = f"/results/{job_id}/dedup_audit.html" if (out_dir / "dedup_audit.html").exists() else None
        result["report"] = f"/results/{job_id}/report.html" if (out_dir / "report.html").exists() else None
    else:
        result["overlay_image"] = f"/results/{job_id}/overlay.jpg" if (out_dir / "overlay.jpg").exists() else None

    return {"record": record, "result": result}


@app.post("/api/jobs/{job_id}/exports")
async def add_export(job_id: str, payload: ExportPayload, record: dict = Depends(get_owned_job)):
    exports_dir = EXPORTS_DIR / job_id
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(payload.filename).name or "export.html"
    saved_name = f"{timestamp}_{safe_name}"
    # payload.html is arbitrary user-supplied content (the frontend sends a
    # client-built report), so it is written as-is -- the Stored XSS this
    # previously enabled came from serving it inline via `/results`, not
    # from storing it. Reaching it only through /api/jobs/{job_id}/exports/
    # {filename} below, which always sets Content-Disposition: attachment,
    # means a browser downloads rather than renders it even if it contains
    # a <script> tag.
    with open(exports_dir / saved_name, "w", encoding="utf-8") as f:
        f.write(payload.html)
    history.add_export(job_id, saved_name)
    return {"saved": f"/api/jobs/{job_id}/exports/{saved_name}"}


@app.get("/api/jobs/{job_id}/exports/{filename}")
async def download_export(filename: str, record: dict = Depends(get_owned_job)):
    # The route's job_id path segment is consumed by get_owned_job's own
    # job_id parameter (FastAPI matches dependency params against the path
    # by name), so by the time we get here it's already validated (shape +
    # ownership) -- record["job_id"] below is that same, now-trusted value.
    safe_name = Path(filename).name
    path = EXPORTS_DIR / record["job_id"] / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="unknown export file")
    return FileResponse(path, media_type="text/html", filename=safe_name,
                        content_disposition_type="attachment")


@app.get("/results/{job_id}/{filename:path}")
async def get_result_file(filename: str, record: dict = Depends(get_owned_job)):
    """Serves a job's result files (overlay image/video, report.html,
    dedup_audit.html, live_preview/*.jpg) -- previously a bare StaticFiles
    mount at this same /results prefix, reachable by ANYONE who knew or
    guessed a job_id with no auth check at all. That stayed exploitable even
    after /api/jobs/{job_id} itself was locked down with get_owned_job,
    because the frontend's result URLs point directly at /results/... and
    nothing was gating that path (reported as a recurrence of the IDOR
    finding). Routing it through get_owned_job here closes that: every
    result file now requires the same logged-in + owns-this-job_id check as
    the API that returns its URL.

    `filename` can contain a subdirectory (e.g. "live_preview/latest.jpg"),
    hence the `:path` converter; is_relative_to re-confirms the resolved
    path still lands inside this job's own results directory before it's
    served, as a second layer under get_owned_job's job_id-shape check."""
    job_dir = (RESULTS_DIR / record["job_id"]).resolve()
    target = (job_dir / filename).resolve()
    if not target.is_relative_to(job_dir) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target)


# ── Video analysis (subprocess -> run_video_pipeline.py) ───────────────

def _reencode_for_browser(out_dir: Path, job_id: str) -> str | None:
    """pqdai.video.run writes overlay.mp4 with OpenCV's 'mp4v' fourcc,
    which most browsers cannot play back directly. If ffmpeg is available,
    re-encode to H.264 so the <video> tag in the frontend actually plays it;
    otherwise fall back to serving the raw file (downloadable, may not
    preview inline)."""
    overlay_path = out_dir / "overlay.mp4"
    if not overlay_path.exists():
        return None
    if shutil.which("ffmpeg"):
        web_path = out_dir / "overlay_web.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(overlay_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(web_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and web_path.exists():
            return f"/results/{job_id}/overlay_web.mp4"
    return f"/results/{job_id}/overlay.mp4"


def run_pipeline(job_id: str, video_path: Path, out_dir: Path) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "running"

    cmd = [sys.executable, "-u", "-m", "pqdai.video.run", "--video", str(video_path), "--out", str(out_dir),
           "--domain", os.environ.get("PQDAI_DOMAIN", "weld")]
    # Optional overrides so a deployment can point at custom checkpoints.
    detector_weights = os.environ.get("PQDAI_DETECTOR_WEIGHTS")
    if detector_weights:
        cmd += ["--detector-weights", detector_weights]
    classifier_weights = os.environ.get("PQDAI_CLASSIFIER_WEIGHTS")
    if classifier_weights:
        cmd += ["--classifier-weights", classifier_weights]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        log_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            with jobs_lock:
                jobs[job_id]["log"] = log_lines[-300:]
                progress = jobs[job_id]["progress"]
                total_match = TOTAL_FRAMES_RE.search(line)
                if total_match:
                    progress["total"] = int(total_match.group(1))
                match = PROGRESS_RE.search(line)
                if match:
                    progress["phase"] = "processing"
                    progress["current"] = int(match.group(1))
                    progress["total"] = int(match.group(2))
                    progress["elapsed_sec"] = float(match.group(3))
                    progress["eta_sec"] = float(match.group(4))
                    progress["rate_fps"] = float(match.group(5))
                elif "Loading models" in line:
                    progress["phase"] = "loading_models"
        proc.wait()
        if proc.returncode != 0:
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"pipeline exited with code {proc.returncode} (see log)"
            history.update_status(job_id, "error")
            return
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
        history.update_status(job_id, "error")
        return

    with jobs_lock:
        jobs[job_id]["progress"]["phase"] = "encoding_video"
    video_url = _reencode_for_browser(out_dir, job_id)
    with jobs_lock:
        jobs[job_id]["progress"]["phase"] = "done"
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "overlay_video": video_url,
            "dedup_audit": f"/results/{job_id}/dedup_audit.html" if (out_dir / "dedup_audit.html").exists() else None,
            "report": f"/results/{job_id}/report.html" if (out_dir / "report.html").exists() else None,
        }
    history.update_status(job_id, "done")


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), user: dict = Depends(auth.get_current_user)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="no file provided")

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    # .name strips any directory components from the client-supplied
    # filename so it can't escape job_upload_dir (e.g. "../../evil.sh").
    video_path = job_upload_dir / Path(video.filename).name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with jobs_lock:
        jobs[job_id] = {
            "emp_no": user["empNo"],  # checked by /api/status/{job_id} below (IDOR guard)
            "status": "queued",
            "log": [],
            "error": None,
            "result": None,
            "progress": {
                "phase": "starting",  # starting -> loading_models -> processing -> encoding_video -> done
                "current": 0,
                "total": None,
                "elapsed_sec": None,
                "eta_sec": None,
                "rate_fps": None,
            },
        }
    history.add_record(job_id, user["empNo"], user["name"], kind="video", filename=video.filename, status="queued")

    with jobs_lock:
        queued_job_ids.append(job_id)
    video_queue.put((job_id, video_path, out_dir))
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str, user: dict = Depends(auth.get_current_user)):
    # Separate ownership check from get_owned_job's: `jobs` is the in-memory
    # dict of still-running jobs (wiped on restart), not the history.json
    # record get_owned_job checks, so it needs its own emp_no comparison
    # (IDOR guard) -- same 404-for-both-cases rule as get_owned_job.
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="unknown job_id")
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or job["emp_no"] != user["empNo"]:
        raise HTTPException(status_code=404, detail="unknown job_id")

    progress = dict(job["progress"])
    live_preview_path = RESULTS_DIR / job_id / "live_preview" / "latest.jpg"
    progress["live_preview_url"] = (
        f"/results/{job_id}/live_preview/latest.jpg" if live_preview_path.exists() else None
    )

    queue_position = None
    if job["status"] == "queued" and job_id in queued_job_ids:
        queue_position = queued_job_ids.index(job_id) + 1  # 1-indexed: "1st in line" etc.

    return {
        "status": job["status"],
        "progress": progress,
        "queue_position": queue_position,
        "log_tail": job["log"][-25:],
        "error": job["error"],
        "result": job["result"],
    }


# ── Image analysis (in-process, synchronous) ────────────────────────────

@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...), user: dict = Depends(auth.get_current_user)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="no file provided")

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_upload_dir / Path(image.filename).name
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = image_pipeline.get_pipeline()
    # The frontend already sends multi-image batches one at a time (see
    # app.js), but this lock is a safety net against the shared detector/
    # classifier models being called concurrently from two requests at once
    # (e.g. two browser tabs) -- they were never verified thread-safe.
    with image_pipeline_lock:
        verdict = pipeline.judge_image(str(image_path))

    source_image = cv2.imread(str(image_path))
    if source_image is None:
        raise HTTPException(status_code=400, detail="could not read image file")
    overlay = image_visualize.draw_verdict(source_image, verdict)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)

    welds = [
        {
            "class_name": w.class_name,
            "confidence": float(w.cnn_probabilities[0] if w.class_name == verdict.unsafe_class else w.cnn_probabilities[-1]),
            "used_retrieval": w.used_retrieval,
        }
        for w in verdict.objects
    ]

    history.add_record(job_id, user["empNo"], user["name"], kind="image", filename=image.filename, status="done")

    return {
        "job_id": job_id,
        "overlay_image": f"/results/{job_id}/overlay.jpg",
        "overall_class": verdict.overall_class_name,
        "objects": welds,
    }


# NOTE: no StaticFiles mount at /results anymore -- that unconditionally
# served every job's result files to anyone who knew/guessed a job_id, with
# no auth at all. get_result_file above now owns this entire prefix with an
# ownership check; FastAPI resolves routes before the mount below, but the
# mount is on a disjoint prefix ("/") so there's no ordering dependency
# either way.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
