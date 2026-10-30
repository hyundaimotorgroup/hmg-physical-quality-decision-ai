"""Per-user analysis history, persisted to data/history.json.

Kept as a flat JSON list (not a DB) -- consistent with this app's scope
(local single-machine tool, everything else is JSON/in-memory too). Survives
server restarts, unlike the in-memory `jobs`/session state, since "record
management" is the whole point of this file.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
HISTORY_PATH = DATA_DIR / "history.json"

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write(records: list[dict]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def add_record(job_id: str, emp_no: str, user_name: str, kind: str, filename: str, status: str) -> None:
    with _lock:
        records = _read()
        records.append({
            "job_id": job_id,
            "emp_no": emp_no,
            "user_name": user_name,
            "kind": kind,  # "video" | "image"
            "filename": filename,
            "created_at": _now_iso(),
            "status": status,
            "exports": [],
        })
        _write(records)


def update_status(job_id: str, status: str) -> None:
    with _lock:
        records = _read()
        for r in records:
            if r["job_id"] == job_id:
                r["status"] = status
                break
        _write(records)


def add_export(job_id: str, filename: str) -> bool:
    """Returns False if job_id isn't a known history record."""
    with _lock:
        records = _read()
        for r in records:
            if r["job_id"] == job_id:
                r["exports"].append({"exported_at": _now_iso(), "filename": filename})
                _write(records)
                return True
        return False


def list_for_user(emp_no: str) -> list[dict]:
    return [r for r in _read() if r["emp_no"] == emp_no]


def get_record(job_id: str) -> Optional[dict]:
    for r in _read():
        if r["job_id"] == job_id:
            return r
    return None
