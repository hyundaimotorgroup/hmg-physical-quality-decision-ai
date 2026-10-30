"""Password-less login: pick your name/emp_no from data/users.json
(falls back to data/users.example.json when users.json does not exist).

THIS IS NOT ACCESS CONTROL: there is no password, hashing or DB. It only
records "who ran what" for a trusted, single-site deployment.
Sessions live in memory only (same lifetime tradeoff as the `jobs` dict in
main.py: gone on server restart, fine for a local single-machine tool).
"""

import json
import secrets
import threading
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_PATH = DATA_DIR / "users.json"
EXAMPLE_USERS_PATH = DATA_DIR / "users.example.json"

SESSION_COOKIE = "session_id"

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def load_users() -> list[dict]:
    path = USERS_PATH if USERS_PATH.exists() else EXAMPLE_USERS_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_user(emp_no: str) -> Optional[dict]:
    for user in load_users():
        if user["empNo"] == emp_no:
            return user
    return None


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = user
    return token


def destroy_session(token: str) -> None:
    with _sessions_lock:
        _sessions.pop(token, None)


def get_session_user(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    with _sessions_lock:
        return _sessions.get(token)


def get_current_user(request: Request) -> dict:
    """FastAPI dependency -- raises 401 if not logged in."""
    user = get_session_user(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(status_code=401, detail="login required")
    return user
