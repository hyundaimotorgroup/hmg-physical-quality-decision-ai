@echo off
chcp 65001 >nul
REM Physical Quality Decision AI - web app launcher (Windows).
REM Optional: set PQDAI_DETECTOR_WEIGHTS / PQDAI_CLASSIFIER_WEIGHTS to override checkpoints.
REM NOTE: the login is a name-picker without a password; do NOT expose to untrusted networks.

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment...
    py -3 -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -e ".[web]"
)

cd webapp\backend
echo Server will start at http://127.0.0.1:8000  (Ctrl+C to stop)
..\..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
