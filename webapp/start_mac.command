#!/bin/bash
# Physical Quality Decision AI - web app launcher (macOS). Double-click or run: bash start_mac.command
# Optional: PQDAI_DETECTOR_WEIGHTS / PQDAI_CLASSIFIER_WEIGHTS to override checkpoints.
# NOTE: the login is a name-picker without a password; do NOT expose to untrusted networks.
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ (https://www.python.org or 'brew install python')."
  read -p "Press Enter to close..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[setup] First run - creating virtual environment (this can take a few minutes)..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[web]"
else
  source .venv/bin/activate
fi

command -v ffmpeg >/dev/null 2>&1 || echo "NOTE: ffmpeg not found (optional). 'brew install ffmpeg' enables inline video preview."

cd webapp/backend
echo "Starting server at http://localhost:8000  (Ctrl+C to stop)"
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000

read -p "Press Enter to close..."
