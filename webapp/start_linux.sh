#!/usr/bin/env bash
# Physical Quality Decision AI - web app launcher (Linux).
#
# Usage:  bash webapp/start_linux.sh
# Optional environment variables:
#   PQDAI_DETECTOR_WEIGHTS    path to a detector checkpoint (default: the domain release in weights/<domain>/)
#   PQDAI_CLASSIFIER_WEIGHTS  path to a classifier checkpoint (default: the domain release in weights/<domain>/)
#   PQDAI_DOMAIN              domain plugin (default: weld)
#   HOST / PORT              bind address (default 127.0.0.1:8000)
#
# NOTE: the login is a name-picker without a password. Do NOT expose this
# server to untrusted networks.
set -e
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "[setup] creating virtual environment (.venv) ..."
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[web]"
else
  . .venv/bin/activate
fi

command -v ffmpeg >/dev/null 2>&1 || echo "NOTE: ffmpeg not found (optional) - without it the result video may not preview inline."

cd webapp/backend
echo "Starting server at http://${HOST:-127.0.0.1}:${PORT:-8000}"
python3 -m uvicorn main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
