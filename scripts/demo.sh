#!/usr/bin/env bash
# TerraLens live demo (Linux / macOS / Git Bash). CPU only, no API keys, no downloads.
# Usage (from anywhere):  bash scripts/demo.sh [--skip-tests] [--limit N]
# See DEMO.md and scripts/demo.py for what each step does.
set -euo pipefail

cd "$(dirname "$0")/.."

# Offline: use only what is already in the Hugging Face cache; never download.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=python3
fi

exec "$PY" scripts/demo.py "$@"
