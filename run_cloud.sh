#!/usr/bin/env bash
# ==============================================================================
# NeuroDINO® Diagnostic Platform - Cloud Studio Runner (Lightning AI / Linux)
# Automatically starts the platform with PyTorch hardware acceleration
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Starting NeuroDINO Diagnostic Engine ==="
echo "Installing/verifying required dependencies..."
python3 -m pip install -r requirements.txt

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Launching FastAPI server on http://$HOST:$PORT ..."
exec python3 brain_tumor_frontend/app.py --host "$HOST" --port "$PORT"

