#!/usr/bin/env bash
# Quick runner for Brain Tumor MRI Live Inference Demo
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="/Users/shashankchaurasia/miniconda3/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$DIR/predictions_showcase_50/run_live_inference_demo.py" "$@"
