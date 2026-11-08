#!/usr/bin/env bash
# ==============================================================================
# NeuroDINO® Diagnostic Platform - Launcher from Project Root
# Forwards execution to the modern frontend runner in brain_tumor_frontend
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/brain_tumor_frontend/run_project.sh" ]; then
    FRONTEND_DIR="$SCRIPT_DIR/brain_tumor_frontend"
elif [ -f "$SCRIPT_DIR/frontend/run_project.sh" ]; then
    FRONTEND_DIR="$SCRIPT_DIR/frontend"
elif [ -f "$SCRIPT_DIR/../brain_tumor_frontend/run_project.sh" ]; then
    FRONTEND_DIR="$SCRIPT_DIR/../brain_tumor_frontend"
else
    FRONTEND_DIR="/Users/shashankchaurasia/Documents/brain_tumor_frontend"
fi

if [ -f "$FRONTEND_DIR/run_project.sh" ]; then
    exec "$FRONTEND_DIR/run_project.sh" "$@"
else
    echo "Error: Frontend runner not found at $FRONTEND_DIR/run_project.sh"
    exit 1
fi

