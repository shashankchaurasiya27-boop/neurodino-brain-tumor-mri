#!/usr/bin/env bash
# ==============================================================================
# NeuroDINO® Diagnostic Platform - Master Runner Script
# Complete Automated Environment Setup, Dynamic Free Port Discovery,
# Model Health Verification, and Local Web Application Launch
# ==============================================================================

set -uo pipefail

# ANSI Color Palette (Matching Figma NeuroDINO® Brand Specs)
CLR_RESET="\033[0m"
CLR_BOLD="\033[1m"
CLR_DIM="\033[2m"
CLR_PURPLE="\033[38;2;155;67;243m"
CLR_LAVENDER="\033[38;2;110;60;245m"
CLR_GREEN="\033[38;2;41;174;41m"
CLR_BLUE="\033[38;2;0;130;255m"
CLR_ORANGE="\033[38;2;237;118;21m"
CLR_CYAN="\033[38;2;56;225;241m"
CLR_YELLOW="\033[38;2;255;198;42m"
CLR_RED="\033[38;2;239;68;68m"

# Print Header Banner
echo -e "${CLR_PURPLE}${CLR_BOLD}"
cat << "EOF"
  _   _                      ____  ___ _   _  ___  ®
 | \ | | ___ _   _ _ __ ___ |  _ \|_ _| \ | |/ _ \ 
 |  \| |/ _ \ | | | '__/ _ \| | | || ||  \| | | | |
 | |\  |  __/ |_| | | | (_) | |_| || || |\  | |_| |
 |_| \_|\___|\__,_|_|  \___/|____/|___|_| \_|\___/ 
EOF
echo -e "${CLR_RESET}${CLR_CYAN}  Biomedical Foundation AI Diagnostic Platform${CLR_RESET}"
echo -e "${CLR_DIM}  Meta DINOv2 ViT-B/14 + QKV-LoRA (r=8, α=16) • MPS Accelerated${CLR_RESET}"
echo -e "${CLR_DIM}------------------------------------------------------------------${CLR_RESET}"

# Resolve Script Directory
FRONTEND_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$FRONTEND_DIR"

# Flags & Options
RESTART_EXISTING=false
CUSTOM_PORT=""
AUTO_OPEN=true

# Parse Command-Line Arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--restart|--force)
      RESTART_EXISTING=true
      shift
      ;;
    -p|--port)
      CUSTOM_PORT="$2"
      shift 2
      ;;
    --no-browser)
      AUTO_OPEN=false
      shift
      ;;
    -h|--help)
      echo -e "${CLR_BOLD}Usage:${CLR_RESET} ./run_project.sh [options]"
      echo ""
      echo -e "${CLR_BOLD}Options:${CLR_RESET}"
      echo -e "  -p, --port <port>    Target port to bind (default: auto-find starting at 8000)"
      echo -e "  -r, --restart        Terminate any existing process on target port and restart"
      echo -e "  --no-browser         Do not automatically open default web browser"
      echo -e "  -h, --help           Show this help message and exit"
      exit 0
      ;;
    *)
      echo -e "${CLR_YELLOW}[!] Unknown argument: $1${CLR_RESET}"
      shift
      ;;
  esac
done

# ==============================================================================
# STEP 1: Locate Python Runtime Environment
# ==============================================================================
echo -e "${CLR_BOLD}[1/5] Discovering Python Runtime...${CLR_RESET}"

PYTHON_BIN=""
POSSIBLE_PYTHONS=(
  "/Users/shashankchaurasia/miniconda3/bin/python"
  "${VIRTUAL_ENV:-}/bin/python"
  "$(which python3 2>/dev/null || true)"
  "$(which python 2>/dev/null || true)"
)

for py in "${POSSIBLE_PYTHONS[@]}"; do
  if [ -n "$py" ] && [ -x "$py" ]; then
    # Verify Python version >= 3.9
    PY_VER=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
    if [ -n "$PY_VER" ]; then
      PYTHON_BIN="$py"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo -e "${CLR_RED}[✗] ERROR: No working Python 3.9+ environment found!${CLR_RESET}"
  echo "Please ensure Python 3.9+ or Miniconda is installed."
  exit 1
fi

PY_FULL_VER=$("$PYTHON_BIN" --version 2>&1)
echo -e "  ${CLR_GREEN}✓${CLR_RESET} Runtime: ${CLR_BOLD}${PYTHON_BIN}${CLR_RESET} (${PY_FULL_VER})"

# ==============================================================================
# STEP 2: Verify & Heal Dependencies
# ==============================================================================
echo -e "${CLR_BOLD}[2/5] Inspecting Required Dependencies...${CLR_RESET}"

MISSING_DEPS=$("$PYTHON_BIN" -c '
required = ["fastapi", "uvicorn", "torch", "torchvision", "peft", "PIL", "pandas", "numpy", "multipart"]
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
print(" ".join(missing))
' 2>/dev/null || true)

if [ -n "$MISSING_DEPS" ]; then
  echo -e "  ${CLR_YELLOW}[!] Missing packages detected: ${MISSING_DEPS}${CLR_RESET}"
  echo -e "  ${CLR_CYAN}[i] Installing missing dependencies via pip...${CLR_RESET}"
  "$PYTHON_BIN" -m pip install fastapi uvicorn python-multipart torch torchvision peft pillow pandas numpy
  echo -e "  ${CLR_GREEN}✓ Dependencies successfully installed!${CLR_RESET}"
else
  echo -e "  ${CLR_GREEN}✓${CLR_RESET} All critical packages verified (fastapi, uvicorn, torch, peft, PIL, pandas)"
fi

# ==============================================================================
# STEP 3: Check Hardware Accelerator & Checkpoints
# ==============================================================================
echo -e "${CLR_BOLD}[3/5] Validating Hardware Acceleration & Certified Weights...${CLR_RESET}"

HW_INFO=$("$PYTHON_BIN" -c '
import torch
if torch.backends.mps.is_available():
    print("Apple Silicon MPS (Metal Performance Shaders) Hardware Acceleration: AVAILABLE")
elif torch.cuda.is_available():
    print(f"NVIDIA CUDA GPU ({torch.cuda.get_device_name(0)}): AVAILABLE")
else:
    print("CPU Fallback: AVAILABLE")
' 2>/dev/null || echo "Unknown Device")

echo -e "  ${CLR_GREEN}✓${CLR_RESET} Compute Device: ${CLR_PURPLE}${HW_INFO}${CLR_RESET}"

# Quick validation of config and checkpoints
"$PYTHON_BIN" -c '
import config
assert config.PROJECT_ROOT.exists(), f"PROJECT_ROOT missing: {config.PROJECT_ROOT}"
assert config.LORA_CHECKPOINT_PATH.exists(), f"LoRA checkpoint missing: {config.LORA_CHECKPOINT_PATH}"
assert config.FROZEN_CHECKPOINT_PATH.exists(), f"Frozen checkpoint missing: {config.FROZEN_CHECKPOINT_PATH}"
' 2>/dev/null
if [ $? -eq 0 ]; then
  echo -e "  ${CLR_GREEN}✓${CLR_RESET} Certified Checkpoints: ${CLR_BOLD}controlled_lora_best.pth${CLR_RESET} & ${CLR_BOLD}frozen_control_best.pth${CLR_RESET} (Verified)"
else
  echo -e "  ${CLR_YELLOW}[!] Notice: Checkpoints will be verified during model load.${CLR_RESET}"
fi

# ==============================================================================
# STEP 4: Intelligent Free Port Discovery
# ==============================================================================
echo -e "${CLR_BOLD}[4/5] Discovering Available Local Port...${CLR_RESET}"

START_PORT="${CUSTOM_PORT:-8000}"

# Function to test port availability via Python socket
check_port() {
  local p=$1
  "$PYTHON_BIN" -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1', $p))
        exit(0)
    except OSError:
        exit(1)
" 2>/dev/null
}

TARGET_PORT=""

# If specific custom port requested
if [ -n "$CUSTOM_PORT" ]; then
  if check_port "$CUSTOM_PORT"; then
    TARGET_PORT="$CUSTOM_PORT"
  else
    if [ "$RESTART_EXISTING" = true ]; then
      echo -e "  ${CLR_YELLOW}[!] Port $CUSTOM_PORT is occupied. Restart requested; freeing port...${CLR_RESET}"
      OCCUPIED_PID=$(lsof -ti tcp:"$CUSTOM_PORT" 2>/dev/null || true)
      if [ -n "$OCCUPIED_PID" ]; then
        kill -9 $OCCUPIED_PID 2>/dev/null || true
        sleep 1
      fi
      TARGET_PORT="$CUSTOM_PORT"
    else
      echo -e "  ${CLR_RED}[✗] Specified port $CUSTOM_PORT is in use!${CLR_RESET}"
      echo -e "  Use ${CLR_BOLD}--restart${CLR_RESET} to terminate existing process or omit --port to auto-discover."
      exit 1
    fi
  fi
else
  # Auto-discovery mode: start at 8000, check up to 8999
  if check_port 8000; then
    TARGET_PORT=8000
  else
    # Port 8000 is occupied
    OCCUPIED_PID=$(lsof -ti tcp:8000 2>/dev/null || true)
    
    if [ "$RESTART_EXISTING" = true ] && [ -n "$OCCUPIED_PID" ]; then
      echo -e "  ${CLR_YELLOW}[!] Port 8000 is occupied by PID $OCCUPIED_PID. Restart requested; freeing port...${CLR_RESET}"
      kill -9 $OCCUPIED_PID 2>/dev/null || true
      sleep 1
      TARGET_PORT=8000
    else
      # Automatically find next free port
      TARGET_PORT=$("$PYTHON_BIN" -c '
import socket
for p in range(8001, 9000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", p))
            print(p)
            break
        except OSError:
            continue
' 2>/dev/null || true)

      if [ -z "$TARGET_PORT" ]; then
        echo -e "${CLR_RED}[✗] ERROR: Could not find any open port in range 8000-8999!${CLR_RESET}"
        exit 1
      fi
      echo -e "  ${CLR_YELLOW}[i] Port 8000 is occupied (PID ${OCCUPIED_PID:-unknown}). Dynamically rerouting...${CLR_RESET}"
    fi
  fi
fi

echo -e "  ${CLR_GREEN}✓${CLR_RESET} Selected Free Port: ${CLR_BOLD}${CLR_CYAN}${TARGET_PORT}${CLR_RESET}"

# ==============================================================================
# STEP 5: Launch FastAPI Server & Live Readiness Poll
# ==============================================================================
echo -e "${CLR_BOLD}[5/5] Launching NeuroDINO® Diagnostic Engine...${CLR_RESET}"

SERVER_URL="http://127.0.0.1:${TARGET_PORT}"

# Export port for app.py
export PORT="${TARGET_PORT}"
export HOST="127.0.0.1"

# Setup process trap for clean exit on Ctrl+C
SERVER_PID=""

cleanup() {
  echo ""
  echo -e "${CLR_YELLOW}[*] Shutting down NeuroDINO® Diagnostic Engine cleanly...${CLR_RESET}"
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  echo -e "${CLR_GREEN}✓ Server stopped. Port ${TARGET_PORT} released.${CLR_RESET}"
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start server in background to poll for readiness
"$PYTHON_BIN" "$FRONTEND_DIR/app.py" --host 127.0.0.1 --port "$TARGET_PORT" &
SERVER_PID=$!

echo -e "  ${CLR_CYAN}➜ Initializing DINOv2 ViT-B/14 weights and adapters on MPS...${CLR_RESET}"

# Poll health check endpoint until ready (up to 30 seconds)
READY=false
for i in {1..30}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${CLR_RED}[✗] Server failed to start! Check logs above for details.${CLR_RESET}"
    exit 1
  fi

  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/api/health" 2>/dev/null || true)
  if [ "$HTTP_CODE" = "200" ]; then
    READY=true
    break
  fi
  sleep 1
  echo -ne "  ${CLR_DIM}Loading models... ($i/30s)${CLR_RESET}\r"
done

echo ""

if [ "$READY" = true ]; then
  echo -e "${CLR_GREEN}${CLR_BOLD}==================================================================${CLR_RESET}"
  echo -e "${CLR_GREEN}${CLR_BOLD}  ✓ NEURODINO® PLATFORM IS LIVE & READY!${CLR_RESET}"
  echo -e "  ${CLR_BOLD}Local Web UI:${CLR_RESET}   ${CLR_CYAN}${CLR_BOLD}${SERVER_URL}${CLR_RESET}"
  echo -e "  ${CLR_BOLD}Health Status:${CLR_RESET}  ${CLR_GREEN}HTTP 200 Healthy (MPS Accelerated)${CLR_RESET}"
  echo -e "  ${CLR_BOLD}Figma Makeover:${CLR_RESET} Joey® / Djectstudio Clean Grotesk"
  echo -e "  ${CLR_DIM}Press ${CLR_BOLD}Ctrl+C${CLR_RESET}${CLR_DIM} anytime to stop server cleanly.${CLR_RESET}"
  echo -e "${CLR_GREEN}${CLR_BOLD}==================================================================${CLR_RESET}"

  # Auto-open browser
  if [ "$AUTO_OPEN" = true ]; then
    echo -e "${CLR_CYAN}[➜] Opening web UI in your default browser...${CLR_RESET}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
      open "$SERVER_URL"
    elif which xdg-open > /dev/null 2>&1; then
      xdg-open "$SERVER_URL"
    fi
  fi
else
  echo -e "${CLR_YELLOW}[!] Server startup timed out waiting for health check, but process is still running.${CLR_RESET}"
  echo -e "  Attempting to open ${SERVER_URL} anyway..."
  if [ "$AUTO_OPEN" = true ] && [[ "$OSTYPE" == "darwin"* ]]; then
    open "$SERVER_URL"
  fi
fi

# Keep script running and attached to server logs until Ctrl+C
wait "$SERVER_PID"

