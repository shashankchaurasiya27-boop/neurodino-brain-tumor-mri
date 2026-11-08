import sys
from pathlib import Path

# Add project root and frontend directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "brain_tumor_frontend"

if FRONTEND_DIR.exists() and str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI application instance
from app import app

