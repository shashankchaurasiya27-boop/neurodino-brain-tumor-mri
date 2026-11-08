import sys
from pathlib import Path

# Add frontend directory to Python path
FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

# Import the FastAPI application instance
from app import app

