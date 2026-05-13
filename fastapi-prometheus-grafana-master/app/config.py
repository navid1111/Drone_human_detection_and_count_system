"""Configuration module for the FastAPI app."""

import os
import tempfile
from pathlib import Path

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://app_user:app_password@mysql:3306/predictions_db",
)

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.onnx")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
INPUT_HW = (640, 640)  # Default input height, width for ONNX model

# File uploads
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # In CI/test environments, fallback to a writable temp location.
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "retail_detection_uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Database setup
DB_WAIT_SECONDS = int(os.getenv("DB_WAIT_SECONDS", "240"))
