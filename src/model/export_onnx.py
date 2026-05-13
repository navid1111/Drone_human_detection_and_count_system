"""ONNX model export module."""

import os
import sys
from pathlib import Path

# Add project root to path for standalone execution
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO

from src.config import get_settings


def export_to_onnx(
    model_path=None,
    imgsz=None,
    dynamic=False,
    simplify=True,
    opset=None,
):
    """
    Export YOLO model to ONNX format for production deployment.
    
    Args:
        model_path: Path to trained model checkpoint
        imgsz: Input image size
        dynamic: Enable dynamic input shapes
        simplify: Simplify ONNX model
        opset: ONNX opset version
        
    Returns:
        Path to exported ONNX model
    """
    runtime = get_settings().runtime
    
    model_path = model_path or os.path.join(
        runtime.train_project,
        runtime.train_name,
        "weights",
        "best.pt",
    )
    imgsz = imgsz if imgsz is not None else runtime.imgsz
    opset = opset or 12  # Default ONNX opset version
    
    print(f"\n── Exporting Model to ONNX ──")
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}")
    print(f"Dynamic shapes: {dynamic}")
    print(f"Simplify: {simplify}")
    print(f"Opset: {opset}")
    
    model = YOLO(model_path)
    
    # Export to ONNX
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
    )
    
    onnx_path = str(export_path)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print("\n── Export Complete ──")
    print(f"ONNX model: {onnx_path}")
    print(f"File size: {onnx_size_mb:.2f} MB")
    
    return onnx_path


if __name__ == '__main__':
    export_to_onnx()
