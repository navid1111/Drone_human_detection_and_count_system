"""Model benchmarking module for inference speed profiling."""

import os
import sys
import time
from pathlib import Path

# Add project root to path for standalone execution
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from ultralytics import YOLO

from src.config import get_settings


def run_benchmark(
    model_path=None,
    imgsz=None,
    batch=1,
    device=None,
    warmup_runs=10,
    test_runs=100,
):
    """
    Benchmark YOLO model inference speed.
    
    Args:
        model_path: Path to trained model checkpoint
        imgsz: Input image size
        batch: Batch size for inference
        device: Device to run on (cuda/cpu)
        warmup_runs: Number of warmup iterations
        test_runs: Number of timed iterations
        
    Returns:
        Dict with benchmark metrics (FPS, latency, throughput)
    """
    runtime = get_settings().runtime
    
    model_path = model_path or os.path.join(
        runtime.train_project,
        runtime.train_name,
        "weights",
        "best.pt",
    )
    imgsz = imgsz if imgsz is not None else runtime.imgsz
    device = device or runtime.device
    
    # Convert device string to proper format for PyTorch
    if device.isdigit():
        device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" or device.startswith("cuda:"):
        device_str = device if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    
    torch_device = torch.device(device_str)
    
    print(f"\n── Benchmarking Model ──")
    print(f"Model: {model_path}")
    print(f"Device: {device_str}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    
    model = YOLO(model_path)
    
    # Create dummy input
    dummy_input = torch.randn(batch, 3, imgsz, imgsz).to(torch_device)
    
    # Warmup
    print(f"\nWarming up ({warmup_runs} iterations)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.predict(dummy_input, verbose=False, device=device_str)
    
    # Benchmark
    print(f"Running benchmark ({test_runs} iterations)...")
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(test_runs):
        with torch.no_grad():
            _ = model.predict(dummy_input, verbose=False, device=device_str)
    
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_latency_ms = (total_time / test_runs) * 1000
    fps = test_runs / total_time
    throughput = (test_runs * batch) / total_time
    
    metrics = {
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "throughput_images_per_sec": throughput,
        "batch_size": batch,
        "image_size": imgsz,
        "device": device_str,
    }
    
    print("\n── Benchmark Results ──")
    print(f"Average Latency : {avg_latency_ms:.2f} ms")
    print(f"FPS             : {fps:.2f}")
    print(f"Throughput      : {throughput:.2f} images/sec")
    
    return metrics


if __name__ == '__main__':
    run_benchmark()
