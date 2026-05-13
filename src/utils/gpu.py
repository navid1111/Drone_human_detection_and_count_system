"""GPU availability and configuration utilities."""

import torch


def check_gpu_availability():
    """Check GPU availability and print device information."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.cuda.is_available()


if __name__ == "__main__":
    check_gpu_availability()
