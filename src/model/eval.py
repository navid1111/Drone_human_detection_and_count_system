"""Model evaluation module."""

import os
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.config import get_settings


def _resolve_base_dir(yaml_dir: str, data: dict) -> str:
    base = data.get("path")
    if not base:
        return yaml_dir
    if os.path.isabs(base):
        return base
    return os.path.normpath(os.path.join(yaml_dir, base))


def _resolve_split_path(base_dir: str, yaml_dir: str, split_value: str) -> str:
    if os.path.isabs(split_value):
        return split_value

    candidate = os.path.normpath(os.path.join(base_dir, split_value))
    if os.path.exists(candidate):
        return candidate

    fallback = os.path.normpath(os.path.join(yaml_dir, split_value))
    if os.path.exists(fallback):
        return fallback

    return candidate


def run_evaluation(
    model_path=None,
    data_yaml=None,
    conf=None,
    iou=None,
):
    """Validate a YOLO model and return a metrics dict."""
    runtime = get_settings().runtime
    model_path = model_path or os.path.join(
        runtime.train_project,
        runtime.train_name,
        "weights",
        "best.pt",
    )
    data_yaml = data_yaml or runtime.dataset_yaml
    conf = conf if conf is not None else runtime.eval_conf
    iou = iou if iou is not None else runtime.eval_iou

    model = YOLO(model_path)

    fps = 0.0
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
        base_dir = _resolve_base_dir(yaml_dir, data)
        val_images = _resolve_split_path(base_dir, yaml_dir, data.get("val", ""))
        val_img_dir = Path(val_images)
        sample_imgs = list(val_img_dir.glob("*.jpg"))[:100]
        if sample_imgs:
            start = time.time()
            for img in sample_imgs:
                model.predict(str(img), verbose=False, conf=conf)
            elapsed = time.time() - start
            if elapsed > 0:
                fps = len(sample_imgs) / elapsed
    except Exception as e:
        print(f"ℹ️  FPS measurement skipped: {e}")

    results = model.val(
        data=data_yaml,
        conf=conf,
        iou=iou,
        verbose=False,
        workers=0,
    )

    metrics = {
        "precision": float(results.results_dict["metrics/precision(B)"]),
        "recall":    float(results.results_dict["metrics/recall(B)"]),
        "mAP50":     float(results.results_dict["metrics/mAP50(B)"]),
        "mAP50_95":  float(results.results_dict["metrics/mAP50-95(B)"]),
        "fps":       round(fps, 2),
    }

    print("\n── Evaluation Results ──")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"mAP50     : {metrics['mAP50']:.3f}")
    print(f"mAP50-95  : {metrics['mAP50_95']:.3f}")
    print(f"FPS       : {metrics['fps']:.2f}")

    return metrics


if __name__ == '__main__':
    run_evaluation()
