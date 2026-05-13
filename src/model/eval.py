"""Model evaluation module."""

import os

from ultralytics import YOLO

from src.config import get_settings


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
    data_yaml = data_yaml or runtime.stratified_yaml
    conf = conf if conf is not None else runtime.eval_conf
    iou = iou if iou is not None else runtime.eval_iou

    model = YOLO(model_path)

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
    }

    print("\n── Evaluation Results ──")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"mAP50     : {metrics['mAP50']:.3f}")
    print(f"mAP50-95  : {metrics['mAP50_95']:.3f}")

    return metrics


if __name__ == '__main__':
    run_evaluation()
