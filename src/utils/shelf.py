"""Share of Shelf Analytics - Calculate product shelf presence metrics."""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

from src.config import get_settings


def load_results(json_path: str) -> list:
    """Load YOLO prediction results from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def count_detections(results: list) -> Counter:
    """Count total detections per class across all images."""
    class_counts = Counter()
    for entry in results:
        # Support both formats: list of dicts or dict with 'predictions' key
        predictions = entry.get("predictions", entry) if isinstance(entry, dict) else entry
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict):
                    label = pred.get("class_name") or pred.get("name") or pred.get("label") or str(pred.get("class", "unknown"))
                    class_counts[label] += 1
    return class_counts


def calculate_share_of_shelf(class_counts: Counter) -> dict:
    """Calculate percentage share of shelf for each class."""
    total = sum(class_counts.values())
    if total == 0:
        return {}
    return {cls: (count / total) * 100 for cls, count in class_counts.items()}


def plot_share_of_shelf(share: dict, counts: Counter, output_path: str = "share_of_shelf.png"):
    """Generate a clean horizontal bar chart of Share of Shelf."""
    # Sort by percentage descending
    sorted_items = sorted(share.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    percentages = [item[1] for item in sorted_items]
    raw_counts = [counts[lbl] for lbl in labels]

    n = len(labels)
    fig_height = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Color gradient from deep blue to light blue
    colors = plt.cm.Blues(np.linspace(0.85, 0.35, n))

    bars = ax.barh(range(n), percentages, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Add percentage + count labels on each bar
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, raw_counts)):
        x_pos = bar.get_width()
        label_text = f"  {pct:.1f}%  (n={cnt})"
        ax.text(
            x_pos + 0.1, i, label_text,
            va="center", ha="left", fontsize=8,
            color="#333333", fontweight="bold"
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Share of Shelf (%)", fontsize=12, labelpad=10)
    ax.set_title(
        "Share of Shelf Analytics\nPercentage of Each SKU Across Test Dataset",
        fontsize=14, fontweight="bold", pad=20
    )

    ax.set_xlim(0, max(percentages) * 1.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total = sum(counts.values())
    fig.text(
        0.99, 0.01,
        f"Total detections: {total:,}  |  Classes: {len(labels)}",
        ha="right", va="bottom", fontsize=9, color="gray"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Chart saved to: {output_path}")
    plt.close()


def run_from_model(
    model_path: str,
    test_images: str,
    output_json: str = "final_results.json",
    output_chart: str = "share_of_shelf.png",
) -> None:
    """Run inference on test images and generate shelf analytics."""
    model = YOLO(model_path)

    # Run predictions
    results_raw = model.predict(source=test_images, verbose=False)

    # Convert to JSON format
    results_list = []
    for result in results_raw:
        entry = {
            "image_path": str(result.path),
            "predictions": [
                {
                    "class": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist() if len(box.xyxy) > 0 else [],
                }
                for box in result.boxes
            ],
        }
        results_list.append(entry)

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"✅ Results saved to: {output_json}")

    # Count and plot
    class_counts = count_detections(results_list)
    share = calculate_share_of_shelf(class_counts)

    # Print summary
    print("\n── Detection Summary ──")
    for cls, pct in sorted(share.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {pct:.2f}% (n={class_counts[cls]})")

    # Plot
    plot_share_of_shelf(share, class_counts, output_chart)


if __name__ == "__main__":
    runtime = get_settings().runtime
    run_from_model(
        model_path=str(Path(runtime.train_project) / runtime.train_name / "weights" / "best.pt"),
        test_images=runtime.shelf_test_images,
        output_json=runtime.shelf_output_json,
        output_chart=runtime.shelf_output_chart,
    )
