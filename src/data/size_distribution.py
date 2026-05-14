# src/data/size_distribution.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from src.config import get_settings

# COCO standard thresholds (normalized to image area)
# small:  area < 32²  / (640²)  → < 0.0025  (relative)
# medium: 32² < area < 96²     → 0.0025 – 0.0225
# large:  area > 96²            → > 0.0225

SMALL_THRESH  = (32  / 640) ** 2   # 0.0025
MEDIUM_THRESH = (96  / 640) ** 2   # 0.0225

CLASS_NAMES = {0: "person", 1: "car"}


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

def analyze_size_distribution(data_yaml_path, split="train", save_dir="docs"):
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    base_dir = _resolve_base_dir(yaml_dir, data)
    image_path = _resolve_split_path(base_dir, yaml_dir, data[split])
    label_path = image_path.replace("images", "labels")

    widths, heights, areas = [], [], []
    size_counts = {"small": 0, "medium": 0, "large": 0}
    per_class = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})

    for label_file in Path(label_path).glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            area = bw * bh  # normalized area (0–1)

            widths.append(bw)
            heights.append(bh)
            areas.append(area)

            if area < SMALL_THRESH:
                size_counts["small"] += 1
                per_class[CLASS_NAMES.get(cls, cls)]["small"] += 1
            elif area < MEDIUM_THRESH:
                size_counts["medium"] += 1
                per_class[CLASS_NAMES.get(cls, cls)]["medium"] += 1
            else:
                size_counts["large"] += 1
                per_class[CLASS_NAMES.get(cls, cls)]["large"] += 1

    total = sum(size_counts.values())
    widths  = np.array(widths)
    heights = np.array(heights)
    areas   = np.array(areas)

    # ── Print Summary ──────────────────────────────────────────────
    print(f"\n{'─'*45}")
    print(f"  Object Size Distribution — {split.upper()} SET")
    print(f"{'─'*45}")
    print(f"  Total objects analyzed : {total:,}")
    print(f"  Small  (< 32px eq.)    : {size_counts['small']:,}  ({100*size_counts['small']/total:.1f}%)")
    print(f"  Medium (32–96px eq.)   : {size_counts['medium']:,}  ({100*size_counts['medium']/total:.1f}%)")
    print(f"  Large  (> 96px eq.)    : {size_counts['large']:,}  ({100*size_counts['large']/total:.1f}%)")
    print(f"\n  Median box width       : {np.median(widths)*100:.2f}% of image width")
    print(f"  Median box height      : {np.median(heights)*100:.2f}% of image height")
    print(f"  Median box area        : {np.median(areas)*100:.4f}% of image area")
    print(f"\n  Per-class breakdown:")
    for cls_name, counts in per_class.items():
        cls_total = sum(counts.values())
        print(f"    {cls_name:>8}: small={counts['small']/cls_total*100:.1f}%  "
              f"medium={counts['medium']/cls_total*100:.1f}%  "
              f"large={counts['large']/cls_total*100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"VisDrone Object Size Distribution — {split.upper()} SET", fontsize=14)

    axes[0].hist(widths * 100,  bins=80, color="#4C9BE8", edgecolor="none")
    axes[0].set_xlabel("Box Width (% of image width)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Width Distribution")
    axes[0].axvline(32/640*100,  color="orange", linestyle="--", label="32px @ 640")
    axes[0].axvline(96/640*100,  color="red",    linestyle="--", label="96px @ 640")
    axes[0].legend(fontsize=8)

    axes[1].hist(heights * 100, bins=80, color="#E87B4C", edgecolor="none")
    axes[1].set_xlabel("Box Height (% of image height)")
    axes[1].set_title("Height Distribution")
    axes[1].axvline(32/640*100,  color="orange", linestyle="--", label="32px @ 640")
    axes[1].axvline(96/640*100,  color="red",    linestyle="--", label="96px @ 640")
    axes[1].legend(fontsize=8)

    labels = ["Small\n(<32px)", "Medium\n(32–96px)", "Large\n(>96px)"]
    values = [size_counts["small"], size_counts["medium"], size_counts["large"]]
    colors = ["#E84C4C", "#E8C44C", "#4CE87B"]
    bars = axes[2].bar(labels, values, color=colors, edgecolor="none")
    axes[2].set_title("Size Category Counts")
    axes[2].set_ylabel("Object Count")
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + total*0.005,
                     f"{val/total*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"size_distribution_{split}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n  Saved → {out_path}")

    return size_counts, per_class


if __name__ == "__main__":
    settings = get_settings()
    analyze_size_distribution(settings.runtime.dataset_yaml, split="train")