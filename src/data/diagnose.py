"""Dataset quality analysis and diagnostics."""

import os
from collections import Counter
import yaml

from src.config import get_settings


def analyze_dataset(data_yaml_path):
    """Analyze class distribution and imbalance in dataset splits."""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    results = {}

    for split in ['train', 'val', 'test']:
        if split not in data:
            continue

        image_path = data[split]
        image_path = os.path.normpath(os.path.join(yaml_dir, image_path))
        label_path = image_path.replace("images", "labels")

        class_counts = Counter()

        print(f"\n{split.upper()} SET:")

        for label_file in os.listdir(label_path):
            label_file_path = os.path.join(label_path, label_file)

            if not label_file.endswith(".txt"):
                continue

            with open(label_file_path) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

        total = sum(class_counts.values())
        num_classes = len(class_counts)
        print(f"  Total objects: {total}")
        print(f"  Classes present: {num_classes}")

        if class_counts:
            max_cls = max(class_counts.values())
            min_cls = min(class_counts.values())
            print(f"  Imbalance ratio: {max_cls/min_cls:.1f}x")
            print(f"  Per-class counts: {dict(class_counts)}")

            results[split] = {
                "num_classes":     num_classes,
                "total_objects":   total,
                "max_cls":         max_cls,
                "min_cls":         min_cls,
                "imbalance_ratio": round(max_cls / min_cls, 1),
                "class_counts":    dict(class_counts),
            }

    return results  # e.g. {"train": {...}, "val": {...}, "test": {...}}


if __name__ == "__main__":
    results = analyze_dataset(get_settings().runtime.dataset_yaml)

    # Example: access individual values
    print("\n── Summary ──")
    for split, stats in results.items():
        print(f"{split}: {stats['num_classes']} classes | "
              f"imbalance {stats['imbalance_ratio']}x | "
              f"{stats['total_objects']} objects")
