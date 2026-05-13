"""Validate class coverage across dataset splits."""

import os
from collections import Counter
import yaml

from src.config import get_settings


def check_class_coverage(data_yaml_path):
    """Check if all training classes appear in validation and test splits."""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    
    split_counts = {}
    
    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
        image_path = data[split]
        image_path = os.path.normpath(os.path.join(yaml_dir, image_path))
        label_path = image_path.replace("images", "labels")
        
        class_counts = Counter()
        for label_file in os.listdir(label_path):
            if not label_file.endswith(".txt"):
                continue
            with open(os.path.join(label_path, label_file)) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
        split_counts[split] = class_counts
    
    train_classes = set(split_counts['train'].keys())
    val_classes = set(split_counts['val'].keys())
    test_classes = set(split_counts['test'].keys())
    
    missing_from_val = train_classes - val_classes
    missing_from_test = train_classes - test_classes
    
    print(f"Total classes in train: {len(train_classes)}")
    print(f"Total classes in val:   {len(val_classes)}")
    print(f"Total classes in test:  {len(test_classes)}")
    
    print(f"\nClasses in train but MISSING from val ({len(missing_from_val)}):")
    print(sorted(missing_from_val))
    
    print(f"\nClasses in train but MISSING from test ({len(missing_from_test)}):")
    print(sorted(missing_from_test))
    
    print(f"\nClasses with <10 training examples (hardest to learn):")
    for cls, count in sorted(split_counts['train'].items(), key=lambda x: x[1]):
        if count < 10:
            print(f"  Class {cls}: {count} examples")


if __name__ == "__main__":
    check_class_coverage(get_settings().runtime.stratified_yaml)
