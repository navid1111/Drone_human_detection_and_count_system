"""Remap VisDrone labels to 2 classes (person, car) in a new dataset root."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


KEEP_AS_PERSON = {0, 1}  # pedestrian + people (crowd)
KEEP_AS_CAR = {3, 4}     # car + van

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def _remap_label_lines(lines: list[str]) -> list[str]:
    remapped: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if not parts:
            continue
        cls_id = int(parts[0])
        coords = parts[1:]
        if cls_id in KEEP_AS_PERSON:
            remapped.append("0 " + " ".join(coords))
        elif cls_id in KEEP_AS_CAR:
            remapped.append("1 " + " ".join(coords))
    return remapped


def _link_or_copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        # Windows often blocks symlinks without Dev Mode/Admin rights
        shutil.copytree(src, dst)


def remap_dataset(src_root: Path, dst_root: Path, link_images: bool) -> None:
    for _split, folder in SPLITS.items():
        src_split = src_root / folder
        src_images = src_split / "images"
        src_labels = src_split / "labels"

        if not src_split.exists():
            print(f"⚠️ Skipping {_split}: {src_split} does not exist.")
            continue

        dst_split = dst_root / folder
        dst_images = dst_split / "images"
        dst_labels = dst_split / "labels"
        dst_labels.mkdir(parents=True, exist_ok=True)

        if link_images:
            _link_or_copy_dir(src_images, dst_images)
        else:
            if not dst_images.exists():
                shutil.copytree(src_images, dst_images)

        for label_file in src_labels.glob("*.txt"):
            lines = label_file.read_text().splitlines()
            remapped = _remap_label_lines(lines)
            (dst_labels / label_file.name).write_text("\n".join(remapped))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap VisDrone labels to 2 classes.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("./VisDrone_Dataset"),  # Relative path avoids WinError 3
        help="Source dataset root (contains VisDrone2019-DET-*/).",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("../dataset/VisDrone_2class"),
        help="Destination dataset root for remapped labels.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of linking them (recommended on Windows).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    remap_dataset(args.src_root, args.dst_root, link_images=not args.copy_images)
    print(f"✅ Remapped labels written to: {args.dst_root}")