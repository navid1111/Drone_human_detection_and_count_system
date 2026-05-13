"""Centralized runtime and build-time configuration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import yaml


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: str
    dataset_yaml: str
    stratified_yaml: str
    source_dataset_dir: str
    stratified_output_dir: str
    dataset_version: str
    expected_classes: int
    imbalance_threshold: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    model_base: str
    train_project: str
    train_name: str
    epochs: int
    imgsz: int
    batch: int
    use_wandb: bool
    device: str
    workers: int
    conf: float
    iou: float
    cls: float
    augment: bool
    optimizer: str
    lr0: float
    lrf: float
    warmup_epochs: int
    patience: int
    eval_conf: float
    eval_iou: float
    shelf_test_images: str
    shelf_output_json: str
    shelf_output_chart: str
    wandb_entity: str
    wandb_project: str


@dataclass(frozen=True)
class BuildConfig:
    airflow_base_image: str
    torch_version: str
    torchvision_version: str
    torchaudio_version: str
    ultralytics_version: str
    opencv_headless_version: str
    pyyaml_version: str
    pillow_version: str
    matplotlib_version: str
    scipy_version: str
    install_wandb: bool


@dataclass(frozen=True)
class AppSettings:
    runtime: RuntimeConfig
    build: BuildConfig


_ENV_MAPPINGS = {
    "runtime.project_root": ("APP_PROJECT_ROOT", str),
    "runtime.dataset_yaml": ("APP_DATASET_YAML", str),
    "runtime.stratified_yaml": ("APP_STRATIFIED_YAML", str),
    "runtime.source_dataset_dir": ("APP_SOURCE_DATASET_DIR", str),
    "runtime.stratified_output_dir": ("APP_STRATIFIED_OUTPUT_DIR", str),
    "runtime.dataset_version": ("APP_DATASET_VERSION", str),
    "runtime.expected_classes": ("APP_EXPECTED_CLASSES", int),
    "runtime.imbalance_threshold": ("APP_IMBALANCE_THRESHOLD", float),
    "runtime.train_ratio": ("APP_TRAIN_RATIO", float),
    "runtime.val_ratio": ("APP_VAL_RATIO", float),
    "runtime.test_ratio": ("APP_TEST_RATIO", float),
    "runtime.random_seed": ("APP_RANDOM_SEED", int),
    "runtime.model_base": ("APP_MODEL_BASE", str),
    "runtime.train_project": ("APP_TRAIN_PROJECT", str),
    "runtime.train_name": ("APP_TRAIN_NAME", str),
    "runtime.epochs": ("APP_TRAIN_EPOCHS", int),
    "runtime.imgsz": ("APP_TRAIN_IMGSZ", int),
    "runtime.batch": ("APP_TRAIN_BATCH", int),
    "runtime.use_wandb": ("APP_USE_WANDB", _to_bool),
    "runtime.device": ("APP_TRAIN_DEVICE", str),
    "runtime.workers": ("APP_TRAIN_WORKERS", int),
    "runtime.conf": ("APP_TRAIN_CONF", float),
    "runtime.iou": ("APP_TRAIN_IOU", float),
    "runtime.cls": ("APP_TRAIN_CLS", float),
    "runtime.augment": ("APP_TRAIN_AUGMENT", _to_bool),
    "runtime.optimizer": ("APP_TRAIN_OPTIMIZER", str),
    "runtime.lr0": ("APP_TRAIN_LR0", float),
    "runtime.lrf": ("APP_TRAIN_LRF", float),
    "runtime.warmup_epochs": ("APP_TRAIN_WARMUP_EPOCHS", int),
    "runtime.patience": ("APP_TRAIN_PATIENCE", int),
    "runtime.eval_conf": ("APP_EVAL_CONF", float),
    "runtime.eval_iou": ("APP_EVAL_IOU", float),
    "runtime.shelf_test_images": ("APP_SHELF_TEST_IMAGES", str),
    "runtime.shelf_output_json": ("APP_SHELF_OUTPUT_JSON", str),
    "runtime.shelf_output_chart": ("APP_SHELF_OUTPUT_CHART", str),
    "runtime.wandb_entity": ("APP_WANDB_ENTITY", str),
    "runtime.wandb_project": ("APP_WANDB_PROJECT", str),
    "build.airflow_base_image": ("BUILD_AIRFLOW_BASE_IMAGE", str),
    "build.torch_version": ("BUILD_TORCH_VERSION", str),
    "build.torchvision_version": ("BUILD_TORCHVISION_VERSION", str),
    "build.torchaudio_version": ("BUILD_TORCHAUDIO_VERSION", str),
    "build.ultralytics_version": ("BUILD_ULTRALYTICS_VERSION", str),
    "build.opencv_headless_version": ("BUILD_OPENCV_HEADLESS_VERSION", str),
    "build.pyyaml_version": ("BUILD_PYYAML_VERSION", str),
    "build.pillow_version": ("BUILD_PILLOW_VERSION", str),
    "build.matplotlib_version": ("BUILD_MATPLOTLIB_VERSION", str),
    "build.scipy_version": ("BUILD_SCIPY_VERSION", str),
    "build.install_wandb": ("BUILD_INSTALL_WANDB", _to_bool),
}


def _load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()

            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _load_dotenv() -> None:
    """Load .env from cwd and project root candidates, if present."""
    candidates = [
        Path.cwd() / ".env",
        Path(os.getenv("APP_PROJECT_ROOT", "")) / ".env" if os.getenv("APP_PROJECT_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            _load_dotenv_file(candidate)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cur = config
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _resolve_path(project_root: str, path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((Path(project_root) / path).resolve())


def _default_values() -> dict[str, Any]:
    project_root = (
        os.getenv("APP_PROJECT_ROOT")
        or os.getenv("PYTHONPATH")
        or str(Path.cwd())
    )
    return {
        "runtime": {
            "project_root": project_root,
            "dataset_yaml": "dataset/dataset/data.yaml",
            "stratified_yaml": "dataset/dataset_stratified/data.yaml",
            "source_dataset_dir": "dataset/dataset",
            "stratified_output_dir": "dataset/dataset_stratified",
            "dataset_version": os.getenv("APP_DATASET_VERSION") or os.getenv("DATASET_VERSION") or "unknown",
            "expected_classes": 70,
            "imbalance_threshold": 10.0,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": 42,
            "model_base": "yolo11m.pt",
            "train_project": "runs/train",
            "train_name": "pipeline_run",
            "epochs": 100,
            "imgsz": 640,
            "batch": 8,
            "use_wandb": True,
            "device": "0",
            "workers": 0,
            "conf": 0.5,
            "iou": 0.5,
            "cls": 1.5,
            "augment": False,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "patience": 15,
            "eval_conf": 0.5,
            "eval_iou": 0.5,
            "shelf_test_images": "dataset/dataset_stratified/test/images",
            "shelf_output_json": "final_try_results.json",
            "shelf_output_chart": "share_of_shelf.png",
            "wandb_entity": "navidkamal-islamic-university-of-technology",
            "wandb_project": "retail_object-detection",
        },
        "build": {
            "airflow_base_image": "apache/airflow:3.1.6",
            "torch_version": "2.6.0",
            "torchvision_version": "0.21.0",
            "torchaudio_version": "2.6.0",
            "ultralytics_version": "8.4.18",
            "opencv_headless_version": "4.13.0.92",
            "pyyaml_version": "6.0.3",
            "pillow_version": "12.1.1",
            "matplotlib_version": "3.10.8",
            "scipy_version": "1.17.1",
            "install_wandb": True,
        },
    }


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    for dotted_path, (env_name, caster) in _ENV_MAPPINGS.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        _set_path(out, dotted_path, caster(raw))
    return out


def _to_runtime(cfg: dict[str, Any]) -> RuntimeConfig:
    project_root = str(Path(cfg["project_root"]).resolve())

    dataset_yaml = _resolve_path(project_root, cfg["dataset_yaml"])
    stratified_yaml = _resolve_path(project_root, cfg["stratified_yaml"])
    source_dataset_dir = _resolve_path(project_root, cfg["source_dataset_dir"])
    stratified_output_dir = _resolve_path(project_root, cfg["stratified_output_dir"])
    model_base = _resolve_path(project_root, cfg["model_base"])
    train_project = _resolve_path(project_root, cfg["train_project"])
    shelf_test_images = _resolve_path(project_root, cfg["shelf_test_images"])
    shelf_output_json = _resolve_path(project_root, cfg["shelf_output_json"])
    shelf_output_chart = _resolve_path(project_root, cfg["shelf_output_chart"])

    return RuntimeConfig(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        stratified_yaml=stratified_yaml,
        source_dataset_dir=source_dataset_dir,
        stratified_output_dir=stratified_output_dir,
        dataset_version=str(cfg["dataset_version"]),
        expected_classes=int(cfg["expected_classes"]),
        imbalance_threshold=float(cfg["imbalance_threshold"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        random_seed=int(cfg["random_seed"]),
        model_base=model_base,
        train_project=train_project,
        train_name=str(cfg["train_name"]),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        use_wandb=bool(cfg["use_wandb"]),
        device=str(cfg["device"]),
        workers=int(cfg["workers"]),
        conf=float(cfg["conf"]),
        iou=float(cfg["iou"]),
        cls=float(cfg["cls"]),
        augment=bool(cfg["augment"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["lr0"]),
        lrf=float(cfg["lrf"]),
        warmup_epochs=int(cfg["warmup_epochs"]),
        patience=int(cfg["patience"]),
        eval_conf=float(cfg["eval_conf"]),
        eval_iou=float(cfg["eval_iou"]),
        shelf_test_images=shelf_test_images,
        shelf_output_json=shelf_output_json,
        shelf_output_chart=shelf_output_chart,
        wandb_entity=str(cfg["wandb_entity"]),
        wandb_project=str(cfg["wandb_project"]),
    )


def _to_build(cfg: dict[str, Any]) -> BuildConfig:
    return BuildConfig(
        airflow_base_image=str(cfg["airflow_base_image"]),
        torch_version=str(cfg["torch_version"]),
        torchvision_version=str(cfg["torchvision_version"]),
        torchaudio_version=str(cfg["torchaudio_version"]),
        ultralytics_version=str(cfg["ultralytics_version"]),
        opencv_headless_version=str(cfg["opencv_headless_version"]),
        pyyaml_version=str(cfg["pyyaml_version"]),
        pillow_version=str(cfg["pillow_version"]),
        matplotlib_version=str(cfg["matplotlib_version"]),
        scipy_version=str(cfg["scipy_version"]),
        install_wandb=bool(cfg["install_wandb"]),
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load app settings with precedence: defaults < yaml file < env vars."""
    _load_dotenv()
    config = _default_values()

    config_file = os.getenv("APP_CONFIG_FILE", "config/runtime.yaml")
    yaml_config = _load_yaml_file(Path(config_file))
    config = _deep_merge(config, yaml_config)
    config = _apply_env_overrides(config)

    return AppSettings(
        runtime=_to_runtime(config["runtime"]),
        build=_to_build(config["build"]),
    )
