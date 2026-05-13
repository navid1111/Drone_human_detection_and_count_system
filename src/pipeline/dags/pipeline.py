"""
Retail Object-Detection Pipeline
─────────────────────────────────
check_data_quality
        ↓
   [BranchOperator]
    ↙           ↘
rebuild_splits   skip_rebuild
    ↘           ↙
   [join / gate]
        ↓
   train_model
        ↓
  benchmark_model
        ↓
  export_to_onnx
        ↓
  evaluate_model
        ↓
 run_share_of_shelf
        ↓
  push_to_registry
"""

from airflow.sdk import dag, task
import os
import sys

# Ensure the project root is on sys.path so modules are importable
# In containers, PYTHONPATH is set to /opt/airflow/project; in local dev, use cwd
_PROJECT_ROOT = os.environ.get("PYTHONPATH") or os.environ.get("APP_PROJECT_ROOT") or "/opt/airflow/project"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import from src package
from src.data.diagnose import analyze_dataset
from src.data.rebuild_splits import main as rebuild_main
from src.model.train import run_training
from src.model.benchmark import run_benchmark
from src.model.export_onnx import export_to_onnx
from src.model.eval import run_evaluation
from src.utils.shelf import run_from_model
from src.utils.gpu import check_gpu_availability
from src.config import get_settings

# ── Configuration (loaded from src.config.settings) ───────
runtime = get_settings().runtime
# Use _PROJECT_ROOT (from PYTHONPATH) to resolve relative paths, ensuring container compatibility
DATASET_YAML = os.path.join(_PROJECT_ROOT, runtime.dataset_yaml) if not os.path.isabs(runtime.dataset_yaml) else runtime.dataset_yaml
STRATIFIED_YAML = os.path.join(_PROJECT_ROOT, runtime.stratified_yaml) if not os.path.isabs(runtime.stratified_yaml) else runtime.stratified_yaml
EXPECTED_CLASSES = runtime.expected_classes

MODEL_BASE = os.path.join(_PROJECT_ROOT, runtime.model_base) if not os.path.isabs(runtime.model_base) else runtime.model_base
TRAIN_PROJECT = os.path.join(_PROJECT_ROOT, runtime.train_project) if not os.path.isabs(runtime.train_project) else runtime.train_project
TRAIN_NAME = runtime.train_name
SHELF_TEST_IMAGES = os.path.join(_PROJECT_ROOT, runtime.shelf_test_images) if not os.path.isabs(runtime.shelf_test_images) else runtime.shelf_test_images
SHELF_OUTPUT_JSON = os.path.join(_PROJECT_ROOT, runtime.shelf_output_json) if not os.path.isabs(runtime.shelf_output_json) else runtime.shelf_output_json
SHELF_OUTPUT_CHART = os.path.join(_PROJECT_ROOT, runtime.shelf_output_chart) if not os.path.isabs(runtime.shelf_output_chart) else runtime.shelf_output_chart


@dag(dag_id="branch_dag")
def branch_dag():

    # ── 1. Check data quality ────────────────────────────────
    @task.python
    def check_data_quality():
        """Analyse class balance / coverage and decide if splits need rebuilding."""
        needs_rebuild = not os.path.exists(STRATIFIED_YAML)

        if needs_rebuild:
            results = analyze_dataset(DATASET_YAML)
        else:
            results = analyze_dataset(STRATIFIED_YAML)

        for _split, stats in results.items():
            if stats.get("num_classes", 0) < EXPECTED_CLASSES:
                needs_rebuild = True

        return {"needs_rebuild": needs_rebuild, "diagnostics": results}

    # ── 2. Branch decision ───────────────────────────────────
    @task.branch
    def decide_rebuild(quality_result):
        """Route to rebuild_splits or skip_rebuild based on quality check."""
        if quality_result["needs_rebuild"]:
            return "rebuild_splits"
        return "skip_rebuild"

    # ── 3a. Rebuild splits (stratified) ──────────────────────
    @task.python
    def rebuild_splits():
        """Re-create train/val/test with stratified sampling."""
        rebuild_main(base_dir=_PROJECT_ROOT)

    # ── 3b. Skip rebuild ─────────────────────────────────────
    @task.python
    def skip_rebuild():
        """No-op — data quality is acceptable."""
        print("Data quality acceptable — skipping rebuild.")

    # ── 4. Join gate ─────────────────────────────────────────
    @task.python(trigger_rule="none_failed_min_one_success")
    def join():
        """Rejoin the two branches before training."""
        print("Branch complete — continuing pipeline.")

    # ── 4.5. Check GPU ───────────────────────────────────────
    @task.python
    def check_gpu():
        """Check GPU availability before training."""
        print(f"✓ GPU available: {check_gpu_availability()}")
        return check_gpu_availability()

    # ── 5. Train model ───────────────────────────────────────
    @task.python
    def train_model():
        """Train YOLO on the (possibly rebuilt) stratified dataset."""
        best = run_training(
            data_yaml=STRATIFIED_YAML,
            model_base=MODEL_BASE,
            project=TRAIN_PROJECT,
            name=TRAIN_NAME,
            use_wandb=runtime.use_wandb,
        )
        return best

    # ── 6. Benchmark model ───────────────────────────────────
    @task.python
    def benchmark_model(model_path: str):
        """Profile inference speed and throughput."""
        metrics = run_benchmark(
            model_path=model_path,
            imgsz=runtime.imgsz,
            batch=1,
            device=runtime.device,
        )
        return {"model_path": model_path, "benchmark": metrics}

    # ── 7. Export to ONNX ────────────────────────────────────
    @task.python
    def export_onnx(benchmark_result: dict):
        """Convert trained model to ONNX format."""
        onnx_path = export_to_onnx(
            model_path=benchmark_result["model_path"],
            imgsz=runtime.imgsz,
            dynamic=False,
            simplify=False,
        )
        return {
            "model_path": benchmark_result["model_path"],
            "onnx_path": onnx_path,
            "benchmark": benchmark_result["benchmark"],
        }

    # ── 8. Evaluate model ────────────────────────────────────
    @task.python
    def evaluate_model(export_result: dict):
        """Validate the trained model and return key metrics."""
        metrics = run_evaluation(
            model_path=export_result["model_path"],
            data_yaml=STRATIFIED_YAML,
        )
        return {
            "model_path": export_result["model_path"],
            "onnx_path": export_result["onnx_path"],
            "metrics": metrics,
            "benchmark": export_result["benchmark"],
        }

    # ── 9. Share-of-Shelf analytics ──────────────────────────
    @task.python
    def run_share_of_shelf(eval_result: dict):
        """Run predictions on test images and compute shelf-share percentages."""
        run_from_model(
            model_path=eval_result["model_path"],
            test_images=SHELF_TEST_IMAGES,
            output_json=SHELF_OUTPUT_JSON,
            output_chart=SHELF_OUTPUT_CHART,
        )
        return eval_result

    # ── 10. Push to model registry ───────────────────────────
    @task.python
    def push_to_registry(result: dict):
        """Push the best checkpoint and ONNX model to registry."""
        print(f"Pushing PyTorch model: {result['model_path']}")
        print(f"Pushing ONNX model: {result['onnx_path']}")
        print(f"Benchmark - FPS: {result['benchmark']['fps']:.2f}")
        print(f"Metrics - mAP50: {result['metrics']['mAP50']:.3f}")
        # TODO: implement actual push (e.g. MLflow, W&B, S3)
        print("Models registered successfully (placeholder).")

    # ── DAG wiring ───────────────────────────────────────────
    quality         = check_data_quality()
    branch          = decide_rebuild(quality)
    do_rebuild      = rebuild_splits()
    do_skip         = skip_rebuild()
    gate            = join()
    gpu_info        = check_gpu()
    model_path      = train_model()
    benchmark_res   = benchmark_model(model_path)
    export_res      = export_onnx(benchmark_res)
    eval_result     = evaluate_model(export_res)
    shelf_result    = run_share_of_shelf(eval_result)
    push_to_registry(shelf_result)

    branch >> [do_rebuild, do_skip] >> gate >> gpu_info >> model_path


# Instantiate the DAG
branch_dag()

