# Project Architecture & Workflow

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RETAIL OBJECT DETECTION PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  RAW DATASET     │
                         │  (Roboflow)      │
                         └────────┬─────────┘
                                  │
                    DVC.ACK (data versioning)
                                  │
                                  ▼
                  ┌────────────────────────────────┐
                  │  src/data/diagnose.py          │
                  │  Analyze class distribution    │
                  └────────────┬───────────────────┘
                               │
                         Quality OK?
                         /          \
                        /            \
                       ▼              ▼
                  Rebuild              Keep existing
                  splits              splits
                      │                  │
                      ▼                  ▼
           ┌──────────────────────────────────────┐
           │ src/data/rebuild_splits.py           │
           │ Create stratified splits             │
           │ (all classes in all splits)          │
           └────────────┬─────────────────────────┘
                        │
                        ▼
          ┌────────────────────────────────┐
          │ src/utils/gpu.py               │
          │ Check GPU availability         │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌────────────────────────────────┐
          │ src/model/train.py             │
          │ Train YOLO 11 (100 epochs)     │
          │ Log to W&B                     │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌────────────────────────────────┐
          │ src/model/eval.py              │
          │ Evaluate on validation set     │
          │ Get: Precision, Recall, mAP    │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌────────────────────────────────┐
          │ src/utils/shelf.py             │
          │ Share-of-Shelf Analytics       │
          │ Predict on test set            │
          │ Generate visualization         │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌────────────────────────────────┐
          │ Push to Model Registry         │
          │ (MLflow/W&B/S3)                │
          └────────────────────────────────┘
```

## Data Flow

```
┌──────────────────┐
│ dataset/dataset/ │  (DVC tracked)
│ (raw Roboflow)   │
└────────┬─────────┘
         │
         │ [analyze quality]
         ▼
┌────────────────────────────┐
│ Check class coverage       │
│ - Classes in each split?   │
│ - Imbalance ratio?         │
└────────┬───────────────────┘
         │
         │ [if needs rebuild]
         ▼
┌────────────────────────────────────────┐
│ dataset/dataset_stratified/            │
│ ├── train/  (80%, 1604 images)        │
│ ├── valid/  (10%, 200 images)         │
│ └── test/   (10%, 199 images)         │
│                                        │
│ All 70 classes in all splits!         │
└────────┬───────────────────────────────┘
         │
         │ [train model]
         ▼
┌────────────────────────────────────────┐
│ runs/train/pipeline_run/               │
│ └── weights/                           │
│     ├── best.pt    ← Use this         │
│     └── last.pt                       │
└────────┬───────────────────────────────┘
         │
         │ [validate & inference]
         ▼
┌────────────────────────────────────────┐
│ final_try_results.json                 │
│ share_of_shelf.png                    │
│ + W&B experiment tracking              │
└────────────────────────────────────────┘
```

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│                   External Dependencies             │
│ ultralytics | torch | wandb | dvc | airflow        │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴─────────────┬──────────────┐
          │                          │              │
          ▼                          ▼              ▼
    ┌──────────────┐          ┌─────────────┐  ┌──────────────┐
    │ src/model/   │          │ src/data/   │  │ src/utils/   │
    ├─ train.py   │          ├─ diagnose   │  ├─ gpu.py      │
    └─ eval.py    │          ├─ rebuild    │  └─ shelf.py   │
                              └─ check      │
                                            
          ▼                          ▼              ▼
    ┌──────────────────────────────────────────────────┐
    │           src/pipeline/dags/pipeline.py          │
    │  (Airflow DAG orchestrating all above)          │
    └──────────────────────────────────────────────────┘
```

## Workflow Execution

### Manual (Python Scripts)
```bash
# 1. Check data
python -m src.data.diagnose

# 2. Rebuild if needed
python -m src.data.rebuild_splits

# 3. Train
python -m src.model.train

# 4. Evaluate
python -m src.model.eval

# 5. Analyze shelf
python -m src.utils.shelf
```

### Automated (Airflow)
```bash
airflow dags trigger branch_dag

# Executes:
# check_data_quality() 
#   ├─→ decide_rebuild (branch)
#   │   ├─→ rebuild_splits OR skip_rebuild
#   │   └─→ join (gate)
#   └─→ check_gpu()
#       └─→ train_model()
#           └─→ evaluate_model()
#               └─→ run_share_of_shelf()
#                   └─→ push_to_registry()
```

## File Organization Benefits

| Aspect | Benefit |
|--------|---------|
| **Maintainability** | Each module has a single responsibility |
| **Testability** | Isolated modules → easier unit tests |
| **Scalability** | Easy to add new models, data processors, utils |
| **Reusability** | Import functions from anywhere in codebase |
| **Team Collaboration** | Clear ownership: data owner, model owner, infra owner |
| **CI/CD** | Easy to set up automated tests per module |
| **Documentation** | Each `__init__.py` documents public API |

## Key Statistics

### Dataset
- **Total Images**: 2,003
- **Unique Classes**: 70
- **Train Split**: 1,604 (80%)
- **Val Split**: 200 (10%)
- **Test Split**: 199 (10%)
- **Imbalance Ratio**: ~10.5x

### Model
- **Base Model**: YOLO 11 Medium
- **Training Time**: ~2-3 hours (GPU)
- **Expected mAP50**: 0.5-0.7 (varies by class)
- **Monitoring**: Weights & Biases

### Code
- **Source Files**: 13 (7 modules + 6 __init__)
- **Package Modules**: 4 (model, data, utils, pipeline)
- **DAGs**: 1 (branch_dag)
- **Test Files**: 1 (test_pipeline.py - template)

## Next Steps

1. **Verify Structure**
   ```bash
   ls -la src/
   tree src/
   ```

2. **Test Imports**
   ```bash
   python -c "from src.model import run_training; print('OK')"
   ```

3. **Run a Module**
   ```bash
   python -m src.data.diagnose
   ```

4. **Set Up Environment**
   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   # Or: pip install -e .
   ```

5. **Run Airflow Pipeline**
   ```bash
   airflow webui -p 8080
   # Access at http://localhost:8080
   ```

## Troubleshooting Quick Reference

| Problem | Answer |
|---------|--------|
| Can't import `src`? | Set PYTHONPATH or run `pip install -e .` |
| CUDA not found? | Run `python -m src.utils.gpu` |
| Data not available? | Run `dvc pull` |
| Airflow DAG not visible? | Check `AIRFLOW_HOME` and `PYTHONPATH` |
| Tests failing? | Run `pytest tests/ -v` for details |

---

**Documentation Updated**: 2026-03-06
**Structure Version**: 1.0.0
