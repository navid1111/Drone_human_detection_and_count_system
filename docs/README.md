# Project Overview

This is a comprehensive **Retail Object Detection** platform built on:
- **YOLO 11** for object detection
- **DVC** for data versioning
- **Airflow** for ML pipeline orchestration
- **W&B** for experiment tracking
- **Stratified Data Splitting** to ensure all classes are represented in train/val/test

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo>
cd retail-object-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e ".[dev]"
```

### Key Workflows

#### 1. Analyze Dataset Quality
```bash
python -m src.data.diagnose
```
Outputs:
- Class distribution per split
- Imbalance ratios
- Missing classes across splits

#### 2. Rebuild Stratified Splits
```bash
python -m src.data.rebuild_splits
```
Ensures:
- All classes appear in all splits (train/val/test)
- Proper distribution ratios (80/10/10)
- Reproducible splits

#### 3. Train Model
```bash
python -m src.model.train
```
- Uses YOLO 11m base model
- Logs to Weights & Biases
- Saves to `runs/train/`

#### 4. Evaluate Model
```bash
python -m src.model.eval
```
Returns:
- Precision, Recall, mAP50, mAP50-95

#### 5. Run Airflow Pipeline
```bash
airflow dags trigger branch_dag
```
Executes full workflow with data quality checks and conditional branching.

## Data Structure

### DVC-Tracked Data
```
dataset/
├── dataset/              # Raw (Roboflow export)
│   ├── train/
│   ├── valid/
│   └── test/
└── dataset_stratified/   # Generated (stratified splits)
    ├── train/
    ├── valid/
    └── test/
```

### Training Artifacts
```
runs/
├── train/                # Model checkpoints
│   └── pipeline_run/
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── detect/               # Inference results
```

## Module Reference

### `src.data` - Data Processing
| Function | Purpose |
|----------|---------|
| `analyze_dataset()` | Check class balance & coverage |
| `rebuild_splits()` | Create stratified train/val/test |
| `check_class_coverage()` | Validate split quality |

### `src.model` - Training & Evaluation
| Function | Purpose |
|----------|---------|
| `run_training()` | Train YOLO with W&B logging |
| `run_evaluation()` | Validate model & return metrics |

### `src.utils` - Utilities
| Function | Purpose |
|----------|---------|
| `check_gpu_availability()` | Check CUDA/GPU status |
| `run_from_model()` | Inference + share-of-shelf analytics |

### `src.pipeline.dags` - Orchestration
| DAG | Purpose |
|-----|---------|
| `branch_dag` | Full ML pipeline with conditional branching |

## Configuration

### Environment Variables
```bash
PYTHONPATH=/path/to/project
WANDB_API_KEY=your_key_here
CUDA_VISIBLE_DEVICES=0
```

### Model Parameters (in `src/model/train.py`)
- **Model**: `yolo11m.pt` (medium)
- **Epochs**: 100
- **Batch Size**: 8
- **Image Size**: 640x640
- **Device**: GPU (CUDA:0)

### Dataset Configuration (in `src/data/rebuild_splits.py`)
- **Train Split**: 80%
- **Val Split**: 10%
- **Test Split**: 10%
- **Random Seed**: 42 (reproducibility)

## Development

### Running Tests
```bash
pytest tests/
pytest tests/ --cov=src
```

### Code Style
```bash
black src/
flake8 src/
mypy src/
```

### Adding New Features
1. Create module in appropriate `src/` subdirectory
2. Add `__init__.py` with imports
3. Update documentation
4. Write tests in `tests/`

## Troubleshooting

### Import Errors
Ensure `PYTHONPATH` includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

### GPU Not Available
```bash
python -m src.utils.gpu
# Check output for CUDA availability
```

### Dataset Not Found
Ensure DVC is initialized:
```bash
dvc pull
```

### Airflow DAG Not Visible
Check DAG file path and `AIRFLOW_HOME`:
```bash
export AIRFLOW_HOME=/path/to/project
airflow dags list
```

## Performance Metrics

### Model Performance (mAP on 2003 images)
- **Train Set**: 1604 images, 70 classes
- **Val Set**: 200 images, 70 classes (all classes covered)
- **Test Set**: 199 images, 70 classes (all classes covered)

### Data Quality
- **Class Imbalance**: ~10.5x (highest to lowest class)
- **Missing Classes**: 0 (perfect stratification)

## License

MIT

## References

- [YOLO 11 Documentation](https://github.com/ultralytics/ultralytics)
- [DVC Documentation](https://dvc.org)
- [Apache Airflow](https://airflow.apache.org)
- [Weights & Biases](https://wandb.ai)
