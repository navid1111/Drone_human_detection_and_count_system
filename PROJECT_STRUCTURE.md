# Project Structure

This project is organized with a proper Python package structure to ensure maintainability, scalability, and easy testing.

## Directory Organization

```
project/
├── src/                       # Main source code package
│   ├── __init__.py
│   ├── model/                 # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py           # Training logic
│   │   └── eval.py            # Evaluation/validation logic
│   ├── data/                  # Data processing & analysis
│   │   ├── __init__.py
│   │   ├── diagnose.py        # Dataset quality analysis
│   │   ├── rebuild_splits.py  # Stratified split creation
│   │   └── check_splits.py    # Split validation
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── gpu.py             # GPU utilities
│   │   └── shelf.py           # Share-of-shelf analytics
│   └── pipeline/              # Orchestration & workflows
│       ├── __init__.py
│       └── dags/              # Airflow DAGs
│           ├── __init__.py
│           └── pipeline.py    # Main DAG definition
│
├── dataset/                   # Data directory (DVC tracked)
│   ├── dataset/               # Raw dataset
│   └── dataset_stratified/    # Stratified splits (generated)
│
├── runs/                      # Training outputs
│   ├── train/                 # Model checkpoints
│   └── detect/                # Prediction outputs
│
├── models/                    # Pre-trained & saved models
├── logs/                      # Airflow logs & pipeline logs
├── tests/                     # Unit tests
├── docs/                      # Documentation
│
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Modern Python packaging
├── Dockerfile                 # Container setup
├── docker-compose.yml         # Multi-container orchestration
├── .dvc/                      # DVC configuration
├── .git/                      # Git repository
├── .gitignore                 # Git ignore rules
├── README.md                  # Project overview
└── setup.py                   # Legacy packaging (optional)
```

## Module Organization

### `src/model/`
Handles all machine learning operations:
- **train.py**: `run_training()` - YOLO training with W&B integration
- **eval.py**: `run_evaluation()` - Model validation and metrics

### `src/data/`
Data processing and validation:
- **diagnose.py**: `analyze_dataset()` - Class distribution, imbalance detection
- **rebuild_splits.py**: `main()` - Stratified train/val/test sampling
- **check_splits.py**: `check_class_coverage()` - Validates split quality

### `src/utils/`
Supporting utilities:
- **gpu.py**: `check_gpu_availability()` - CUDA/GPU detection
- **shelf.py**: `run_from_model()` - Share-of-shelf analytics

### `src/pipeline/`
ML workflow orchestration:
- **dags/pipeline.py**: Airflow DAG with branching logic

## Importing from Modules

### Within the project
```python
from src.model import run_training, run_evaluation
from src.data import analyze_dataset, rebuild_splits
from src.utils import check_gpu_availability, run_from_model
```

### From scripts in project root
```python
import sys
sys.path.insert(0, '.')
from src.data.diagnose import analyze_dataset
```

### From Airflow (automatic)
When running via Airflow with correct PYTHONPATH, imports work directly.

## Configuration Files

### `requirements.txt`
Lists all Python dependencies:
```
ultralytics>=8.0
torch
torchvision
wandb
pyyaml
pydantic
```

### `pyproject.toml` (Modern Python packaging)
```toml
[project]
name = "retail-object-detection"
version = "1.0.0"
requires-python = ">=3.9"
```

### `.gitignore`
Ensures git ignores:
- `__pycache__/`
- `*.pyc`, `*.pyo`
- `runs/`
- `dataset/dataset/`
- `.venv/`, `*.egg-info/`
- `.dvc/cache/`

## Running Code

### Training
```bash
cd project
python -m src.model.train
# or
from src.model import run_training
run_training()
```

### Data Diagnostics
```bash
cd project
python -m src.data.diagnose
# or
from src.data import analyze_dataset
results = analyze_dataset("dataset/dataset/data.yaml")
```

### Rebuilding Splits
```bash
cd project
python -m src.data.rebuild_splits
# or
from src.data import rebuild_splits
rebuild_splits()
```

### Airflow Pipeline
```bash
airflow dags list
airflow dags trigger branch_dag
```

## Benefits of This Structure

✅ **Clear separation of concerns** - Each module has a well-defined purpose  
✅ **Easy testing** - Isolated modules are easier to unit test  
✅ **Scalability** - Easy to add new modules or features  
✅ **Package management** - Can be installed as `pip install -e .`  
✅ **Import clarity** - No ambiguous relative imports  
✅ **IDE support** - Better autocomplete and type hints  
✅ **DVC compatibility** - Works seamlessly with DVC for data versioning  
✅ **Airflow integration** - Clean DAG definitions with proper imports  

## Migration Notes

Old file locations → New locations:
- `train.py` → `src/model/train.py`
- `baseline_eval.py` → `src/model/eval.py`
- `diagnose.py` → `src/data/diagnose.py`
- `rebuild_splits.py` → `src/data/rebuild_splits.py`
- `check_splits.py` → `src/data/check_splits.py`
- `check_gpu.py` → `src/utils/gpu.py`
- `shelf.py` → `src/utils/shelf.py`
- `dags/pipeline.py` → `src/pipeline/dags/pipeline.py`

Old scripts in project root are kept for backward compatibility but should use imports from `src/` package.
