# Migration Guide: Old to New Structure

This document guides you through updating code that relied on the old flat structure to use the new organized package structure.

## Old Structure → New Structure Mapping

```
OLD                              NEW
├── train.py              →      src/model/train.py
├── baseline_eval.py      →      src/model/eval.py
├── diagnose.py           →      src/data/diagnose.py
├── rebuild_splits.py     →      src/data/rebuild_splits.py
├── check_splits.py       →      src/data/check_splits.py
├── check_gpu.py          →      src/utils/gpu.py
├── shelf.py              →      src/utils/shelf.py
└── dags/pipeline.py      →      src/pipeline/dags/pipeline.py
```

## Updating Imports

### Before (Old Structure)
```python
# Direct imports from root
from train import run_training
from baseline_eval import run_evaluation
from diagnose import analyze_dataset
from rebuild_splits import main as rebuild_main
from check_gpu import check_gpu_availability
from shelf import run_from_model
```

### After (New Structure)
```python
# Imports from src package
from src.model import run_training, run_evaluation
from src.data import analyze_dataset, rebuild_splits, check_class_coverage
from src.utils import check_gpu_availability, run_from_model
```

## Common Migration Patterns

### Pattern 1: Training Script
**Before:**
```python
import sys
sys.path.insert(0, '.')
from train import run_training

best_model = run_training(data_yaml="path/to/data.yaml")
```

**After:**
```python
from src.model import run_training

best_model = run_training(data_yaml="path/to/data.yaml")
```

### Pattern 2: Data Analysis
**Before:**
```python
from diagnose import analyze_dataset
from rebuild_splits import main as rebuild_splits

results = analyze_dataset("data.yaml")
if needs_rebuild:
    rebuild_splits(base_dir="/path/to/project")
```

**After:**
```python
from src.data import analyze_dataset, rebuild_splits

results = analyze_dataset("data.yaml")
if needs_rebuild:
    rebuild_splits(base_dir="/path/to/project")
```

### Pattern 3: Utility Functions
**Before:**
```python
from check_gpu import check_gpu_availability
from shelf import run_from_model

if check_gpu_availability():
    run_from_model(model="best.pt", images="test/")
```

**After:**
```python
from src.utils import check_gpu_availability, run_from_model

if check_gpu_availability():
    run_from_model(model="best.pt", images="test/")
```

### Pattern 4: Airflow DAG
**Before:**
```python
from train import run_training
from baseline_eval import run_evaluation
from diagnose import analyze_dataset

@task.python
def train():
    run_training()
```

**After:**
```python
from src.model import run_training, run_evaluation
from src.data import analyze_dataset

@task.python
def train():
    run_training()
```

## Running Code After Migration

### From Project Root
```bash
cd /path/to/project

# Python module execution
python -m src.data.diagnose
python -m src.model.train
python -m src.utils.gpu

# Python scripts
python -c "from src.data import analyze_dataset; analyze_dataset('data.yaml')"
```

### From Any Directory
```bash
export PYTHONPATH="/path/to/project:$PYTHONPATH"
python -c "from src.model import run_training; run_training()"
```

### Airflow
```bash
export PYTHONPATH="/path/to/project:$PYTHONPATH"
airflow dags trigger branch_dag
```

## Installation Approaches

### Approach 1: Editable Install (Recommended for Development)
```bash
cd /path/to/project
pip install -e ".[dev]"
```
Now you can import `src` from anywhere without setting PYTHONPATH.

### Approach 2: PYTHONPATH Environment Variable
```bash
export PYTHONPATH="/path/to/project:$PYTHONPATH"
```
Add to `.bashrc` or `.zshrc` for persistence.

### Approach 3: Docker
Update Dockerfile to set PYTHONPATH:
```dockerfile
ENV PYTHONPATH="/opt/airflow/project:$PYTHONPATH"
WORKDIR /opt/airflow/project
```

## Backward Compatibility

### Legacy Scripts
If you have scripts in the project root that import from modules, they'll break. Options:

1. **Move to new structure** (Recommended)
   ```bash
   # Move script into src/
   mv my_script.py src/scripts/
   # Update imports
   ```

2. **Update root scripts with new imports**
   ```python
   from src.data import analyze_dataset
   from src.model import run_training
   ```

3. **Keep compatibility aliases** (Not recommended)
   ```python
   # In project_root/train.py
   from src.model.train import run_training
   __all__ = ["run_training"]
   ```

## Testing Migration

After migrating, verify imports work:

```bash
# Test individual imports
python -c "from src.model import run_training; print('✓ model imports OK')"
python -c "from src.data import analyze_dataset; print('✓ data imports OK')"
python -c "from src.utils import check_gpu_availability; print('✓ utils imports OK')"

# Run test suite
pytest tests/ -v
```

## Troubleshooting Migration Issues

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:**
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Set it
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Or install editable
pip install -e .
```

### Issue: `ImportError: cannot import name 'run_training'`
**Solution:**
Check the import path. It should be:
```python
from src.model import run_training  # ✓ Correct
# not
from src.model.train import run_training  # ✗ Works but less clean
```

### Issue: Airflow can't find modules
**Solution:**
```bash
# In docker-compose.yml or airflow config:
environment:
  PYTHONPATH: /opt/airflow/project

# Or in airflow.cfg:
[core]
dags_folder = /opt/airflow/project/src/pipeline/dags
```

### Issue: Old scripts still run but with warnings
**Solution:**
The old files still exist in root for backward compatibility but are deprecated. Update imports to `src/` package gradually.

## Timeline for Full Migration

1. **Phase 1** (Now): New structure available, old files still work
2. **Phase 2** (Optional): Deprecation warnings if importing from root
3. **Phase 3** (Future): Remove old files from root

For safety, keep old files until all scripts are migrated.

## Summary Checklist

- [ ] Update imports in main scripts
- [ ] Update DAG imports in `src/pipeline/dags/pipeline.py`
- [ ] Test imports: `python -c "from src.model import run_training"`
- [ ] Set PYTHONPATH or install with `pip install -e .`
- [ ] Run tests: `pytest tests/`
- [ ] Update Dockerfile/docker-compose.yml PYTHONPATH
- [ ] Update Airflow configuration if needed
- [ ] Document new import paths in team wiki/docs

## Questions?

Refer to:
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed structure
- [docs/README.md](docs/README.md) - Quick reference
- [pyproject.toml](pyproject.toml) - Package configuration
