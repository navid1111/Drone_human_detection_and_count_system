#!/usr/bin/env bash
# Quick Setup Script - Execute this after project reorganization

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "   Retail Object Detection - Setup & Verification"
echo "════════════════════════════════════════════════════════════════"

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1)
echo "✓ $python_version"

# Create/activate virtual environment
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "✓ Created .venv"
else
    echo "✓ .venv already exists"
fi

# Activate venv (note: for scripts, may need to do this separately)
# source .venv/bin/activate  # On Unix
# .venv\Scripts\activate     # On Windows

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ Installed from requirements.txt"

# Install in editable mode
echo ""
echo "[4/5] Installing package in editable mode..."
pip install -e ".[dev]"
echo "✓ Installed in editable mode (pip install -e .)"

# Verify imports
echo ""
echo "[5/5] Verifying imports..."
python -c "from src.model import run_training; print('✓ src.model imports OK')"
python -c "from src.data import analyze_dataset; print('✓ src.data imports OK')"
python -c "from src.utils import check_gpu_availability; print('✓ src.utils imports OK')"
python -c "from src.pipeline.dags import pipeline; print('✓ src.pipeline imports OK')"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "Next steps:"
echo "  1. Check data quality:"
echo "     python -m src.data.diagnose"
echo ""
echo "  2. Run training:"
echo "     python -m src.model.train"
echo ""
echo "  3. Start Airflow pipeline:"
echo "     airflow dags trigger branch_dag"
echo ""
echo "📖 Documentation:"
echo "  - PROJECT_STRUCTURE.md        ← Detailed structure"
echo "  - MIGRATION.md                ← Import migration"
echo "  - ARCHITECTURE.md             ← System design"
echo "  - docs/README.md              ← Quick reference"
echo ""
echo "For GPU check:"
echo "  python -m src.utils.gpu"
echo ""
echo "════════════════════════════════════════════════════════════════"
