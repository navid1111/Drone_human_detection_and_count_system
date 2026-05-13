@echo off
REM Quick Setup Script for Windows

echo.
echo ========================================================================
echo    Retail Object Detection - Setup and Verification (Windows)
echo ========================================================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    exit /b 1
)
echo.

REM Check if venv exists
echo [2/5] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo Created .venv
) else (
    echo .venv already exists
)
echo.

REM Activate venv
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Install dependencies
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
echo Successfully installed from requirements.txt
echo.

REM Install in editable mode
echo [5/5] Installing package in editable mode...
python -m pip install -e ".[dev]"
echo Successfully installed in editable mode
echo.

REM Verify imports
echo [6/6] Verifying imports...
python -c "from src.model import run_training; print('✓ src.model imports OK')"
if errorlevel 1 goto error

python -c "from src.data import analyze_dataset; print('✓ src.data imports OK')"
if errorlevel 1 goto error

python -c "from src.utils import check_gpu_availability; print('✓ src.utils imports OK')"
if errorlevel 1 goto error

python -c "from src.pipeline.dags import pipeline; print('✓ src.pipeline imports OK')"
if errorlevel 1 goto error

echo.
echo ========================================================================
echo ✅ Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Check data quality:
echo      python -m src.data.diagnose
echo.
echo   2. Run training:
echo      python -m src.model.train
echo.
echo   3. Start Airflow pipeline:
echo      airflow dags trigger branch_dag
echo.
echo 📖 Documentation:
echo    - PROJECT_STRUCTURE.md        ^<- Detailed structure
echo    - MIGRATION.md                ^<- Import migration
echo    - ARCHITECTURE.md             ^<- System design
echo    - docs\README.md              ^<- Quick reference
echo.
echo For GPU check:
echo   python -m src.utils.gpu
echo.
echo ========================================================================
echo.
goto end

:error
echo.
echo ❌ Error during import verification!
echo Please check the error messages above.
exit /b 1

:end
pause
