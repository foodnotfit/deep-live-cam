@echo off
echo ============================================================
echo  DeepFake Lab
echo ============================================================
echo.

:: Check setup was run
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Please run setup_windows.bat first!
    echo.
    pause
    exit /b 1
)

:: Check model exists
if not exist "models\inswapper_128_fp16.onnx" (
    if not exist "models\inswapper_128.onnx" (
        echo [ERROR] Face swap model not found in models\ folder.
        echo         Please run setup_windows.bat first!
        echo.
        pause
        exit /b 1
    )
)

:: Activate venv and launch
call .venv\Scripts\activate.bat
echo Starting DeepFake Lab...
echo.
python run.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed with error code %errorlevel%.
    echo         If this is your first run, try setup_windows.bat again.
    echo.
    pause
)
