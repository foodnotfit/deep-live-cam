@echo off
echo Starting DeepFake Lab...
call .venv\Scripts\activate.bat
python launcher.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed. Make sure you ran setup_windows.bat first.
    pause
)
