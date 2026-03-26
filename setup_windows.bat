@echo off
echo ============================================================
echo  DeepFake Lab - Windows Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Download from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found
python --version

:: Create virtual environment
echo.
echo [1/4] Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)

:: Activate and install deps
echo [2/4] Activating environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install numpy opencv-python onnx onnxruntime insightface psutil ^
            customtkinter pillow protobuf typing-extensions ^
            cv2_enumerate_cameras

echo [4/4] Checking model files...
if not exist "models\inswapper_128_fp16.onnx" (
    echo.
    echo [!] Face swap model not found. Downloading now (~265MB)...
    python -c "import urllib.request; print('Downloading...'); urllib.request.urlretrieve('https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true', 'models/inswapper_128_fp16.onnx'); print('Done!')"
) else (
    echo [OK] Model found
)

echo.
echo ============================================================
echo  Setup complete! Run launch.bat to start the app.
echo ============================================================
pause
