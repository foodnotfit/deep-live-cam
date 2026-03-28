@echo off
echo ============================================================
echo  DeepFake Lab - Windows Setup
echo  Supports: CPU, NVIDIA (CUDA), AMD (DirectML)
echo ============================================================
echo.

:: Check Python 3.10-3.12 (3.13 not supported by onnxruntime on Windows yet)
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo         Download Python 3.11 from https://python.org
    echo         IMPORTANT: Check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found

:: Check Python version is 3.10-3.12
python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Python 3.13+ detected. onnxruntime requires Python 3.10-3.12.
    echo           Download Python 3.11 from https://python.org
    pause
    exit /b 1
)

:: Create virtual environment
echo.
echo [1/5] Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate
echo [2/5] Activating environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip --quiet

:: Detect GPU
echo [3/5] Detecting GPU...
set GPU_PROVIDER=cpu
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected - will install CUDA support
    set GPU_PROVIDER=cuda
    goto :install_deps
)

:: Check for AMD/Intel DirectML
python -c "import subprocess; r=subprocess.run(['wmic','path','win32_VideoController','get','name'],capture_output=True,text=True); print(r.stdout)" 2>nul | findstr /i "AMD Radeon" >nul
if %errorlevel% equ 0 (
    echo [OK] AMD GPU detected - will install DirectML support
    set GPU_PROVIDER=directml
)

:install_deps
echo.
echo [4/5] Installing dependencies (this may take several minutes)...

:: Base dependencies always needed
pip install --quiet numpy pillow customtkinter psutil requests ^
    protobuf typing-extensions tqdm

:: OpenCV
pip install --quiet opencv-python

:: onnx base
pip install --quiet onnx

:: onnxruntime - GPU-specific
if "%GPU_PROVIDER%"=="cuda" (
    echo     Installing CUDA-accelerated onnxruntime...
    pip install --quiet onnxruntime-gpu
) else if "%GPU_PROVIDER%"=="directml" (
    echo     Installing DirectML-accelerated onnxruntime...
    pip install --quiet onnxruntime-directml
) else (
    echo     Installing CPU onnxruntime...
    pip install --quiet onnxruntime
)

:: insightface (face detection + analysis)
pip install --quiet insightface

:: Windows camera enumeration
pip install --quiet opencv-contrib-python

echo.
echo [5/5] Checking model files...
if not exist "models" mkdir models

if not exist "models\inswapper_128_fp16.onnx" (
    if not exist "models\inswapper_128.onnx" (
        echo [!] Face swap model not found. Downloading (~265MB)...
        python -c "import urllib.request, os, ssl; ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE; print('Downloading inswapper_128_fp16.onnx...'); urllib.request.urlretrieve('https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx', 'models\\inswapper_128_fp16.onnx'); print('Done!')"
        if %errorlevel% neq 0 (
            echo [ERROR] Download failed. Please manually download from:
            echo         https://huggingface.co/hacksider/deep-live-cam
            echo         and place in the models\ folder.
        )
    ) else (
        echo [OK] inswapper_128.onnx found
    )
) else (
    echo [OK] inswapper_128_fp16.onnx found
)

echo.
echo ============================================================
echo  Setup complete!
echo.
if "%GPU_PROVIDER%"=="cuda" (
    echo  GPU: NVIDIA CUDA - Run launch.bat or run-cuda.bat
) else if "%GPU_PROVIDER%"=="directml" (
    echo  GPU: AMD DirectML - Run launch.bat or run-directml.bat
) else (
    echo  GPU: CPU only - Run launch.bat
    echo  NOTE: For faster performance, install NVIDIA or AMD GPU drivers.
)
echo.
echo  To start: Double-click launch.bat
echo ============================================================
pause
