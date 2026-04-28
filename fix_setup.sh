#!/bin/bash
# Family Day setup fix for deep-live-cam on Apple Silicon
# Rebuilds .venv on Python 3.11 with corrected dependencies.

set -e  # exit on first error

echo "=========================================="
echo " DeepFake Lab - Family Day Setup Fix"
echo "=========================================="

# --- 1. Homebrew dependencies ---------------------------------------------
echo ""
echo "[1/7] Installing Python 3.11, Tcl/Tk, ffmpeg via Homebrew..."
brew install python@3.11 python-tk@3.11 ffmpeg

# --- 2. Find python3.11 ---------------------------------------------------
PY311="/opt/homebrew/opt/python@3.11/bin/python3.11"
if [ ! -x "$PY311" ]; then
    PY311="$(which python3.11)"
fi
if [ ! -x "$PY311" ]; then
    echo "ERROR: python3.11 not found after brew install."
    exit 1
fi
echo "Using: $PY311"
"$PY311" --version

# --- 3. Wipe old venv -----------------------------------------------------
echo ""
echo "[2/7] Removing any old virtual environment..."
cd "$(dirname "$0")"
rm -rf .venv .venv312

# --- 4. Create fresh venv -------------------------------------------------
echo ""
echo "[3/7] Creating fresh .venv with Python 3.11..."
"$PY311" -m venv .venv
source .venv/bin/activate

# --- 5. Sanity check Tk ---------------------------------------------------
echo ""
echo "[4/7] Verifying tkinter works..."
python -c "import tkinter; print('Tk OK, version:', tkinter.TkVersion)"

# --- 6. Install dependencies ----------------------------------------------
echo ""
echo "[5/7] Installing Python dependencies (this takes a few minutes)..."
pip install --upgrade pip wheel
pip install \
    "numpy>=1.23.5,<2" \
    "typing-extensions>=4.8.0" \
    "opencv-python==4.10.0.84" \
    "onnx==1.18.0" \
    "onnxruntime" \
    "insightface==0.7.3" \
    "psutil==5.9.8" \
    "customtkinter==5.2.2" \
    "pillow" \
    "protobuf==4.25.1" \
    "cv2_enumerate_cameras" \
    "pyobjc-framework-AVFoundation"

# --- 7. Download model ----------------------------------------------------
echo ""
echo "[6/7] Checking for face-swap model..."
mkdir -p models
if [ ! -f "models/inswapper_128_fp16.onnx" ]; then
    echo "Downloading inswapper_128_fp16.onnx (~265 MB)..."
    curl -L -o models/inswapper_128_fp16.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
else
    echo "Model already present."
fi

# --- 8. Verify model size -------------------------------------------------
echo ""
echo "[7/7] Verifying model file..."
MODEL_SIZE=$(stat -f%z models/inswapper_128_fp16.onnx 2>/dev/null || echo 0)
echo "Model size: $MODEL_SIZE bytes"
if [ "$MODEL_SIZE" -lt 100000000 ]; then
    echo ""
    echo "WARNING: Model file is smaller than 100 MB."
    echo "Hugging Face may have gated the download."
    echo "Tell Claude the size and we will use an alternate source."
else
    echo "Model size looks correct (should be ~265 MB)."
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " To launch the app, run:"
echo "   ./launch.sh"
echo ""
echo " On first launch, macOS will ask for camera"
echo " permission. Click OK. If you miss it, go to"
echo " System Settings -> Privacy & Security -> Camera"
echo " and enable Terminal."
echo "=========================================="
