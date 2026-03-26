#!/bin/bash
echo "============================================================"
echo " DeepFake Lab - macOS/Linux Setup"
echo "============================================================"
echo ""

# Check Python 3.13+
PYTHON=$(which python3.13 || which python3 || which python)
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3 not found. Install via: brew install python@3.13"
    exit 1
fi
echo "[OK] Python: $($PYTHON --version)"

# Create venv
echo ""
echo "[1/4] Creating virtual environment..."
$PYTHON -m venv .venv

echo "[2/4] Activating..."
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install numpy opencv-python onnx onnxruntime insightface psutil \
            customtkinter pillow protobuf typing-extensions \
            cv2_enumerate_cameras

echo "[4/4] Checking model..."
if [ ! -f "models/inswapper_128_fp16.onnx" ]; then
    echo "[!] Downloading face swap model (~265MB)..."
    curl -L "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true" \
         -o models/inswapper_128_fp16.onnx
else
    echo "[OK] Model found"
fi

echo ""
echo "============================================================"
echo " Setup complete! Run: ./launch.sh"
echo ""
echo " NOTE (macOS): Grant camera access to Terminal:"
echo "   System Settings → Privacy & Security → Camera → Terminal ✓"
echo "============================================================"
