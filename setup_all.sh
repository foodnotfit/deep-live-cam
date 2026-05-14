#!/bin/bash
# One-shot setup for a fresh Mac. Idempotent — safe to re-run; each step
# checks whether it's already done before doing it.
#
# This is the master script that calls all the smaller fix scripts in
# order, plus installs the PyTorch+MPS deps and downloads the faces.
#
# Usage:
#   cd ~/Desktop/deep-live-cam-main
#   bash setup_all.sh
#
# Takes ~15-20 minutes start to finish (most of that is package downloads).

set -e
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo " Deep Live Cam - Full Setup (Family Day demo)"
echo "============================================================"
echo ""

# ----------------------------------------------------------------------
# Step 1: Homebrew + Python 3.11 + Tcl/Tk + ffmpeg
# ----------------------------------------------------------------------
echo "[1/7] Checking Homebrew and base dependencies..."

if ! command -v brew >/dev/null 2>&1; then
    echo "  Homebrew not found. Install it first from https://brew.sh"
    echo "  Then re-run this script."
    exit 1
fi
echo "  Homebrew: OK ($(brew --version | head -1))"

# Install missing brew packages quietly
NEEDED_PKGS=()
for pkg in "python@3.11" "python-tk@3.11" "ffmpeg"; do
    if ! brew list "$pkg" >/dev/null 2>&1; then
        NEEDED_PKGS+=("$pkg")
    fi
done

if [ ${#NEEDED_PKGS[@]} -gt 0 ]; then
    echo "  Installing: ${NEEDED_PKGS[*]}"
    brew install "${NEEDED_PKGS[@]}"
else
    echo "  python@3.11, python-tk@3.11, ffmpeg: all already installed"
fi

# Find python3.11
PY311="/opt/homebrew/opt/python@3.11/bin/python3.11"
if [ ! -x "$PY311" ]; then
    PY311="$(which python3.11)"
fi
if [ ! -x "$PY311" ]; then
    echo "  ERROR: python3.11 not found after brew install."
    exit 1
fi
echo "  Using: $PY311 ($("$PY311" --version))"

# ----------------------------------------------------------------------
# Step 2: Python venv with base packages
# ----------------------------------------------------------------------
echo ""
echo "[2/7] Setting up Python virtual environment..."

if [ -d ".venv" ] && [ -f ".venv/bin/activate" ] && [ -f ".venv/bin/python" ]; then
    echo "  .venv already exists, skipping creation"
else
    rm -rf .venv .venv312
    "$PY311" -m venv .venv
    echo "  Created .venv with $PY311"
fi
source .venv/bin/activate

# Verify Tk works
python -c "import tkinter; print(f'  Tk version: {tkinter.TkVersion} (OK)')"

# ----------------------------------------------------------------------
# Step 3: Base pip packages (the parent project's runtime deps)
# ----------------------------------------------------------------------
echo ""
echo "[3/7] Installing base Python packages..."

pip install --upgrade pip wheel --quiet

# Install only what's missing
REQUIRED_PIP_PACKAGES=(
    "typing-extensions>=4.8.0"
    "opencv-python==4.10.0.84"
    "onnx==1.18.0"
    "onnxruntime"
    "insightface==0.7.3"
    "psutil==5.9.8"
    "customtkinter==5.2.2"
    "pillow"
    "protobuf==4.25.1"
    "cv2_enumerate_cameras"
    "pyobjc-framework-AVFoundation"
)
pip install --quiet "${REQUIRED_PIP_PACKAGES[@]}"
echo "  Base packages installed"

# ----------------------------------------------------------------------
# Step 4: InsightFace buffalo_l face detection models (~280 MB)
# ----------------------------------------------------------------------
echo ""
echo "[4/7] Installing InsightFace buffalo_l models..."

BUFFALO_DIR="$HOME/.insightface/models/buffalo_l"
if [ -f "$BUFFALO_DIR/det_10g.onnx" ] && [ -f "$BUFFALO_DIR/w600k_r50.onnx" ]; then
    echo "  buffalo_l models already installed at $BUFFALO_DIR"
else
    bash fix_buffalo.sh
fi

# ----------------------------------------------------------------------
# Step 5: Inswapper face-swap model (~265 MB)
# ----------------------------------------------------------------------
echo ""
echo "[5/7] Installing inswapper face-swap model..."

mkdir -p models
INSWAPPER="models/inswapper_128_fp16.onnx"
if [ -f "$INSWAPPER" ] && [ "$(stat -f%z "$INSWAPPER" 2>/dev/null || echo 0)" -gt 100000000 ]; then
    echo "  inswapper model already installed ($(du -h "$INSWAPPER" | cut -f1))"
else
    echo "  Downloading inswapper_128_fp16.onnx (~265 MB)..."
    curl -L -o "$INSWAPPER" \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    SIZE=$(stat -f%z "$INSWAPPER" 2>/dev/null || echo 0)
    if [ "$SIZE" -lt 100000000 ]; then
        echo "  WARNING: downloaded file is too small ($SIZE bytes)."
        echo "  Hugging Face may have gated the URL. Download manually and place at:"
        echo "    $INSWAPPER"
        exit 1
    fi
    echo "  Downloaded ($(du -h "$INSWAPPER" | cut -f1))"
fi

# ----------------------------------------------------------------------
# Step 6: PyTorch + onnx2torch (the MPS rewrite's deps)
# ----------------------------------------------------------------------
echo ""
echo "[6/7] Installing PyTorch + onnx2torch (MPS backend)..."

# Check if torch is already there at the right version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [ "$TORCH_VERSION" = "2.7.0" ]; then
    echo "  torch 2.7.0 already installed"
else
    pip install --quiet "torch==2.7.0" "torchvision==0.22.0" onnx2torch
    echo "  torch 2.7.0 installed"
fi

# Verify MPS is available
python -c "
import torch
if torch.backends.mps.is_available():
    print(f'  MPS available: YES (PyTorch {torch.__version__})')
else:
    raise SystemExit('  MPS available: NO — this Mac does not support GPU acceleration')
"

# ----------------------------------------------------------------------
# Step 7: numpy linked to Apple Accelerate (critical for speed)
# ----------------------------------------------------------------------
echo ""
echo "[7/7] Verifying numpy is linked to Apple Accelerate..."

if python -c "
import numpy as np
cfg = str(np.show_config(mode='dicts')).lower()
import sys
sys.exit(0 if ('accelerate' in cfg or 'veclib' in cfg) else 1)
"; then
    echo "  numpy + Accelerate: OK"
else
    echo "  numpy NOT linked to Accelerate. Rebuilding from source..."
    bash fix_numpy.sh
fi

# ----------------------------------------------------------------------
# Final smoke test
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Setup complete. Running final smoke test..."
echo "============================================================"

python -c "
import torch, numpy, cv2, insightface, onnx2torch
print(f'  PyTorch:    {torch.__version__}  MPS={torch.backends.mps.is_available()}')
print(f'  numpy:      {numpy.__version__}')
print(f'  OpenCV:     {cv2.__version__}')
print(f'  insightface: {insightface.__version__}')
print(f'  onnx2torch: {onnx2torch.__version__}')
"

echo ""
echo "============================================================"
echo " All systems go!"
echo ""
echo " Next steps:"
echo "   1. Grant camera permission:"
echo "      System Settings -> Privacy & Security -> Camera -> Terminal"
echo "      (Quit Terminal entirely and reopen after enabling)"
echo ""
echo "   2. (Optional) Download celebrity face images:"
echo "      python download_faces.py"
echo ""
echo "   3. Launch the demo:"
echo "      ./launch.sh"
echo ""
echo "============================================================"
