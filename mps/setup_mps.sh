#!/bin/bash
# One-time setup for the PyTorch + MPS rewrite of deep-live-cam.
# Creates a fresh .venv with Python 3.11 and installs the minimum
# dependencies needed to run Phase 1 (the smoke test).

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " Deep Live Cam (MPS rewrite) - Phase 1 Setup"
echo "============================================================"

# --- Find python3.11 -------------------------------------------------
PY311="/opt/homebrew/opt/python@3.11/bin/python3.11"
if [ ! -x "$PY311" ]; then
    PY311="$(which python3.11)"
fi
if [ ! -x "$PY311" ]; then
    echo "ERROR: python3.11 not found. Install via:"
    echo "    brew install python@3.11 python-tk@3.11"
    exit 1
fi
echo "Using: $PY311 ($("$PY311" --version))"

# --- Fresh venv ------------------------------------------------------
echo ""
echo "[1/3] Creating .venv with Python 3.11..."
rm -rf .venv
"$PY311" -m venv .venv
source .venv/bin/activate

# --- Install dependencies --------------------------------------------
# numpy compiled from source so it links against Apple Accelerate
# (the silent BLAS bug we hit on the onnxruntime build).
echo ""
echo "[2/3] Installing dependencies..."
pip install --upgrade pip wheel
pip install --no-binary numpy "numpy>=1.26,<2"
pip install \
    torch \
    torchvision \
    onnx \
    onnx2torch \
    onnxruntime \
    opencv-python \
    pillow

# --- Verify Apple Accelerate + MPS ----------------------------------
echo ""
echo "[3/3] Verifying numpy/Accelerate and PyTorch/MPS..."
python <<'PY'
import sys

# numpy + Accelerate
import numpy as np
cfg = str(np.show_config(mode='dicts')).lower()
if "accelerate" in cfg or "veclib" in cfg:
    print("  numpy + Accelerate:  OK")
else:
    print("  numpy + Accelerate:  MISSING (rebuild needed)")
    sys.exit(1)

# PyTorch + MPS
import torch
print(f"  PyTorch version:     {torch.__version__}")
if torch.backends.mps.is_available():
    print("  MPS available:       YES")
    if torch.backends.mps.is_built():
        print("  MPS built into torch: YES")
    # Quick MPS sanity test
    try:
        x = torch.randn(4, 4, device="mps")
        y = x @ x.T
        _ = y.cpu().numpy()
        print("  MPS sanity test:     PASSED")
    except Exception as e:
        print(f"  MPS sanity test:     FAILED ({e})")
        sys.exit(1)
else:
    print("  MPS available:       NO -- this Mac does not support MPS")
    sys.exit(1)

# onnx2torch
import onnx2torch
print(f"  onnx2torch version:  {onnx2torch.__version__}")
PY

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next, run the Phase 1 smoke test:"
echo "    source .venv/bin/activate"
echo "    python phase1_smoke.py"
echo "============================================================"
