#!/bin/bash
# Reinstall numpy so it's linked against Apple's Accelerate framework.
# This is the #1 cause of "ML is slow on Apple Silicon" — the default pip
# wheel sometimes ships with a generic BLAS that's 10-50x slower.

set -e

cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv missing. Run fix_setup.sh first."
    exit 1
fi
source .venv/bin/activate

echo "=========================================="
echo " Reinstalling numpy with Apple Accelerate"
echo "=========================================="
echo ""
echo "[1/3] Removing existing numpy..."
pip uninstall -y numpy

echo ""
echo "[2/3] Reinstalling numpy 1.26.4 from a fresh wheel..."
# 1.26.4 ARM64 wheel ships with Accelerate by default; force-reinstall and
# clear cache to make sure we get the official wheel, not a stale local copy.
pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

echo ""
echo "[3/3] Verifying Accelerate linkage..."
python <<'PY'
import numpy as np
print("numpy version:", np.__version__)
cfg = np.show_config(mode='dicts')

found_accelerate = False
for section_name, section in cfg.items():
    text = str(section).lower()
    if "accelerate" in text or "veclib" in text:
        print(f"  FOUND Accelerate in section: {section_name}")
        found_accelerate = True

if found_accelerate:
    print("\n  SUCCESS: numpy is linked to Apple Accelerate.")
    print("  Expect a 10x+ speedup in the live demo.")
else:
    print("\n  WARNING: Accelerate not detected in this wheel.")
    print("  Falling back to compile-from-source...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--no-cache-dir", "--no-binary", "numpy",
                           "numpy==1.26.4"])
    print("  Done. Re-run this script to verify.")
PY

echo ""
echo "=========================================="
echo " Now relaunch the app:"
echo "   ./launch.sh"
echo "=========================================="
