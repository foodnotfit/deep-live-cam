#!/bin/bash
# Family Day launcher for Deep Live Cam on Apple Silicon.
# Activates the venv, sanity-checks the models, and launches the upstream UI
# with CoreML (Neural Engine) acceleration and 4 execution threads.

cd "$(dirname "$0")" || exit 1

# Pretty banner
echo ""
echo "============================================================"
echo "  Deep Live Cam - Family Day Launcher"
echo "============================================================"

# --- Pre-flight: virtual environment ------------------------------
if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo "[ERROR] Virtual environment not found at .venv/"
    echo "        Run setup first:  bash fix_setup.sh"
    exit 1
fi
source .venv/bin/activate

# --- Pre-flight: inswapper face-swap model ------------------------
if [ ! -f "models/inswapper_128_fp16.onnx" ]; then
    echo ""
    echo "[ERROR] Face swap model missing at models/inswapper_128_fp16.onnx"
    echo "        Re-run setup:  bash fix_setup.sh"
    exit 1
fi

# --- Pre-flight: buffalo_l face detection pack --------------------
if [ ! -f "$HOME/.insightface/models/buffalo_l/det_10g.onnx" ]; then
    echo ""
    echo "[ERROR] InsightFace buffalo_l models not installed."
    echo "        Run:  bash fix_buffalo.sh"
    exit 1
fi

echo "  Models:  OK"
echo "  Mode:    PyTorch + MPS (Apple GPU) for the swap, CPU for detection"
echo ""
echo "  After exhausting onnxruntime+CoreML and onnxruntime+CPU paths, we"
echo "  bypass onnxruntime entirely for the inswapper and run it on Apple's"
echo "  GPU via PyTorch's MPS backend. ~79 ms per swap (vs 654 ms on CPU)."
echo "  Expected end-to-end: 6-10 FPS on M3 Pro."
echo ""
echo "  In the UI: click 'Select a face' -> pick from faces/"
echo "             then click 'Live' to start the webcam swap."
echo ""
echo "============================================================"
echo ""

# --- Launch --------------------------------------------------------
exec python run.py --execution-provider cpu --execution-threads 4
