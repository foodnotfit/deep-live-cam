#!/bin/bash
# Manually download InsightFace's buffalo_l face-detection model pack
# from the official deepinsight/insightface GitHub releases.

set -e

echo "=========================================="
echo " Installing buffalo_l face detection pack"
echo "=========================================="

DEST_DIR="$HOME/.insightface/models"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [ -d "buffalo_l" ] && [ -f "buffalo_l/det_10g.onnx" ]; then
    echo "buffalo_l already installed at $DEST_DIR/buffalo_l"
    ls -lh buffalo_l/
    exit 0
fi

echo ""
echo "[1/3] Downloading buffalo_l.zip (~281 MB)..."
echo "      Source: github.com/deepinsight/insightface releases"
curl -L -o buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

ZIP_SIZE=$(stat -f%z buffalo_l.zip 2>/dev/null || echo 0)
echo "Downloaded $ZIP_SIZE bytes."
if [ "$ZIP_SIZE" -lt 100000000 ]; then
    echo "ERROR: download is too small. Network or URL issue."
    rm -f buffalo_l.zip
    exit 1
fi

echo ""
echo "[2/3] Extracting..."
mkdir -p buffalo_l
unzip -o buffalo_l.zip -d buffalo_l
rm -f buffalo_l.zip

echo ""
echo "[3/3] Verifying contents..."
ls -lh buffalo_l/
EXPECTED=("det_10g.onnx" "w600k_r50.onnx" "2d106det.onnx" "genderage.onnx" "1k3d68.onnx")
ALL_OK=true
for f in "${EXPECTED[@]}"; do
    if [ ! -f "buffalo_l/$f" ]; then
        echo "MISSING: $f"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "=========================================="
    echo " buffalo_l installed successfully!"
    echo ""
    echo " Now relaunch the app:"
    echo "   cd ~/Desktop/deep-live-cam-main"
    echo "   ./launch.sh"
    echo "=========================================="
else
    echo "=========================================="
    echo " Some files are missing - extraction may have"
    echo " produced a nested folder. Check the listing"
    echo " above and tell Claude what you see."
    echo "=========================================="
fi
