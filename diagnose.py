#!/usr/bin/env python3
"""Diagnose why face swap isn't working. Bypasses launcher's silent
error handling and prints the actual exception."""

import os
import sys
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Inject venv if launcher would
import sysconfig
pyver = f"python{sysconfig.get_python_version()}"
for vn in (".venv312", ".venv"):
    site = os.path.join(ROOT, vn, "lib", pyver, "site-packages")
    if os.path.isdir(site):
        sys.path.insert(1, site)
        print(f"[venv] using {site}")
        break

print("\n=== Step 1: filesystem check ===")
inswap = os.path.join(ROOT, "models", "inswapper_128_fp16.onnx")
buf_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
print(f"inswapper:    {inswap}")
print(f"  exists: {os.path.exists(inswap)}, size: {os.path.getsize(inswap) if os.path.exists(inswap) else 0:,}")
print(f"buffalo_l:    {buf_dir}")
if os.path.isdir(buf_dir):
    for f in sorted(os.listdir(buf_dir)):
        full = os.path.join(buf_dir, f)
        if os.path.isfile(full):
            print(f"  file: {f}  ({os.path.getsize(full):,} bytes)")
        elif os.path.isdir(full):
            print(f"  SUBDIR: {f}/  <-- nested! files should be in buffalo_l/ directly")
            for sf in sorted(os.listdir(full)):
                print(f"    {sf}")
else:
    print("  MISSING")

# Also check for the zipped form some installers leave behind
for cand in ("buffalo_l.zip", "buffalo_l/buffalo_l.zip"):
    p = os.path.join(os.path.expanduser("~/.insightface/models"), cand)
    if os.path.exists(p):
        print(f"  STRAY ZIP at {p} -- needs manual extraction")

print("\n=== Step 2: import core libs ===")
for mod in ["numpy", "cv2", "onnxruntime", "insightface"]:
    try:
        m = __import__(mod)
        print(f"  OK   {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {mod:14s} {type(e).__name__}: {e}")

print("\n=== Step 3: load InsightFace FaceAnalysis (buffalo_l) ===")
try:
    import insightface
    fa = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition', 'landmark_2d_106']
    )
    fa.prepare(ctx_id=0, det_size=(640, 640))
    print("  OK FaceAnalysis loaded")
except Exception as e:
    print("  FAIL FaceAnalysis:")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 4: detect face in source image ===")
try:
    import cv2
    src = os.path.join(ROOT, "faces", "obama.jpg")
    if not os.path.exists(src):
        # pick any face image
        for f in os.listdir(os.path.join(ROOT, "faces")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(ROOT, "faces", f)
                break
    print(f"  using: {src}")
    img = cv2.imread(src)
    print(f"  image shape: {img.shape if img is not None else None}")
    faces = fa.get(img)
    print(f"  faces detected: {len(faces)}")
    if not faces:
        print("  WARN: source image has no detectable face")
except Exception as e:
    print("  FAIL detect:")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 5: load inswapper face swap model ===")
try:
    swapper = insightface.model_zoo.get_model(
        os.path.join(ROOT, "models", "inswapper_128_fp16.onnx"),
        providers=['CPUExecutionProvider']
    )
    print(f"  OK swapper loaded: {type(swapper).__name__}")
except Exception as e:
    print("  FAIL swapper load:")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 6: end-to-end swap on dummy frame ===")
try:
    import numpy as np
    target = np.zeros((480, 640, 3), dtype=np.uint8)
    # paste source image as a fake "webcam frame" so we have a face to swap
    h, w = img.shape[:2]
    if h > 480 or w > 640:
        scale = min(480/h, 640/w)
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        img_small = img
    yh, yw = img_small.shape[:2]
    target[0:yh, 0:yw] = img_small
    target_faces = fa.get(target)
    print(f"  target faces detected: {len(target_faces)}")
    if target_faces and faces:
        result = swapper.get(target, target_faces[0], faces[0], paste_back=True)
        print(f"  swap result shape: {result.shape}")
        out = os.path.join(ROOT, "diagnose_swap_test.jpg")
        cv2.imwrite(out, result)
        print(f"  saved: {out}")
        print("\n=== ALL CHECKS PASSED ===")
        print("If the live app still does not swap, the problem is camera frames")
        print("(no face visible to camera, lighting, or backend issues).")
    else:
        print("  cannot run end-to-end (no face in target)")
except Exception as e:
    print("  FAIL swap:")
    traceback.print_exc()
    sys.exit(1)
