#!/usr/bin/env python3
"""
Phase 2: port buffalo_l face detection to PyTorch + MPS.

After Phase 1.7 confirmed that the inswapper has a hard CoreML compatibility
ceiling (only 34/273 ops supported), PyTorch+MPS at 79 ms is our baseline
for the swap stage. To hit ~10 FPS for the full live pipeline, detection
needs to come in at <50 ms.

The buffalo_l face detector is a RetinaFace variant — much smaller than
inswapper (16 MB vs 265 MB) with cleaner ops. It should accelerate well.

What this does:
  1. Loads ~/.insightface/models/buffalo_l/det_10g.onnx
  2. Converts to a torch.nn.Module via onnx2torch with FIXED input shape
     (320x320 — matches face_analyser's det_size we set earlier)
  3. Times 30 forward passes on MPS and CPU as comparison
  4. Sanity check: numerical agreement with the ORT CPU reference

If MPS detection is < 50 ms and outputs match, Phase 2 passes and we
move to Phase 3 (wire both models into a swap_face function).
"""

import os
import sys
import time
import statistics
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

DET_ONNX = Path(os.path.expanduser(
    "~/.insightface/models/buffalo_l/det_10g.onnx"))

print("=" * 60)
print(" Phase 2: PyTorch+MPS face detection (det_10g.onnx)")
print("=" * 60)

if not DET_ONNX.exists():
    print(f"ERROR: {DET_ONNX} not found.")
    print("  Run the parent project's fix_buffalo.sh to install detection models.")
    sys.exit(1)

print(f"Model:    {DET_ONNX}")
print(f"PyTorch:  {torch.__version__}")
print(f"MPS:      {torch.backends.mps.is_available()}")

# --- Load ONNX -------------------------------------------------------
print("\n[1/4] Loading ONNX...")
import onnx
onnx_model = onnx.load(str(DET_ONNX))

# Show input/output shapes — note '?' for dynamic dims
for i, inp in enumerate(onnx_model.graph.input):
    dims = [d.dim_value if d.dim_value > 0 else "?"
            for d in inp.type.tensor_type.shape.dim]
    print(f"  input {i}:  '{inp.name}' shape={dims}")
for i, out in enumerate(onnx_model.graph.output):
    dims = [d.dim_value if d.dim_value > 0 else "?"
            for d in out.type.tensor_type.shape.dim]
    print(f"  output {i}: '{out.name}' shape={dims}")

# --- Convert to PyTorch with FIXED shape ----------------------------
# RetinaFace's input is dynamic [1, 3, '?', '?']. Static shape will let
# onnx2torch and MPS optimize better. We use 320x320 to match face_analyser.
print("\n[2/4] Converting to PyTorch (fixed input 320x320)...")
from onnx import shape_inference

# Force the input shape to 320x320 so the converter has a static graph
for inp in onnx_model.graph.input:
    dims = inp.type.tensor_type.shape.dim
    if len(dims) == 4:
        dims[2].dim_value = 320
        dims[3].dim_value = 320
        # clear any dim_param
        dims[2].dim_param = ""
        dims[3].dim_param = ""

onnx_model = shape_inference.infer_shapes(onnx_model)

from onnx2torch import convert as onnx2torch_convert
t0 = time.perf_counter()
torch_model = onnx2torch_convert(onnx_model)
torch_model.eval()
print(f"  conversion: {(time.perf_counter()-t0)*1000:.0f} ms")
n_params = sum(p.numel() for p in torch_model.parameters())
print(f"  parameters: {n_params:,}")

# --- Build dummy input ---------------------------------------------
np.random.seed(0)
inp_np = np.random.randn(1, 3, 320, 320).astype(np.float32)

# --- ORT CPU reference for correctness ------------------------------
print("\n[3/4] ORT CPU reference (for correctness)...")
import onnxruntime as ort

# Reload original ONNX (without our shape edit) for fair reference
ref_onnx = onnx.load(str(DET_ONNX))
sess = ort.InferenceSession(ref_onnx.SerializeToString(),
                            providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
t0 = time.perf_counter()
ort_outs = sess.run(output_names, {input_name: inp_np})
print(f"  ORT CPU call: {(time.perf_counter()-t0)*1000:.0f} ms")
print(f"  outputs: {len(ort_outs)} tensors  shapes={[o.shape for o in ort_outs]}")

# --- Time each device -----------------------------------------------
def time_torch(model, inp_tensor, label, n=30):
    with torch.inference_mode():
        # warmup
        t0 = time.perf_counter()
        out = model(inp_tensor)
        if inp_tensor.device.type == "mps":
            torch.mps.synchronize()
        warmup = (time.perf_counter() - t0) * 1000
        times = []
        for _ in range(n):
            if inp_tensor.device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            out = model(inp_tensor)
            if inp_tensor.device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    print(f"  {label}: warmup {warmup:.0f} ms  "
          f"median {med:.1f} ms  min {min(times):.1f}  max {max(times):.1f}  "
          f"FPS {1000.0/med:.1f}")
    return med, out

print("\n[4/4] Timing detection model...")

# CPU baseline (skip if you suspect it'll hang — this model is small enough
# that CPU should finish in seconds, but flag it just in case)
print("\n  [CPU PyTorch]")
inp_cpu = torch.from_numpy(inp_np)
cpu_med, cpu_out = time_torch(torch_model, inp_cpu, "CPU", n=10)

# MPS
print("\n  [MPS]")
device = torch.device("mps")
torch_model_mps = torch_model.to(device)
inp_mps = inp_cpu.to(device)
mps_med, mps_out = time_torch(torch_model_mps, inp_mps, "MPS", n=30)

# --- Correctness ----------------------------------------------------
print("\n[5/5] Correctness — comparing MPS to ORT CPU reference...")
# Convert tuple outputs to a comparable form
if isinstance(mps_out, (tuple, list)):
    mps_outs = [t.detach().cpu().numpy() for t in mps_out]
else:
    mps_outs = [mps_out.detach().cpu().numpy()]

ok = True
for i, (mps_arr, ort_arr) in enumerate(zip(mps_outs, ort_outs)):
    if mps_arr.shape != ort_arr.shape:
        print(f"  output[{i}]: shape mismatch {mps_arr.shape} vs {ort_arr.shape}")
        ok = False
        continue
    diff = np.abs(mps_arr - ort_arr)
    print(f"  output[{i}] shape={mps_arr.shape}  "
          f"max diff={diff.max():.2e}  mean diff={diff.mean():.2e}")
    if diff.max() > 5e-2:
        ok = False

# --- Verdict --------------------------------------------------------
print("\n" + "=" * 60)
PASS_SPEED = mps_med < 50
PASS_NUMERIC = ok
if PASS_SPEED and PASS_NUMERIC:
    print(f" RESULT: PASS — MPS detection at {mps_med:.0f} ms.")
    print(f" Combined budget: 79 (swap) + {mps_med:.0f} (det) = "
          f"{79 + mps_med:.0f} ms = {1000.0/(79 + mps_med):.1f} FPS overall.")
    print(" Proceed to Phase 3 (build the swap_face function in PyTorch).")
elif PASS_NUMERIC:
    print(f" RESULT: PARTIAL — MPS at {mps_med:.0f} ms is over 50 ms target.")
    print(f" Overall: {1000.0/(79 + mps_med):.1f} FPS. Still build Phase 3 but tighter.")
else:
    print(f" RESULT: FAIL — numeric divergence on detection output.")
    print(" Investigate which op produced the mismatch before Phase 3.")
print("=" * 60)
