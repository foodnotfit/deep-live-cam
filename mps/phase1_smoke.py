#!/usr/bin/env python3
"""
Phase 1 smoke test for the PyTorch + MPS rewrite (MPS-only edition).

Skips the PyTorch-CPU baseline (which appears to be either hung or
extremely slow on this machine — same underlying issue as the
onnxruntime CPU bottleneck). For correctness, we compare MPS output
to the original ONNX model run via onnxruntime CPU as a quick sanity
check, but only one pass — not enough volume to hit whatever the
CPU pathology is.

Pass criteria:
  - MPS forward pass < 50 ms  (= 20+ FPS ceiling for swap stage alone)
  - MPS output ≈ onnxruntime CPU output (max abs diff < 5e-2; the
    onnx model is fp16 and onnx2torch upcasts to fp32, so we expect
    some slack)

Total runtime target: under 30 seconds.
"""

import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
INSWAPPER_ONNX = PARENT / "models" / "inswapper_128_fp16.onnx"

print("=" * 60)
print(" Phase 1: PyTorch + MPS smoke test (MPS-only)")
print("=" * 60)

# --- Pre-flight ------------------------------------------------------
if not INSWAPPER_ONNX.exists():
    print(f"ERROR: model not found at {INSWAPPER_ONNX}")
    sys.exit(1)
print(f"Model:         {INSWAPPER_ONNX}")
print(f"PyTorch:       {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    print("MPS unavailable. Cannot run smoke test.")
    sys.exit(1)

# --- Load ONNX -------------------------------------------------------
print("\n[1/4] Loading ONNX model...")
import onnx
t0 = time.perf_counter()
onnx_model = onnx.load(str(INSWAPPER_ONNX))
print(f"  loaded in {time.perf_counter()-t0:.2f}s")

for i, inp in enumerate(onnx_model.graph.input):
    dims = [d.dim_value if d.dim_value > 0 else "?"
            for d in inp.type.tensor_type.shape.dim]
    print(f"  input {i}: '{inp.name}' shape={dims}")
for i, out in enumerate(onnx_model.graph.output):
    dims = [d.dim_value if d.dim_value > 0 else "?"
            for d in out.type.tensor_type.shape.dim]
    print(f"  output {i}: '{out.name}' shape={dims}")

# --- Build dummy inputs ---------------------------------------------
np.random.seed(0)
target_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
source_np = np.random.randn(1, 512).astype(np.float32)
source_np = source_np / np.linalg.norm(source_np, axis=1, keepdims=True)

# --- Run via onnxruntime as the correctness reference ---------------
print("\n[2/4] Reference inference via onnxruntime CPU (1 call)...")
import onnxruntime as ort
ort_sess = ort.InferenceSession(
    str(INSWAPPER_ONNX),
    providers=["CPUExecutionProvider"],
)
t0 = time.perf_counter()
ort_out = ort_sess.run(None, {"target": target_np, "source": source_np})[0]
print(f"  ort CPU call: {(time.perf_counter()-t0)*1000:.0f} ms")
print(f"  reference output shape: {ort_out.shape}, dtype: {ort_out.dtype}")

# --- Convert ONNX -> PyTorch and move to MPS ------------------------
print("\n[3/4] Converting ONNX -> PyTorch and moving to MPS...")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnx2torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from onnx2torch import convert

t0 = time.perf_counter()
torch_model = convert(onnx_model)
torch_model.eval()
print(f"  conversion: {(time.perf_counter()-t0)*1000:.0f} ms")

device = torch.device("mps")
t0 = time.perf_counter()
torch_model_mps = torch_model.to(device)
print(f"  move to MPS: {(time.perf_counter()-t0)*1000:.0f} ms")
n_params = sum(p.numel() for p in torch_model_mps.parameters())
print(f"  parameters: {n_params:,}")

target_mps = torch.from_numpy(target_np).to(device)
source_mps = torch.from_numpy(source_np).to(device)

# --- MPS timing ------------------------------------------------------
print("\n[4/4] MPS timing (warmup + 30 passes)...")
with torch.inference_mode():
    t0 = time.perf_counter()
    out0 = torch_model_mps(target_mps, source_mps)
    torch.mps.synchronize()
    print(f"  warmup (kernel compile): {(time.perf_counter()-t0)*1000:.0f} ms")

    mps_times = []
    for _ in range(30):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = torch_model_mps(target_mps, source_mps)
        torch.mps.synchronize()
        mps_times.append((time.perf_counter() - t0) * 1000)

mps_median = statistics.median(mps_times)
mps_min = min(mps_times)
mps_max = max(mps_times)

# Correctness: compare last MPS output to the onnxruntime reference
mps_np = out.detach().cpu().numpy()
diff = np.abs(mps_np - ort_out)
max_diff = float(diff.max())
mean_diff = float(diff.mean())

print(f"\n  MPS per-call:    median={mps_median:.1f} ms  "
      f"min={mps_min:.1f}  max={mps_max:.1f}")
print(f"  Implied ceiling: {1000.0/mps_median:.1f} FPS for swap stage alone")
print(f"  vs ort CPU ref:  max abs diff={max_diff:.2e}  mean={mean_diff:.2e}")

PASS_NUMERIC = max_diff < 5e-2
PASS_SPEED = mps_median < 50.0

print("\n" + "=" * 60)
if PASS_NUMERIC and PASS_SPEED:
    print(" RESULT: PASS")
    print(f" Swap inference at ~{mps_median:.0f} ms ({1000/mps_median:.0f} FPS ceiling).")
    print(" Proceed to Phase 2 (port face detection).")
elif PASS_NUMERIC:
    print(" RESULT: PARTIAL")
    print(f" Numerics correct, but {mps_median:.0f} ms is over 50 ms target.")
    print(" Project still viable; budget tighter.")
else:
    print(" RESULT: FAIL — numeric divergence")
    print(f" max abs diff {max_diff:.2e} (threshold 5e-2).")
    print(" Likely an unsupported MPS op or precision issue. Check warnings above.")
print("=" * 60)
