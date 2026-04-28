#!/usr/bin/env python3
"""
Phase 1.6: try Apple's native Core ML runtime instead of PyTorch+MPS.

Phase 1.5 found that PyTorch's MPS backend is only partially using the
GPU on this machine — sanity matmul showed 1.0x CPU/MPS ratio, fp16 was
a no-op, torch.compile hit graph breaks. The real Apple Silicon path is
Core ML, which decides per-op whether to use the Neural Engine, the GPU,
or the CPU, and it actually engages the Neural Engine.

What this does:
  1. Converts inswapper_128_fp16.onnx -> .mlpackage via coremltools
     (cached after first conversion)
  2. Loads the .mlpackage with coremltools.models.MLModel
  3. Times 30 inferences using each compute target:
       - ALL  (Apple chooses: typically NE > GPU > CPU)
       - CPU_AND_GPU  (skip NE, use GPU)
       - CPU_ONLY  (baseline)
  4. Reports median ms and implied FPS for each
  5. Compares output to ONNX reference for correctness

Pass criteria:
  - CoreML 'ALL' median < 50 ms  (== 20+ FPS ceiling)
  - Numerics match ONNX (max abs diff < 5e-2)
"""

import os
import sys
import time
import statistics
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
INSWAPPER_ONNX = PARENT / "models" / "inswapper_128_fp16.onnx"
ML_PACKAGE_OUT = HERE / "inswapper_128.mlpackage"

print("=" * 60)
print(" Phase 1.6: Apple Core ML runtime")
print("=" * 60)

# --- Pre-flight ------------------------------------------------------
try:
    import coremltools as ct
    print(f"coremltools: {ct.__version__}")
except ImportError:
    print("coremltools not installed. Run:")
    print("    pip install coremltools")
    sys.exit(1)

print(f"Source ONNX: {INSWAPPER_ONNX}")

# --- Convert (or load cached) ---------------------------------------
if ML_PACKAGE_OUT.exists():
    print(f"\n[1/3] Using cached .mlpackage at {ML_PACKAGE_OUT}")
else:
    print("\n[1/3] Converting ONNX -> PyTorch -> CoreML .mlpackage (one-time, ~1-2 min)...")
    print("       (coremltools 9.0 dropped direct ONNX support; routing through")
    print("        onnx2torch + TorchScript trace, which is the supported path.)")
    import torch
    import onnx
    from onnx2torch import convert as onnx2torch_convert

    onnx_model = onnx.load(str(INSWAPPER_ONNX))
    t0 = time.perf_counter()
    torch_model = onnx2torch_convert(onnx_model).eval()
    print(f"  onnx -> pytorch: {time.perf_counter()-t0:.1f}s")

    # Trace with example inputs so coremltools sees a static graph
    example_target = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    example_source = torch.randn(1, 512, dtype=torch.float32)
    example_source = example_source / example_source.norm(dim=1, keepdim=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(torch_model, (example_target, example_source),
                                 strict=False)
    print(f"  torchscript trace: {time.perf_counter()-t0:.1f}s")

    # Now convert the traced module
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
        inputs=[
            ct.TensorType(name="target", shape=(1, 3, 128, 128), dtype=np.float32),
            ct.TensorType(name="source", shape=(1, 512), dtype=np.float32),
        ],
    )
    print(f"  pytorch -> coreml: {time.perf_counter()-t0:.1f}s")
    mlmodel.save(str(ML_PACKAGE_OUT))
    print(f"  saved to {ML_PACKAGE_OUT}")

# --- Run ONNX reference once for correctness comparison -------------
print("\n[2/3] ONNX reference inference (correctness baseline)...")
import onnxruntime as ort
sess = ort.InferenceSession(str(INSWAPPER_ONNX), providers=["CPUExecutionProvider"])
np.random.seed(0)
target_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
source_np = np.random.randn(1, 512).astype(np.float32)
source_np = source_np / np.linalg.norm(source_np, axis=1, keepdims=True)
t0 = time.perf_counter()
ort_out = sess.run(None, {"target": target_np, "source": source_np})[0]
print(f"  ort CPU call: {(time.perf_counter()-t0)*1000:.0f} ms")

# --- Time CoreML on each compute target -----------------------------
print("\n[3/3] Timing CoreML across compute targets...")

def time_with(compute_units, n=30):
    model = ct.models.MLModel(str(ML_PACKAGE_OUT), compute_units=compute_units)
    inp = {"target": target_np, "source": source_np}
    # warmup
    t0 = time.perf_counter()
    out = model.predict(inp)
    warmup = (time.perf_counter() - t0) * 1000
    # timed loop
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = model.predict(inp)
        times.append((time.perf_counter() - t0) * 1000)
    median = statistics.median(times)
    out_arr = list(out.values())[0]
    return {
        "warmup_ms": warmup,
        "median_ms": median,
        "min_ms": min(times),
        "max_ms": max(times),
        "fps": 1000.0 / median,
        "out": out_arr,
    }

targets = [
    ("ALL (NE+GPU+CPU)", ct.ComputeUnit.ALL),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
]
results = {}
for name, cu in targets:
    print(f"\n  [{name}]")
    try:
        r = time_with(cu)
        results[name] = r
        diff = np.abs(r["out"] - ort_out)
        max_d, mean_d = float(diff.max()), float(diff.mean())
        print(f"    warmup: {r['warmup_ms']:.0f} ms  "
              f"median: {r['median_ms']:.1f} ms  "
              f"FPS: {r['fps']:.1f}")
        print(f"    vs ORT: max diff={max_d:.2e}  mean={mean_d:.2e}")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")

# --- Summary --------------------------------------------------------
print("\n" + "=" * 60)
print(" Summary (sorted fastest first)")
print("=" * 60)
ranked = sorted(results.items(), key=lambda kv: kv[1]["median_ms"])
for name, r in ranked:
    print(f"  {name:22s}  median {r['median_ms']:6.1f} ms   "
          f"({r['fps']:5.1f} FPS ceiling)")

if ranked:
    best_name, best = ranked[0]
    print()
    if best["median_ms"] < 50:
        print(f" WIN: {best_name} at {best['median_ms']:.0f} ms ({best['fps']:.0f} FPS).")
        print(" CoreML is the right runtime. Phase 2 should target this.")
    else:
        print(f" Best: {best_name} at {best['median_ms']:.0f} ms.")
        print(" Above 50 ms target. CoreML model still likely beats PyTorch+MPS;")
        print(" we may need a smaller model or to accept this as the ceiling.")
