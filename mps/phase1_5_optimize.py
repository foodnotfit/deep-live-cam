#!/usr/bin/env python3
"""
Phase 1.5: optimize the MPS swap inference past the 50 ms target.

Phase 1 landed at 79 ms (12.7 FPS ceiling). With face detection added
that's around 8-9 FPS for the full live pipeline — watchable but not
great. This script tries three orthogonal optimizations and reports
which gives the biggest win:

  A. fp16 (half precision)        — typically 1.5-2x speedup on MPS
  B. torch.compile (PT 2.0+)      — graph fusion, reduce launch overhead
  C. fp16 + torch.compile         — both stacked

Also includes a sanity check that MPS is actually running on the GPU
and not silently falling back to CPU for ops it can't dispatch.

We're aiming for sub-50 ms median per call. Each variant reports
median + implied FPS ceiling.
"""

import os
import sys
import time
import statistics
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
INSWAPPER_ONNX = PARENT / "models" / "inswapper_128_fp16.onnx"

print("=" * 60)
print(" Phase 1.5: MPS optimization sprint")
print("=" * 60)

if not torch.backends.mps.is_available():
    sys.exit("MPS unavailable.")
device = torch.device("mps")

# --- MPS GPU sanity check -------------------------------------------
# If MPS is real GPU, a big matmul should be much faster than the same
# matmul on CPU. If it's silently CPU-fallback, the times match.
print("\n[Sanity] Verifying MPS is actually on the GPU...")
N = 4096
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)
t0 = time.perf_counter(); _ = a_cpu @ b_cpu; cpu_ms = (time.perf_counter()-t0)*1000

a_mps = a_cpu.to(device); b_mps = b_cpu.to(device)
torch.mps.synchronize()
t0 = time.perf_counter(); _ = a_mps @ b_mps; torch.mps.synchronize()
mps_ms = (time.perf_counter()-t0)*1000

print(f"  CPU  4096x4096 matmul: {cpu_ms:.0f} ms")
print(f"  MPS  4096x4096 matmul: {mps_ms:.0f} ms")
ratio = cpu_ms / max(mps_ms, 1e-3)
print(f"  speedup ratio: {ratio:.1f}x")
if ratio < 2.0:
    print("  WARNING: MPS not significantly faster than CPU here.")
    print("  Some ops may be silently falling back. Optimizations will be limited.")
elif ratio < 5.0:
    print("  OK — MPS is on GPU but smallish speedup; partial fallback possible.")
else:
    print("  GOOD — MPS is using the Apple GPU correctly.")

# --- Load ONNX, convert to PyTorch ----------------------------------
print("\n[Load] Loading ONNX and converting to torch.nn.Module...")
import onnx
from onnx2torch import convert
onnx_model = onnx.load(str(INSWAPPER_ONNX))
torch_fp32 = convert(onnx_model).to(device).eval()

np.random.seed(0)
target_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
source_np = np.random.randn(1, 512).astype(np.float32)
source_np = source_np / np.linalg.norm(source_np, axis=1, keepdims=True)
target_fp32 = torch.from_numpy(target_np).to(device)
source_fp32 = torch.from_numpy(source_np).to(device)

# --- Helper: time N forward passes -----------------------------------
def time_model(model, target, source, n=30, name=""):
    with torch.inference_mode():
        # warmup
        t0 = time.perf_counter()
        _ = model(target, source)
        torch.mps.synchronize()
        warmup = (time.perf_counter()-t0)*1000
        # timed loop
        times = []
        for _ in range(n):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(target, source)
            torch.mps.synchronize()
            times.append((time.perf_counter()-t0)*1000)
    med = statistics.median(times)
    return {
        "name": name, "warmup_ms": warmup,
        "median_ms": med, "min_ms": min(times), "max_ms": max(times),
        "fps": 1000.0/med if med > 0 else float("inf"),
    }

# --- Variant A: baseline fp32 --------------------------------------
print("\n[A] Baseline fp32...")
res_a = time_model(torch_fp32, target_fp32, source_fp32, name="fp32")
print(f"  warmup: {res_a['warmup_ms']:.0f} ms  "
      f"median: {res_a['median_ms']:.1f} ms  "
      f"fps: {res_a['fps']:.1f}")

# --- Variant B: fp16 (half precision) -----------------------------
print("\n[B] fp16 (half precision)...")
try:
    torch_fp16 = convert(onnx_model).to(device).half().eval()
    target_fp16 = target_fp32.half()
    source_fp16 = source_fp32.half()
    res_b = time_model(torch_fp16, target_fp16, source_fp16, name="fp16")
    print(f"  warmup: {res_b['warmup_ms']:.0f} ms  "
          f"median: {res_b['median_ms']:.1f} ms  "
          f"fps: {res_b['fps']:.1f}")
except Exception as e:
    print(f"  fp16 failed: {type(e).__name__}: {e}")
    res_b = None

# --- Variant C: torch.compile (fp32) -------------------------------
print("\n[C] torch.compile (fp32)...")
try:
    torch_comp = torch.compile(
        convert(onnx_model).to(device).eval(),
        mode="reduce-overhead",
        backend="aot_eager",  # MPS-friendly; "inductor" is x86/CUDA-focused
    )
    res_c = time_model(torch_comp, target_fp32, source_fp32, name="compile-fp32")
    print(f"  warmup: {res_c['warmup_ms']:.0f} ms  "
          f"median: {res_c['median_ms']:.1f} ms  "
          f"fps: {res_c['fps']:.1f}")
except Exception as e:
    print(f"  compile fp32 failed: {type(e).__name__}: {e}")
    res_c = None

# --- Variant D: torch.compile + fp16 ------------------------------
print("\n[D] torch.compile + fp16...")
try:
    base_fp16 = convert(onnx_model).to(device).half().eval()
    torch_comp16 = torch.compile(
        base_fp16, mode="reduce-overhead", backend="aot_eager"
    )
    res_d = time_model(torch_comp16, target_fp16, source_fp16, name="compile-fp16")
    print(f"  warmup: {res_d['warmup_ms']:.0f} ms  "
          f"median: {res_d['median_ms']:.1f} ms  "
          f"fps: {res_d['fps']:.1f}")
except Exception as e:
    print(f"  compile fp16 failed: {type(e).__name__}: {e}")
    res_d = None

# --- Summary ------------------------------------------------------
print("\n" + "=" * 60)
print(" Summary (sorted fastest first)")
print("=" * 60)
all_results = [r for r in [res_a, res_b, res_c, res_d] if r]
all_results.sort(key=lambda r: r["median_ms"])
for r in all_results:
    print(f"  {r['name']:18s}  median {r['median_ms']:6.1f} ms   "
          f"({r['fps']:5.1f} FPS ceiling)")

best = all_results[0]
print()
if best["median_ms"] < 50:
    print(f" WIN: '{best['name']}' hits {best['median_ms']:.0f} ms — under 50 ms target.")
    print(" Proceed to Phase 2 with this configuration.")
else:
    print(f" Best is '{best['name']}' at {best['median_ms']:.0f} ms.")
    print(" Above 50 ms target. Still workable; deeper optimization needed for headroom.")
