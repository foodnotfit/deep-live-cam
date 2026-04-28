#!/usr/bin/env python3
"""
Phase 1.7: try onnxruntime's CoreML EP on the inswapper alone.

Background:
  - Phase 1 (PyTorch+MPS):   79 ms,  12.7 FPS, partial GPU utilization
  - Phase 1.6 (coremltools): conversion hangs at torch.jit.trace step
  - In the parent project, onnxruntime+CoreML crashed *on buffalo_l*
    (a model with dynamic input shape: [1, 3, '?', '?'])

The inswapper has fully *static* shapes — [1, 3, 128, 128] and [1, 512].
CoreML EP is happy with static shapes; the shape-rank bug it hit on
buffalo_l shouldn't apply. This is a 30-second test, not a rewrite.

If this gives sub-50 ms inference with correct numerics, we have our
runtime for the swap stage. Detection stays on CPU (where it works
fine), inswapper runs through onnxruntime + CoreML EP directly, and
we don't need PyTorch, MPS, or coremltools at all.
"""

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

print("=" * 60)
print(" Phase 1.7: onnxruntime + CoreML EP on inswapper alone")
print("=" * 60)

import onnxruntime as ort
print(f"onnxruntime: {ort.__version__}")
print(f"available providers: {ort.get_available_providers()}")

if "CoreMLExecutionProvider" not in ort.get_available_providers():
    print("CoreMLExecutionProvider not available. Bailing.")
    sys.exit(1)

# --- Inputs ---------------------------------------------------------
np.random.seed(0)
target = np.random.randn(1, 3, 128, 128).astype(np.float32)
source = np.random.randn(1, 512).astype(np.float32)
source = source / np.linalg.norm(source, axis=1, keepdims=True)

# --- Reference: CPU --------------------------------------------------
print("\n[1/3] Reference (CPU)...")
sess_cpu = ort.InferenceSession(
    str(INSWAPPER_ONNX),
    providers=["CPUExecutionProvider"],
)
# warmup
_ = sess_cpu.run(None, {"target": target, "source": source})
times = []
for _ in range(5):
    t0 = time.perf_counter()
    cpu_out = sess_cpu.run(None, {"target": target, "source": source})[0]
    times.append((time.perf_counter() - t0) * 1000)
print(f"  CPU median: {statistics.median(times):.0f} ms")

# --- CoreML EP -------------------------------------------------------
print("\n[2/3] CoreML EP (Apple Neural Engine + GPU)...")
try:
    sess_cml = ort.InferenceSession(
        str(INSWAPPER_ONNX),
        providers=[
            ("CoreMLExecutionProvider", {
                # Plain config — older onnxruntime CoreML options hardcoded
                # in the parent project's face_swapper.py crashed because they
                # included unsupported keys. Defaults are best on 1.20+.
            }),
            "CPUExecutionProvider",
        ],
    )
    print("  session created OK")
    # warmup (first call compiles the CoreML cache, can take 5-10s)
    t0 = time.perf_counter()
    out_warmup = sess_cml.run(None, {"target": target, "source": source})[0]
    print(f"  warmup (first call, includes CoreML compile): "
          f"{(time.perf_counter() - t0)*1000:.0f} ms")

    # timed loop
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        cml_out = sess_cml.run(None, {"target": target, "source": source})[0]
        times.append((time.perf_counter() - t0) * 1000)
    median = statistics.median(times)
    print(f"  CoreML median: {median:.1f} ms  "
          f"min: {min(times):.1f}  max: {max(times):.1f}  "
          f"FPS ceiling: {1000.0/median:.1f}")

    # correctness vs CPU
    diff = np.abs(cml_out - cpu_out)
    print(f"  vs CPU: max abs diff={diff.max():.2e}  mean={diff.mean():.2e}")

except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    median = None

# --- Verdict ---------------------------------------------------------
print("\n" + "=" * 60)
if median is None:
    print(" RESULT: FAIL — CoreML EP couldn't run inswapper either.")
    print(" Pivot: skip CoreML, port detection to PyTorch+MPS, ship 8-10 FPS.")
elif median < 50:
    print(f" RESULT: WIN — {median:.0f} ms ({1000/median:.0f} FPS ceiling).")
    print(" Use onnxruntime+CoreML for the swap, keep CPU for detection.")
    print(" Skip the entire PyTorch rewrite. Phase 2 can wire this directly")
    print(" into the parent project's modules/processors/frame/face_swapper.py.")
else:
    print(f" RESULT: PARTIAL — {median:.0f} ms ({1000/median:.0f} FPS).")
    print(" Better than nothing but not the win we wanted. Compare to PyTorch+MPS")
    print(" baseline (79 ms / 12.7 FPS) — pick whichever is faster.")
print("=" * 60)
