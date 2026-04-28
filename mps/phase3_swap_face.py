#!/usr/bin/env python3
"""
Phase 3: end-to-end swap_face on a still image.

Strategy:
  - Keep insightface's FaceAnalysis (on CPU) for face detection, landmark
    detection, and source-face embedding. These are fast on CPU and have
    well-tested post-processing (anchor decoding, NMS, alignment).
  - Replace ONLY the inswapper's onnxruntime session with a PyTorch+MPS
    session that has a matching interface. insightface's INSwapper.get()
    method handles all the alignment and paste-back; we just hijack the
    inference call.

Smoke test:
  - Load Trump as the source face, Obama as the target image
  - Run the full pipeline: detect target face, get source embedding, swap
  - Save the result alongside the original
  - Report per-stage timing

Pass criteria:
  - Output image looks like a real swap (Trump's features on Obama's body)
  - End-to-end pipeline under 200 ms (= 5+ FPS)
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import cv2

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
INSWAPPER_ONNX = PARENT / "models" / "inswapper_128_fp16.onnx"
FACES_DIR = PARENT / "faces"

# Test images: source = the face we want to wear, target = the photo we paste into
SOURCE_IMG = FACES_DIR / "trump.jpg"
TARGET_IMG = FACES_DIR / "obama.jpg"
OUTPUT_IMG = HERE / "phase3_swap_result.jpg"

print("=" * 60)
print(" Phase 3: end-to-end swap_face on still image")
print("=" * 60)

# --- Pre-flight ------------------------------------------------------
for p in [INSWAPPER_ONNX, SOURCE_IMG, TARGET_IMG]:
    if not p.exists():
        print(f"ERROR: missing {p}")
        sys.exit(1)
print(f"Source face:  {SOURCE_IMG.name}")
print(f"Target image: {TARGET_IMG.name}")
print(f"Inswapper:    {INSWAPPER_ONNX.name}")
print(f"PyTorch:      {torch.__version__}  MPS={torch.backends.mps.is_available()}")

# --- MPSSession: drop-in replacement for ort.InferenceSession --------
print("\n[1/5] Building MPSSession wrapper...")
import onnx
from onnx2torch import convert as onnx2torch_convert

class _MockIO:
    def __init__(self, name): self.name = name

class MPSSession:
    """Mimics onnxruntime.InferenceSession but runs on Apple GPU via MPS.

    insightface's INSwapper calls session.run(output_names, {input_name: blob, ...})
    and expects a list of numpy arrays back. That's the only interface we need."""
    def __init__(self, onnx_path: str):
        self.device = torch.device("mps")
        onnx_model = onnx.load(onnx_path)
        self.torch_model = onnx2torch_convert(onnx_model).to(self.device).eval()
        self._input_names = [i.name for i in onnx_model.graph.input]
        self._output_names = [o.name for o in onnx_model.graph.output]
        # warm up (compiles MPS kernels — first call would otherwise add ~1s latency)
        dummy_inputs = []
        for inp in onnx_model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 1
                    for d in inp.type.tensor_type.shape.dim]
            dummy_inputs.append(torch.randn(*dims, device=self.device, dtype=torch.float32))
        with torch.inference_mode():
            _ = self.torch_model(*dummy_inputs)
            torch.mps.synchronize()

    def run(self, output_names, input_dict):
        # Convert all inputs to torch tensors on MPS
        torch_inputs = []
        for name in self._input_names:
            arr = input_dict[name]
            torch_inputs.append(torch.from_numpy(arr).to(self.device))
        with torch.inference_mode():
            out = self.torch_model(*torch_inputs)
            torch.mps.synchronize()
        # insightface's INSwapper expects a list back
        if isinstance(out, (tuple, list)):
            return [t.detach().cpu().numpy() for t in out]
        return [out.detach().cpu().numpy()]

    def get_inputs(self):
        return [_MockIO(n) for n in self._input_names]

    def get_outputs(self):
        return [_MockIO(n) for n in self._output_names]

mps_session = MPSSession(str(INSWAPPER_ONNX))
print(f"  MPSSession ready (warmup done)")

# --- Build the insightface pipeline with MPS-backed swapper ----------
print("\n[2/5] Initializing insightface FaceAnalysis (CPU)...")
import insightface
fa = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition", "landmark_2d_106"],
)
fa.prepare(ctx_id=0, det_size=(320, 320))
print("  FaceAnalysis ready")

print("\n[3/5] Building INSwapper with MPS session injected...")
from insightface.model_zoo.inswapper import INSwapper
# Construct the swapper but bypass its onnxruntime session creation
swapper = INSwapper.__new__(INSwapper)
# Manually set fields — INSwapper.__init__ expects model_file or session
swapper.model_file = str(INSWAPPER_ONNX)
swapper.session = mps_session
swapper.taskname = "inswapper"
# Extract the emap (last initializer in the ONNX graph)
import onnx
from onnx import numpy_helper
_onnx_model = onnx.load(str(INSWAPPER_ONNX))
swapper.emap = numpy_helper.to_array(_onnx_model.graph.initializer[-1])
# Input metadata
swapper.input_mean = 0.0
swapper.input_std = 255.0
inputs = mps_session.get_inputs()
swapper.input_names = [inputs[0].name, inputs[1].name]
# input_size is read from the model's first input (target image)
input_shape = _onnx_model.graph.input[0].type.tensor_type.shape.dim
H, W = input_shape[2].dim_value, input_shape[3].dim_value
swapper.input_size = (W, H)  # (width, height)
swapper.output_names = [o.name for o in mps_session.get_outputs()]
print(f"  INSwapper configured: input_size={swapper.input_size}")

# --- Run the actual pipeline ---------------------------------------
print("\n[4/5] Running end-to-end swap...")

# Load images
src_img = cv2.imread(str(SOURCE_IMG))
tgt_img = cv2.imread(str(TARGET_IMG))
print(f"  source: {src_img.shape}  target: {tgt_img.shape}")

# Detect source face once (this gives us the embedding we need)
t0 = time.perf_counter()
source_faces = fa.get(src_img)
t_src_detect = (time.perf_counter() - t0) * 1000
if not source_faces:
    print("ERROR: no face detected in source image")
    sys.exit(1)
source_face = source_faces[0]
print(f"  source detection:  {t_src_detect:.0f} ms  (1 face)")

# Detect target face(s)
t0 = time.perf_counter()
target_faces = fa.get(tgt_img)
t_tgt_detect = (time.perf_counter() - t0) * 1000
if not target_faces:
    print("ERROR: no face detected in target image")
    sys.exit(1)
target_face = target_faces[0]
print(f"  target detection:  {t_tgt_detect:.0f} ms  ({len(target_faces)} face)")

# Swap — this is what we'd do per frame in live mode
t0 = time.perf_counter()
result = swapper.get(tgt_img, target_face, source_face, paste_back=True)
t_swap = (time.perf_counter() - t0) * 1000
print(f"  MPS swap+paste:    {t_swap:.0f} ms")

# Repeat the swap a few times to get a stable median (warmup is over)
times = []
for _ in range(10):
    t0 = time.perf_counter()
    _ = swapper.get(tgt_img, target_face, source_face, paste_back=True)
    times.append((time.perf_counter() - t0) * 1000)
import statistics
swap_median = statistics.median(times)
print(f"  MPS swap+paste (steady-state, median of 10): {swap_median:.0f} ms")

# Live-mode budget: detection + swap (source is cached; not re-detected per frame)
live_budget = t_tgt_detect + swap_median
print(f"\n  Live-frame budget: {t_tgt_detect:.0f} (det) + {swap_median:.0f} (swap+paste) "
      f"= {live_budget:.0f} ms = {1000.0/live_budget:.1f} FPS")

# --- Save output ----------------------------------------------------
print(f"\n[5/5] Saving result...")
cv2.imwrite(str(OUTPUT_IMG), result)
print(f"  wrote {OUTPUT_IMG}")
print(f"  open it: open {OUTPUT_IMG}")

# --- Verdict --------------------------------------------------------
print("\n" + "=" * 60)
if live_budget < 200:
    print(f" RESULT: PASS — {1000.0/live_budget:.1f} FPS budget for live frames.")
    print(" Inspect phase3_swap_result.jpg — you should see Trump's features")
    print(" on Obama. If it looks correct, Phase 3 is done. Move to Phase 4")
    print(" (wire MPSSession into the parent project's face_swapper.py).")
else:
    print(f" RESULT: PARTIAL — {1000.0/live_budget:.1f} FPS budget.")
    print(" Visual check the output — if it's correct, we still have a working")
    print(" swap pipeline; we'd just want to look at where the time is going.")
print("=" * 60)
