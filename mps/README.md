# Deep Live Cam - PyTorch + MPS Rewrite

The original `deep-live-cam-main` runs inference through `onnxruntime`,
which on Apple Silicon hits a wall: the CoreML EP crashes the detection
threads on a shape-rank bug, and CPU mode tops out around 1 FPS even
after every reasonable optimization. This branch replaces the inference
layer with **PyTorch + MPS** (Apple's Metal Performance Shaders backend),
which is well-tuned for Apple Silicon and should hit 15-30+ FPS on
M-series chips.

The UI, camera capture, and worker thread architecture from the parent
project are unchanged. Only `get_face_analyser()` and `get_face_swapper()`
get re-implementations.

## Phases

| Phase | Goal | Pass criteria |
|-------|------|---------------|
| 1 | Smoke test inswapper on MPS | <50 ms per call, output matches CPU |
| 2 | Port face detection (RetinaFace) to PyTorch | Detects faces in test image on MPS |
| 3 | Wrap full swap pipeline (`swap_face` in PyTorch) | End-to-end swap on a test image |
| 4 | Wire into `modules/ui.py`'s live preview | Live webcam swap at 15+ FPS |

We do not move to the next phase until the previous passes.

## Phase 1 — quickstart

```bash
cd ~/Desktop/deep-live-cam-main/mps
bash setup_mps.sh             # one-time, ~3-5 min, installs PyTorch
source .venv/bin/activate
python phase1_smoke.py
```

The smoke test:
- loads the existing `inswapper_128_fp16.onnx` from the parent project
- converts it to a `torch.nn.Module` via `onnx2torch`
- runs CPU and MPS inference
- compares outputs numerically and times 30 forward passes
- reports an implied FPS ceiling for the swap stage

### Decision tree from the smoke test result

- **PASS** (MPS < 50 ms, outputs match) → write Phase 2.
- **PARTIAL** (outputs match but 50-150 ms) → still viable, write Phase 2 with tighter budget.
- **FAIL** (numeric divergence or > 200 ms) → investigate; common causes are
  unsupported ops on MPS (look at the conversion warnings) or fp16/fp32
  precision mismatch on a particular layer.

## Files

- `setup_mps.sh` - one-time environment setup (Python 3.11 venv, PyTorch, deps)
- `phase1_smoke.py` - Phase 1 smoke test
- `README.md` - this file
- `.venv/` - Python virtual environment (created by setup)

## Why a fresh sub-tree

The parent project's `.venv` is built around `onnxruntime` and has a
mountain of working state we don't want to disturb. Photo-mode demos
(via the parent's Start button) still work fine and are a perfectly
valid Family Day fallback. This sub-tree is a clean engineering branch
that can be evolved independently and merged back (or kept separate)
once it's working.
