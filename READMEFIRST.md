# READMEFIRST — Family Day deepfake demo

**TL;DR — to start the demo:**

```bash
cd ~/Desktop/deep-live-cam-main
./launch.sh
```

In the UI window: click **Select a face** → pick a celebrity from `faces/` → click **Live**. Press **Q** in the preview window to stop. That's it.

If the above doesn't work, scroll to **Troubleshooting** below.

---

## What this is

A real-time face-swap demo for educational use (showing kids how deepfakes work and why they're dangerous). Runs entirely offline on the Mac. Uses Apple Silicon's GPU (via PyTorch's MPS backend) for the heavy face-swap inference. The teaching moment is the kid sitting in front of the webcam and seeing themselves wearing a celebrity's face in real time.

Performance ceiling on M3 Pro is ~7-10 FPS — choppy but functional. That's a known limitation of the inswapper model on Apple Silicon, not a bug. NVIDIA hardware would do 30+, but we work with what we have.

## First-time setup (skip if already done)

Only needed once on a fresh machine. If `~/Desktop/deep-live-cam-main/.venv` already exists with packages installed, skip this section.

```bash
# 1. Install Python 3.11 with Tcl/Tk and ffmpeg via Homebrew
brew install python@3.11 python-tk@3.11 ffmpeg

# 2. Set up the project venv
cd ~/Desktop/deep-live-cam-main
bash fix_setup.sh

# 3. Install the buffalo_l face detection models
bash fix_buffalo.sh

# 4. Install the PyTorch + MPS dependencies (this branch uses Apple GPU)
source .venv/bin/activate
pip install "torch==2.7.0" "torchvision==0.22.0" onnx2torch

# 5. Numpy must link to Apple Accelerate or everything is slow
bash fix_numpy.sh

# Now ./launch.sh should work.
```

---

## Running the demo

**`./launch.sh` is self-contained — it activates the venv internally.** You do NOT need to `source .venv/bin/activate` before launching the demo. Just:

```bash
cd ~/Desktop/deep-live-cam-main
./launch.sh
```

You only need to manually activate the venv when running the **other** Python scripts directly (`download_faces.py`, `validate_faces.py`, `diagnose.py`):

```bash
cd ~/Desktop/deep-live-cam-main
source .venv/bin/activate    # only needed for python <something>.py commands
```

You'll know the venv is active when your prompt starts with `(.venv)`.

**Live mode (the main demo):**

A window opens with **Select a face / Select a target / Start / Live** buttons. For Family Day:

1. Click **Select a face** → navigate to `~/Desktop/deep-live-cam-main/faces/` → pick any celebrity/historical figure
2. Click **Live** → a second preview window opens with your webcam + face swap applied
3. Press **Q** in the preview window to stop
4. To switch faces: close the preview, pick a new face, click Live again

First swap after launch takes ~2 seconds (MPS kernel warmup). Steady-state runs at ~7-10 FPS on M3 Pro. Choppy but recognizable.

**Photo mode (the reliable backup):**

If live mode is acting up at the event, photo mode never fails. It produces higher quality output anyway.

1. Take a phone photo of the kid, AirDrop to Mac
2. **Select a face** → celebrity
3. **Select a target** → kid's photo
4. **Start** (not Live) — processes in 10-15 seconds
5. Show the result on screen

---

## Adding more celebrity / historical faces

Faces live in `~/Desktop/deep-live-cam-main/faces/`. Any `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` is automatically picked up by the UI.

**Bulk download from Wikipedia:**

```bash
python download_faces.py
```

Pulls Creative Commons / public-domain portraits for the curated list inside the script. Already-downloaded files are skipped. Edit `ENTRIES` near the top of `download_faces.py` to add or remove entries — each entry is `(slug, wikipedia_page_title, note)`.

If you hit a 429 "Too many requests", wait 10 minutes and re-run. The script handles retries automatically but Wikimedia will block sustained scraping.

**Manual additions:**

Drop any photo into `faces/` with a sensible filename (`harry_styles.jpg`). The UI displays the filename (without extension) as the label.

What makes a face work as a source:
- Front-facing within ~15° of head-on
- Single face in image (multiple faces confuse the embedding)
- Resolution > 500×500
- Well-lit, neutral or slight-smile expression
- No occlusion (sunglasses, big hats, hands on face)

**Validate every face works:**

```bash
python validate_faces.py
open validation_report.html
```

This walks every image in `faces/`, runs face detection, runs the swap pipeline onto a test target, and produces an HTML grid showing what each source actually looks like swapped. Run this before Family Day to catch any silent failures (low-res sources, profile angles, multi-face confusion). Green = OK, yellow = warning, red = broken.

---

## Troubleshooting

**`zsh: command not found: python`**

You forgot to activate the venv. Run:
```bash
cd ~/Desktop/deep-live-cam-main
source .venv/bin/activate
```

**`./launch.sh: line 3: .venv/bin/activate: No such file or directory`**

The venv was never created or got deleted. Run:
```bash
bash fix_setup.sh
```

**App opens but live preview shows 0.5-1 FPS**

The numpy library lost its Apple Accelerate linkage (this can happen if pip reinstalls numpy from a binary wheel). Verify and fix:
```bash
source .venv/bin/activate
python -c "import numpy as np; cfg=str(np.show_config(mode='dicts')).lower(); print('FOUND Accelerate' if 'accelerate' in cfg else 'NOT FOUND')"
```
If "NOT FOUND", run `bash fix_numpy.sh` to recompile numpy from source.

**Live preview opens but no face swap appears**

InsightFace's buffalo_l face detection models aren't installed (look in `~/.insightface/models/buffalo_l/` — should contain 5 `.onnx` files). Run:
```bash
bash fix_buffalo.sh
```

**`No module named 'modules.mps_inswapper'`**

The MPS bridge file got deleted. It should be at `modules/mps_inswapper.py`. Pull from git or restore from backup.

**`No module named 'torch'`**

PyTorch isn't installed in the parent venv. Run:
```bash
source .venv/bin/activate
pip install "torch==2.7.0" "torchvision==0.22.0" onnx2torch
```

**Camera permission denied**

System Settings → Privacy & Security → Camera → enable **Terminal** (and **Python** if it appears separately) → relaunch the app.

**Stuck at `quote>` prompt in Terminal**

Press **Ctrl-C**. Bash got confused by an unclosed quote (often a smart-quote from copy-pasting). Then re-type the command manually.

**App is hung; can't close window**

In the Terminal that launched it, press **Ctrl-C**. If that doesn't work:
```bash
pkill -f "python run.py"
```

**Face swap is wrong / weird artifacts**

Try a different source face. Some Wikipedia photos have side-angle problems that produce ugly swaps. Run `python validate_faces.py` to see which sources are reliable.

---

## What's where

```
deep-live-cam-main/
├── READMEFIRST.md          ← this file
├── launch.sh               ← starts the demo
├── run.py                  ← the actual app entry point (called by launch.sh)
├── fix_setup.sh            ← one-time setup of the venv
├── fix_buffalo.sh          ← installs InsightFace face detection models
├── fix_numpy.sh            ← rebuilds numpy with Apple Accelerate
├── download_faces.py       ← scrapes Wikipedia for celebrity photos
├── validate_faces.py       ← preflight check, produces HTML report
├── diagnose.py             ← end-to-end pipeline test (used during debugging)
│
├── faces/                  ← the celebrity / historical figure source images
├── models/                 ← inswapper_128_fp16.onnx (the swap model, 265 MB)
├── modules/
│   ├── mps_inswapper.py    ← Apple GPU-backed swap (the big rewrite we did)
│   ├── face_analyser.py    ← face detection / landmarks / recognition
│   ├── ui.py               ← the customtkinter UI
│   └── processors/frame/face_swapper.py  ← face swap orchestration
│
├── snapshots/              ← (not used by upstream UI, ignore)
├── validation_report/      ← generated by validate_faces.py
├── validation_report.html  ← generated by validate_faces.py
│
├── .venv/                  ← parent project's Python environment
└── mps/                    ← engineering scratch from the PyTorch+MPS rewrite
                              (phase1_smoke.py, phase2_detection.py, etc.)
                              Don't delete — useful reference if anything breaks.
```

---

## How this is wired (architecture, in case something breaks)

The parent project (`deep-live-cam-main/`) was originally an `onnxruntime`-based app. Three things were broken on this Mac:

1. `onnxruntime`'s CoreML EP crashes the buffalo_l face detector with a shape-rank validation bug.
2. `onnxruntime`'s CPU path runs the inswapper at ~654 ms/call (broken — should be ~80 ms).
3. The launcher had a wiring bug calling `process_frame(display)` instead of `process_frame(source_face, display)`.

The fix replaces only the inswapper inference with PyTorch + MPS (Apple GPU), keeping everything else (face detection, alignment, paste-back) on the original code path.

**The key file is `modules/mps_inswapper.py`.** It defines:
- `MPSSession` — mimics `onnxruntime.InferenceSession` but runs on Apple GPU via PyTorch
- `get_mps_inswapper(model_path)` — factory that returns an `INSwapper` with the MPS session injected

`modules/processors/frame/face_swapper.py` was edited to call `get_mps_inswapper()` instead of `insightface.model_zoo.get_model()`. That's the only meaningful change to the parent project's logic.

Face detection still uses `onnxruntime` on CPU (forced via `face_analyser.py`) at ~30-60 ms/frame. Combined with ~80 ms swap, total is ~120-150 ms = 7-10 FPS.

If you ever want to push higher: there's a `mps/` subfolder with Phase 2 (`phase2_detection.py`) that proves the buffalo_l detector also runs at 12 ms on MPS. Wrapping that the same way `mps_inswapper.py` wraps the swap would push the demo to ~12-15 FPS. ~30 minutes of work, low risk.

---

## Family Day operational checklist

**Night before:**

- [ ] Run `python download_faces.py` to make sure all sources are downloaded
- [ ] Run `python validate_faces.py` and open `validation_report.html` — replace any red/yellow faces
- [ ] Do a full dry-run: `./launch.sh`, click a face, click Live, swap your face for ~30 seconds
- [ ] Check that camera permissions work (System Settings → Privacy & Security → Camera)
- [ ] Plug in the laptop charger (the demo will heat up the M3 Pro)
- [ ] Close all other apps to maximize available CPU
- [ ] Test the Start (photo mode) backup path in case Live is choppy

**At the event:**

- [ ] Open Terminal, `cd ~/Desktop/deep-live-cam-main`, `source .venv/bin/activate`, `./launch.sh`
- [ ] Pre-warm: pick one face and click Live for a few seconds (compiles MPS kernels) before the first kid sits down
- [ ] Position the camera so the kid's face is well-lit and roughly head-on
- [ ] Plain-ish background helps face detection. Avoid busy patterns or other faces in frame.

**After the event:**

- [ ] Quit the app (close window or Ctrl-C in the Terminal)
- [ ] If you used Photo mode (Start button) and saved any kid faces to disk, delete them:
  ```bash
  ls ~/Desktop/deep-live-cam-main/snapshots/
  rm ~/Desktop/deep-live-cam-main/snapshots/*
  ```
- [ ] Live mode does NOT auto-save anything (verified) — there's nothing to clean up if you only used Live

---

## Privacy notes

- **Live mode does not save any frames to disk.** The webcam feed streams in, gets swapped, displays in the preview window, and goes nowhere. Verified by grepping the source for `imwrite` / `save_frame` and finding no matches in the live preview path.
- **Photo / Start mode does save** — the swap result is written to wherever you point it. If you use Start mode at the event, choose an output location you'll clean up afterward.
- The fork's `launcher.py` (which we replaced with the upstream `run.py`) had a snapshot button that wrote to `snapshots/`. We don't use that launcher, but if you ever launch via `python launcher.py` instead of `./launch.sh`, that path would write files.

---

## When things go really wrong: clean reset

If everything is busted and you want to nuke it and start over:

```bash
cd ~/Desktop/deep-live-cam-main
rm -rf .venv .venv312
bash fix_setup.sh
bash fix_buffalo.sh
source .venv/bin/activate
pip install "torch==2.7.0" "torchvision==0.22.0" onnx2torch
bash fix_numpy.sh
./launch.sh
```

This rebuilds the entire Python environment from scratch. Takes ~10-15 minutes. The model files in `models/` and `~/.insightface/models/buffalo_l/` are preserved (no need to re-download). Your `faces/` folder is preserved.

---

## Useful commands cheat-sheet

| What | Command |
|---|---|
| Activate venv | `source .venv/bin/activate` |
| Launch demo | `./launch.sh` |
| Add new faces | `python download_faces.py` |
| Validate faces | `python validate_faces.py && open validation_report.html` |
| Kill stuck app | `pkill -f "python run.py"` |
| Verify numpy/Accelerate | `python -c "import numpy as np; print('OK' if 'accelerate' in str(np.show_config(mode='dicts')).lower() else 'BROKEN')"` |
| Verify MPS works | `python -c "import torch; print('MPS' if torch.backends.mps.is_available() else 'NO MPS')"` |
| Verify swap pipeline | `python diagnose.py` |
| Clear photo-mode outputs | `rm -f snapshots/*` |
