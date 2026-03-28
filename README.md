<h1 align="center">Deep-Live-Cam 2.1</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

---

## ⚡ Quick Start — Pick Your OS

> **Not sure where to start? Follow the guide for your operating system below.**

---

### 🪟 Windows (Dell / HP / Any PC)

**Requirements:**
- Windows 10 or 11
- Python 3.11 — [Download from python.org](https://www.python.org/downloads/release/python-3119/) ⚠️ Check **"Add Python to PATH"** during install
- Git — [Download from git-scm.com](https://git-scm.com/download/win)
- ffmpeg — run in PowerShell: `iex (irm ffmpeg.tc.ht)`

**Steps:**

```
1. Open a terminal (PowerShell or Command Prompt)
2. git clone https://github.com/foodnotfit/deep-live-cam.git
3. cd deep-live-cam
4. Double-click: setup_windows.bat   ← installs everything automatically
5. Double-click: launch.bat          ← starts the app
```

**GPU Acceleration (optional but faster):**

| Your GPU | File to run |
|----------|-------------|
| NVIDIA (GeForce, RTX, GTX) | `run-cuda.bat` |
| AMD (Radeon) | `run-directml.bat` |
| No GPU / Integrated | `launch.bat` |

> `setup_windows.bat` auto-detects your GPU and installs the right packages.

---

### 🍎 macOS (MacBook, iMac, Mac Mini)

**Requirements:**
- macOS 12 or later
- Homebrew — [brew.sh](https://brew.sh)
- Python 3.11 via Homebrew

**Steps:**

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install dependencies
brew install python@3.11 ffmpeg git

# 3. Clone the repo
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam

# 4. Set up virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Run the app
python run.py
```

**Camera Permission (macOS Sequoia 15+):**

On first run, macOS will ask for camera access. If it doesn't appear automatically:

1. Open **System Settings → Privacy & Security → Camera**
2. Click **`+`** → press **`⌘ Shift G`** → paste:
   ```
   /usr/local/Cellar/python@3.13/3.13.12_1/Frameworks/Python.framework/Versions/3.13/Resources/Python.app
   ```
3. Toggle it **ON**

> See `CAMERA_FIX.md` for full troubleshooting details.

**GPU Acceleration (Apple Silicon M1/M2/M3/M4):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-silicon==1.13.1
python run.py --execution-provider coreml
```

---

### 🐧 Linux

**Requirements:**
- Ubuntu 20.04+ or similar
- Python 3.10–3.12
- ffmpeg

**Steps:**

```bash
# Install system deps
sudo apt update && sudo apt install python3.11 python3.11-venv ffmpeg git -y

# Clone and set up
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
```

**GPU Acceleration (NVIDIA):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.21.0
python run.py --execution-provider cuda
```

---

## 📁 Models (Required)

The face swap model is required. `setup_windows.bat` downloads it automatically on Windows.

**For macOS/Linux** — download manually and place in the `models/` folder:

- [`inswapper_128_fp16.onnx`](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx) (~265MB) — **required**
- [`gfpgan-1024.onnx`](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx) (~350MB) — optional (face enhancement)

---

## 🎮 How to Use

**Live Webcam Mode (Face Swap in Real Time):**
1. Click **Select a face** — pick any image from the `faces/` folder (or add your own)
2. Click **Live** — your webcam will start
3. Your face is now swapped in real time!

**Image / Video Mode:**
1. Click **Select a face** — choose a source face
2. Click **Select a target** — choose an image or video
3. Click **Start** — output saves in the same folder as the target

---

## 🚀 Performance Tips

| Hardware | Speed | Recommendation |
|----------|-------|----------------|
| NVIDIA GPU | ⚡⚡⚡ Fast | Use `run-cuda.bat` on Windows |
| AMD GPU | ⚡⚡ Good | Use `run-directml.bat` on Windows |
| Apple Silicon (M1+) | ⚡⚡ Good | Use `--execution-provider coreml` |
| CPU only | ⚡ Slow | Works, just slower |

---

## ⚠️ Disclaimer

This software is designed as a creative AI tool for artists, content creators, and entertainment. 

- **Always obtain consent** before using someone else's likeness
- **Label deepfakes** when sharing online
- **Do not use** for fraud, deception, or non-consensual content
- Built-in filters block inappropriate/NSFW content

By using this software you agree to these terms.

---

## Features

### Mouth Mask — Retain natural mouth movement
<p align="center"><img src="media/ludwig.gif" alt="mouth-mask"></p>

### Face Mapping — Multiple faces simultaneously
<p align="center"><img src="media/streamers.gif" alt="face-mapping"></p>

### Movie Mode — Watch any movie as any face
<p align="center"><img src="media/movie.gif" alt="movie-mode"></p>

---

## Manual Installation (Advanced)

<details>
<summary>Click to expand — for developers and advanced users</summary>

### Prerequisites

- Python 3.11 (recommended)
- pip, git, ffmpeg
- [Visual Studio 2022 Build Tools (Windows only)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Clone

```bash
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam
```

### Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### GPU-Specific Installs

**NVIDIA CUDA:**
```bash
pip install onnxruntime-gpu==1.21.0
python run.py --execution-provider cuda
```

**AMD DirectML (Windows):**
```bash
pip install onnxruntime-directml==1.21.0
python run.py --execution-provider directml
```

**Apple CoreML (M1/M2/M3):**
```bash
pip install onnxruntime-silicon==1.13.1
python run.py --execution-provider coreml
```

**Intel OpenVINO:**
```bash
pip install onnxruntime-openvino==1.21.0
python run.py --execution-provider openvino
```

### Command Line Arguments

```
python run.py [options]

  -s SOURCE_PATH        source face image
  -t TARGET_PATH        target image or video
  -o OUTPUT_PATH        output file or directory
  --execution-provider  cpu | cuda | directml | coreml | openvino
  --many-faces          process every face in frame
  --mouth-mask          preserve original mouth movement
  --map-faces           map multiple source faces to targets
  --live-mirror         mirror webcam (selfie-style)
  --keep-fps            preserve original video FPS
  --keep-audio          preserve original audio
  --max-memory N        limit RAM usage (GB)
  -v, --version         show version
```

</details>

---

## Press

- [**Ars Technica**](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) — *"Deep-Live-Cam goes viral"*
- [**Yahoo! Tech**](https://www.yahoo.com/tech/ok-viral-ai-live-stream-080041056.html) — *"This viral AI live stream software is truly terrifying"*
- [**TechLinked (Linus Tech Tips)**](https://www.youtube.com/watch?v=wnCghLjqv3s&t=551s) — *"They do a pretty good job matching poses, expression and even lighting"*
- [**IShowSpeed**](https://youtu.be/JbUPRmXRUtE?t=3964) — *"What the F\*\*\*! This shit is crazy!"*

---

## Credits

- [ffmpeg](https://ffmpeg.org/) — video processing
- [deepinsight / insightface](https://github.com/deepinsight/insightface) — face detection & analysis (non-commercial research use only)
- [hacksider/deep-live-cam](https://github.com/hacksider/deep-live-cam) — upstream project
- [s0md3v/roop](https://github.com/s0md3v/roop) — original base
- All contributors ❤️

---

## Stars to the Moon 🚀

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
