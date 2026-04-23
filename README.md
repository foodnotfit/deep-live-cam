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
> 📄 Full step-by-step guide with troubleshooting: [INSTALL.md](INSTALL.md)

---

## 🖥️ Step 0 — Check Your GPU First (Do This Before Installing)

Knowing your GPU determines which version to run.

### Windows — How to Check Your GPU

**Option A — Task Manager (easiest):**
1. Press `Ctrl + Shift + Esc` to open Task Manager
2. Click the **Performance** tab
3. Look at the left panel for entries like:
   - `GPU 0 — NVIDIA GeForce RTX 3060` → use `run-cuda.bat`
   - `GPU 0 — AMD Radeon RX 6600` → use `run-directml.bat`
   - `GPU 0 — Intel UHD Graphics 620` → use `launch.bat`

**Option B — PowerShell (one command):**
```powershell
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM
```

**Option C — Device Manager:**
1. Press `Windows + X` → click **Device Manager**
2. Expand **Display adapters** — your GPU name is listed there

### macOS — How to Check Your GPU

**Option A — About This Mac:**
1. Click the Apple menu → **About This Mac**
2. Look for the **Graphics** or **Chip** row:
   - `Apple M1 / M2 / M3 / M4` → Apple Silicon → use CoreML (fast)
   - `AMD Radeon Pro` → Intel Mac → CPU mode
   - `Intel Iris / UHD` → integrated → CPU mode

**Option B — Terminal (one command):**
```bash
system_profiler SPDisplaysDataType | grep "Chipset Model"
```

### GPU → Launch Mode Reference

| GPU You See | Platform | What to Run |
|---|---|---|
| NVIDIA GeForce / RTX / GTX | Windows | `run-cuda.bat` |
| AMD Radeon | Windows | `run-directml.bat` |
| Intel HD / UHD / Iris | Windows | `launch.bat` |
| No dedicated GPU | Windows | `launch.bat` |
| Apple M1 / M2 / M3 / M4 | macOS | `python run.py --execution-provider coreml` |
| AMD Radeon (Intel Mac) | macOS | `python run.py` |
| Intel (Mac) | macOS | `python run.py` |

---

### 🪟 Windows — Step-by-Step Installation

#### Step 1 — Install Python 3.11

1. Go to: https://www.python.org/downloads/release/python-3119/
2. Scroll to the bottom → click **Windows installer (64-bit)**
3. Run the installer
4. ⚠️ **On the first screen, check "Add Python 3.11 to PATH"** before clicking Install Now — this is critical
5. Click **Install Now** and wait for it to finish → click **Close**

**Verify — open PowerShell and run:**
```powershell
python --version
```
Expected: `Python 3.11.x`

**If you see "not recognized" — fix PATH manually:**
1. Search Windows for **"Edit the system environment variables"**
2. Click **Environment Variables**
3. Under **User variables**, find `Path` → click **Edit**
4. Click **New** → add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\`
5. Click **New** again → add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\Scripts\`
6. Click OK on all windows, close and reopen PowerShell, try `python --version` again

#### Step 2 — Install Git

1. Go to: https://git-scm.com/download/win
2. Download starts automatically — run the installer
3. Click **Next** through all options (defaults are fine) → **Install** → **Finish**

**Verify:**
```powershell
git --version
```

#### Step 3 — Install ffmpeg

1. Open **PowerShell as Administrator** (press `Windows + X` → Terminal/PowerShell Admin)
2. Run:
```powershell
iex (irm ffmpeg.tc.ht)
```
3. Wait for it to complete → close and reopen PowerShell

**Verify:**
```powershell
ffmpeg -version
```

#### Step 4 — Clone the Repository

```powershell
cd C:\Users\YourUsername\Desktop
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam
```

#### Step 5 — Run Setup

1. Open **File Explorer** → navigate to the `deep-live-cam` folder
2. Double-click **`setup_windows.bat`**
3. A terminal opens and runs automatically — this will:
   - Detect your GPU
   - Create a Python virtual environment
   - Install all required packages
   - Download the face swap AI model (~265MB)
4. Wait for completion (5–15 min depending on internet speed)

#### Step 6 — Launch the App

| GPU | Launch file |
|---|---|
| NVIDIA (RTX / GTX / GeForce) | Double-click `run-cuda.bat` |
| AMD Radeon | Double-click `run-directml.bat` |
| Intel / No GPU | Double-click `launch.bat` |

---

### 🍎 macOS — Step-by-Step Installation

#### Step 1 — Install Homebrew

1. Open **Terminal** (`Cmd + Space` → type Terminal → Enter)
2. Paste and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Enter your password when prompted
4. **Apple Silicon only** — after install, run these two lines to add Homebrew to PATH:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**Verify:**
```bash
brew --version
```

#### Step 2 — Install Python 3.11, ffmpeg, and Git

```bash
brew install python@3.11 ffmpeg git
```

**Add Python 3.11 to PATH so it's used by default:**
```bash
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Verify all three:**
```bash
python3.11 --version
ffmpeg -version
git --version
```

#### Step 3 — Clone the Repository

```bash
cd ~/Desktop
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam
```

#### Step 4 — Create Virtual Environment and Install Packages

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Step 5 — Download the Face Model

```bash
mkdir -p models
curl -L "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx" -o models/inswapper_128_fp16.onnx
```
~265MB — wait for the download to complete.

#### Step 6 — Launch the App

**Apple Silicon (M1/M2/M3/M4):**
```bash
pip uninstall onnxruntime -y
pip install onnxruntime-silicon==1.13.1
python run.py --execution-provider coreml
```

**Intel Mac:**
```bash
python run.py
```

#### Step 7 — Camera Permission (macOS only)

On first run, macOS will ask for camera access. If the prompt doesn't appear:
1. Go to **System Settings → Privacy & Security → Camera**
2. Click **+** → press `Cmd + Shift + G` → paste the Python path shown in your terminal
3. Toggle it **ON**

> See `CAMERA_FIX.md` for full troubleshooting details.
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
