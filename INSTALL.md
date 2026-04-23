# Deep Live Cam — Step-by-Step Installation Guide

---

## STEP 0 — Check Your GPU First (Before Installing Anything)

Knowing your GPU determines which version to run. Here's how to check on each OS.

### Windows — Check GPU

**Option A: Task Manager**
1. Press `Ctrl + Shift + Esc` to open Task Manager
2. Click the **Performance** tab
3. Look at the left panel — you'll see entries like:
   - `GPU 0 — NVIDIA GeForce RTX 3060` → NVIDIA → use `run-cuda.bat`
   - `GPU 0 — AMD Radeon RX 6600` → AMD → use `run-directml.bat`
   - `GPU 0 — Intel(R) UHD Graphics 620` → Intel integrated → use `launch.bat`

**Option B: Device Manager**
1. Press `Windows + X` → click **Device Manager**
2. Expand **Display adapters**
3. You'll see your GPU name listed there

**Option C: PowerShell (one command)**
```powershell
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM
```
Run in PowerShell — it prints your GPU name and VRAM.

---

### macOS — Check GPU

**Option A: About This Mac**
1. Click the  Apple menu (top left)
2. Click **About This Mac**
3. Look for the **Graphics** row — e.g.:
   - `Apple M1 Pro` → Apple Silicon → use CoreML
   - `AMD Radeon Pro 5500M` → AMD (Intel Mac) → CPU mode
   - `Intel Iris Plus Graphics` → integrated → CPU mode

**Option B: Terminal (one command)**
```bash
system_profiler SPDisplaysDataType | grep "Chipset Model"
```

---

## GPU → Launch Mode Reference

| GPU You See | Platform | What to Run |
|---|---|---|
| NVIDIA GeForce / RTX / GTX | Windows | `run-cuda.bat` |
| AMD Radeon | Windows | `run-directml.bat` |
| Intel HD / UHD / Iris | Windows | `launch.bat` |
| No dedicated GPU | Windows | `launch.bat` |
| Apple M1 / M2 / M3 / M4 | macOS | `--execution-provider coreml` |
| AMD Radeon (Intel Mac) | macOS | `python run.py` (CPU) |
| Intel (Mac) | macOS | `python run.py` (CPU) |

---

---

# WINDOWS — Full Installation (Step by Step)

---

## Step 1 — Install Python 3.11

1. Go to: https://www.python.org/downloads/release/python-3119/
2. Scroll down to **Files** at the bottom
3. Click **Windows installer (64-bit)** to download
4. Run the installer
5. ⚠️ **CRITICAL: On the first screen, check the box that says "Add Python 3.11 to PATH"** before clicking Install Now
   - If you miss this, Python won't work from the command line
6. Click **Install Now**
7. Wait for it to finish, then click **Close**

**Verify it worked — open PowerShell and run:**
```powershell
python --version
```
You should see: `Python 3.11.x`

If you see "not recognized" — Python wasn't added to PATH. Fix:
1. Search Windows for **"Edit the system environment variables"**
2. Click **Environment Variables**
3. Under **User variables**, find `Path` and click **Edit**
4. Click **New** and add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\`
5. Click **New** again and add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\Scripts\`
6. Click OK on all windows
7. Close and reopen PowerShell, try `python --version` again

---

## Step 2 — Install Git

1. Go to: https://git-scm.com/download/win
2. The download starts automatically — run the installer
3. Click **Next** through all the options (defaults are fine)
4. Click **Install**, then **Finish**

**Verify:**
```powershell
git --version
```
Should show: `git version 2.x.x`

---

## Step 3 — Install ffmpeg

1. Open **PowerShell as Administrator**:
   - Press `Windows + X`
   - Click **Windows PowerShell (Admin)** or **Terminal (Admin)**
2. Paste and run this command:
```powershell
iex (irm ffmpeg.tc.ht)
```
3. It will download and install ffmpeg automatically and add it to PATH
4. Close and reopen PowerShell when done

**Verify:**
```powershell
ffmpeg -version
```
Should show ffmpeg version info.

---

## Step 4 — Clone the Repository

1. Open PowerShell (regular, not admin)
2. Navigate to where you want to install it. Example — put it on your Desktop:
```powershell
cd C:\Users\YourUsername\Desktop
```
3. Clone the repo:
```powershell
git clone https://github.com/foodnotfit/deep-live-cam.git
```
4. Move into the folder:
```powershell
cd deep-live-cam
```

---

## Step 5 — Run Setup

1. Open **File Explorer** and navigate to the `deep-live-cam` folder
2. Double-click **`setup_windows.bat`**
3. A terminal window will open and run automatically — this will:
   - Detect your GPU
   - Create a Python virtual environment
   - Install all required packages
   - Download the face swap AI model (~265MB)
4. Wait for it to complete — this may take 5-15 minutes depending on your internet speed
5. When done, the window will say setup is complete

---

## Step 6 — Launch the App

Based on your GPU (from Step 0):

- **NVIDIA GPU** → double-click **`run-cuda.bat`**
- **AMD GPU** → double-click **`run-directml.bat`**
- **No GPU / Intel** → double-click **`launch.bat`**

The app UI will open in a window.

---

## Step 7 — Using the App

**Live Webcam Mode (real-time face swap):**
1. Click **Select a face** → choose any image from the `faces/` folder (or add your own photo)
2. Click **Live** → your webcam starts
3. Your face is swapped in real time

**Image / Video Mode:**
1. Click **Select a face** → choose source face image
2. Click **Select a target** → choose an image or video file
3. Click **Start** → output saves in the same folder as the target file

---

---

# macOS — Full Installation (Step by Step)

---

## Step 1 — Install Homebrew (if not installed)

1. Open **Terminal** (press `Cmd + Space`, type Terminal, press Enter)
2. Paste and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Follow the prompts — it will ask for your password
4. When done, if you're on Apple Silicon (M1/M2/M3/M4), run the two lines it shows at the end to add Homebrew to PATH:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**Verify:**
```bash
brew --version
```

---

## Step 2 — Install Python 3.11, ffmpeg, and Git

Run all three in one command:
```bash
brew install python@3.11 ffmpeg git
```
Wait for it to complete (may take a few minutes).

**Add Python 3.11 to PATH (so it's used by default):**
```bash
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Verify:**
```bash
python3.11 --version
ffmpeg -version
git --version
```
All three should return version info.

---

## Step 3 — Clone the Repository

```bash
cd ~/Desktop
git clone https://github.com/foodnotfit/deep-live-cam.git
cd deep-live-cam
```

---

## Step 4 — Create Virtual Environment and Install Packages

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
This installs all dependencies. May take a few minutes.

---

## Step 5 — Download the Face Model

The model file is required. Download it and place it in the `models/` folder:

1. Download: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx (~265MB)
2. Move the file to `deep-live-cam/models/inswapper_128_fp16.onnx`

Or via terminal:
```bash
mkdir -p models
curl -L "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx" -o models/inswapper_128_fp16.onnx
```

---

## Step 6 — Launch the App

**Apple Silicon (M1/M2/M3/M4):**
```bash
pip uninstall onnxruntime -y
pip install onnxruntime-silicon==1.13.1
python run.py --execution-provider coreml
```

**Intel Mac (AMD or Intel GPU):**
```bash
python run.py
```

---

## Step 7 — Camera Permission (macOS only)

On first run, macOS may ask for camera access. If the prompt doesn't appear:
1. Go to **System Settings → Privacy & Security → Camera**
2. Click **+**
3. Press `Cmd + Shift + G` and paste the Python path shown in your terminal
4. Toggle it **ON**

---

## Common Errors

| Error | Fix |
|---|---|
| `python not recognized` | Re-check PATH setup in Step 1 |
| `pip not found` | Run `python -m ensurepip` |
| `No module named cv2` | Re-run `pip install -r requirements.txt` |
| `Model file not found` | Make sure `inswapper_128_fp16.onnx` is in the `models/` folder |
| Camera not detected | Check camera permissions (macOS) or try a different USB port |
| Very slow (CPU only) | Normal without GPU — consider using a machine with NVIDIA GPU |
