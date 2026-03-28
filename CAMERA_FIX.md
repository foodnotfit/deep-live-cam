# Camera Fix — macOS Permission Issue

## Root Cause

The venv Python binary (`deep-live-cam/.venv/bin/python3.13`) is **not code-signed**.

On macOS, TCC (Transparency, Consent, and Control) — the camera permission system — **requires a signed binary** to associate a permission grant with an app. When Python is unsigned, macOS either:
- Returns `kTCCAuthorizationStatusNotDetermined` (status 0, silently requesting)
- Or silently denies the request without showing the system camera permission dialog

OpenCV reports: `not authorized to capture video (status 0), requesting...`

This is why the camera never works even if you've granted permission in System Settings — the permission is associated with the wrong binary (or no binary at all).

## Quick Fix

Run this **once** to sign the venv Python binary with camera entitlements:

```bash
cd ~/Desktop/deep-live-cam

# Create entitlements file
cat > /tmp/camera_entitlements.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.camera</key>
    <true/>
</dict>
</plist>
EOF

# Sign the venv python binary
codesign --sign - --entitlements /tmp/camera_entitlements.plist --force \
    .venv/bin/python3.13

echo "Done. Now launch deep-live-cam — macOS will prompt for camera permission."
```

After signing, the first launch will show the macOS camera permission dialog. Grant it, and it will work permanently.

## Why This Happens

- Homebrew Python 3.13 (`/usr/local/opt/python@3.13/bin/python3.13`) is also unsigned
- When `venv` creates a new environment, it copies/symlinks the unsigned binary
- Result: no signing identity → no camera permission dialog → always denied

## Verify It Worked

```bash
cd ~/Desktop/deep-live-cam

# Should now show entitlements
codesign -d --entitlements - .venv/bin/python3.13

# Should open camera
.venv/bin/python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
cap.release()
"
```

## Alternative: Install cv2_enumerate_cameras

The code in `modules/ui.py` already handles `cv2_enumerate_cameras` for safer camera detection (avoids SIGSEGV from probing bad indices on macOS). Install it too:

```bash
cd ~/Desktop/deep-live-cam
.venv/bin/pip install cv2-enumerate-cameras
```

## Test Suite

Run unit tests (no camera hardware needed):
```bash
cd ~/Desktop/deep-live-cam
.venv/bin/python -m pytest tests/test_video_capture.py -v
```
