#!/usr/bin/env python3
"""
Preflight check for the faces/ folder.

Walks every image in faces/ and answers: will this face actually work as
a source in the live demo? Many "good-looking" portraits silently fail
because they're too low-resolution, the face is profile-angled, or
something obscures key landmarks.

For each candidate face this script:
  1. Loads the image and runs face detection (insightface CPU)
  2. If a face is detected, runs the actual swap pipeline (same code
     path as live mode) onto a fixed target image
  3. Records: source thumb, swap result, status, and any warnings

Output: validation_report.html in this folder. Open it in any browser
to see a side-by-side grid of all faces — green for OK, yellow for
warnings, red for failures. Each row also includes the swap result
image so you can see what kids will actually get.

Run from the parent project's venv (the one with torch + insightface).
"""

import os
import sys
import time
import html as _html
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FACES_DIR = HERE / "faces"
REPORT_DIR = HERE / "validation_report"
REPORT_HTML = HERE / "validation_report.html"
INSWAPPER_PATH = HERE / "models" / "inswapper_128_fp16.onnx"

# Use a fixed target so every face is swapped onto the same person.
# obama.jpg is a clean, well-lit, front-facing reference — closest to
# what a kid sitting at the webcam will look like.
TARGET_IMAGE = FACES_DIR / "obama.jpg"

print("=" * 60)
print(f" Validating every source face in {FACES_DIR}")
print("=" * 60)

# Pre-flight
if not TARGET_IMAGE.exists():
    print(f"ERROR: test target {TARGET_IMAGE} not found.")
    sys.exit(1)
if not INSWAPPER_PATH.exists():
    print(f"ERROR: inswapper model {INSWAPPER_PATH} not found.")
    sys.exit(1)

REPORT_DIR.mkdir(exist_ok=True)
# Clear any old report images
for old in REPORT_DIR.glob("*.jpg"):
    old.unlink()

# --- Load pipeline ---------------------------------------------------
print("\n[1/3] Loading swap pipeline (this takes ~10s the first time)...")
import insightface
fa = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition", "landmark_2d_106"],
)
fa.prepare(ctx_id=0, det_size=(320, 320))

# Use the same MPS-backed inswapper the live demo uses
sys.path.insert(0, str(HERE))
from modules.mps_inswapper import get_mps_inswapper
swapper = get_mps_inswapper(str(INSWAPPER_PATH))

# Detect target face once (it's the same for every test)
target_img = cv2.imread(str(TARGET_IMAGE))
target_faces = fa.get(target_img)
if not target_faces:
    print(f"ERROR: no face detected in test target {TARGET_IMAGE}")
    sys.exit(1)
target_face = target_faces[0]
print(f"  target: {TARGET_IMAGE.name} ({target_img.shape[1]}x{target_img.shape[0]})")

# --- Walk faces/ and validate each ----------------------------------
print("\n[2/3] Validating each source face...")
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
candidates = sorted(p for p in FACES_DIR.iterdir() if p.suffix.lower() in exts)

results = []
for src_path in candidates:
    name = src_path.stem
    entry = {"name": name, "src_path": src_path, "status": None,
             "warnings": [], "result_path": None, "src_size": None}

    src = cv2.imread(str(src_path))
    if src is None:
        entry["status"] = "fail"
        entry["warnings"].append("Could not read image (corrupt or unsupported format)")
        results.append(entry)
        print(f"  [FAIL ] {name:25s} unreadable")
        continue

    h, w = src.shape[:2]
    entry["src_size"] = (w, h)
    if min(w, h) < 200:
        entry["warnings"].append(f"Low resolution ({w}x{h}) — face crop will be blurry")

    src_faces = fa.get(src)
    if not src_faces:
        entry["status"] = "fail"
        entry["warnings"].append("No face detected — try a more front-facing photo")
        results.append(entry)
        print(f"  [FAIL ] {name:25s} no face detected")
        continue

    if len(src_faces) > 1:
        entry["warnings"].append(f"{len(src_faces)} faces detected — using largest")

    # Sort by face area, take largest
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    try:
        result = swapper.get(target_img, target_face, src_face, paste_back=True)
        # Save a side-by-side preview: source thumb, swap result
        result_path = REPORT_DIR / f"{name}_result.jpg"
        # Resize source to match result height for the side-by-side
        src_resized = cv2.resize(src, (int(src.shape[1] * 256 / src.shape[0]), 256))
        result_resized = cv2.resize(result, (int(result.shape[1] * 256 / result.shape[0]), 256))
        side_by_side = np.concatenate([src_resized, result_resized], axis=1)
        cv2.imwrite(str(result_path), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 80])
        entry["result_path"] = result_path
        entry["status"] = "ok" if not entry["warnings"] else "warn"
        symbol = "OK   " if entry["status"] == "ok" else "WARN "
        warn_str = f" ({'; '.join(entry['warnings'])})" if entry["warnings"] else ""
        print(f"  [{symbol}] {name:25s} {entry['src_size']}{warn_str}")
    except Exception as e:
        entry["status"] = "fail"
        entry["warnings"].append(f"Swap failed: {type(e).__name__}: {e}")
        results.append(entry)
        print(f"  [FAIL ] {name:25s} {type(e).__name__}: {str(e)[:50]}")
        continue

    results.append(entry)

# --- Generate HTML report -------------------------------------------
print(f"\n[3/3] Writing report to {REPORT_HTML}...")

ok_count = sum(1 for r in results if r["status"] == "ok")
warn_count = sum(1 for r in results if r["status"] == "warn")
fail_count = sum(1 for r in results if r["status"] == "fail")

# Sort: failures first, then warnings, then ok (so problems are at the top)
results.sort(key=lambda r: {"fail": 0, "warn": 1, "ok": 2}[r["status"]])

def cell_for(r):
    src_rel = os.path.relpath(r["src_path"], HERE)
    result_html = ""
    if r["result_path"]:
        result_rel = os.path.relpath(r["result_path"], HERE)
        result_html = (f'<img src="{_html.escape(result_rel)}" '
                       f'alt="swap of {_html.escape(r["name"])}">')
    else:
        result_html = '<div class="no-result">no swap (face not detected)</div>'

    bg_class = {"ok": "row-ok", "warn": "row-warn", "fail": "row-fail"}[r["status"]]
    badge = {"ok": "✓ OK", "warn": "⚠ WARN", "fail": "✗ FAIL"}[r["status"]]
    warn_list = "<br>".join(_html.escape(w) for w in r["warnings"])
    size = f"{r['src_size'][0]}x{r['src_size'][1]}" if r["src_size"] else "?"

    return f"""
    <tr class="{bg_class}">
      <td class="badge">{badge}</td>
      <td class="name">{_html.escape(r["name"])}<br><small>{size}</small></td>
      <td class="result">{result_html}</td>
      <td class="warnings">{warn_list or "&mdash;"}</td>
    </tr>
    """

rows = "".join(cell_for(r) for r in results)
total = len(results)
target_rel = os.path.relpath(TARGET_IMAGE, HERE)

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Face validation report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3; padding: 20px; }}
  h1 {{ color: #39d353; }}
  .stats {{ font-size: 14px; color: #8b949e; margin-bottom: 20px; }}
  .stats span {{ margin-right: 18px; }}
  .stats .ok {{ color: #39d353; }}
  .stats .warn {{ color: #d29922; }}
  .stats .fail {{ color: #f85149; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #30363d; vertical-align: middle; }}
  th {{ background: #161b22; }}
  td.badge {{ font-weight: bold; white-space: nowrap; }}
  tr.row-ok td.badge {{ color: #39d353; }}
  tr.row-warn td.badge {{ color: #d29922; }}
  tr.row-fail td.badge {{ color: #f85149; }}
  td.name {{ width: 180px; }}
  td.name small {{ color: #8b949e; }}
  td.result img {{ height: 256px; border-radius: 4px; }}
  td.result .no-result {{ color: #8b949e; font-style: italic; padding: 20px; }}
  td.warnings {{ width: 280px; color: #8b949e; font-size: 13px; }}
</style></head>
<body>
  <h1>Face validation report</h1>
  <div class="stats">
    <span>Total: <b>{total}</b></span>
    <span class="ok">OK: <b>{ok_count}</b></span>
    <span class="warn">Warnings: <b>{warn_count}</b></span>
    <span class="fail">Failures: <b>{fail_count}</b></span>
    <span>Target: {_html.escape(target_rel)}</span>
  </div>
  <table>
    <thead>
      <tr><th>Status</th><th>Source</th><th>Source &rarr; Swap result</th><th>Notes</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body></html>
"""

REPORT_HTML.write_text(html_doc)

print()
print("=" * 60)
print(f" Done. {ok_count} OK, {warn_count} warnings, {fail_count} failures.")
print(f" Open the report:")
print(f"   open {REPORT_HTML}")
print("=" * 60)
