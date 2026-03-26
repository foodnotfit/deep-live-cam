"""
🎭 DeepFake Lab — Family Edition
A kid-friendly Tkinter launcher for deep-live-cam.
"""

import os
import sys
import threading
import time
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ── ensure project root on path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── optional heavy imports (handled gracefully) ───────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── colour / style constants ──────────────────────────────────────────────────
BG          = "#0d1117"
PANEL       = "#161b22"
ACCENT      = "#39d353"
BORDER_SEL  = ACCENT
BORDER_OFF  = "#30363d"
TEXT_MAIN   = "#e6edf3"
TEXT_DIM    = "#8b949e"
THUMB_SIZE  = 120
CAM_W, CAM_H = 640, 480

FONT_MONO   = ("Courier", 11)
FONT_MONO_B = ("Courier", 13, "bold")
FONT_BODY   = ("Helvetica", 11)
FONT_BODY_B = ("Helvetica", 13, "bold")
FONT_TITLE  = ("Helvetica", 16, "bold")
FONT_SMALL  = ("Courier", 9)

FACES_DIR     = os.path.join(ROOT, "faces")
SNAPSHOTS_DIR = os.path.join(ROOT, "snapshots")

# ── ensure directories exist ──────────────────────────────────────────────────
os.makedirs(FACES_DIR,     exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class DeepFakeLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎭 DeepFake Lab — Family Edition")
        self.geometry("1100x700")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── state ─────────────────────────────────────────────────────────────
        self.selected_face_path: str | None = None
        self.selected_face_name: str | None = None
        self.camera_running   = False
        self.camera_thread    = None
        self.current_frame    = None          # numpy BGR frame
        self.tk_image         = None          # keep reference alive
        self.face_swapper_loaded = False
        self.face_swapper_loading = False
        self.camera_index     = tk.IntVar(value=0)
        self.step_active      = [False, False, False, False]  # bottom bar steps
        self._thumb_refs: list = []           # prevent GC of PhotoImages

        # ── layout ────────────────────────────────────────────────────────────
        self._build_ui()
        self._populate_face_gallery()
        self._populate_camera_list()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        # outer container
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # split: left gallery + right camera
        content = tk.Frame(outer, bg=BG)
        content.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(content)
        self._build_right_panel(content)
        self._build_bottom_bar(outer)

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, width=300, bd=0,
                         highlightbackground=BORDER_OFF, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        frame.pack_propagate(False)

        # title
        tk.Label(frame, text="PICK A FACE", font=FONT_TITLE,
                 bg=PANEL, fg=ACCENT).pack(pady=(12, 6))

        # scrollable thumbnail grid
        scroll_container = tk.Frame(frame, bg=PANEL)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=4)

        self.gallery_canvas = tk.Canvas(scroll_container, bg=PANEL,
                                        highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL,
                                   command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_frame = tk.Frame(self.gallery_canvas, bg=PANEL)
        self.gallery_canvas_window = self.gallery_canvas.create_window(
            (0, 0), window=self.gallery_frame, anchor="nw"
        )
        self.gallery_frame.bind("<Configure>", self._on_gallery_configure)
        self.gallery_canvas.bind("<Configure>", self._on_canvas_configure)

        # mouse-wheel scroll
        self.gallery_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.gallery_canvas.bind("<Button-4>",   self._on_mousewheel)
        self.gallery_canvas.bind("<Button-5>",   self._on_mousewheel)

        # selected face label
        self.selected_label = tk.Label(
            frame, text="None selected", font=FONT_SMALL,
            bg=PANEL, fg=TEXT_DIM, wraplength=260
        )
        self.selected_label.pack(pady=(4, 4))

        # add your own button
        add_btn = tk.Button(
            frame, text="➕  Add Your Own",
            font=FONT_BODY_B, bg=ACCENT, fg=BG,
            activebackground="#2ea843", relief=tk.FLAT, cursor="hand2",
            command=self._add_custom_face
        )
        add_btn.pack(fill=tk.X, padx=12, pady=(0, 12))

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0,
                         highlightbackground=BORDER_OFF, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(frame, text="DEEPFAKE LIVE", font=FONT_TITLE,
                 bg=PANEL, fg=ACCENT).pack(pady=(12, 6))

        # camera canvas
        self.cam_canvas = tk.Canvas(frame, width=CAM_W, height=CAM_H,
                                    bg="#000000", highlightthickness=0)
        self.cam_canvas.pack()
        self._draw_placeholder("No camera  —  press  ▶ START CAMERA")

        # toolbar
        toolbar = tk.Frame(frame, bg=PANEL)
        toolbar.pack(fill=tk.X, padx=12, pady=8)

        self.start_btn = tk.Button(
            toolbar, text="▶  START CAMERA",
            font=FONT_BODY_B, bg=ACCENT, fg=BG,
            activebackground="#2ea843", relief=tk.FLAT, cursor="hand2",
            command=self._toggle_camera, width=16
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.snap_btn = tk.Button(
            toolbar, text="📸  SNAPSHOT",
            font=FONT_BODY_B, bg="#1f6feb", fg=TEXT_MAIN,
            activebackground="#1a5fcd", relief=tk.FLAT, cursor="hand2",
            command=self._take_snapshot, state=tk.DISABLED
        )
        self.snap_btn.pack(side=tk.LEFT, padx=(0, 8))

        # camera selector
        tk.Label(toolbar, text="Cam:", font=FONT_BODY,
                 bg=PANEL, fg=TEXT_DIM).pack(side=tk.LEFT)
        self.cam_dropdown = ttk.Combobox(
            toolbar, textvariable=self.camera_index,
            state="readonly", width=6, font=FONT_BODY
        )
        self.cam_dropdown.pack(side=tk.LEFT, padx=(4, 0))
        self.cam_dropdown.bind("<<ComboboxSelected>>", self._on_camera_change)

        # status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            frame, textvariable=self.status_var,
            font=FONT_MONO, bg="#0a0f15", fg=ACCENT,
            anchor="w", padx=10
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── BOTTOM BAR ────────────────────────────────────────────────────────────
    def _build_bottom_bar(self, parent):
        bar = tk.Frame(parent, bg="#0a0f15", height=40)
        bar.pack(fill=tk.X, pady=(6, 0))
        bar.pack_propagate(False)

        steps = [
            ("1. AI detects your face", 0),
            ("2. Maps 68 facial landmarks", 1),
            ("3. Warps selected face to match", 2),
            ("4. Blends seamlessly onto your head", 3),
        ]

        tk.Label(bar, text="HOW DEEPFAKES WORK: ",
                 font=FONT_SMALL, bg="#0a0f15", fg=TEXT_DIM).pack(side=tk.LEFT, padx=(8, 0))

        self.step_labels = []
        for i, (text, idx) in enumerate(steps):
            lbl = tk.Label(bar, text=text, font=FONT_SMALL,
                           bg="#0a0f15", fg=TEXT_DIM)
            lbl.pack(side=tk.LEFT, padx=4)
            self.step_labels.append(lbl)
            if i < len(steps) - 1:
                tk.Label(bar, text="→", font=FONT_SMALL,
                         bg="#0a0f15", fg=TEXT_DIM).pack(side=tk.LEFT)

    # =========================================================================
    # Gallery helpers
    # =========================================================================

    def _populate_face_gallery(self):
        """Load face images from ./faces/ and display as thumbnails."""
        # clear old widgets
        for w in self.gallery_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        face_files = sorted(
            p for p in Path(FACES_DIR).iterdir()
            if p.suffix.lower() in exts
        )

        cols = 2
        for i, fpath in enumerate(face_files):
            row, col = divmod(i, cols)
            self._add_thumb_cell(fpath, row, col)

        if not face_files:
            tk.Label(self.gallery_frame, text="No faces found\nin ./faces/",
                     font=FONT_BODY, bg=PANEL, fg=TEXT_DIM).grid(
                row=0, column=0, columnspan=2, padx=20, pady=40)

        self.gallery_canvas.update_idletasks()

    def _add_thumb_cell(self, fpath: Path, row: int, col: int):
        name = fpath.stem.replace("_", " ").replace("-", " ").title()

        cell = tk.Frame(self.gallery_frame, bg=PANEL, pady=4, padx=4)
        cell.grid(row=row, column=col, padx=4, pady=4)

        # border frame that we recolour on select
        border = tk.Frame(cell, bg=BORDER_OFF, bd=2)
        border.pack()

        img_label = tk.Label(border, bg=PANEL, cursor="hand2")
        img_label.pack()

        # load thumbnail
        photo = self._load_thumb(str(fpath))
        if photo:
            img_label.configure(image=photo)
            img_label.image = photo          # keep ref on widget too
            self._thumb_refs.append(photo)
        else:
            img_label.configure(text="🖼", font=("Helvetica", 36),
                                 bg=PANEL, fg=TEXT_DIM,
                                 width=THUMB_SIZE // 12, height=THUMB_SIZE // 24)

        tk.Label(cell, text=name, font=FONT_SMALL,
                 bg=PANEL, fg=TEXT_MAIN,
                 wraplength=THUMB_SIZE + 8).pack()

        # click binding
        def on_click(path=str(fpath), n=name, b=border):
            self._select_face(path, n, b)

        for w in (border, img_label):
            w.bind("<Button-1>", lambda e, f=on_click: f())

    def _load_thumb(self, path: str):
        if not PIL_OK:
            return None
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            # pad to square
            square = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (22, 27, 34))
            offset = ((THUMB_SIZE - img.width) // 2,
                      (THUMB_SIZE - img.height) // 2)
            square.paste(img, offset)
            return ImageTk.PhotoImage(square)
        except Exception:
            return None

    def _select_face(self, path: str, name: str, border_frame: tk.Frame):
        """Highlight selected face, deselect others."""
        # reset all borders
        for cell in self.gallery_frame.winfo_children():
            for child in cell.winfo_children():
                if isinstance(child, tk.Frame):
                    child.configure(bg=BORDER_OFF)

        border_frame.configure(bg=BORDER_SEL)
        self.selected_face_path = path
        self.selected_face_name = name
        self.selected_label.configure(
            text=f"Selected: {name}", fg=ACCENT)

        # update globals for face swapper
        try:
            import modules.globals as gm
            gm.source_path = path
        except Exception:
            pass

        self._set_status(f"Face selected: {name}")

    def _add_custom_face(self):
        path = filedialog.askopenfilename(
            title="Pick a face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        import shutil
        dest = os.path.join(FACES_DIR, os.path.basename(path))
        if path != dest:
            shutil.copy2(path, dest)
        self._populate_face_gallery()

    # ── gallery scroll callbacks ──────────────────────────────────────────────
    def _on_gallery_configure(self, event):
        self.gallery_canvas.configure(
            scrollregion=self.gallery_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.gallery_canvas.itemconfig(
            self.gallery_canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.gallery_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.gallery_canvas.yview_scroll(1, "units")
        else:
            self.gallery_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # =========================================================================
    # Camera helpers
    # =========================================================================

    def _populate_camera_list(self):
        if not CV2_OK:
            self.cam_dropdown["values"] = [0]
            self.cam_dropdown.current(0)
            return
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        if not cams:
            cams = [0]
        self.cam_dropdown["values"] = cams
        self.cam_dropdown.current(0)

    def _toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if not CV2_OK:
            self._set_status("OpenCV not installed — cannot start camera")
            return
        self.camera_running = True
        self.start_btn.configure(text="⏹  STOP", bg="#da3633")
        self.snap_btn.configure(state=tk.NORMAL)
        self._set_status("Starting camera…")
        self.camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _stop_camera(self):
        self.camera_running = False
        self.start_btn.configure(text="▶  START CAMERA", bg=ACCENT, fg=BG)
        self.snap_btn.configure(state=tk.DISABLED)
        self._set_status("Camera stopped")
        self._draw_placeholder("Camera stopped  —  press  ▶ START CAMERA")
        self._reset_steps()

    def _on_camera_change(self, event=None):
        if self.camera_running:
            self._stop_camera()

    # =========================================================================
    # Camera loop (daemon thread)
    # =========================================================================

    def _camera_loop(self):
        idx = self.camera_index.get()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self.after(0, lambda: self._set_status(
                "⚠ Camera access denied — go to System Settings → Privacy & Security → Camera → enable Terminal"))
            self.after(0, lambda: self._draw_placeholder(
                "📷 Camera Permission Required\n\n"
                "System Settings → Privacy & Security\n"
                "→ Camera → enable Terminal"))
            self.after(0, self._stop_camera)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        self.after(0, lambda: self._set_status("Camera running…"))
        process_frame_fn = None   # lazy load

        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            display = frame.copy()

            if self.selected_face_path:
                # ── lazy load face swapper ────────────────────────────────────
                if not self.face_swapper_loaded and not self.face_swapper_loading:
                    self.face_swapper_loading = True
                    self.after(0, lambda: self._set_status(
                        "Loading model… (first time ~30s)"))
                    try:
                        self._init_globals()
                        from modules.processors.frame.face_swapper import (
                            pre_check, process_frame as _pf
                        )
                        pre_check()
                        process_frame_fn = _pf
                        self.face_swapper_loaded = True
                        self.face_swapper_loading = False
                        self.after(0, lambda: self._set_status(
                            f"Swapping: {self.selected_face_name}"))
                    except Exception as ex:
                        self.face_swapper_loading = False
                        self.after(0, lambda e=ex: self._set_status(
                            f"Swap load error: {e}"))

                # ── run swap ─────────────────────────────────────────────────
                if self.face_swapper_loaded and process_frame_fn:
                    try:
                        self._init_globals()
                        self._highlight_step(0)  # face detected
                        self._highlight_step(1)  # landmarks
                        swapped = process_frame_fn(display)
                        if swapped is not None:
                            display = swapped
                            self._highlight_step(2)  # warp
                            self._highlight_step(3)  # blend
                    except Exception:
                        pass  # fall back to raw feed silently
                    self.after(0, lambda n=self.selected_face_name:
                               self._set_status(f"Swapping: {n}"))
            else:
                self.after(0, lambda: self._set_status("No face selected"))
                self._reset_steps()

            # convert BGR → RGB → PhotoImage
            self.current_frame = display.copy()
            if PIL_OK:
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                pil_img = Image.fromarray(rgb)
                pil_img = pil_img.resize((CAM_W, CAM_H), Image.BILINEAR)
                photo = ImageTk.PhotoImage(pil_img)
                self.after(0, self._update_canvas, photo)

        cap.release()

    def _update_canvas(self, photo):
        self.tk_image = photo          # prevent GC
        self.cam_canvas.delete("all")
        self.cam_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # =========================================================================
    # Face swapper globals init
    # =========================================================================

    def _init_globals(self):
        try:
            import modules.globals as gm
            gm.source_path        = self.selected_face_path
            gm.execution_providers = ["CPUExecutionProvider"]
            gm.execution_threads  = 4
            gm.many_faces         = False
            gm.map_faces          = False
            gm.color_correction   = False
            gm.nsfw_filter        = False
        except Exception:
            pass

    # =========================================================================
    # Snapshot
    # =========================================================================

    def _take_snapshot(self):
        frame = self.current_frame
        if frame is None:
            self._set_status("No frame to save")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"snapshot_{ts}.png"
        out  = os.path.join(SNAPSHOTS_DIR, name)
        try:
            if CV2_OK:
                cv2.imwrite(out, frame)
                self._set_status(f"📸 Saved: {name}")
            elif PIL_OK:
                import numpy as np
                rgb = frame[..., ::-1]
                Image.fromarray(rgb).save(out)
                self._set_status(f"📸 Saved: {name}")
        except Exception as ex:
            self._set_status(f"Save error: {ex}")

    # =========================================================================
    # Status / step bar helpers
    # =========================================================================

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _highlight_step(self, idx: int):
        if idx < len(self.step_labels):
            self.after(0, lambda: self.step_labels[idx].configure(fg=ACCENT))

    def _reset_steps(self):
        self.after(0, lambda: [
            lbl.configure(fg=TEXT_DIM) for lbl in self.step_labels
        ])

    # =========================================================================
    # Canvas placeholder
    # =========================================================================

    def _draw_placeholder(self, text: str):
        self.cam_canvas.delete("all")
        self.cam_canvas.create_rectangle(
            0, 0, CAM_W, CAM_H, fill="#0a0f15", outline="")
        self.cam_canvas.create_text(
            CAM_W // 2, CAM_H // 2, text=text,
            fill=TEXT_DIM, font=FONT_BODY, width=400, justify=tk.CENTER)

    # =========================================================================
    # Close
    # =========================================================================

    def _on_close(self):
        self.camera_running = False
        # brief pause to let thread exit
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.5)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DeepFakeLab()
    app.mainloop()
