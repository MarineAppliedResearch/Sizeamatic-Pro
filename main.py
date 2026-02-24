# Sizeamatic Pro - GUI Skeleton (Tkinter only)
#
# Purpose:
#   Build the full GUI layout with all required elements and stub callbacks.
#   No video decoding, no calibration loading, no rectification, no measurement math yet.
#
# Notes:
#   - This file intentionally does not depend on OpenCV or PIL.
#   - Canvases show placeholder content.
#   - All commands are wired to stub handlers so you can wire logic later.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class SizeamaticProApp:
    """
    Main GUI container for Sizeamatic Pro.
    """

    def __init__(self, root):
        self.root = root

        # Tkinter is Python, but we still use // per your inline comment rule preference.
        # However: Python does not support // comments as syntax.
        # So in Python we must use # for comments.
        #
        # I will keep comments compact and frequent using #.

        # ---- Window setup ----
        self.root.title("Sizeamatic Pro")
        self.root.minsize(1100, 700)

        # ---- State flags (UI only for now) ----
        self.view_rectified = tk.BooleanVar(value=False)
        self.fit_to_window = tk.BooleanVar(value=True)
        self.show_overlays = tk.BooleanVar(value=True)
        self.show_epipolar = tk.BooleanVar(value=False)
        self.lock_lr = tk.BooleanVar(value=True)

        # ---- File state placeholders ----
        self.left_video_path = None
        self.right_video_path = None
        self.calibration_folder = None

        # ---- Frame counters placeholders ----
        self.left_frame_index = tk.IntVar(value=0)
        self.right_frame_index = tk.IntVar(value=0)
        self.left_frame_max = 0
        self.right_frame_max = 0

        # ---- Root layout ----
        # Row 0: menu (handled by root.config(menu=...))
        # Row 1: toolbar
        # Row 2: main panes
        # Row 3: status bar
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ---- Build UI ----
        self._build_menu()
        self._build_toolbar()
        self._build_viewers()
        self._build_statusbar()

        # ---- Initial UI refresh ----
        self._refresh_status_left()
        self._refresh_placeholder_canvases()

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # ---- File menu ----
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load Left Video…", command=self.on_load_left_video)
        file_menu.add_command(label="Load Right Video…", command=self.on_load_right_video)
        file_menu.add_separator()
        file_menu.add_command(label="Load Calibration Folder…", command=self.on_load_calibration_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # ---- View menu ----
        view_menu = tk.Menu(menubar, tearoff=False)

        # Show Raw / Show Rectified as a single toggle.
        # Disabled behavior (until calibration loaded) can be added later.
        view_menu.add_checkbutton(
            label="Show Rectified",
            variable=self.view_rectified,
            command=self.on_toggle_view_rectified,
        )
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Fit To Window",
            variable=self.fit_to_window,
            command=self.on_toggle_fit_to_window,
        )
        view_menu.add_checkbutton(
            label="Show Overlays",
            variable=self.show_overlays,
            command=self.on_toggle_show_overlays,
        )
        view_menu.add_checkbutton(
            label="Show Epipolar Cursor Line",
            variable=self.show_epipolar,
            command=self.on_toggle_show_epipolar,
        )
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    # -------------------------------------------------------------------------
    # Toolbar
    # -------------------------------------------------------------------------

    def _build_toolbar(self):
        self.toolbar = ttk.Frame(self.root, padding=(8, 6))
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.toolbar.grid_columnconfigure(20, weight=1)

        # ---- Transport buttons ----
        self.btn_to_start = ttk.Button(self.toolbar, text="⏮", width=3, command=self.on_to_start)
        self.btn_step_back = ttk.Button(self.toolbar, text="◀", width=3, command=self.on_step_back)
        self.btn_play_pause = ttk.Button(self.toolbar, text="⏯", width=3, command=self.on_play_pause)
        self.btn_step_fwd = ttk.Button(self.toolbar, text="▶", width=3, command=self.on_step_forward)
        self.btn_to_end = ttk.Button(self.toolbar, text="⏭", width=3, command=self.on_to_end)

        self.btn_to_start.grid(row=0, column=0, padx=(0, 2))
        self.btn_step_back.grid(row=0, column=1, padx=2)
        self.btn_play_pause.grid(row=0, column=2, padx=2)
        self.btn_step_fwd.grid(row=0, column=3, padx=2)
        self.btn_to_end.grid(row=0, column=4, padx=(2, 12))

        # ---- Speed control ----
        ttk.Label(self.toolbar, text="Speed").grid(row=0, column=5, padx=(0, 6))

        self.speed_var = tk.StringVar(value="1x")
        self.speed_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.speed_var,
            values=["0.25x", "0.5x", "1x", "2x", "4x"],
            width=6,
            state="readonly",
        )
        self.speed_combo.grid(row=0, column=6, padx=(0, 12))
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_changed)

        # ---- Lock checkbox ----
        self.lock_check = ttk.Checkbutton(
            self.toolbar,
            text="Lock L and R",
            variable=self.lock_lr,
            command=self.on_toggle_lock,
        )
        self.lock_check.grid(row=0, column=7, padx=(0, 12))

        # ---- Spacer (keeps toolbar left packed, leaves room to add more) ----
        ttk.Frame(self.toolbar).grid(row=0, column=20, sticky="ew")

    # -------------------------------------------------------------------------
    # Viewer panes
    # -------------------------------------------------------------------------

    def _build_viewers(self):
        # ---- Paned window for resizable left/right panes ----
        self.panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.panes.grid(row=2, column=0, sticky="nsew")
        self.root.grid_rowconfigure(2, weight=1)

        # ---- Left pane ----
        self.left_frame = ttk.Frame(self.panes, padding=(8, 8))
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.left_header = ttk.Label(self.left_frame, text="Left", font=("Segoe UI", 10, "bold"))
        self.left_header.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.left_canvas = tk.Canvas(self.left_frame, bg="black", highlightthickness=1, highlightbackground="#333333")
        self.left_canvas.grid(row=1, column=0, sticky="nsew")

        # Slider row: slider + label
        self.left_slider_row = ttk.Frame(self.left_frame)
        self.left_slider_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.left_slider_row.grid_columnconfigure(0, weight=1)

        self.left_slider = ttk.Scale(
            self.left_slider_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_left_slider_changed,
        )
        self.left_slider.grid(row=0, column=0, sticky="ew")

        self.left_frame_label = ttk.Label(self.left_slider_row, text="Frame: 0/0", width=14, anchor="e")
        self.left_frame_label.grid(row=0, column=1, padx=(10, 0))

        # ---- Right pane ----
        self.right_frame = ttk.Frame(self.panes, padding=(8, 8))
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.right_header = ttk.Label(self.right_frame, text="Right", font=("Segoe UI", 10, "bold"))
        self.right_header.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.right_canvas = tk.Canvas(self.right_frame, bg="black", highlightthickness=1, highlightbackground="#333333")
        self.right_canvas.grid(row=1, column=0, sticky="nsew")

        self.right_slider_row = ttk.Frame(self.right_frame)
        self.right_slider_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.right_slider_row.grid_columnconfigure(0, weight=1)

        self.right_slider = ttk.Scale(
            self.right_slider_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_right_slider_changed,
        )
        self.right_slider.grid(row=0, column=0, sticky="ew")

        self.right_frame_label = ttk.Label(self.right_slider_row, text="Frame: 0/0", width=14, anchor="e")
        self.right_frame_label.grid(row=0, column=1, padx=(10, 0))

        # ---- Add panes to PanedWindow ----
        self.panes.add(self.left_frame, weight=1)
        self.panes.add(self.right_frame, weight=1)

        # ---- Bind canvas resize to redraw placeholder ----
        self.left_canvas.bind("<Configure>", lambda e: self._refresh_placeholder_canvases())
        self.right_canvas.bind("<Configure>", lambda e: self._refresh_placeholder_canvases())

        # ---- Mouse bindings placeholders (for future overlays) ----
        self.left_canvas.bind("<Button-1>", lambda e: self.on_canvas_click("L", e))
        self.right_canvas.bind("<Button-1>", lambda e: self.on_canvas_click("R", e))

    # -------------------------------------------------------------------------
    # Status bar
    # -------------------------------------------------------------------------

    def _build_statusbar(self):
        self.status = ttk.Frame(self.root, padding=(8, 6))
        self.status.grid(row=3, column=0, sticky="ew")
        self.status.grid_columnconfigure(1, weight=1)

        # Left: file/cal/view state
        self.status_left = ttk.Label(self.status, text="", anchor="w")
        self.status_left.grid(row=0, column=0, sticky="w")

        # Middle: warnings / messages
        self.status_mid = ttk.Label(self.status, text="", anchor="center")
        self.status_mid.grid(row=0, column=1, sticky="ew")

        # Right: measurement results
        self.status_right = ttk.Label(self.status, text="", anchor="e")
        self.status_right.grid(row=0, column=2, sticky="e")

    # -------------------------------------------------------------------------
    # Stub handlers (menu)
    # -------------------------------------------------------------------------

    def on_load_left_video(self):
        path = filedialog.askopenfilename(
            title="Load Left Video",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if not path:
            return

        # Store path, no decoding yet.
        self.left_video_path = path
        self._set_status_mid("Loaded left video (UI only)")
        self._refresh_status_left()

    def on_load_right_video(self):
        path = filedialog.askopenfilename(
            title="Load Right Video",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if not path:
            return

        self.right_video_path = path
        self._set_status_mid("Loaded right video (UI only)")
        self._refresh_status_left()

    def on_load_calibration_folder(self):
        folder = filedialog.askdirectory(title="Load Calibration Folder")
        if not folder:
            return

        self.calibration_folder = folder
        self._set_status_mid("Loaded calibration folder (UI only)")
        self._refresh_status_left()

        # In real wiring, you will enable "Show Rectified" only after maps load.
        # For now, we leave it togglable to test UI.

    # -------------------------------------------------------------------------
    # Stub handlers (view toggles)
    # -------------------------------------------------------------------------

    def on_toggle_view_rectified(self):
        # In real wiring, this would switch frame pipeline raw vs rectified.
        self._refresh_status_left()
        self._refresh_placeholder_canvases()

    def on_toggle_fit_to_window(self):
        # In real wiring, this would change scaling logic for displayed frames.
        self._set_status_mid("Fit To Window toggled (UI only)")
        self._refresh_placeholder_canvases()

    def on_toggle_show_overlays(self):
        # In real wiring, this would enable/disable drawing points/lines on canvas.
        self._set_status_mid("Show Overlays toggled (UI only)")
        self._refresh_placeholder_canvases()

    def on_toggle_show_epipolar(self):
        # In real wiring, only makes sense when rectified is active.
        self._set_status_mid("Epipolar cursor toggled (UI only)")
        self._refresh_placeholder_canvases()

    # -------------------------------------------------------------------------
    # Stub handlers (toolbar)
    # -------------------------------------------------------------------------

    def on_to_start(self):
        self._jump_frames_locked_or_single(target_index=0)

    def on_to_end(self):
        # For now, "end" means whatever the current slider max is.
        if self.lock_lr.get():
            max_i = int(min(self.left_frame_max, self.right_frame_max))
            self._jump_frames_locked_or_single(target_index=max_i)
        else:
            self.left_frame_index.set(self.left_frame_max)
            self.right_frame_index.set(self.right_frame_max)
            self.left_slider.set(self.left_frame_max)
            self.right_slider.set(self.right_frame_max)
            self._update_frame_labels()
            self._refresh_placeholder_canvases()

    def on_step_back(self):
        self._nudge_frames_locked_or_single(delta=-1)

    def on_step_forward(self):
        self._nudge_frames_locked_or_single(delta=+1)

    def on_play_pause(self):
        # Playback loop comes later.
        messagebox.showinfo("Play/Pause", "Playback not wired yet.\nThis is UI skeleton only.")

    def on_speed_changed(self, _evt=None):
        # Speed affects playback step or timer interval later.
        self._set_status_mid(f"Speed set to {self.speed_var.get()} (UI only)")

    def on_toggle_lock(self):
        # Lock affects slider behavior. We already implement the UI behavior for slider movement.
        self._set_status_mid("Lock toggled")
        self._refresh_status_left()

        # When enabling lock, unify indices to the left slider's current value.
        if self.lock_lr.get():
            master = int(round(self.left_slider.get()))
            self._jump_frames_locked_or_single(target_index=master)

    # -------------------------------------------------------------------------
    # Slider callbacks
    # -------------------------------------------------------------------------

    def on_left_slider_changed(self, _value):
        # ttk.Scale gives float strings; we quantize to int frame index.
        i = int(round(float(self.left_slider.get())))
        if self.lock_lr.get():
            self._jump_frames_locked_or_single(target_index=i)
        else:
            self.left_frame_index.set(i)
            self._update_frame_labels()
            self._refresh_placeholder_canvases()

    def on_right_slider_changed(self, _value):
        i = int(round(float(self.right_slider.get())))
        if self.lock_lr.get():
            self._jump_frames_locked_or_single(target_index=i)
        else:
            self.right_frame_index.set(i)
            self._update_frame_labels()
            self._refresh_placeholder_canvases()

    # -------------------------------------------------------------------------
    # Canvas stubs
    # -------------------------------------------------------------------------

    def on_canvas_click(self, which, event):
        # Placeholder for later measurement interactions.
        # Keep it minimal: show click coords.
        self._set_status_mid(f"{which} click at ({event.x}, {event.y}) (UI only)")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _refresh_status_left(self):
        l = self.left_video_path if self.left_video_path else "(none)"
        r = self.right_video_path if self.right_video_path else "(none)"
        c = self.calibration_folder if self.calibration_folder else "(none)"
        view = "Rectified" if self.view_rectified.get() else "Raw"
        lock = "Locked" if self.lock_lr.get() else "Unlocked"

        self.status_left.config(text=f"L: {self._short_path(l)} | R: {self._short_path(r)} | Cal: {self._short_path(c)} | View: {view} | {lock}")

    def _short_path(self, path, max_len=45):
        if path is None:
            return "(none)"
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1):]

    def _set_status_mid(self, text):
        self.status_mid.config(text=text)

    def _set_status_right(self, text):
        self.status_right.config(text=text)

    def _update_frame_labels(self):
        # Frame max is currently 0 because no video is loaded.
        # Later you will set left_frame_max/right_frame_max from cv2 capture length.
        lmax = max(0, int(self.left_frame_max))
        rmax = max(0, int(self.right_frame_max))

        li = int(self.left_frame_index.get())
        ri = int(self.right_frame_index.get())

        self.left_frame_label.config(text=f"Frame: {li}/{lmax}")
        self.right_frame_label.config(text=f"Frame: {ri}/{rmax}")

    def _refresh_placeholder_canvases(self):
        # Draw a simple placeholder pattern so we can see resizing and overlays.
        self._draw_placeholder(self.left_canvas, "LEFT", self.view_rectified.get())
        self._draw_placeholder(self.right_canvas, "RIGHT", self.view_rectified.get())

        # Keep frame labels consistent.
        self._update_frame_labels()

    def _draw_placeholder(self, canvas, label, rectified):
        canvas.delete("all")

        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())

        # Background is already black; draw border guides.
        canvas.create_rectangle(2, 2, w - 2, h - 2, outline="#444444")

        # Draw a simple grid so "fit to window" and scaling logic later is obvious.
        step = 50
        for x in range(step, w, step):
            canvas.create_line(x, 0, x, h, fill="#222222")
        for y in range(step, h, step):
            canvas.create_line(0, y, w, y, fill="#222222")

        # Central crosshair.
        cx = w // 2
        cy = h // 2
        canvas.create_line(cx, 0, cx, h, fill="#333333")
        canvas.create_line(0, cy, w, cy, fill="#333333")

        # Big label.
        mode = "RECTIFIED" if rectified else "RAW"
        canvas.create_text(
            cx,
            cy - 20,
            text=f"{label} VIEW",
            fill="white",
            font=("Segoe UI", 16, "bold"),
        )
        canvas.create_text(
            cx,
            cy + 15,
            text=mode,
            fill="#cccccc",
            font=("Segoe UI", 12, "bold"),
        )

        # Overlay indicator (just to test the toggle visually).
        if self.show_overlays.get():
            canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, outline="#00ff66", width=2)
            canvas.create_text(cx, cy + 40, text="Overlay ON", fill="#00ff66", font=("Segoe UI", 10, "normal"))
        else:
            canvas.create_text(cx, cy + 40, text="Overlay OFF", fill="#888888", font=("Segoe UI", 10, "normal"))

        # Epipolar cursor indicator stub.
        if self.show_epipolar.get():
            canvas.create_line(0, cy + 80, w, cy + 80, fill="#ffcc00", dash=(6, 4))
            canvas.create_text(90, cy + 65, text="Epipolar Line", fill="#ffcc00", font=("Segoe UI", 9, "normal"))

    def _clamp(self, x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _nudge_frames_locked_or_single(self, delta):
        # Adjust current frame(s) by delta, respecting lock mode and clamp behavior.
        if self.lock_lr.get():
            # Locked: clamp to shorter max.
            max_i = int(min(self.left_frame_max, self.right_frame_max))
            cur = int(round(self.left_slider.get()))
            nxt = self._clamp(cur + delta, 0, max_i)
            self._jump_frames_locked_or_single(nxt)
        else:
            # Unlocked: each side clamps independently.
            li = self._clamp(int(round(self.left_slider.get())) + delta, 0, int(self.left_frame_max))
            ri = self._clamp(int(round(self.right_slider.get())) + delta, 0, int(self.right_frame_max))

            self.left_frame_index.set(li)
            self.right_frame_index.set(ri)

            self.left_slider.set(li)
            self.right_slider.set(ri)

            self._update_frame_labels()
            self._refresh_placeholder_canvases()

    def _jump_frames_locked_or_single(self, target_index):
        # Jump to target_index in lock mode or update only the active slider.
        if self.lock_lr.get():
            max_i = int(min(self.left_frame_max, self.right_frame_max))
            i = self._clamp(int(target_index), 0, max_i)

            self.left_frame_index.set(i)
            self.right_frame_index.set(i)

            # Keep both sliders visually synced.
            self.left_slider.set(i)
            self.right_slider.set(i)

            self._update_frame_labels()
            self._refresh_placeholder_canvases()
        else:
            # If unlocked, this helper is used for start/end operations.
            # We treat it as applying to both sides for toolbar actions.
            li = self._clamp(int(target_index), 0, int(self.left_frame_max))
            ri = self._clamp(int(target_index), 0, int(self.right_frame_max))

            self.left_frame_index.set(li)
            self.right_frame_index.set(ri)

            self.left_slider.set(li)
            self.right_slider.set(ri)

            self._update_frame_labels()
            self._refresh_placeholder_canvases()


def main():
    root = tk.Tk()

    # ttk theme defaults are OK. If you want a darker theme later, we can style it.
    app = SizeamaticProApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()