# Sizeamatic Pro - 
#


import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import base64

import numpy as np


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

         # ---- File state ----
        self.left_video_path = None
        self.right_video_path = None
        self.calibration_folder = None

        # ---- OpenCV video captures ----
        # These remain open for the lifetime of the app, so seeking is fast.
        self.capL = None
        self.capR = None

        # ---- Video metadata ----
        # Each meta dict contains: fps, width, height, frame_count
        self.metaL = None
        self.metaR = None

        # ---- Timeline state ----
        # Frame indices are the master timeline values.
        self.left_frame_index = tk.IntVar(value=0)
        self.right_frame_index = tk.IntVar(value=0)

        # Max frame index inclusive for each stream.
        # These are updated after loading video and when lock mode changes.
        self.left_frame_max = 0
        self.right_frame_max = 0

        # ---- Playback loop state ----
        self.is_playing = False
        self.play_after_id = None

        # ---- Tk image handles ----
        # Tk will garbage collect images unless we keep a reference.
        self.tkimg_left = None
        self.tkimg_right = None

        # Prevents recursive slider callbacks when we update slider positions in code.
        self._suppress_slider_callbacks = False

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

        # When locked, the videos stay aligned by a fixed frame offset.
        # Definition: offset = right_index - left_index.
        # Example: if right is 12 frames ahead of left, offset = +12.
        self.lock_offset_frames = 0

        # Debounce handle for resize redraw.
        # Resize events can fire dozens of times per second while dragging the window.
        self._resize_after_id = None

        # Per pane point lists stored in IMAGE PIXEL coordinates.
        # These are the authoritative coordinates for measurement.
        self.ptsL = []
        self.ptsR = []

        # Current cap for points per pane.
        # We start with 2 for a line, but later we can raise this for curves.
        self.max_points_per_pane = 2

        # Handle radius in SCREEN pixels (after scaling).
        # Handles are large so you can click them directly without hit test math.
        self.handle_radius_px = 8

        # Drag state for moving an existing handle.
        # drag_index is the point index we are moving (0, 1, ...).
        self.drag_active = False
        self.drag_which = None
        self.drag_index = None

        # Per pane view state for zoom and pan.
        # zoom is unitless scale multiplier applied on top of fit-to-window scaling.
        # off_x/off_y are screen-pixel offsets applied after scaling.
        self.viewL = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0}
        self.viewR = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0}

        # Zoom limits.
        self.zoom_min = 1.0
        self.zoom_max = 10.0

        # Zoom factor per mouse wheel notch.
        self.zoom_step = 1.10

        # Calibration bundle loaded from NPZ files.
        # None means not loaded or invalid.
        self.cal = None

    # Redraws overlay items for both panes.
    # For now, this draws a single test handle and label in each pane.
    def _redraw_overlays(self):
        # Draw left overlay.
        self._draw_overlay_for_pane("L", self.left_overlay_canvas, self.ptsL)

        # Draw right overlay.
        self._draw_overlay_for_pane("R", self.right_overlay_canvas, self.ptsR)

    # Draws handles, labels, and connecting segments for one pane.
    # Points are stored in IMAGE pixel coordinates and mapped to SCREEN coords using the same scale as the video.
    def _draw_overlay_for_pane(self, which, canvas, pts):
        # Clear only overlay items so the video frame stays intact.
        canvas.delete("overlay")

        # Compute image->screen scale so overlay tracks the displayed video.
        scale = self._get_pane_scale(which, canvas)

        # Draw line segments first so handles sit on top.
        # For N points, draw segments (0-1), (1-2), ...
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]

                sx0, sy0 = self._image_to_screen(which, canvas, x0, y0)
                sx1, sy1 = self._image_to_screen(which, canvas, x1, y1)

                canvas.create_line(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    width=2,
                    fill="#00ff66",
                    tags=("overlay",),
                )

        # Draw handles and index labels.
        r = int(self.handle_radius_px)

        for i, (x, y) in enumerate(pts):

            # Translate to screen coordinates
            sx, sy = self._image_to_screen(which, canvas, x, y)

            # Handle oval:
            # Tag it as "handle" so _get_handle_index_under_cursor can recognize it.
            # Tag idx:<n> so we can identify which point index was clicked.
            canvas.create_oval(
                sx - r,
                sy - r,
                sx + r,
                sy + r,
                outline="#00ff66",
                width=2,
                fill="",
                tags=("overlay", "handle", f"idx:{i}"),
            )

            # Index label near the handle.
            canvas.create_text(
                sx + r + 6,
                sy - r - 6,
                text=str(i),
                fill="#00ff66",
                font=("Segoe UI", 11, "bold"),
                tags=("overlay",),
            )

        # Mouse wheel zoom for a pane, anchored under the cursor.
    def on_mouse_wheel(self, which, event):
        canvas = self.left_overlay_canvas if which == "L" else self.right_overlay_canvas

        # Require metadata so we know how to map coords.
        if self._get_image_size(which) is None:
            return

        view = self._get_view(which)

        # Convert cursor position to image coords before zoom changes.
        ix, iy = self._screen_to_image(which, canvas, event.x, event.y)

        # Wheel direction (Windows: event.delta is typically ±120 per notch).
        if event.delta > 0:
            new_zoom = float(view["zoom"]) * float(self.zoom_step)
        else:
            new_zoom = float(view["zoom"]) / float(self.zoom_step)

        # Clamp zoom.
        new_zoom = max(float(self.zoom_min), min(float(self.zoom_max), new_zoom))
        view["zoom"] = new_zoom

        # After zoom changes, compute new total scale.
        S_new = self._get_total_scale(which, canvas)

        # Keep the same image point under the cursor.
        # event.x/event.y are in canvas coords, so subtract the display rect origin first.
        dx, dy, _dw, _dh = self._get_display_rect(which, canvas)

        view["off_x"] = (float(event.x) - float(dx)) - float(ix) * S_new
        view["off_y"] = (float(event.y) - float(dy)) - float(iy) * S_new

        # Redraw everything using the new transform.
        self._render_current_frames()

    # Closes OpenCV windows and releases captures before exiting.
    def on_app_close(self):
        # Stop playback loop.
        self.is_playing = False
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

        # Release capture objects if open.
        if self.capL:
            self.capL.release()
            self.capL = None
        if self.capR:
            self.capR.release()
            self.capR = None

        # Close any OpenCV windows.
        cv2.destroyAllWindows()

        # Close the Tk app.
        self.root.destroy()

    # Called when either canvas is resized.
    # We debounce redraw to avoid decoding and encoding on every resize event.
    def on_canvas_resized(self, _event):
        # If Fit To Window is off, resizing the window does not change the image size.
        # In that case, we can ignore resize events entirely.
        if not self.fit_to_window.get():
            return

        # Cancel any pending redraw so we only redraw once after resizing settles.
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = None

        # Schedule a redraw shortly in the future.
        # 50 ms is short enough to feel responsive but avoids resize storm spam.
        self._resize_after_id = self.root.after(50, self._redraw_after_resize)

    # Runs after the debounce delay to redraw the current frames.
    def _redraw_after_resize(self):
        # Clear the pending handle first.
        self._resize_after_id = None

        # If no videos are loaded yet, keep the placeholders.
        if not self.capL and not self.capR:
            self._refresh_placeholder_canvases()
            return

        # Otherwise, redraw the current decoded frames.
        # This will re run the Fit To Window scaling logic.
        self._render_current_frames()

    # Returns the view dict for a pane.
    def _get_view(self, which):
        if which == "L":
            return self.viewL
        return self.viewR

    # Returns the source image width/height for a pane.
    def _get_image_size(self, which):
        if which == "L":
            if not self.metaL:
                return None
            return int(self.metaL["width"]), int(self.metaL["height"])
        else:
            if not self.metaR:
                return None
            return int(self.metaR["width"]), int(self.metaR["height"])

    # Computes the base fit-to-window scale (by width only).
    def _get_fit_scale(self, which, canvas):
        # If Fit To Window is off, base scale is 1.
        if not self.fit_to_window.get():
            return 1.0

        size = self._get_image_size(which)
        if size is None:
            return 1.0

        img_w, _img_h = size

        # Use the actual draw rect width, not the full canvas width.
        _dx, _dy, dw, _dh = self._get_display_rect(which, canvas)
        return float(dw) / float(img_w)

    # Computes the total scale used for both video and overlays: S = fit_scale * zoom.
    def _get_total_scale(self, which, canvas):
        view = self._get_view(which)
        fit_scale = self._get_fit_scale(which, canvas)
        return fit_scale * float(view["zoom"])

    # Converts image pixel coords to screen coords for a pane.
    def _image_to_screen(self, which, canvas, ix, iy):
        view = self._get_view(which)

        # Display rect defines where the video lives inside the canvas.
        dx, dy, _dw, _dh = self._get_display_rect(which, canvas)

        # Total scale includes Fit To Window scale and zoom.
        S = self._get_total_scale(which, canvas)

        # off_x/off_y are pan offsets in screen pixels relative to the display rect.
        sx = float(dx) + float(ix) * S + float(view["off_x"])
        sy = float(dy) + float(iy) * S + float(view["off_y"])
        return sx, sy

    # Converts screen coords to image pixel coords for a pane.
    def _screen_to_image(self, which, canvas, sx, sy):
        view = self._get_view(which)

        dx, dy, _dw, _dh = self._get_display_rect(which, canvas)

        S = self._get_total_scale(which, canvas)
        if S <= 0.0:
            S = 1.0

        # Convert screen->image by undoing display rect origin and offsets first.
        ix = (float(sx) - float(dx) - float(view["off_x"])) / S
        iy = (float(sy) - float(dy) - float(view["off_y"])) / S
        return ix, iy

    """ # Converts screen coordinates (canvas pixels) to image pixel coordinates.
    # This must invert the same scale used to draw the video and overlays.
    def _screen_to_image(self, which, canvas, sx, sy):
        # Scale maps image->screen. We invert it to map screen->image.
        scale = self._get_pane_scale(which, canvas)

        # Guard against divide by zero.
        if scale <= 0.0:
            scale = 1.0

        # Convert to image pixel coords.
        ix = float(sx) / scale
        iy = float(sy) / scale

        return ix, iy """

    # Returns the point list for a pane.
    def _get_points_list(self, which):
        if which == "L":
            return self.ptsL
        return self.ptsR
    
    # Attempts to extract a handle index from the canvas item under the cursor.
    # Returns an integer index if the current item is a handle, else returns None.
    def _get_handle_index_under_cursor(self, canvas):
        # "current" is the canvas item under the mouse pointer at event time.
        items = canvas.find_withtag("current")
        if not items:
            return None

        item_id = items[0]

        # Read the item's tags and look for the "handle" marker and an "idx:<n>" tag.
        tags = canvas.gettags(item_id)

        # Only treat this as a draggable point if it is tagged as a handle.
        if "handle" not in tags:
            return None

        # Parse an index tag formatted like "idx:0", "idx:1", etc.
        for t in tags:
            if t.startswith("idx:"):
                try:
                    return int(t.split(":", 1)[1])
                except ValueError:
                    return None

        return None
    
    # Left mouse button pressed on overlay.
    def on_overlay_left_down(self, which, event):
        # Choose the correct overlay canvas for this pane.
        canvas = self.left_overlay_canvas if which == "L" else self.right_overlay_canvas

        # If the user clicked a handle, start dragging that handle.
        idx = self._get_handle_index_under_cursor(canvas)
        if idx is not None:
            # Begin drag mode.
            self.drag_active = True
            self.drag_which = which
            self.drag_index = idx
            return

        # Otherwise, treat this as a "place a new point" click.
        pts = self._get_points_list(which)

        # If we are already at the point cap, ignore clicks on empty space.
        # This is important because later we will raise the cap for curves.
        if len(pts) >= int(self.max_points_per_pane):
            return

        # Convert the click from screen coords to image coords.
        ix, iy = self._screen_to_image(which, canvas, event.x, event.y)

        # Append as the next point.
        pts.append((ix, iy))

        # Trigger overlay redraw and measurement stub.
        self._on_points_changed()

    # Mouse moved while left button is held.
    def on_overlay_left_drag(self, which, event):
        # Only drag if we are actively dragging a handle.
        if not self.drag_active:
            return

        # Only respond if the drag belongs to this pane.
        if self.drag_which != which:
            return

        canvas = self.left_overlay_canvas if which == "L" else self.right_overlay_canvas
        pts = self._get_points_list(which)

        # Validate index.
        if self.drag_index is None:
            return
        if self.drag_index < 0 or self.drag_index >= len(pts):
            return

        # Convert cursor position to image coords.
        ix, iy = self._screen_to_image(which, canvas, event.x, event.y)

        # Update the dragged point.
        pts[self.drag_index] = (ix, iy)

        # Redraw overlays and update measurement stub continuously while dragging.
        self._on_points_changed()

    # Left mouse button released.
    def on_overlay_left_up(self, which, _event):
        # End drag mode cleanly.
        if self.drag_active and self.drag_which == which:
            self.drag_active = False
            self.drag_which = None
            self.drag_index = None

    

    # Called whenever points are added, moved, or deleted.
    # This is the single place that triggers overlay redraw and measurement refresh.
    def _on_points_changed(self):
        # Redraw overlays (function will be updated next step to draw real points/lines).
        # For now, keep calling it so the pipeline is correct.
        self._redraw_overlays()

        # Update measurement status text (stub for now).
        self._update_measurement_status_stub()

    # Updates the status bar with a simple "ready" message.
    def _update_measurement_status_stub(self):
        l_count = len(self.ptsL)
        r_count = len(self.ptsR)

        # If counts differ, we are missing a matching point on one side.
        if l_count != r_count:
            self._set_status_right(f"Point pair incomplete: L={l_count} R={r_count}")
            return

        # If we have at least one matched point, we will later compute depth.
        if l_count >= 1:
            if l_count == 1:
                self._set_status_right("Ready: depth from point 0 (needs triangulation)")
                return

        # If we have at least two matched points, we will later compute length.
        if l_count >= 2:
            self._set_status_right("Ready: length from points 0-1 (needs triangulation)")
            return

        # No points set.
        self._set_status_right("")

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

        # Clears all measurement points in both panes.
        # This is the only delete mechanism for now (simple and safe).
        self.btn_clear_points = ttk.Button(
            self.toolbar,
            text="Clear Points",
            command=self.on_clear_points,
        )
        self.btn_clear_points.grid(row=0, column=8, padx=(0, 12))

        # ---- Spacer (keeps toolbar left packed, leaves room to add more) ----
        ttk.Frame(self.toolbar).grid(row=0, column=20, sticky="ew")

    # Clears all measurement points in both panes.
    def on_clear_points(self):
        # Clear both point lists to keep pairing consistent.
        self.ptsL.clear()
        self.ptsR.clear()

        # Cancel any active drag state.
        self.drag_active = False
        self.drag_which = None
        self.drag_index = None

        # Redraw overlays to remove handles and lines.
        self._redraw_overlays()

        # Update measurement status text.
        self._update_measurement_status_stub()

        # Show a short confirmation in the center status area.
        self._set_status_mid("Points cleared")

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

        # Left: replace the single canvas with a viewport that contains two stacked canvases.
        # Bottom canvas draws video, top canvas draws overlays later.

        # Create a container frame in the exact grid cell where the old canvas lived.
        self.left_viewport = ttk.Frame(self.left_frame)
        self.left_viewport.grid(row=1, column=0, sticky="nsew")

        # Allow row 1 (the video area) to grow when the window grows.
        self.left_frame.grid_rowconfigure(1, weight=1)

        # Allow column 0 (the only column) to grow when the window grows.
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Make the viewport frame expand to fill its parent cell.
        self.left_viewport.grid_rowconfigure(0, weight=1)
        self.left_viewport.grid_columnconfigure(0, weight=1)

        # Bottom canvas: this is where we draw the video image.
        self.left_video_canvas = tk.Canvas(
            self.left_viewport,
            bg="black",                    # Fill background when no frame is drawn.
            highlightthickness=1,          # Thin border for visibility.
            highlightbackground="#333333", # Border color.
        )
        self.left_video_canvas.grid(row=0, column=0, sticky="nsew")  # Fill the viewport.

        # Top canvas: overlays live here (points, lines, labels).
        # We do NOT draw video here, so we can change video rendering later without touching overlays.
        self.left_overlay_canvas = tk.Canvas(
            self.left_viewport,
            bg="black",          # Tk requires a valid color; we fake transparency later.
            highlightthickness=0,
            bd=0,
        )

        # Use place so the overlay canvas always covers the video canvas exactly.
        # relwidth/relheight = 1 makes it track the viewport size automatically.
        self.left_overlay_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

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

        # Right: same stacked canvas setup as left.

        self.right_viewport = ttk.Frame(self.right_frame)
        self.right_viewport.grid(row=1, column=0, sticky="nsew")

        # Let the right pane's video row expand with window size.
        self.right_frame.grid_rowconfigure(1, weight=1)

        # Let the right pane's single column expand with window size.
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Let the viewport expand inside that growing area.
        self.right_viewport.grid_rowconfigure(0, weight=1)
        self.right_viewport.grid_columnconfigure(0, weight=1)

        # Bottom canvas draws video frames.
        self.right_video_canvas = tk.Canvas(
            self.right_viewport,
            bg="black",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.right_video_canvas.grid(row=0, column=0, sticky="nsew")

        # Top canvas draws overlay shapes and handles later.
        self.right_overlay_canvas = tk.Canvas(
            self.right_viewport,
            bg="black",
            highlightthickness=0,
            bd=0,
        )
        self.right_overlay_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

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

        # When the canvas size changes, we need to redraw the current frames.
        # We debounce because resize events fire rapidly while dragging the window.
        self.left_video_canvas.bind("<Configure>", self.on_canvas_resized)
        self.right_video_canvas.bind("<Configure>", self.on_canvas_resized)

        self.left_overlay_canvas.bind("<Configure>", self.on_canvas_resized)
        self.right_overlay_canvas.bind("<Configure>", self.on_canvas_resized)

        # Left overlay canvas receives user input.
        self.left_overlay_canvas.bind("<Button-1>", lambda e: self.on_overlay_left_down("L", e))
        self.left_overlay_canvas.bind("<B1-Motion>", lambda e: self.on_overlay_left_drag("L", e))
        self.left_overlay_canvas.bind("<ButtonRelease-1>", lambda e: self.on_overlay_left_up("L", e))

        # Right overlay canvas receives user input.
        self.right_overlay_canvas.bind("<Button-1>", lambda e: self.on_overlay_left_down("R", e))
        self.right_overlay_canvas.bind("<B1-Motion>", lambda e: self.on_overlay_left_drag("R", e))
        self.right_overlay_canvas.bind("<ButtonRelease-1>", lambda e: self.on_overlay_left_up("R", e))

        # Mouse wheel zoom for each pane (Windows uses <MouseWheel> with event.delta).
        self.left_overlay_canvas.bind("<MouseWheel>", lambda e: self.on_mouse_wheel("L", e))
        self.right_overlay_canvas.bind("<MouseWheel>", lambda e: self.on_mouse_wheel("R", e))

       

    # -------------------------------------------------------------------------
    # Status bar
    # -------------------------------------------------------------------------

    def _build_statusbar(self):
        self.status = ttk.Frame(self.root, padding=(8, 6))
        self.status.grid(row=3, column=0, sticky="ew")
        self.status.grid_columnconfigure(1, weight=1)

        # Left: file/cal/view state.
        # width is in characters, used to stop the label from resizing the window.
        self.status_left = ttk.Label(self.status, text="", anchor="w", width=90)
        self.status_left.grid(row=0, column=0, sticky="w")

        # Middle: warnings / messages.
        # sticky="ew" lets it stretch inside the fixed grid column.
        self.status_mid = ttk.Label(self.status, text="", anchor="center", width=40)
        self.status_mid.grid(row=0, column=1, sticky="ew")

        # Right: measurement results.
        self.status_right = ttk.Label(self.status, text="", anchor="e", width=60)
        self.status_right.grid(row=0, column=2, sticky="e")

    # -------------------------------------------------------------------------
    # Stub handlers (menu)
    # -------------------------------------------------------------------------

    # Loads the left video and updates UI state.
    def on_load_left_video(self):
        # Ask user to choose a left MP4 file.
        path = filedialog.askopenfilename(
            title="Load Left Video",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if not path:
            return

        # Close any previous capture so we do not leak file handles.
        if self.capL:
            self.capL.release()
            self.capL = None
            self.metaL = None

        # Open the new capture and read its metadata.
        cap, meta = self._open_video_capture(path)
        if cap is None:
            messagebox.showerror("Load Left Video", "Failed to open the selected video file.")
            return

        # Save state.
        self.left_video_path = path
        self.capL = cap
        self.metaL = meta

        # Update header with metadata so you can verify the file quickly.
        self.left_header.config(
            text=f"Left  ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )

        # Reset left index to 0 on new load to avoid seeking into nonsense.
        self.left_frame_index.set(0)

        # Update slider ranges based on lock mode and which videos are loaded.
        self._update_slider_ranges()

        # Render whichever frames are available.
        self._render_current_frames()

        # Update status.
        self._set_status_mid("Loaded left video")
        self._refresh_status_left()




    # Loads the right video and updates UI state.
    def on_load_right_video(self):
        # Ask user to choose a right MP4 file.
        # We do not assume both videos are loaded at once, so this must work independently.
        path = filedialog.askopenfilename(
            title="Load Right Video",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if not path:
            # User cancelled the dialog.
            return

        # If we already had a right capture open, release it.
        # This avoids file handle leaks and lets the user reload different files safely.
        if self.capR:
            self.capR.release()
            self.capR = None
            self.metaR = None

        # Open the new capture and read container metadata.
        # We do this immediately so the UI can show fps, resolution, and frame count.
        cap, meta = self._open_video_capture(path)
        if cap is None:
            # If OpenCV cannot open it, inform the user with a clear error.
            messagebox.showerror("Load Right Video", "Failed to open the selected video file.")
            return

        # Save state so the rest of the app can render frames from this capture.
        self.right_video_path = path
        self.capR = cap
        self.metaR = meta

        # Update the right header text so it is obvious what file was loaded.
        # This is important for debugging when videos are mismatched.
        self.right_header.config(
            text=f"Right  ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )

        # Reset the right timeline to frame 0 on load.
        # This avoids "seek into the middle" behavior that is confusing during testing.
        self.right_frame_index.set(0)

        # Update slider max ranges and clamping rules.
        # If lock is enabled and both videos exist, we clamp to the shorter length here.
        self._update_slider_ranges()

        # Draw the current frames (left if present, right always now).
        # This makes it immediately obvious that loading worked.
        self._render_current_frames()

        # Update status line.
        self._set_status_mid("Loaded right video")
        self._refresh_status_left()

    # Called when the user decides to load a new calibration folder.
    def on_load_calibration_folder(self):
        folder = filedialog.askdirectory(title="Load Calibration Folder")
        if not folder:
            return

        # Store the folder path for status display.
        self.calibration_folder = folder

        # Build expected file paths.
        intr_path = os.path.join(folder, "calibration_intrinsics.npz")
        extr_path = os.path.join(folder, "calibration_extrinsics.npz")
        rect_path = os.path.join(folder, "calibration_rectification.npz")
        maps_path = os.path.join(folder, "calibration_maps.npz")

        # Verify required files exist.
        missing = []
        for p in [intr_path, extr_path, rect_path, maps_path]:
            if not os.path.isfile(p):
                missing.append(os.path.basename(p))

        # Handle the case where a calibration file doesn't exist
        if missing:
            self.cal = None
            self.view_rectified.set(False)
            self._set_status_mid(f"Missing calibration files: {', '.join(missing)}")
            self._refresh_status_left()
            return

        try:
            intr = np.load(intr_path)
            rect = np.load(rect_path)
            maps = np.load(maps_path)

            # Pull required matrices/maps.
            PL = rect["PL"]
            PR = rect["PR"]
            Q = rect["Q"]

            mapLx = maps["mapLx"]
            mapLy = maps["mapLy"]
            mapRx = maps["mapRx"]
            mapRy = maps["mapRy"]

            # Intrinsics file stores expected calibration resolution.
            cal_w = int(intr["image_width"])
            cal_h = int(intr["image_height"])

        except Exception as e:
            self.cal = None
            self.view_rectified.set(False)
            self._set_status_mid(f"Failed to load calibration: {e}")
            self._refresh_status_left()
            return

        # If we have a loaded video, enforce resolution match now.
        # Rectification maps must match the decoded frame size.
        if self.metaL:
            if self.metaL["width"] != cal_w or self.metaL["height"] != cal_h:
                self.cal = None
                self.view_rectified.set(False)
                self._set_status_mid("Calibration resolution does not match LEFT video")
                self._refresh_status_left()
                return

        if self.metaR:
            if self.metaR["width"] != cal_w or self.metaR["height"] != cal_h:
                self.cal = None
                self.view_rectified.set(False)
                self._set_status_mid("Calibration resolution does not match RIGHT video")
                self._refresh_status_left()
                return

        # Store calibration bundle.
        self.cal = {
            "PL": PL,
            "PR": PR,
            "Q": Q,
            "mapLx": mapLx,
            "mapLy": mapLy,
            "mapRx": mapRx,
            "mapRy": mapRy,
            "w": cal_w,
            "h": cal_h,
        }

        self._set_status_mid("Calibration loaded")
        self._refresh_status_left()

        # Trigger redraw so rectified mode can be enabled immediately.
        self._render_current_frames()

        # In real wiring, you will enable "Show Rectified" only after maps load.
        # For now, we leave it togglable to test UI.

    # -------------------------------------------------------------------------
    # Stub handlers (view toggles)
    # -------------------------------------------------------------------------

    # Toggle whether the user is watching recitfied stereo video, or raw stereo video
    def on_toggle_view_rectified(self):
        # If user turned rectified on, ensure calibration is ready.
        if self.view_rectified.get():
            if self.cal is None:
                # Force it off and warn.
                self.view_rectified.set(False)
                self._set_status_mid("Rectified view requires calibration")
                self._refresh_status_left()
                return

            # If videos are loaded, ensure sizes match calibration.
            if self.metaL:
                if self.metaL["width"] != self.cal["w"] or self.metaL["height"] != self.cal["h"]:
                    self.view_rectified.set(False)
                    self._set_status_mid("Rectified view disabled: LEFT video resolution mismatch")
                    self._refresh_status_left()
                    return

            if self.metaR:
                if self.metaR["width"] != self.cal["w"] or self.metaR["height"] != self.cal["h"]:
                    self.view_rectified.set(False)
                    self._set_status_mid("Rectified view disabled: RIGHT video resolution mismatch")
                    self._refresh_status_left()
                    return

        self._refresh_status_left()
        self._render_current_frames()

    # Toggles fit-to-window rendering and redraws the current frames.
    def on_toggle_fit_to_window(self):
        # Fit-to-window changes the display size calculation.
        # It does not change the underlying frame indices.
        self._set_status_mid("Fit To Window toggled")

        # Redraw using the new scale rule.
        # If videos are not loaded yet, _render_current_frames() is a no-op.
        self._render_current_frames()

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

    # Steps one frame backward.
    def on_step_back(self):
        # If we are locked and both videos are loaded, step the master timeline
        # and keep the stored offset alignment.
        if self.lock_lr.get() and self._both_videos_loaded():
            # Left is the master timeline for transport controls.
            li = int(self.left_frame_index.get())
            self._jump_frames_locked_with_offset("L", li - 1)
            return

        # Otherwise, fall back to the old behavior.
        self._nudge_frames_locked_or_single(delta=-1)

    # Steps one frame forward.
    def on_step_forward(self):
        # If we are locked and both videos are loaded, step the master timeline
        # and keep the stored offset alignment.
        if self.lock_lr.get() and self._both_videos_loaded():
            li = int(self.left_frame_index.get())
            self._jump_frames_locked_with_offset("L", li + 1)
            return

        # Otherwise, fall back to the old behavior.
        self._nudge_frames_locked_or_single(delta=+1)

    # Toggles playback on and off using a Tk after loop.
    def on_play_pause(self):
        # Do nothing unless at least one video is loaded.
        if not self.capL and not self.capR:
            return

        # Toggle playback state.
        self.is_playing = not self.is_playing

        # If enabling playback, start the loop immediately.
        if self.is_playing:
            self._playback_tick()
        else:
            # If disabling, cancel any scheduled tick.
            if self.play_after_id is not None:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None

    def on_speed_changed(self, _evt=None):
        # Speed affects playback step or timer interval later.
        self._set_status_mid(f"Speed set to {self.speed_var.get()} (UI only)")

    # Toggles lock mode.
    # When enabling lock, capture the current alignment as a fixed frame offset.
    def on_toggle_lock(self):
        # Update status UI.
        self._set_status_mid("Lock toggled")
        self._refresh_status_left()

        # Only define an offset when BOTH videos are loaded.
        # If only one video is loaded, lock is effectively meaningless.
        if self.lock_lr.get() and self._both_videos_loaded():
            # Read the current indices.
            li = int(self.left_frame_index.get())
            ri = int(self.right_frame_index.get())

            # Store offset so that future locked moves preserve the current alignment.
            # offset = R - L
            self.lock_offset_frames = ri - li

            # Do not jump any frames here.
            # The current point in time is already aligned by the user's manual scrubbing.
            self._set_status_mid(f"Lock enabled (offset {self.lock_offset_frames:+d} frames)")
            return

        # If disabling lock, we keep each slider where it is and do nothing else.
        if not self.lock_lr.get():
            self._set_status_mid("Lock disabled")

        # When enabling lock, unify indices to the left slider's current value.
        if self.lock_lr.get():
            master = int(round(self.left_slider.get()))
            self._jump_frames_locked_or_single(target_index=master)

    # Advances the timeline and schedules the next playback tick.
    def _playback_tick(self):
        # If playback was turned off between ticks, stop immediately.
        if not self.is_playing:
            return

        # Determine per tick frame step based on speed setting.
        # We implement speed by skipping frames rather than changing decode rate.
        step = 1
        if self.speed_var.get() == "0.25x":
            # 0.25x is implemented as a slower tick, not fractional frames.
            step = 1
            delay_ms = 160
        elif self.speed_var.get() == "0.5x":
            step = 1
            delay_ms = 80
        elif self.speed_var.get() == "1x":
            step = 1
            delay_ms = 40
        elif self.speed_var.get() == "2x":
            step = 2
            delay_ms = 40
        else:
            step = 4
            delay_ms = 40

        # Compute maximum index depending on lock mode.
        if self.lock_lr.get() and self.metaL and self.metaR:
            max_i = min(self.left_frame_max, self.right_frame_max)
            cur = int(self.left_frame_index.get())
            nxt = cur + step

            # Stop at the end.
            if nxt > max_i:
                self.is_playing = False
                self.play_after_id = None
                return

            # Advance both indices in lock mode.
            self.left_frame_index.set(nxt)
            self.right_frame_index.set(nxt)
            self.left_slider.set(nxt)
            self.right_slider.set(nxt)
        else:
            # Unlocked playback advances each loaded stream independently.
            if self.metaL:
                curL = int(self.left_frame_index.get())
                nxtL = curL + step
                nxtL = self._clamp(nxtL, 0, int(self.left_frame_max))
                self.left_frame_index.set(nxtL)
                self.left_slider.set(nxtL)

            if self.metaR:
                curR = int(self.right_frame_index.get())
                nxtR = curR + step
                nxtR = self._clamp(nxtR, 0, int(self.right_frame_max))
                self.right_frame_index.set(nxtR)
                self.right_slider.set(nxtR)

        # Render the new frames.
        self._render_current_frames()

        # Schedule the next tick.
        self.play_after_id = self.root.after(delay_ms, self._playback_tick)

    # -------------------------------------------------------------------------
    # Slider callbacks
    # -------------------------------------------------------------------------

    # Called whenever the user drags the left slider.
    # This is the primary scrubbing mechanism for the left timeline.
    def on_left_slider_changed(self, _value):
        # If we are moving the slider in code, ignore this callback.
        # This prevents recursion when lock mode updates both sliders.
        if self._suppress_slider_callbacks:
            return

        i = int(round(float(self.left_slider.get())))

        # In lock mode, left slider drives the master and right follows with offset.
        if self.lock_lr.get() and self._both_videos_loaded():
            self._jump_frames_locked_with_offset("L", i)
            return

        # If unlocked, the left slider only controls the left timeline.
        # We update the stored index so future renders use this frame.
        self.left_frame_index.set(i)

        # Update the numeric "Frame: i/max" label under the slider.
        self._update_frame_labels()

        # Render frames so the left pane updates immediately as the slider moves.
        # Right pane will render too if the right video is loaded, but it stays on its own index.
        self._render_current_frames()

    # Called whenever the user drags the right slider.
    # This is the primary scrubbing mechanism for the right timeline.
    def on_right_slider_changed(self, _value):
        # If we are moving the slider in code, ignore this callback.
        # This prevents recursion when lock mode updates both sliders.
        if self._suppress_slider_callbacks:
            return
    
        # Quantize slider float to an integer frame index.
        i = int(round(float(self.right_slider.get())))

        # In lock mode, right slider drives the master and left follows with offset.
        if self.lock_lr.get() and self._both_videos_loaded():
            self._jump_frames_locked_with_offset("R", i)
            return

        # Unlocked mode means right slider controls right video only.
        self.right_frame_index.set(i)

        # Update the numeric labels under each slider.
        self._update_frame_labels()

        # Render so the right pane updates immediately.
        self._render_current_frames()

    # Jumps timelines in lock mode while preserving the stored frame offset.
    # master_side indicates which slider the user is driving: "L" or "R".
    def _jump_frames_locked_with_offset(self, master_side, target_index):
        # Guard: lock mode requires both videos.
        if not self._both_videos_loaded():
            return

        # Convert target to int frame index.
        target = int(target_index)

        # Compute desired indices using the offset definition:
        # offset = R - L
        if master_side == "L":
            # User is driving left.
            li = target
            ri = li + int(self.lock_offset_frames)
        else:
            # User is driving right.
            ri = target
            li = ri - int(self.lock_offset_frames)

        # Clamp using Option A:
        # If one side hits an end stop, shift the other side to preserve offset.
        #
        # Left legal range is [0, left_frame_max]
        # Right legal range is [0, right_frame_max]
        lmax = int(self.left_frame_max)
        rmax = int(self.right_frame_max)

        # Clamp left first, and adjust right accordingly.
        if li < 0:
            li = 0
            ri = li + int(self.lock_offset_frames)
        elif li > lmax:
            li = lmax
            ri = li + int(self.lock_offset_frames)

        # Now clamp right, and adjust left accordingly.
        if ri < 0:
            ri = 0
            li = ri - int(self.lock_offset_frames)
        elif ri > rmax:
            ri = rmax
            li = ri - int(self.lock_offset_frames)

        # Final safety clamp in case adjustment pushed the other side slightly out.
        # This keeps indices always valid even in extreme offset cases.
        li = self._clamp(li, 0, lmax)
        ri = self._clamp(ri, 0, rmax)

        # Save indices.
        self.left_frame_index.set(li)
        self.right_frame_index.set(ri)

        # Update sliders without triggering callbacks.
        self._suppress_slider_callbacks = True
        try:
            self.left_slider.set(li)
            self.right_slider.set(ri)
        finally:
            self._suppress_slider_callbacks = False

        # Redraw.
        self._update_frame_labels()
        self._render_current_frames()

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

        # Only show an offset when lock is enabled and both videos are loaded.
        # This keeps the status line clean when you are still loading files.
        offset_txt = ""
        if self.lock_lr.get() and self._both_videos_loaded():
            offset_txt = f" | Offset: {self.lock_offset_frames:+d}f"

        self.status_left.config(
            text=f"L: {self._short_path(l)} | R: {self._short_path(r)} | Cal: {self._short_path(c)} | View: {view} | {lock}{offset_txt}"
        )

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

    # Draw placeholders only when we do not have video content to display.
    def _refresh_placeholder_canvases(self):
        # If either capture is loaded, we should be showing real frames, not placeholders.
        if self.capL or self.capR:
            self._render_current_frames()
            return

        self._draw_placeholder(self.left_overlay_canvas, "LEFT", self.view_rectified.get())
        self._draw_placeholder(self.right_overlay_canvas, "RIGHT", self.view_rectified.get())

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

            # Keep the frame counters under the sliders correct.
            self._update_frame_labels()

            # Now that we can decode frames, render actual video content.
            # This replaces placeholder drawing.
            self._render_current_frames()

    def _jump_frames_locked_or_single(self, target_index):
        # Jump to target_index in lock mode or update only the active slider.
        if self.lock_lr.get():
            max_i = int(min(self.left_frame_max, self.right_frame_max))
            i = self._clamp(int(target_index), 0, max_i)

            self.left_frame_index.set(i)
            self.right_frame_index.set(i)

            # Programmatically moving the sliders triggers their callbacks.
            # We suppress callbacks here to prevent recursive lock updates.
            self._suppress_slider_callbacks = True
            try:
                self.left_slider.set(i)
                self.right_slider.set(i)
            finally:
                self._suppress_slider_callbacks = False

            # Update the labels under the sliders so they reflect the new indices.
            self._update_frame_labels()

            # Render the current frames after the jump so the UI updates immediately.
            self._render_current_frames()
        else:
            # If unlocked, this helper is used for start/end operations.
            # We treat it as applying to both sides for toolbar actions.
            li = self._clamp(int(target_index), 0, int(self.left_frame_max))
            ri = self._clamp(int(target_index), 0, int(self.right_frame_max))

            self.left_frame_index.set(li)
            self.right_frame_index.set(ri)

            # Programmatically moving the sliders triggers their callbacks.
            # We suppress callbacks here to prevent recursive lock updates.
            self._suppress_slider_callbacks = True
            try:
                self.left_slider.set(li)
                self.right_slider.set(ri)
            finally:
                self._suppress_slider_callbacks = False

            self._update_frame_labels()
            self._refresh_placeholder_canvases()

    # Opens a video file and returns (cap, meta) or (None, None) on failure.
    def _open_video_capture(self, path):
        # Create the capture object.
        cap = cv2.VideoCapture(path)

        # Validate that the capture opened successfully.
        if not cap.isOpened():
            return None, None

        # Read metadata from the container.
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Guard against weird containers that report 0 frames.
        if frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            return None, None

        meta = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }

        return cap, meta
    

    # Updates slider max ranges and clamps indices based on lock mode and loaded videos.
    def _update_slider_ranges(self):
        # Compute per stream maximum indices.
        # Index is inclusive, so max = frame_count - 1.
        self.left_frame_max = (self.metaL["frame_count"] - 1) if self.metaL else 0
        self.right_frame_max = (self.metaR["frame_count"] - 1) if self.metaR else 0

        # If locked and both videos are loaded, clamp both to the shorter stream.
        if self.lock_lr.get() and self.metaL and self.metaR:
            master_max = min(self.left_frame_max, self.right_frame_max)

            # Clamp stored indices to valid range.
            li = self._clamp(int(self.left_frame_index.get()), 0, master_max)
            ri = self._clamp(int(self.right_frame_index.get()), 0, master_max)

            # Force both sides to the same master index (left is the master).
            self.left_frame_index.set(li)
            self.right_frame_index.set(li)

            # Update both slider ranges to match the clamped master range.
            self.left_slider.configure(to=master_max)
            self.right_slider.configure(to=master_max)

            # Updating slider position here should not invoke the slider callbacks.
            self._suppress_slider_callbacks = True
            try:
                self.left_slider.set(li)
                self.right_slider.set(li)
            finally:
                self._suppress_slider_callbacks = False
        else:
            # Unlocked mode uses independent ranges.
            # Each slider range is based on its own stream if loaded, else 0.
            self.left_slider.configure(to=int(self.left_frame_max))
            self.right_slider.configure(to=int(self.right_frame_max))

            # Clamp and apply each index independently.
            li = self._clamp(int(self.left_frame_index.get()), 0, int(self.left_frame_max))
            ri = self._clamp(int(self.right_frame_index.get()), 0, int(self.right_frame_max))

            self.left_frame_index.set(li)
            self.right_frame_index.set(ri)

            # Updating slider position here should not invoke the slider callbacks.
            self._suppress_slider_callbacks = True
            try:
                self.left_slider.set(li)
                self.right_slider.set(ri)
            finally:
                self._suppress_slider_callbacks = False

        # Always refresh the numeric labels under the sliders.
        self._update_frame_labels()


    # Seeks to a specific frame index and reads a single frame.
    def _read_frame_at(self, cap, index):
        # Seek to the requested frame index.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        # Decode a single frame.
        ok, frame_bgr = cap.read()
        if not ok:
            return None

        return frame_bgr
    

    # Displays a BGR frame on a Tk canvas using Tk's PNG decoder.
    # This avoids Pillow and avoids PPM decoding quirks in some Tk builds.
    def _display_bgr_on_canvas(self, canvas, frame_bgr, which):
        # Compute where the video should be drawn inside this canvas.
        dx, dy, dw, dh = self._get_display_rect(which, canvas)

        # If we have no metadata yet, just show a simple fit-by-width render.
        size = self._get_image_size(which)
        if size is None:
            # Fall back: draw the full frame scaled to the display width, preserve aspect.
            h, w = frame_bgr.shape[0], frame_bgr.shape[1]
            if dw > 0 and w > 0:
                scale = float(dw) / float(w)
                out_w = int(round(w * scale))
                out_h = int(round(h * scale))
                if out_w > 0 and out_h > 0:
                    frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            return

        # Total scale for this pane.
        view = self._get_view(which)
        S = self._get_total_scale(which, canvas)
        if S <= 0.0:
            S = 1.0

        # Compute visible ROI in IMAGE coords for the display rect (dw x dh).
        # We treat the display rect as the screen coordinate region for mapping.
        off_x = float(view["off_x"])
        off_y = float(view["off_y"])

        ix0 = (0.0 - off_x) / S
        iy0 = (0.0 - off_y) / S
        ix1 = (float(dw) - off_x) / S
        iy1 = (float(dh) - off_y) / S

        x0 = min(ix0, ix1)
        x1 = max(ix0, ix1)
        y0 = min(iy0, iy1)
        y1 = max(iy0, iy1)

        img_w, img_h = size

        # Clamp ROI to image bounds.
        x0 = max(0.0, min(float(img_w), x0))
        x1 = max(0.0, min(float(img_w), x1))
        y0 = max(0.0, min(float(img_h), y0))
        y1 = max(0.0, min(float(img_h), y1))

        rx0 = int(x0)
        ry0 = int(y0)
        rx1 = int(x1 + 0.9999)
        ry1 = int(y1 + 0.9999)

        # If ROI is invalid, draw a black frame layer.
        if rx1 <= rx0 or ry1 <= ry0:
            canvas.delete("frame")
            canvas.create_rectangle(0, 0, int(canvas.winfo_width()), int(canvas.winfo_height()), fill="black", outline="", tags=("frame",))
            return

        # Crop and scale to the display rect size (preserves aspect because dw/dh preserves it).
        crop = frame_bgr[ry0:ry1, rx0:rx1]
        crop = cv2.resize(crop, (int(dw), int(dh)), interpolation=cv2.INTER_LINEAR)

        # Encode to PNG for Tk PhotoImage.
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        ok, png_bytes = cv2.imencode(".png", crop, encode_params)
        if not ok:
            self._draw_missing_frame(canvas, "ENCODE", 0)
            return

        png_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
        tk_img = tk.PhotoImage(data=png_b64)

        if which == "L":
            self.tkimg_left = tk_img
        else:
            self.tkimg_right = tk_img

        # Replace only frame items: first clear frame layer.
        canvas.delete("frame")

        # Draw a black background so letterbox areas look clean.
        cw = int(max(1, canvas.winfo_width()))
        ch = int(max(1, canvas.winfo_height()))
        canvas.create_rectangle(0, 0, cw, ch, fill="black", outline="", tags=("frame",))

        # Draw the image inside the display rect.
        canvas.create_image(int(dx), int(dy), anchor="nw", image=tk_img, tags=("frame",))

     # Renders current left and right frames based on the current indices.
    def _render_current_frames(self):
        # Left side render.
        if self.capL:
            li = int(self.left_frame_index.get())
            frameL = self._read_frame_at(self.capL, li)

            # Do Stereo Rectification on this frame if show rectified is set
            if self.view_rectified.get() and self.cal is not None:
                frameL = cv2.remap(frameL, self.cal["mapLx"], self.cal["mapLy"], interpolation=cv2.INTER_LINEAR)

            if frameL is None:
                self._draw_missing_frame(self.left_overlay_canvas, "LEFT", li)
            else:
                self._display_bgr_on_canvas(self.left_overlay_canvas, frameL, "L")

        # Right side render.
        if self.capR:
            ri = int(self.right_frame_index.get())
            frameR = self._read_frame_at(self.capR, ri)

            # Do Stereo Rectification on this frame if show rectified is set
            if self.view_rectified.get() and self.cal is not None:
                frameR = cv2.remap(frameR, self.cal["mapRx"], self.cal["mapRy"], interpolation=cv2.INTER_LINEAR)

            if frameR is None:
                self._draw_missing_frame(self.right_overlay_canvas, "RIGHT", ri)
            else:
                self._display_bgr_on_canvas(self.right_overlay_canvas, frameR, "R")

        # Update the slider frame labels after rendering.
        self._update_frame_labels()

        # Draw overlay over frame
        self._redraw_overlays()

    # Computes the current image->screen scale factor for the given pane.
    # Fit To Window scales by width only, so we match that here.
    def _get_pane_scale(self, which, canvas):
        # Default scale is 1.0 (native pixel mapping).
        if not self.fit_to_window.get():
            return 1.0

        # We need source image width to compute scale.
        # If meta is missing, fall back to 1.0.
        if which == "L":
            if not self.metaL:
                return 1.0
            src_w = float(self.metaL["width"])
        else:
            if not self.metaR:
                return 1.0
            src_w = float(self.metaR["width"])

        # Canvas width is the target width under Fit To Window.
        canvas_w = float(max(1, canvas.winfo_width()))

        # Scale by width only.
        return canvas_w / src_w
    
    # Computes the on-canvas rectangle where the video should be drawn while preserving aspect ratio.
    # Returns (dx, dy, dw, dh) in SCREEN pixels.
    def _get_display_rect(self, which, canvas):
        # Canvas size in screen pixels.
        cw = int(max(1, canvas.winfo_width()))
        ch = int(max(1, canvas.winfo_height()))

        # If we do not know the image size yet, fall back to full canvas.
        size = self._get_image_size(which)
        if size is None:
            return 0, 0, cw, ch

        img_w, img_h = size

        # If Fit To Window is off, draw at native size anchored at top-left.
        # Clamp to canvas so we do not exceed widget bounds.
        if not self.fit_to_window.get():
            dw = min(cw, int(img_w))
            dh = min(ch, int(img_h))
            return 0, 0, dw, dh

        # Fit To Window means: fit by width, but preserve aspect.
        # Compute the height implied by fitting the image width to the canvas width.
        dw = cw
        dh = int(round(dw * (float(img_h) / float(img_w))))

        # If that height does not fit, instead fit by height (still preserving aspect).
        if dh > ch:
            dh = ch
            dw = int(round(dh * (float(img_w) / float(img_h))))

        # Center the draw rect within the canvas (letterboxing).
        dx = (cw - dw) // 2
        dy = (ch - dh) // 2

        return dx, dy, dw, dh

    # Draws a clear error message on a canvas when a frame cannot be decoded.
    def _draw_missing_frame(self, canvas, label, frame_index):
        # Only delete the frame layer so overlay items can persist on top.
        canvas.delete("frame")

        # Canvas size is needed to center the message.
        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())

        # Draw a border rectangle tagged as frame content.
        canvas.create_rectangle(
            2,
            2,
            w - 2,
            h - 2,
            outline="#444444",
            tags=("frame",),
        )

        # Draw the main missing frame text tagged as frame content.
        canvas.create_text(
            w // 2,
            h // 2 - 10,
            text=f"{label} FRAME MISSING",
            fill="white",
            font=("Segoe UI", 14, "bold"),
            tags=("frame",),
        )

        # Draw the frame index text tagged as frame content.
        canvas.create_text(
            w // 2,
            h // 2 + 18,
            text=f"Frame {frame_index}",
            fill="#cccccc",
            font=("Segoe UI", 11, "normal"),
            tags=("frame",),
        )

    # Returns True only when BOTH captures and metadata exist.
    def _both_videos_loaded(self):
        # Require both captures.
        if self.capL is None:
            return False
        if self.capR is None:
            return False

        # Require both metadata dicts.
        if self.metaL is None:
            return False
        if self.metaR is None:
            return False

        return True


def main():
    root = tk.Tk()

    # ttk theme defaults are OK. If you want a darker theme later, we can style it.
    app = SizeamaticProApp(root)

    root.protocol("WM_DELETE_WINDOW", app.on_app_close)

    root.mainloop()


if __name__ == "__main__":
    main()