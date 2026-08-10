"""Sizeamatic Pro main application module.

Sizeamatic Pro is a desktop tool for measuring real world distances from
stereo video. This module defines `SizeamaticProApp`, the Tkinter GUI
container that owns window/menu construction, video playback and timeline
state, calibration loading, and the top-level event wiring that ties
together the supporting feature modules (`stereo_matching`,
`measurement_window`, `calibration_summary`, `anaglyph_preview`,
`video_overlay`).

See `ARCHITECTURE.md` for how responsibilities are currently split across
files, and `README.md` for the user-facing description of the app.
"""

import os
import sys # Used for icon resources

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2

from PIL import Image, ImageTk

import ctypes # For App Model ID

# Local Imports
import stereo_matching
import measurement_window  # measurement_window contains the Tkinter measurement results window and update helpers.
import anaglyph_preview    # Manages the anaglyph_preview functionality
import calibration_summary # calibration_summary contains the Tkinter calibration summary window and update helpers.
import calibration_io      # Loads and validates calibration NPZ files, without the directory-chooser dialog.
import video_overlay # Manages drawing the overlay on the video


def resource_path(relative_path):
    """Get the correct path to a bundled resource file.

    When running normally, this returns a path relative to the source
    folder. When running from PyInstaller, this returns a path inside the
    bundled app.

    Args:
        relative_path (str): The file path relative to the project root.

    Returns:
        str: The absolute path to the requested resource.
    """

    # PyInstaller stores bundled files in a temporary/internal folder exposed here.
    if hasattr(sys, "_MEIPASS"):

        # Build the path to the bundled resource.
        return os.path.join(sys._MEIPASS, relative_path)

    # Otherwise, build the path from the normal source directory.
    return os.path.join(os.path.abspath("."), relative_path)


class SizeamaticProApp:
    """Main GUI container for Sizeamatic Pro.

    Owns the Tkinter window, menu, toolbar, video viewer panes, and status
    bar, plus all playback/timeline/zoom/measurement state for the app.
    Video decoding and rendering, calibration loading, and top-level event
    handling live directly on this class; stereo math, the measurement and
    calibration summary popup windows, the anaglyph preview, and overlay
    point interaction are delegated to the supporting feature modules
    imported at the top of this file (each function in those modules
    receives this instance as `app` to read/mutate its state).

    See the inline comments in `__init__` for what each instance attribute
    group is for (window/file/video/timeline/playback/zoom/measurement
    state, etc.) — they're intentionally dense per this project's
    commenting convention rather than restated here.
    """

    def __init__(self, root):
        self.root = root
        """The Tk root window passed in from `main()`. Owns every widget in
        the app — menu, toolbar, viewer panes, status bar all attach to
        this."""

        self.video_overlay = video_overlay.VideoOverlay(self)
        """Owns the overlay canvases and overlay interaction state. See
        `video_overlay.VideoOverlay`. Constructed this early (before
        `_build_menu`/`_build_viewers` below) because `_build_viewers`
        calls `self.video_overlay.create_canvases()` — unlike
        `self.cal_summary_window`/`self.measurement_window`/
        `self.anaglyph_preview`, which only need to exist before a user
        action first opens them."""

        # ---- Window setup ----
        self.root.title("Sizeamatic Pro")
        self.root.minsize(1100, 700)

        # ---- State flags (UI only for now) ----
        self.view_rectified = tk.BooleanVar(value=False)
        """Whether the video panes show rectified (calibration-aligned) or
        raw camera frames. Gated by `on_toggle_view_rectified`, which
        forces this back to False if calibration isn't loaded or its
        resolution doesn't match the loaded video(s)."""

        self.fit_to_window = tk.BooleanVar(value=True)
        """Whether each pane scales its video to fit the current window
        size (True) or shows the video at native pixel size (False). Read
        throughout the display-rect/scale calculations in
        `_get_fit_scale`/`_get_display_rect`."""

        self.show_overlays = tk.BooleanVar(value=True)
        """Whether measurement point overlays are shown. Currently only
        wired up for the placeholder canvases (`_draw_placeholder`) — see
        `on_toggle_show_overlays`'s docstring for the UI-only caveat."""

        self.show_epipolar = tk.BooleanVar(value=False)
        """Whether the epipolar cursor line indicator is shown. Currently
        only wired up for the placeholder canvases; only really makes
        conceptual sense in rectified view, but that isn't enforced yet."""

        self.lock_lr = tk.BooleanVar(value=True)
        """Whether the left and right timelines are locked together at a
        fixed frame offset (`self.lock_offset_frames`), so scrubbing one
        side moves the other in sync."""

         # ---- File state ----
        self.left_video_path = None
        """Path to the loaded left video file, or None if not loaded yet."""

        self.right_video_path = None
        """Path to the loaded right video file, or None if not loaded yet."""

        self.calibration_folder = None
        """Path to the loaded calibration folder, or None if not loaded
        yet. Kept around purely for status-bar display
        (`_refresh_status_left`) — the actual calibration data lives in
        `self.cal`."""

        # ---- OpenCV video captures ----
        self.capL = None
        """The left video's `cv2.VideoCapture`, or None if not loaded.
        Stays open for the lifetime of the app once loaded, so re-opening
        the file isn't needed on every frame — released and reopened on a
        fresh load (`on_load_left_video`) or on app close
        (`on_app_close`). Note that keeping this open does **not** by
        itself make seeking fast — `cap.set(CAP_PROP_POS_FRAMES)` forces
        an expensive keyframe seek regardless; see `_read_frame_at`, which
        checks the capture's own reported position before deciding
        whether to seek at all."""

        self.capR = None
        """The right video's `cv2.VideoCapture`, or None if not loaded.
        Mirrors `self.capL` for the right side."""

        # ---- Video metadata ----
        self.metaL = None
        """Metadata dict for the left video (`{"fps", "width", "height",
        "frame_count"}`), or None if not loaded. Always set together with
        `self.capL` — code elsewhere (e.g. `_display_bgr_on_canvas`)
        assumes that pairing holds."""

        self.metaR = None
        """Metadata dict for the right video, or None if not loaded.
        Mirrors `self.metaL` for the right side."""

        # ---- Timeline state ----
        self.left_frame_index = tk.IntVar(value=0)
        """Current frame index on the left timeline. This is the
        authoritative "where are we" value for the left pane — the slider
        position is derived from/synced to it, not the other way around."""

        self.right_frame_index = tk.IntVar(value=0)
        """Current frame index on the right timeline. Mirrors
        `self.left_frame_index` for the right side."""

        self.left_frame_max = 0
        """Highest valid left frame index (`frame_count - 1`), or 0 if no
        left video is loaded. Recomputed by `_update_slider_ranges`
        whenever a video loads or lock mode changes — in lock mode this
        gets clamped down to match the shorter of the two streams."""

        self.right_frame_max = 0
        """Highest valid right frame index. Mirrors `self.left_frame_max`
        for the right side."""

        # ---- Playback loop state ----
        self.is_playing = False
        """Whether the playback loop (`_playback_tick`) is currently
        running."""

        self.play_after_id = None
        """Tkinter `after()` job ID for the next scheduled playback tick,
        so it can be cancelled when playback stops. None when not
        playing."""

        # ---- Tk image handles ----
        self.tkimg_left = None
        """The `tkinter.PhotoImage` currently shown in the left pane. Tk
        does not keep its own strong reference to image data drawn on a
        canvas, so this attribute exists purely to keep the image alive —
        if it weren't stored somewhere, Tk would garbage-collect it and
        the canvas would go blank."""

        self.tkimg_right = None
        """The `tkinter.PhotoImage` currently shown in the right pane.
        Mirrors `self.tkimg_left` for the right side."""

        self._suppress_slider_callbacks = False
        """Set True while code (not the user) is moving a slider, so
        `on_left_slider_changed`/`on_right_slider_changed` can tell the
        difference and avoid recursive updates when lock mode programmatically
        repositions both sliders."""

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

        self.lock_offset_frames = 0
        """When `self.lock_lr` is True, the fixed frame offset that keeps
        the two timelines aligned, defined as `right_index - left_index`
        (e.g. +12 means right is 12 frames ahead of left). Captured at the
        moment lock is enabled (`on_toggle_lock`) from whatever alignment
        the user had already scrubbed to manually — enabling lock never
        jumps either timeline itself."""

        self._resize_after_id = None
        """Tkinter `after()` job ID for the debounced resize redraw, so a
        pending redraw can be cancelled and rescheduled. Resize events can
        fire dozens of times per second while dragging the window, so
        `on_canvas_resized` debounces through this rather than redrawing
        on every single event."""

        self.ptsL = []
        """Left-pane measurement points, in image pixel coordinates. This
        (along with `self.ptsR`) is the authoritative measurement state —
        everything else (overlays, the measurement window, triangulation)
        is derived from these two lists."""

        self.ptsR = []
        """Right-pane measurement points, in image pixel coordinates.
        Mirrors `self.ptsL` for the right side; index *i* in each list is
        expected to be the same physical point, matched between the two
        views."""

        self.max_points_per_pane = 2
        """Point cap per pane. Starts at 2 (a single line/segment); raising
        this later would let `self.video_overlay`'s `draw_pane` and
        `_update_measurement_status_stub`'s segment math extend naturally
        into a multi-point polyline, since both already connect points as
        a consecutive chain rather than independent pairs."""

        self.handle_radius_px = 8
        """Point handle radius, in screen pixels (after scaling). Kept
        fairly large so handles are easy to click directly without needing
        precise hit-test math — `self.video_overlay`'s
        `get_nearest_handle_index` also uses a multiple of this as a
        forgiving fallback hit radius."""

        self.drag_active = False
        """Vestigial. Conceptually "whether a left-button point handle
        drag is active", but the actual drag handling lives on
        `self.video_overlay` (its own `drag_active` attribute) and never
        reads this copy — see finding 7 in `FINDINGS.md`. Only
        `on_clear_points` still writes to it, so clicking "Clear Points"
        mid-drag doesn't actually stop a real drag (harmless:
        `self.video_overlay`'s own drag handlers bounds-check the point
        index anyway, so a stale drag just no-ops once the point list is
        cleared)."""

        self.drag_which = None
        """Vestigial, same as `self.drag_active` — see finding 7 in
        `FINDINGS.md`."""

        self.drag_index = None
        """Vestigial, same as `self.drag_active` — see `FINDINGS.md` #7."""

        self.refine_drag_active = False
        """Vestigial, same as `self.drag_active` but for the explicit
        right-button refinement drag — see `FINDINGS.md` #7."""

        self.refine_drag_which = None
        """Vestigial, same as `self.refine_drag_active` — see
        `FINDINGS.md` #7."""

        self.refine_drag_index = None
        """Vestigial, same as `self.refine_drag_active` — see
        `FINDINGS.md` #7."""

        self.viewL = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0}
        """Left pane's pan/zoom view state. "zoom" is a unitless multiplier
        applied on top of fit-to-window scaling (see
        `_get_total_scale`); "off_x"/"off_y" are pan offsets in screen
        pixels, applied after scaling, and get recentered on the cursor
        during mouse-wheel zoom (`on_mouse_wheel`)."""

        self.viewR = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0}
        """Right pane's pan/zoom view state. Mirrors `self.viewL` for the
        right side."""

        self.zoom_min = 1.0
        """Minimum allowed zoom multiplier for either pane."""

        self.zoom_max = 10.0
        """Maximum allowed zoom multiplier for either pane."""

        self.zoom_step = 1.10
        """Zoom multiplier applied per mouse wheel notch (each notch
        multiplies or divides the current zoom by this factor)."""

        self.cal = None
        """Loaded stereo calibration bundle, or None if not loaded or
        invalid. When set, this is a dict with keys for intrinsics
        ("mtxL"/"distL"/"mtxR"/"distR"), extrinsics ("R"/"T"/"E"/"F"/
        "stereo_rms"), rectification ("RL"/"RR"/"PL"/"PR"/"Q"/"roiL"/
        "roiR"), remap arrays ("mapLx"/"mapLy"/"mapRx"/"mapRy"), and
        calibrated size ("w"/"h") — see
        `calibration_io.load_calibration_bundle`, which builds this dict;
        `on_load_calibration_folder` is the only caller."""

        self.measurement_window = measurement_window.MeasurementWindow(self)
        """Owns the measurement results Toplevel window and its widgets.
        See `measurement_window.MeasurementWindow`."""

        self.click_sigma_px = 3.0
        """Assumed user click-placement uncertainty, in image pixels. An
        explicit modeling assumption (not a measured value) fed into
        `stereo_matching.py`'s perturbation-based uncertainty estimates
        (`estimate_point_sigma_mm`, `estimate_segment_sigma_len_mm`) to
        translate pixel-level click imprecision into millimeter-level
        depth/length uncertainty estimates."""

        self.cal_summary_window = calibration_summary.CalibrationSummaryWindow(self)
        """Owns the calibration summary Toplevel window and its widgets.
        See `calibration_summary.CalibrationSummaryWindow`."""

        self.anaglyph_preview = anaglyph_preview.AnaglyphPreview(self)
        """Owns the anaglyph preview's OpenCV window and playback state.
        See `anaglyph_preview.AnaglyphPreview`."""

        # Rename the preview window title to match this app rather than
        # that class's generic default.
        self.anaglyph_preview.window_name = "Sizeamatic Pro - Anaglyph 3D"

    def on_toggle_anaglyph_preview(self):
        """Start or stop the anaglyph preview window.

        Requires both videos to be loaded. Toggles based on the current
        `self.anaglyph_preview.active` state.

        Returns:
            None
        """
        # Require both videos loaded.
        if not self._both_videos_loaded():
            self._set_status_mid("Load both videos to use anaglyph preview")
            return

        # Toggle behavior.
        if self.anaglyph_preview.active:
            self.anaglyph_preview.stop()
            return

        self.anaglyph_preview.start()

    def on_show_calibration_summary(self):
        """Open (or focus) the calibration summary window.

        Requires calibration to already be loaded.

        Returns:
            None
        """
        if self.cal is None:
            self._set_status_mid("Load calibration first")
            return

        self.cal_summary_window.ensure_window()
        self.cal_summary_window.update_window()

    def on_mouse_wheel(self, which, event):
        """Handle mouse wheel zoom for a pane, anchored under the cursor.

        Adjusts the pane's zoom level and pans so the image point that was
        under the cursor before zooming stays under the cursor afterward.

        Args:
            which (str): Which pane received the wheel event, "L" or "R".
            event (tkinter.Event): The Tkinter mouse wheel event. Uses
                `event.delta` (Windows: typically ±120 per notch) and
                `event.x`/`event.y` for the cursor position.

        Returns:
            None
        """
        canvas = self.video_overlay.left_canvas if which == "L" else self.video_overlay.right_canvas

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

    def on_app_close(self):
        """Close OpenCV windows and release captures before exiting.

        Stops the playback loop, stops the anaglyph preview if running,
        releases both video captures, destroys any OpenCV windows, and
        destroys the Tk root window.

        Returns:
            None
        """
        # Stop playback loop.
        self.is_playing = False
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

        # Close the anaglyph viewer if it is running.
        if self.anaglyph_preview.active:
            self.anaglyph_preview.stop()

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

    def on_canvas_resized(self, _event):
        """Handle a video/overlay canvas resize event.

        Debounces redraw to avoid decoding and encoding on every resize
        event, since resize events can fire dozens of times per second
        while dragging the window.

        Args:
            _event (tkinter.Event): The Tkinter configure event (unused).

        Returns:
            None
        """
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

    def _redraw_after_resize(self):
        """Redraw the current frames after the resize debounce delay elapses.

        Returns:
            None
        """
        # Clear the pending handle first.
        self._resize_after_id = None

        # If no videos are loaded yet, keep the placeholders.
        if not self.capL and not self.capR:
            self._refresh_placeholder_canvases()
            return

        # Otherwise, redraw the current decoded frames.
        # This will re run the Fit To Window scaling logic.
        self._render_current_frames()

    def _get_view(self, which):
        """Get the pan/zoom view state dict for a pane.

        Args:
            which (str): Which pane's view state to return, "L" or "R".

        Returns:
            dict: `self.viewL` or `self.viewR`, each with keys "zoom",
            "off_x", "off_y".
        """
        if which == "L":
            return self.viewL
        return self.viewR

    def _get_image_size(self, which):
        """Get the source image width/height for a pane.

        Args:
            which (str): Which pane's image size to return, "L" or "R".

        Returns:
            tuple[int, int] | None: `(width, height)` if that pane's video
            metadata is loaded, otherwise None.
        """
        if which == "L":
            if not self.metaL:
                return None
            return int(self.metaL["width"]), int(self.metaL["height"])
        else:
            if not self.metaR:
                return None
            return int(self.metaR["width"]), int(self.metaR["height"])

    def _get_fit_scale(self, which, canvas):
        """Compute the base fit-to-window scale (by width only) for a pane.

        Args:
            which (str): Which pane to compute the scale for, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for that pane.

        Returns:
            float: The fit-to-window scale factor, or 1.0 if Fit To Window
            is disabled or image size is not yet known.
        """
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

    def _get_total_scale(self, which, canvas):
        """Compute the total image-to-screen scale for a pane.

        Total scale combines the fit-to-window base scale and the user's
        zoom level: `S = fit_scale * zoom`.

        Args:
            which (str): Which pane to compute the scale for, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for that pane.

        Returns:
            float: The total image-to-screen scale factor.
        """
        view = self._get_view(which)
        fit_scale = self._get_fit_scale(which, canvas)
        return fit_scale * float(view["zoom"])

    def _image_to_screen(self, which, canvas, ix, iy):
        """Convert image pixel coordinates to overlay screen coordinates.

        Args:
            which (str): Which pane the point belongs to, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for that pane.
            ix (float): Image X pixel coordinate.
            iy (float): Image Y pixel coordinate.

        Returns:
            tuple[float, float]: The corresponding (sx, sy) screen
            coordinates.
        """
        view = self._get_view(which)

        # Display rect defines where the video lives inside the canvas.
        dx, dy, _dw, _dh = self._get_display_rect(which, canvas)

        # Total scale includes Fit To Window scale and zoom.
        S = self._get_total_scale(which, canvas)

        # off_x/off_y are pan offsets in screen pixels relative to the display rect.
        sx = float(dx) + float(ix) * S + float(view["off_x"])
        sy = float(dy) + float(iy) * S + float(view["off_y"])
        return sx, sy

    def _screen_to_image(self, which, canvas, sx, sy):
        """Convert overlay screen coordinates to image pixel coordinates.

        Args:
            which (str): Which pane the point belongs to, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for that pane.
            sx (float): Screen X pixel coordinate.
            sy (float): Screen Y pixel coordinate.

        Returns:
            tuple[float, float]: The corresponding (ix, iy) image pixel
            coordinates.
        """
        view = self._get_view(which)

        dx, dy, _dw, _dh = self._get_display_rect(which, canvas)

        S = self._get_total_scale(which, canvas)
        if S <= 0.0:
            S = 1.0

        # Convert screen->image by undoing display rect origin and offsets first.
        ix = (float(sx) - float(dx) - float(view["off_x"])) / S
        iy = (float(sy) - float(dy) - float(view["off_y"])) / S
        return ix, iy

    def _update_measurement_status_stub(self):
        """Recompute measurements from current points and refresh the UI.

        Triangulates all currently paired left/right points, builds the
        point diagnostics and segment rows, updates the status bar's right
        section with a short summary (or the reason measurement isn't
        available), and refreshes the measurement results window.

        Note:
            If a point in the middle of the list fails to triangulate, the
            loop below stops there (via `break`) but the function still
            continues on to report a summary count for whatever points
            triangulated successfully beforehand — the partial-failure
            `err_msg` is passed on to the measurement popup window (which
            does display it), but is not shown in this window's own status
            bar, which instead gets overwritten with the "Measured N pts"
            summary. Worth being aware of if a click intermittently fails
            to triangulate: the main status bar won't say why.

        Returns:
            None
        """
        l_count = len(self.ptsL)
        r_count = len(self.ptsR)

        # Quick gating messages stay in the status bar.
        if l_count == 0 and r_count == 0:
            self._set_status_right("")
            return

        if l_count != r_count:
            self._set_status_right(f"Point pair incomplete: L={l_count} R={r_count}")
            return

        if not self.view_rectified.get():
            self._set_status_right("Enable rectified view to measure")
            return

        if self.cal is None:
            self._set_status_right("Load calibration to measure")
            return

        n = min(l_count, r_count)

        # Compute 3D points as many as we can.
        pts3d = []
        err_msg = ""

        for i in range(n):
            P, err = stereo_matching.triangulate_point_pair(self, i)
            if err is not None:
                err_msg = f"Point {i} failed: {err}"
                break
            pts3d.append(P)

        # If we got nothing, show the error and return.
        if len(pts3d) == 0:
            self._set_status_right(err_msg if err_msg else "No valid points")
            return

        # Build rows for the points table.
        points_rows = []
        sigma_px = float(self.click_sigma_px)

        for i, (X, Y, Z) in enumerate(pts3d):
            R = (X * X + Y * Y + Z * Z) ** 0.5

            # Assumption-free quality metric.
            erms = stereo_matching.reprojection_rms_px(self, i)
            erms_str = f"{erms:.2f}" if erms is not None else ""

            # Assumption-based uncertainty in mm.
            sig = stereo_matching.estimate_point_sigma_mm(self, i, sigma_px)
            if sig is None:
                sZ_str = ""
                sR_str = ""
            else:
                sZ, sR = sig
                sZ_str = f"{sZ:.1f}"
                sR_str = f"{sR:.1f}"

            # Read the clicked left and right pixels for this point.
            xL, yL = self.ptsL[i]
            xR, yR = self.ptsR[i]

            # Compute disparity, which drives stereo depth.
            disp = xL - xR

            # Compute rectified Y mismatch between left and right clicks.
            dy = yR - yL

            # Store the formatted point row for the UI table and copy block.
            points_rows.append((
                str(i),
                f"{X:.1f}",
                f"{Y:.1f}",
                f"{Z:.1f}",
                f"{R:.1f}",
                f"{disp:.2f}",
                f"{dy:.2f}",
                erms_str,
                sZ_str,
                sR_str,
            ))

        # Build rows for the segments table.
        seg_rows = []
        if len(pts3d) >= 2:
            for i in range(1, len(pts3d)):
                X0, Y0, Z0 = pts3d[i - 1]
                X1, Y1, Z1 = pts3d[i]
                dX = X1 - X0
                dY = Y1 - Y0
                dZ = Z1 - Z0
                L = (dX * dX + dY * dY + dZ * dZ) ** 0.5

                # Segment sigma length estimate.
                seg_est = stereo_matching.estimate_segment_sigma_len_mm(self, i - 1, i, sigma_px)
                if seg_est is None:
                    sL_str = ""
                else:
                    _L0, sL = seg_est
                    sL_str = f"{sL:.1f}"

                seg_rows.append((
                    f"{i-1}-{i}",
                    f"{dX:.1f}",
                    f"{dY:.1f}",
                    f"{dZ:.1f}",
                    f"{L:.1f}",
                    sL_str,
                ))

            self._set_status_right(f"Measured {len(pts3d)} pts, {len(seg_rows)} segs")
        else:
            self._set_status_right("Measured 1 point")

        # Update popup window (creates it on first valid measurement).
        self.measurement_window.update_window(points_rows, seg_rows, err_msg)

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------

    def _build_menu(self):
        """Build the File and View menus and attach them to the root window.

        Returns:
            None
        """
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

        view_menu.add_command(label="Anaglyph 3D Preview…", command=self.on_toggle_anaglyph_preview)

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

        view_menu.add_separator()
        view_menu.add_command(label="Calibration Summary…", command=self.on_show_calibration_summary)

        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    # -------------------------------------------------------------------------
    # Toolbar
    # -------------------------------------------------------------------------

    def _build_toolbar(self):
        """Build the transport/speed/lock/clear-points toolbar.

        Returns:
            None
        """
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

    def on_clear_points(self):
        """Clear all measurement points in both panes.

        Cancels any active drag state, redraws overlays, refreshes
        measurement status, and shows a confirmation in the status bar.
        This is currently the only point-delete mechanism.

        Returns:
            None
        """
        # Clear both point lists to keep pairing consistent.
        self.ptsL.clear()
        self.ptsR.clear()

        # Cancel any active drag state.
        self.drag_active = False
        self.drag_which = None
        self.drag_index = None

        # Redraw overlays to remove handles and lines.
        self.video_overlay.redraw()

        # Update measurement status text.
        self._update_measurement_status_stub()

        # Show a short confirmation in the center status area.
        self._set_status_mid("Points cleared")

    def _fmt_mm(self, v):
        """Format a float millimeter value for display.

        Args:
            v (float): The value, in millimeters.

        Returns:
            str: The formatted value with one decimal place and a unit
            suffix, e.g. "12.3 mm".
        """
        # Use one decimal place to keep it readable, but still precise enough.
        return f"{v:.1f} mm"

    # -------------------------------------------------------------------------
    # Viewer panes
    # -------------------------------------------------------------------------

    def _build_viewers(self):
        """Build the left/right video viewer panes, sliders, and overlays.

        Creates the resizable paned window containing the left and right
        video viewports (each a stacked video canvas with an overlay
        canvas on top, created via `self.video_overlay.create_canvases`),
        plus the frame-scrubbing slider and frame label under each pane.

        Returns:
            None
        """
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

        # Create overlay canvases for point drawing and point interaction.
        self.video_overlay.create_canvases()




    # -------------------------------------------------------------------------
    # Status bar
    # -------------------------------------------------------------------------

    def _build_statusbar(self):
        """Build the three-section status bar (left/mid/right labels).

        Returns:
            None
        """
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

    def on_load_left_video(self):
        """Prompt for and load the left video, updating UI state.

        Releases any previously open left capture, opens the newly
        selected file, updates the header/slider/frame state, and
        re-renders.

        Returns:
            None
        """
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

    def on_load_right_video(self):
        """Prompt for and load the right video, updating UI state.

        Releases any previously open right capture, opens the newly
        selected file, updates the header/slider/frame state, and
        re-renders. Mirrors `on_load_left_video` for the right pane and
        works independently of whether the left video is loaded.

        Returns:
            None
        """
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

    def on_load_calibration_folder(self):
        """Prompt for and load a stereo calibration folder.

        Delegates the actual file loading/validation to
        `calibration_io.load_calibration_bundle` — this method just
        handles the directory dialog and updating UI state from the
        result. On any failure, clears `self.cal`, disables rectified
        view, and shows a status message explaining why.

        Returns:
            None
        """
        folder = filedialog.askdirectory(title="Load Calibration Folder")
        if not folder:
            return

        # Store the folder path for status display.
        self.calibration_folder = folder

        cal, err = calibration_io.load_calibration_bundle(folder, self.metaL, self.metaR)

        if err is not None:
            self.cal = None
            self.view_rectified.set(False)
            self._set_status_mid(err)
            self._refresh_status_left()
            return

        self.cal = cal

        self._set_status_mid("Calibration loaded")
        self._refresh_status_left()

        # Trigger redraw so rectified mode can be enabled immediately.
        self._render_current_frames()

        # In real wiring, you will enable "Show Rectified" only after maps load.
        # For now, we leave it togglable to test UI.

    # -------------------------------------------------------------------------
    # Stub handlers (view toggles)
    # -------------------------------------------------------------------------

    def on_toggle_view_rectified(self):
        """Toggle between raw and rectified stereo video display.

        If turning rectified view on, validates that calibration is loaded
        and (if videos are loaded) that their resolution matches the
        calibrated resolution, forcing the toggle back off with a status
        message if not.

        Returns:
            None
        """
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

    def on_toggle_fit_to_window(self):
        """Toggle fit-to-window rendering and redraw the current frames.

        Returns:
            None
        """
        # Fit-to-window changes the display size calculation.
        # It does not change the underlying frame indices.
        self._set_status_mid("Fit To Window toggled")

        # Redraw using the new scale rule.
        # If videos are not loaded yet, _render_current_frames() is a no-op.
        self._render_current_frames()

    def on_toggle_show_overlays(self):
        """Toggle the "Show Overlays" UI flag and refresh placeholders.

        Note:
            Currently UI-only for the placeholder canvases; does not yet
            enable/disable actual point/line drawing during real video
            playback.

        Returns:
            None
        """
        # In real wiring, this would enable/disable drawing points/lines on canvas.
        self._set_status_mid("Show Overlays toggled (UI only)")
        self._refresh_placeholder_canvases()

    def on_toggle_show_epipolar(self):
        """Toggle the "Show Epipolar Cursor Line" UI flag and refresh placeholders.

        Note:
            Currently UI-only; only makes conceptual sense when rectified
            view is active, but that is not yet enforced here.

        Returns:
            None
        """
        # In real wiring, only makes sense when rectified is active.
        self._set_status_mid("Epipolar cursor toggled (UI only)")
        self._refresh_placeholder_canvases()

    # -------------------------------------------------------------------------
    # Stub handlers (toolbar)
    # -------------------------------------------------------------------------

    def on_to_start(self):
        """Jump the timeline (or locked timelines) to frame 0.

        Returns:
            None
        """
        self._jump_frames_locked_or_single(target_index=0)

    def on_to_end(self):
        """Jump the timeline (or locked timelines) to the current max frame.

        "End" means whatever the current slider max is for each stream
        (or the shorter of the two streams, when locked).

        Returns:
            None
        """
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
        """Step the timeline one frame backward.

        In lock mode with both videos loaded, steps the left timeline (the
        master) and lets the right timeline follow via the stored offset.
        Otherwise falls back to independent per-side stepping.

        Returns:
            None
        """
        # If we are locked and both videos are loaded, step the master timeline
        # and keep the stored offset alignment.
        if self.lock_lr.get() and self._both_videos_loaded():
            # Left is the master timeline for transport controls.
            li = int(self.left_frame_index.get())
            self._jump_frames_locked_with_offset("L", li - 1)
            return

        # Otherwise, fall back to the old behavior.
        self._nudge_frames_locked_or_single(delta=-1)

    def on_step_forward(self):
        """Step the timeline one frame forward.

        In lock mode with both videos loaded, steps the left timeline (the
        master) and lets the right timeline follow via the stored offset.
        Otherwise falls back to independent per-side stepping.

        Returns:
            None
        """
        # If we are locked and both videos are loaded, step the master timeline
        # and keep the stored offset alignment.
        if self.lock_lr.get() and self._both_videos_loaded():
            li = int(self.left_frame_index.get())
            self._jump_frames_locked_with_offset("L", li + 1)
            return

        # Otherwise, fall back to the old behavior.
        self._nudge_frames_locked_or_single(delta=+1)

    def on_play_pause(self):
        """Toggle playback on/off, driven by a Tk `after()` loop.

        Does nothing if no video is loaded. Starts `_playback_tick`
        immediately when enabling playback; cancels the scheduled tick when
        disabling it.

        Returns:
            None
        """
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
        """Handle a playback speed selection change.

        Note:
            Currently UI-only status text; the actual speed/step logic
            lives in `_playback_tick`, which reads `self.speed_var`
            directly rather than through this handler.

        Args:
            _evt (tkinter.Event | None): The combobox selection event
                (unused).

        Returns:
            None
        """
        # Speed affects playback step or timer interval later.
        self._set_status_mid(f"Speed set to {self.speed_var.get()} (UI only)")

    def on_toggle_lock(self):
        """Toggle lock mode between the left/right timelines.

        When enabling lock with both videos loaded, captures the current
        alignment as a fixed frame offset (`right_index - left_index`) so
        future locked moves preserve it, without jumping either timeline.

        Returns:
            None
        """
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

    def _playback_tick(self):
        """Advance the timeline by one playback step and schedule the next tick.

        Reads the current speed setting to determine the per-tick frame
        step and delay, advances the locked master timeline (or each
        unlocked stream independently), re-renders, and reschedules itself
        via `root.after` unless playback has stopped or hit the end.

        Returns:
            None
        """
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

    def on_left_slider_changed(self, _value):
        """Handle the user dragging the left timeline slider.

        The primary scrubbing mechanism for the left timeline. In lock
        mode with both videos loaded, drives the master/offset jump logic;
        otherwise updates only the left timeline.

        Args:
            _value (str): The new slider value as a string (Tkinter scale
                callback convention); unused, `self.left_slider.get()` is
                read directly instead.

        Returns:
            None
        """
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

    def on_right_slider_changed(self, _value):
        """Handle the user dragging the right timeline slider.

        The primary scrubbing mechanism for the right timeline. In lock
        mode with both videos loaded, drives the master/offset jump logic;
        otherwise updates only the right timeline.

        Args:
            _value (str): The new slider value as a string (Tkinter scale
                callback convention); unused, `self.right_slider.get()` is
                read directly instead.

        Returns:
            None
        """
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

    def _jump_frames_locked_with_offset(self, master_side, target_index):
        """Jump both timelines in lock mode, preserving the stored frame offset.

        Computes the desired left/right indices from `target_index` and
        `self.lock_offset_frames` (defined as `offset = R - L`), then
        clamps using "Option A": if one side would hit an end stop, the
        other side is shifted to preserve the offset rather than letting
        the offset itself change. A final safety clamp keeps both indices
        valid even in extreme offset cases (which can slightly break exact
        offset preservation at the very ends of the shorter stream — a
        known, accepted simplification, not a silent defect).

        Args:
            master_side (str): Which slider the user is driving, "L" or
                "R".
            target_index (int): The requested frame index for the driving
                side.

        Returns:
            None
        """
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
        """Handle a raw canvas click (placeholder for future interactions).

        Note:
            This is currently unused/superseded by the overlay canvas
            click handling in `self.video_overlay`
            (`on_left_down`/`on_right_down`), which is bound to the
            overlay canvases instead of this handler. Kept as a minimal
            placeholder that just reports click coordinates.

        Args:
            which (str): Which pane was clicked, "L" or "R".
            event (tkinter.Event): The Tkinter mouse click event.

        Returns:
            None
        """
        # Placeholder for later measurement interactions.
        # Keep it minimal: show click coords.
        self._set_status_mid(f"{which} click at ({event.x}, {event.y}) (UI only)")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _refresh_status_left(self):
        """Refresh the status bar's left section with file/view/lock state.

        Returns:
            None
        """
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
        """Truncate a file path for compact status bar display.

        Args:
            path (str | None): The path to shorten, or None.
            max_len (int): Maximum displayed length before truncating with
                a leading ellipsis.

        Returns:
            str: "(none)" if `path` is None, the path unchanged if it fits
            within `max_len`, otherwise an ellipsis-prefixed suffix of the
            path.
        """
        if path is None:
            return "(none)"
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1):]

    def _set_status_mid(self, text):
        """Set the status bar's center (message/warning) text.

        Args:
            text (str): The text to display.

        Returns:
            None
        """
        self.status_mid.config(text=text)

    def _set_status_right(self, text):
        """Set the status bar's right (measurement results) text.

        Args:
            text (str): The text to display.

        Returns:
            None
        """
        self.status_right.config(text=text)

    def _update_frame_labels(self):
        """Refresh the "Frame: i/max" labels under both sliders.

        Returns:
            None
        """
        # Frame max is currently 0 because no video is loaded.
        # Later you will set left_frame_max/right_frame_max from cv2 capture length.
        lmax = max(0, int(self.left_frame_max))
        rmax = max(0, int(self.right_frame_max))

        li = int(self.left_frame_index.get())
        ri = int(self.right_frame_index.get())

        self.left_frame_label.config(text=f"Frame: {li}/{lmax}")
        self.right_frame_label.config(text=f"Frame: {ri}/{rmax}")

    def _refresh_placeholder_canvases(self):
        """Draw placeholder graphics, or render real frames if videos are loaded.

        Draws placeholders only when there is no video content to display;
        if either capture is already loaded, delegates to
        `_render_current_frames` instead.

        Returns:
            None
        """
        # If either capture is loaded, we should be showing real frames, not placeholders.
        if self.capL or self.capR:
            self._render_current_frames()
            return

        self._draw_placeholder(self.video_overlay.left_canvas, "LEFT", self.view_rectified.get())
        self._draw_placeholder(self.video_overlay.right_canvas, "RIGHT", self.view_rectified.get())

        self._update_frame_labels()

    def _draw_placeholder(self, canvas, label, rectified):
        """Draw a placeholder grid/label graphic on a pane's canvas.

        Used before any video is loaded, so the pane isn't just blank —
        shows the pane label, raw/rectified mode, and current
        overlay/epipolar toggle state for visual confirmation while wiring
        up the UI.

        Args:
            canvas (tkinter.Canvas): The canvas to draw on.
            label (str): The pane label to display, e.g. "LEFT" or
                "RIGHT".
            rectified (bool): Whether to show "RECTIFIED" or "RAW" mode
                text.

        Returns:
            None
        """
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
        """Clamp a value into an inclusive [lo, hi] range.

        Args:
            x: The value to clamp.
            lo: The inclusive lower bound.
            hi: The inclusive upper bound.

        Returns:
            The clamped value.
        """
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _nudge_frames_locked_or_single(self, delta):
        """Adjust the current frame(s) by a small step, respecting lock mode.

        In lock mode, steps the shared master timeline (clamped to the
        shorter stream) via `_jump_frames_locked_or_single`. Otherwise,
        steps each side's timeline independently, each clamped to its own
        max.

        Args:
            delta (int): The signed number of frames to step by.

        Returns:
            None
        """
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
        """Jump to a target frame index in lock mode, or update the active slider.

        In lock mode, jumps both timelines to the same clamped index
        (clamped to the shorter stream's max). When unlocked, this helper
        is used for start/end toolbar actions and applies the target index
        to both sides independently, each clamped to its own max.

        Args:
            target_index (int): The requested frame index.

        Returns:
            None
        """
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

    def _open_video_capture(self, path):
        """Open a video file and read its container metadata.

        Args:
            path (str): Path to the video file to open.

        Returns:
            tuple[cv2.VideoCapture, dict] | tuple[None, None]: The opened
            capture and a metadata dict with keys "fps", "width",
            "height", "frame_count", or `(None, None)` if the file could
            not be opened or reports an invalid (zero or negative)
            width/height/frame count.
        """
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

    def _update_slider_ranges(self):
        """Update slider max ranges and clamp indices based on lock mode.

        Recomputes `left_frame_max`/`right_frame_max` from loaded video
        metadata. In lock mode with both videos loaded, clamps both
        sliders' ranges to the shorter stream and forces both indices to
        match (left is master). Otherwise, each slider's range and index
        is set independently.

        Returns:
            None
        """
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

    def _read_frame_at(self, cap, index):
        """Decode the frame at a specific index, seeking only if needed.

        Skips the explicit `cap.set(CAP_PROP_POS_FRAMES)` seek when the
        capture's own reported position (`cap.get(CAP_PROP_POS_FRAMES)`)
        is already at the requested index — checking the position is
        essentially free (~0.0002ms measured), while `cap.set` forces an
        expensive keyframe seek even for a one-frame advance, benchmarked
        at ~19x slower than reading sequentially (54ms vs 2.8ms per frame
        on a real 1080p capture). This was the dominant remaining cost in
        rectified playback fps, bigger than the render path itself; see
        ROADMAP.md Phase 5.

        Checking the capture's actual reported position (rather than
        tracking "the last index this method itself read") matters
        because more than one part of the app can read from the same
        capture — the main render loop and `anaglyph_preview.py`'s
        independent preview tick both call this with `app.capL`/
        `app.capR`, each with their own frame index sequence. Tracking
        only this method's own last call would get confused by an
        interleaved read from a different sequence; asking the capture
        directly is correct regardless of who else touched it in between.

        Args:
            cap (cv2.VideoCapture): The capture to read from.
            index (int): The zero-based frame index to read.

        Returns:
            numpy.ndarray | None: The decoded BGR frame, or None if the
            seek/decode failed.
        """
        index = int(index)

        # Only seek if the capture isn't already positioned to read this
        # exact frame next.
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        # Decode a single frame.
        ok, frame_bgr = cap.read()
        if not ok:
            return None

        return frame_bgr

    def _display_bgr_on_canvas(self, canvas, frame_bgr, which):
        """Display a BGR frame on a Tk canvas via Pillow's ImageTk.

        Crops to the currently visible (pan/zoom) region in image space,
        resizes to the pane's display rect, converts BGR to RGB, and draws
        it via `PIL.Image.fromarray` + `ImageTk.PhotoImage` — wrapping the
        numpy array directly with no encode/decode round-trip.

        Note:
            This used to encode each frame to PNG, base64-encode that, and
            hand the base64 string to `tkinter.PhotoImage` (deliberately
            avoiding Pillow, per a comment in an earlier version of this
            method). That round-trip, redone on every single render for
            both panes, was the actual cause of the "unacceptably slow"
            rectified rendering the README used to warn about — not
            Tkinter itself. Switching to Pillow's direct-numpy-array path
            fixed it; see ROADMAP.md Phase 5.

        Note:
            If this pane's image size isn't known yet (`_get_image_size`
            returns None), this function computes a width-fit resized
            frame but then returns without ever drawing it — that resize
            result is discarded. In the current app flow this branch
            should be unreachable in practice, since `capL`/`metaL` (and
            `capR`/`metaR`) are always set together when a video loads, so
            image size is already known by the time this is called with a
            decoded frame. Worth fixing if that invariant ever changes.

        Args:
            canvas (tkinter.Canvas): The canvas to draw on.
            frame_bgr (numpy.ndarray): The decoded BGR video frame.
            which (str): Which pane this is for, "L" or "R".

        Returns:
            None
        """
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

        # Convert BGR (OpenCV) to RGB (PIL) and wrap directly as a Tk image.
        # No encode/decode round-trip: this is what actually fixed the
        # "unacceptably slow" rectified rendering — the previous PNG-encode
        # + base64 + Tk-parses-base64 path re-encoded a full frame on every
        # single render, for both panes.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(crop_rgb))

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

    def _render_current_frames(self):
        """Render the current left and right frames based on the current indices.

        Reads and (if rectified view is enabled) remaps the frame at each
        pane's current index, caches it as `self.current_frameL`/
        `self.current_frameR` for use by stereo matching, displays it (or
        a missing-frame placeholder on decode failure), updates the frame
        labels, and redraws the point overlays.

        Returns:
            None
        """
        # Left side render.
        if self.capL:
            li = int(self.left_frame_index.get())
            frameL = self._read_frame_at(self.capL, li)

            # Do Stereo Rectification on this frame if show rectified is set
            if self.view_rectified.get() and self.cal is not None:
                frameL = cv2.remap(frameL, self.cal["mapLx"], self.cal["mapLy"], interpolation=cv2.INTER_LINEAR)

            if frameL is None:
                # Clear the cached left frame because there is no valid image to match against.
                self.current_frameL = None

                # Draw the missing-frame placeholder on the left pane.
                self._draw_missing_frame(self.video_overlay.left_canvas, "LEFT", li)
            else:
                # Cache the exact left image currently being displayed.
                # If rectified view is enabled, this is the rectified frame.
                self.current_frameL = frameL

                # Display the current left frame on the left pane.
                self._display_bgr_on_canvas(self.video_overlay.left_canvas, frameL, "L")

        # Right side render.
        if self.capR:
            ri = int(self.right_frame_index.get())
            frameR = self._read_frame_at(self.capR, ri)

            # Do Stereo Rectification on this frame if show rectified is set
            if self.view_rectified.get() and self.cal is not None:
                frameR = cv2.remap(frameR, self.cal["mapRx"], self.cal["mapRy"], interpolation=cv2.INTER_LINEAR)

            if frameR is None:
                # Clear the cached right frame because there is no valid image to match against.
                self.current_frameR = None

                # Draw the missing-frame placeholder on the right pane.
                self._draw_missing_frame(self.video_overlay.right_canvas, "RIGHT", ri)
            else:
                # Cache the exact right image currently being displayed.
                # If rectified view is enabled, this is the rectified frame.
                self.current_frameR = frameR

                # Display the current right frame on the right pane.
                self._display_bgr_on_canvas(self.video_overlay.right_canvas, frameR, "R")

        # Update the slider frame labels after rendering.
        self._update_frame_labels()

        # Draw overlay over frame
        self.video_overlay.redraw()

    def _get_display_rect(self, which, canvas):
        """Compute the on-canvas rectangle where video should be drawn.

        Preserves aspect ratio. When Fit To Window is off, draws at native
        size anchored top-left (clamped to canvas bounds). When on, fits by
        width first, falling back to fitting by height if that would
        overflow the canvas, then centers the result (letterboxing).

        Args:
            which (str): Which pane's display rect to compute, "L" or "R".
            canvas (tkinter.Canvas): The canvas being measured.

        Returns:
            tuple[int, int, int, int]: `(dx, dy, dw, dh)` in screen pixels.
        """
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

    def _draw_missing_frame(self, canvas, label, frame_index):
        """Draw a clear error message on a canvas when a frame can't be decoded.

        Only clears the "frame" tagged canvas layer so overlay items can
        persist on top.

        Args:
            canvas (tkinter.Canvas): The canvas to draw on.
            label (str): The pane label to display, e.g. "LEFT" or
                "RIGHT".
            frame_index (int): The frame index that failed to decode, shown
                to the user for context.

        Returns:
            None
        """
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

    def _both_videos_loaded(self):
        """Check whether both left and right video captures and metadata exist.

        Returns:
            bool: True only when both `capL`/`capR` and `metaL`/`metaR`
            are set.
        """
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
    """Entry point: build the Tk root window and run the application.

    Sets the Windows taskbar application identity (so the app groups under
    its own taskbar icon rather than a generic Python one), creates the Tk
    root window, applies the window icon if available, constructs
    `SizeamaticProApp`, wires up the close protocol, and starts the Tk
    event loop.

    Returns:
        None
    """

    # Set the Windows taskbar application identity.
    if sys.platform == "win32":

        # Use a stable unique ID for Windows taskbar grouping.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SizeamaticPro.SizeamaticPro.App"
        )

    root = tk.Tk()

    # Set the application window icon, if one is present.
    # assets/icon.ico is not currently committed to the repo, so this is
    # best-effort: fall back to the default Tk icon rather than crashing.
    try:
        root.iconbitmap(resource_path("assets/icon.ico"))
    except tk.TclError:
        pass

    # ttk theme defaults are OK. If you want a darker theme later, we can style it.
    app = SizeamaticProApp(root)

    root.protocol("WM_DELETE_WINDOW", app.on_app_close)

    root.mainloop()


if __name__ == "__main__":
    main()
