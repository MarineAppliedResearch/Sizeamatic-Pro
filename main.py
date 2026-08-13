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

import datetime # Used for the real-world time anchor/sync feature.
import os
import sys # Used for icon resources
import time # Used to enforce the startup splash's minimum display duration.
import tomllib # Reads pyproject.toml's version for the project file's app_version field.

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
import perform_calibration  # Owns the Perform Calibration window, capturing calibration frame pairs (ROADMAP.md Phase 10).
import calibration_io      # Loads and validates calibration NPZ files, without the directory-chooser dialog.
import project_io          # Saves/loads a project manifest (video paths, calibration folder, resync offset).
import recent_projects      # Persists the File > Recent Projects submenu's list of project paths.
import prepare_splash_image # Flattens/labels the startup splash image (works from source or packaged).
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


def get_app_version():
    """Read this app's version string from `pyproject.toml`.

    A standalone module-level function (rather than only a method on
    `SizeamaticProApp`) so the startup splash — shown before that app
    object is even constructed — can also read the version, for the
    "vX.Y.Z" text drawn onto it (ROADMAP.md Phase 9). Reads via
    `resource_path` rather than a path relative to `__file__`, since a
    packaged build's `__file__` resolves inside PyInstaller's temporary
    extraction folder, not next to a real `pyproject.toml` — the build
    (`sizeamatic.spec`) bundles `pyproject.toml` as a data file
    specifically so this still resolves correctly there too.

    Returns:
        str: The version string (e.g. "0.1.0"), or "unknown" if
        `pyproject.toml` can't be found or parsed for any reason.
    """
    try:
        with open(resource_path("pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
        return str(data["project"]["version"])
    except Exception:
        return "unknown"


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
        self.current_project_name = None
        """The loaded project's file name, without its directory or
        ".json" extension, or None if no project has been saved/opened
        this session yet. Set by `on_save_project`/
        `_open_project_from_path`; read by `_app_window_title` to show
        which project every window (main, Measurement, Calibration
        Summary) belongs to. Declared here, before the first
        `self.root.title(...)` call just below, rather than in the
        later scattered-init section — the same FINDINGS.md #6
        attribute-ordering pitfall as `offset_var`/`real_time_anchor_*`
        before it: `_app_window_title` reads it immediately below."""

        self.root.title(self._app_window_title())
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

        self.offset_var = tk.IntVar(value=0)
        """Tk-bound mirror of `self.lock_offset_frames` (declared much
        later below, near the rest of the transport/lock state — this one
        has to live here instead, before `_build_toolbar()` runs, since
        that method's Spinbox references it as a `textvariable`; see
        FINDINGS.md #6 for the general scattered-init-order issue this
        runs into). Kept in sync in both directions: `on_toggle_lock`
        updates it when lock capture computes a new offset, and
        `on_offset_changed` updates `self.lock_offset_frames` when the
        user edits it directly."""

        # Real-world time anchor entry: six separate plain text boxes (year,
        # month, day, hour, minute, second) rather than one free-text field to
        # parse, or spinners — the project owner specifically didn't want
        # either. Start empty, not pre-filled with "now": the six boxes exist
        # to be typed into, matching whatever's burned into the video, not
        # edited from a default that has nothing to do with the footage.
        # Declared here for the same before-`_build_toolbar()` reason as
        # `self.offset_var` above.
        self.real_time_year_var = tk.StringVar(value="")
        """Tk-bound year text box for the real-time anchor entry. Starts
        empty; only read (and validated) when "Set Time Sync" is pressed."""

        self.real_time_month_var = tk.StringVar(value="")
        """Tk-bound month text box (expected 1-12) for the real-time anchor entry."""

        self.real_time_day_var = tk.StringVar(value="")
        """Tk-bound day-of-month text box for the real-time anchor entry.
        Not validated against the specific month/year as you type —
        `on_real_time_entered` catches the `ValueError` from constructing
        the actual `datetime.datetime` (e.g. day 31 in a 30-day month) and
        reports it rather than crashing."""

        self.real_time_hour_var = tk.StringVar(value="")
        """Tk-bound hour text box (expected 0-23) for the real-time anchor entry."""

        self.real_time_minute_var = tk.StringVar(value="")
        """Tk-bound minute text box (expected 0-59) for the real-time anchor entry."""

        self.real_time_second_var = tk.StringVar(value="")
        """Tk-bound second text box (expected 0-59) for the real-time
        anchor entry. Together with the five boxes above, read by
        "Set Time Sync" (`on_real_time_entered`) to build the anchor
        `datetime.datetime` — see `self.real_time_anchor_dt` below."""

        self.real_time_anchor_frame = None
        """The left-timeline frame index the user was on when they last
        set the real-world time anchor (`on_real_time_entered`), or None
        if no anchor has been set. One shared anchor referenced to the
        left/master timeline — consistent with how measurements already
        treat left as the reference (see `_current_measurement_context`)
        — not a separate anchor per pane. Declared here (rather than with
        the rest of the measurement-point state below) because
        `_update_frame_labels` — which reads it via `_format_actual_time`
        — already runs during `__init__` itself (via
        `_refresh_placeholder_canvases`), before that later, scattered
        section runs; see FINDINGS.md #6."""

        self.real_time_anchor_dt = None
        """The `datetime.datetime` the user typed in at
        `self.real_time_anchor_frame`, read off whatever real-world clock
        is burned into the video image itself — or None if no anchor has
        been set. Together with `self.real_time_anchor_frame` and the
        left video's fps, this is what `_format_actual_time` uses to
        calculate the real-world time at any other frame: anchor time +
        (frame - anchor frame) / fps. See ROADMAP.md Phase 8's
        video-time-sync item."""

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

        self.current_frameL = None
        """The exact left image currently displayed (post-rectification,
        if enabled), cached by `_render_current_frames` for use by stereo
        matching and by `_redisplay_current_frames` (panning). None until
        the first successful render, or if the last decode attempt
        failed. Initialized here (rather than only coming into existence
        on first render) so code that reads it before any frame has ever
        been rendered gets a clean None instead of an AttributeError."""

        self.current_frameR = None
        """The exact right image currently displayed. Mirrors
        `self.current_frameL` for the right side."""

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
        # Row 3: shared Frame/Video Time/Actual Time readout (by the scrub bars)
        # Row 4: status bar
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
        jumps either timeline itself. Also directly editable via the
        resync offset Spinbox (`self.offset_var`, `on_offset_changed`) for
        manually correcting a misaligned pair — see ROADMAP.md Phase 7."""

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

        self.last_recorded_snapshot = None
        """Enough state to restore the most recently *Recorded*
        measurement from a project file later: a dict with keys
        "left_frame_index", "right_frame_index", "ptsL", "ptsR", or None
        if nothing's been recorded this session yet. Updated by
        `_on_measurement_recorded`, called from
        `measurement_window.py`'s `record_current_measurement` right
        after it successfully appends to the Log."""

        self.max_points_per_pane = 20
        """Point cap per pane. A generous fixed ceiling rather than a
        precisely-reasoned limit — high enough that no realistic
        multi-segment measurement chain hits it, while still bounding
        worst-case UI/computation cost. Raised from the original cap of 2
        (a single segment) once `self.video_overlay`'s `draw_pane` and
        `_update_measurement_status_stub`'s segment math were confirmed to
        already extend naturally into a multi-point polyline — both
        already connect points as a consecutive chain rather than
        independent pairs — see ROADMAP.md Phase 7's measurement output
        item."""

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

        self.pan_active = False
        """Whether a middle-mouse-button pan drag is currently active.
        Deliberately a separate button from point placement (left) and
        explicit point refinement (right) so panning never collides with
        either — see `on_pan_down`."""

        self.pan_which = None
        """Which pane, "L" or "R", owns the pan drag currently in
        progress. `None` when no pan is active."""

        self.pan_last_x = 0
        """Screen X coordinate of the pan drag's most recent mouse event,
        used to compute the incremental delta on the next `on_pan_drag`
        call."""

        self.pan_last_y = 0
        """Screen Y coordinate of the pan drag's most recent mouse event.
        Mirrors `self.pan_last_x` for the Y axis."""

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

        self.perform_calibration_window = perform_calibration.PerformCalibrationWindow(self)
        """Owns the Perform Calibration Toplevel window and its widgets -
        capturing calibration frame pairs from the currently loaded
        left/right video (ROADMAP.md Phase 10). See
        `perform_calibration.PerformCalibrationWindow`."""

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

    def on_perform_calibration(self):
        """Open (or focus) the Perform Calibration window.

        Unlike `on_show_calibration_summary`, this doesn't require an
        existing calibration to already be loaded — capturing frame
        pairs is how a *new* calibration gets built in the first place
        (ROADMAP.md Phase 10).

        Returns:
            None
        """
        self.perform_calibration_window.ensure_window()

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

    def on_pan_down(self, which, event):
        """Begin a middle-mouse-button pan drag for one pane.

        Records the starting cursor position so `on_pan_drag` can compute
        an incremental delta on each subsequent move event.

        Args:
            which (str): Which pane received the button press, "L" or "R".
            event (tkinter.Event): The Tkinter mouse event.

        Returns:
            None
        """
        self.pan_active = True
        self.pan_which = which
        self.pan_last_x = event.x
        self.pan_last_y = event.y

    def on_pan_drag(self, which, event):
        """Continue a middle-mouse-button pan drag for one pane.

        Nudges the pane's view offset by however far the cursor moved
        since the last event, then redraws using the already-decoded
        current frame rather than re-reading from the video capture — see
        `_redisplay_current_frames` for why that distinction matters here.

        Args:
            which (str): Which pane the drag event belongs to, "L" or
                "R". Ignored unless it matches the pane the drag started
                on.
            event (tkinter.Event): The Tkinter mouse event.

        Returns:
            None
        """
        # Ignore drag events unless this pane owns the active pan.
        if not self.pan_active or self.pan_which != which:
            return

        view = self._get_view(which)

        # Nudge the pan offset by the on-screen distance moved since the last event.
        view["off_x"] += float(event.x - self.pan_last_x)
        view["off_y"] += float(event.y - self.pan_last_y)

        # Remember this position as the baseline for the next drag event.
        self.pan_last_x = event.x
        self.pan_last_y = event.y

        # Redraw with the new offset without re-decoding the video.
        self._redisplay_current_frames()

    def on_pan_up(self, which, _event):
        """End a middle-mouse-button pan drag for one pane.

        Args:
            which (str): Which pane received the button release, "L" or
                "R". Ignored unless it matches the pane the drag started
                on.
            _event (tkinter.Event): The Tkinter mouse event (unused).

        Returns:
            None
        """
        if self.pan_active and self.pan_which == which:
            self.pan_active = False
            self.pan_which = None

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

    def _format_timestamp(self, frame_index, fps):
        """Format a frame index as an HH:MM:SS:FF timecode.

        The trailing "FF" is the frame number *within* that second
        (0-based, wrapping at the video's own fps) — not a fraction of a
        second — so scrubbing to a specific frame shows exactly which
        frame that is, the same way professional video timecode does,
        rather than a decimal fraction that doesn't map onto anything
        the frame slider actually understands.

        Args:
            frame_index (int): Zero-based frame index.
            fps (float | None): The video's frames-per-second, or None/0
                if unknown.

        Returns:
            str: The formatted timecode, or "?" if `fps` isn't a usable
            positive number (e.g. no video loaded yet).
        """
        if not fps or fps <= 0:
            return "?"

        # Round fps to the nearest whole number of frames-per-second for the
        # purposes of bucketing frames into seconds — exact integer
        # arithmetic on frame_index itself, no floating-point seconds
        # involved, so there's no rounding drift between this and the
        # frame slider's own integer frame count.
        fps_int = max(1, round(float(fps)))
        frame_index = int(frame_index)

        whole_seconds, frame_in_second = divmod(frame_index, fps_int)
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame_in_second:02d}"

    def _format_actual_time(self, frame_index):
        """Calculate and format the real-world time at a given frame.

        Uses the real-world time anchor (`self.real_time_anchor_frame`/
        `self.real_time_anchor_dt`, set by `on_real_time_entered`) plus
        the left video's fps to project forward/backward from that one
        known point, in whole frames rather than fractional seconds — see
        `_format_timestamp`'s docstring for why. See ROADMAP.md Phase 8's
        video-time-sync item.

        Args:
            frame_index (int): The left-timeline frame index to calculate
                the real-world time for.

        Returns:
            str: The calculated real-world time as
            "YYYY-MM-DD HH:MM:SS:FF" (FF = frame number within that
            second), or "(not set)" if no anchor has been set yet or the
            left video's fps isn't known.
        """
        if self.real_time_anchor_frame is None or self.real_time_anchor_dt is None:
            return "(not set)"

        fps = self.metaL["fps"] if self.metaL else None
        if not fps or fps <= 0:
            return "(not set)"

        fps_int = max(1, round(float(fps)))
        frame_delta = int(frame_index) - int(self.real_time_anchor_frame)

        # divmod floors toward negative infinity for a positive divisor, so a
        # negative frame_delta still lands on a frame_in_second in [0, fps_int)
        # rather than a negative frame count - e.g. one frame before the
        # anchor is "the second before, frame fps_int - 1", not "-1 frames".
        whole_seconds_delta, frame_in_second = divmod(frame_delta, fps_int)

        actual_dt = self.real_time_anchor_dt + datetime.timedelta(seconds=whole_seconds_delta)
        return actual_dt.strftime("%Y-%m-%d %H:%M:%S") + f":{frame_in_second:02d}"

    def _advance_real_time_focus(self, entry, next_entry, max_len):
        """Move focus to the next real-time-anchor box once this one looks full.

        Purely a typing convenience bound to each box's `<KeyRelease>` —
        does not validate or apply anything; that only happens when
        "Set Time Sync" is pressed (`on_real_time_entered`).

        Args:
            entry (ttk.Entry): The box that was just typed into.
            next_entry (ttk.Entry | None): The box to focus next, or None
                if this is the last one (seconds).
            max_len (int): How many characters this box is expected to
                hold (e.g. 4 for year, 2 for the rest) before advancing.

        Returns:
            None
        """
        if next_entry is not None and len(entry.get()) >= max_len:
            next_entry.focus_set()
            next_entry.select_range(0, "end")

    def on_real_time_entered(self, _evt=None):
        """Handle the user pressing "Set Time Sync".

        Reads the six year/month/day/hour/minute/second text boxes and,
        if they form a valid date/time, anchors it to the left
        timeline's current frame index — from then on,
        `_format_actual_time` can calculate the real-world time at any
        other frame. Requires the left video to already be loaded (its
        fps is needed for that calculation). Nothing is validated or
        applied by typing alone — only this explicit action does that,
        deliberately, so a half-typed date never triggers a premature
        error dialog.

        Args:
            _evt (tkinter.Event | None): Unused; present only so this can
                also be bound directly to a widget event if ever needed.

        Returns:
            None
        """
        try:
            year = int(self.real_time_year_var.get().strip())
            month = int(self.real_time_month_var.get().strip())
            day = int(self.real_time_day_var.get().strip())
            hour = int(self.real_time_hour_var.get().strip())
            minute = int(self.real_time_minute_var.get().strip())
            second = int(self.real_time_second_var.get().strip())
            parsed = datetime.datetime(year, month, day, hour, minute, second)
        except (ValueError, tk.TclError):
            # ValueError: a box is empty/non-numeric, or the values parsed as
            # ints fine but don't form a real date (e.g. day 31 in a 30-day
            # month). Either way, there's nothing safe to anchor yet.
            messagebox.showerror(
                "Real Time",
                "That's not a valid date/time — check that every box is "
                "filled in and the day of month is valid.",
            )
            return

        if not self.metaL:
            self._set_status_mid("Load the left video before setting a real-time anchor")
            return

        self.real_time_anchor_frame = int(self.left_frame_index.get())
        self.real_time_anchor_dt = parsed

        self._set_status_mid(f"Real time anchored at frame {self.real_time_anchor_frame}")
        self._show_time_sync_indicator()
        self._update_frame_labels()

    def _show_time_sync_indicator(self):
        """Show the checkmark next to "Set Time Sync" confirming an anchor is set.

        Returns:
            None
        """
        self.time_sync_indicator.config(text="✓ Synced")

    def _refresh_real_time_entries(self, frame_index):
        """Update the six real-time anchor text boxes to the calculated
        actual time at a given frame.

        A no-op if no anchor is set yet — so the boxes stay exactly as
        the user is typing them until "Set Time Sync" actually establishes
        an anchor; once one exists, this keeps the boxes live-tracking the
        calculated real-world time as the frame changes (scrubbing,
        playback, stepping), not frozen at the original anchor value.
        Called from `_update_frame_labels` (every frame change) and from
        `on_open_project` (right after restoring a saved anchor, with
        `frame_index` equal to the anchor's own frame, so the boxes show
        exactly what was saved).

        Args:
            frame_index (int): The left-timeline frame index to display
                the calculated actual time for.

        Returns:
            None
        """
        if self.real_time_anchor_frame is None or self.real_time_anchor_dt is None:
            return

        fps = self.metaL["fps"] if self.metaL else None
        if not fps or fps <= 0:
            return

        elapsed_seconds = (float(frame_index) - float(self.real_time_anchor_frame)) / float(fps)
        current_dt = self.real_time_anchor_dt + datetime.timedelta(seconds=elapsed_seconds)

        self.real_time_year_var.set(f"{current_dt.year:04d}")
        self.real_time_month_var.set(f"{current_dt.month:02d}")
        self.real_time_day_var.set(f"{current_dt.day:02d}")
        self.real_time_hour_var.set(f"{current_dt.hour:02d}")
        self.real_time_minute_var.set(f"{current_dt.minute:02d}")
        self.real_time_second_var.set(f"{current_dt.second:02d}")

    def _current_measurement_context(self):
        """Build the video/frame/timestamp identifying info for the current measurement.

        This is what lets a copied-and-pasted measurement row still mean
        something once it's sitting in a spreadsheet with no other
        context — see ROADMAP.md Phase 7's measurement output item.
        Always reads the *left* timeline/video, since measurements are
        computed in the rectified left camera coordinate frame (see
        `README.md`'s "Measurement notes").

        Returns:
            dict: Keys "video_name" (str), "frame_index" (int),
            "timestamp" (str, elapsed video time since frame 0), and
            "actual_time" (str, the calculated real-world time if a
            real-time anchor is set — see ROADMAP.md Phase 8's
            video-time-sync item — or "" if not).
        """
        if self.left_video_path:
            video_name = os.path.basename(self.left_video_path)
        else:
            video_name = "(no video)"

        frame_index = int(self.left_frame_index.get())
        fps = self.metaL["fps"] if self.metaL else None

        actual_time = self._format_actual_time(frame_index)
        if actual_time == "(not set)":
            # A spreadsheet column should be empty when there's nothing to
            # show, not carry a placeholder string as if it were real data.
            actual_time = ""

        return {
            "video_name": video_name,
            "frame_index": frame_index,
            "timestamp": self._format_timestamp(frame_index, fps),
            "actual_time": actual_time,
        }

    def _update_measurement_status_stub(self):
        """Recompute measurements from current points and refresh the UI.

        Triangulates all currently paired left/right points, builds one
        unified list of result rows (a "Point" row per point, a "Segment"
        row per consecutive pair, and — for 2+ points — one "Total" row
        summing the connected chain's segment lengths), updates the
        status bar's right section with a short summary (or the reason
        measurement isn't available), and refreshes the measurement
        results window. See `measurement_window.py`'s module docstring
        for why points/segments/total are one flat, `Type`-tagged table
        rather than separate shapes.

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

        # Video/frame/timestamp context, shared by every row this call produces.
        ctx = self._current_measurement_context()
        video_col = ctx["video_name"]
        frame_col = str(ctx["frame_index"])
        time_col = ctx["timestamp"]
        actual_time_col = ctx["actual_time"]

        rows = []
        sigma_px = float(self.click_sigma_px)

        # Build one "Point" row per clicked point pair.
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

            rows.append((
                video_col, frame_col, time_col, actual_time_col, "",
                "Point", str(i),
                f"{X:.1f}", f"{Y:.1f}", f"{Z:.1f}", f"{R:.1f}",
                f"{disp:.2f}", f"{dy:.2f}", erms_str, sZ_str, sR_str,
            ))

        # Build one "Segment" row per consecutive point pair (the chain is a
        # single connected polyline: 0-1, 1-2, 2-3, ... — see
        # video_overlay.py's draw_pane docstring for why), plus a running
        # total length and quadrature-summed sigma across the whole chain.
        total_len_mm = 0.0
        total_var_mm2 = 0.0
        have_total_sigma = True

        if len(pts3d) >= 2:
            for i in range(1, len(pts3d)):
                X0, Y0, Z0 = pts3d[i - 1]
                X1, Y1, Z1 = pts3d[i]
                dX = X1 - X0
                dY = Y1 - Y0
                dZ = Z1 - Z0
                L = (dX * dX + dY * dY + dZ * dZ) ** 0.5
                total_len_mm += L

                # Segment sigma length estimate.
                seg_est = stereo_matching.estimate_segment_sigma_len_mm(self, i - 1, i, sigma_px)
                if seg_est is None:
                    sL_str = ""
                    # Can't propagate a total sigma if any segment along the
                    # chain is missing one.
                    have_total_sigma = False
                else:
                    _L0, sL = seg_est
                    sL_str = f"{sL:.1f}"
                    total_var_mm2 += sL * sL

                rows.append((
                    video_col, frame_col, time_col, actual_time_col, "",
                    "Segment", f"{i-1}-{i}",
                    f"{dX:.1f}", f"{dY:.1f}", f"{dZ:.1f}", f"{L:.1f}",
                    "", "", "", sL_str, "",
                ))

            # Total: sum of the connected chain's segment lengths. Segment
            # sigmas are each estimated independently (see
            # estimate_segment_sigma_len_mm), so a sum of independent errors
            # adds in quadrature: sigma_total = sqrt(sum(sigma_i^2)).
            total_sigma_str = f"{total_var_mm2 ** 0.5:.1f}" if have_total_sigma else ""
            rows.append((
                video_col, frame_col, time_col, actual_time_col, "",
                "Total", "",
                "", "", "", f"{total_len_mm:.1f}",
                "", "", "", total_sigma_str, "",
            ))

            self._set_status_right(
                f"Measured {len(pts3d)} pts, {len(pts3d) - 1} segs, total {total_len_mm:.1f}mm"
            )
        else:
            self._set_status_right("Measured 1 point")

        # Update popup window (creates it on first valid measurement).
        self.measurement_window.update_window(rows, err_msg)

    def _on_measurement_recorded(self):
        """Snapshot enough state to restore this exact measurement later.

        Called by `self.measurement_window.record_current_measurement`
        right after it successfully appends to the Log. Records which
        frame each timeline was on and the exact clicked points at this
        moment, so a saved project file can jump back to "the very last
        place that was recorded" and show those same points again on
        reopen (`on_open_project`) — see ROADMAP.md Phase 7's project
        file item.

        Returns:
            None
        """
        self.last_recorded_snapshot = {
            "left_frame_index": int(self.left_frame_index.get()),
            "right_frame_index": int(self.right_frame_index.get()),
            "ptsL": [list(p) for p in self.ptsL],
            "ptsR": [list(p) for p in self.ptsR],
        }

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------

    def _build_menu(self):
        """Build the File, View, and Calibration menus and attach them to
        the root window.

        Returns:
            None
        """
        menubar = tk.Menu(self.root)

        # ---- File menu ----
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load Left Video…", command=self.on_load_left_video)
        file_menu.add_command(label="Load Right Video…", command=self.on_load_right_video)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project…", command=self.on_save_project)
        file_menu.add_command(label="Open Project…", command=self.on_open_project)

        # Rebuilt fresh every time it's about to be shown (via postcommand),
        # not once at startup - so a project moved/renamed/deleted since the
        # last time this menu opened just quietly drops out of the list
        # instead of showing an entry that would only error if clicked.
        self.recent_projects_menu = tk.Menu(file_menu, tearoff=False)
        self.recent_projects_menu.config(postcommand=self._refresh_recent_projects_menu)
        file_menu.add_cascade(label="Recent Projects", menu=self.recent_projects_menu)

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

        # ---- Calibration menu ----
        # A dedicated top-level menu (ROADMAP.md Phase 10) rather than
        # burying calibration actions under File/View - "Load
        # Calibration…" moved here from the File menu, and "Perform
        # Calibration…" (new) opens the frame-pair capture window.
        # "Calibration Report…" isn't added yet - it has no real
        # behavior to wire up until a later Phase 10 step actually
        # builds it.
        calibration_menu = tk.Menu(menubar, tearoff=False)
        calibration_menu.add_command(label="Load Calibration…", command=self.on_load_calibration_folder)
        calibration_menu.add_command(label="Perform Calibration…", command=self.on_perform_calibration)
        menubar.add_cascade(label="Calibration", menu=calibration_menu)

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

        # ---- Resync offset control ----
        # Directly editable mirror of self.lock_offset_frames, for correcting a
        # pair that's out of sync without having to re-scrub both timelines and
        # re-toggle Lock just to capture a new offset.
        ttk.Label(self.toolbar, text="Offset:").grid(row=0, column=8, padx=(0, 2))
        self.offset_spin = ttk.Spinbox(
            self.toolbar,
            from_=-100000,
            to=100000,
            textvariable=self.offset_var,
            width=6,
            command=self.on_offset_changed,
        )
        self.offset_spin.grid(row=0, column=9, padx=(0, 12))
        self.offset_spin.bind("<Return>", self.on_offset_changed)
        self.offset_spin.bind("<FocusOut>", self.on_offset_changed)

        # Clears all measurement points in both panes.
        # This is the only delete mechanism for now (simple and safe).
        self.btn_clear_points = ttk.Button(
            self.toolbar,
            text="Clear Points",
            command=self.on_clear_points,
        )
        self.btn_clear_points.grid(row=0, column=10, padx=(0, 12))

        # ---- Rectified/not-rectified indicator ----
        # Measurements and clicked points are only real-world-accurate in
        # rectified view, so make raw view visually unmistakable at a
        # glance (ROADMAP.md Phase 8) - kept in sync by
        # _refresh_rectified_indicator, called everywhere
        # _refresh_status_left already is (view toggled, video loaded,
        # project opened, etc.).
        self.rectified_indicator = ttk.Label(self.toolbar, font=("Segoe UI", 10, "bold"))
        self.rectified_indicator.grid(row=0, column=11, padx=(0, 12))

        # ---- Real-world time anchor ----
        # Type in a date+time matching whatever real-world clock is burned
        # into the video image at the current frame, so the app can
        # calculate real-world time at any other frame too (ROADMAP.md
        # Phase 8's video-time-sync item). Six separate plain text boxes,
        # each labeled below it, not spinners and not pre-filled with
        # "now" — deliberate, per the project owner. Nothing here is
        # validated/applied until "Set Time Sync" is pressed; typing alone
        # only moves focus to the next box once a box looks full.
        real_time_frame = ttk.Frame(self.toolbar)
        real_time_frame.grid(row=0, column=12, padx=(0, 12))

        real_time_box_specs = [
            (self.real_time_year_var, 4, "YYYY"),
            (self.real_time_month_var, 2, "MM"),
            (self.real_time_day_var, 2, "DD"),
            (self.real_time_hour_var, 2, "HH"),
            (self.real_time_minute_var, 2, "MM"),
            (self.real_time_second_var, 2, "SS"),
        ]
        # Separator text drawn between consecutive boxes (index i sits between
        # box i and box i+1) — one shorter than the number of boxes.
        real_time_separators = ["-", "-", "  ", ":", ":"]

        self.real_time_entries = []
        """The six real-time-anchor Entry widgets (year/month/day/hour/
        minute/second, in that order) — kept so each box's `<KeyRelease>`
        auto-advance handler can focus the *next* one, and so tests can
        drive them uniformly without naming each one."""

        col = 0
        for i, (var, max_len, label_text) in enumerate(real_time_box_specs):
            entry = ttk.Entry(real_time_frame, textvariable=var, width=max_len + 1)
            entry.grid(row=0, column=col, padx=(0, 1))
            ttk.Label(real_time_frame, text=label_text).grid(row=1, column=col)
            self.real_time_entries.append(entry)
            col += 1

            if i < len(real_time_separators):
                ttk.Label(real_time_frame, text=real_time_separators[i]).grid(row=0, column=col)
                col += 1

        # Auto-advance to the next box once this one looks full - purely a
        # focus convenience, not validation (nothing is checked/applied here).
        for i, entry in enumerate(self.real_time_entries):
            max_len = real_time_box_specs[i][1]
            next_entry = self.real_time_entries[i + 1] if i + 1 < len(self.real_time_entries) else None
            entry.bind(
                "<KeyRelease>",
                lambda _evt, e=entry, n=next_entry, m=max_len: self._advance_real_time_focus(e, n, m),
            )

        self.btn_set_time_sync = ttk.Button(
            real_time_frame,
            text="Set Time Sync",
            command=self.on_real_time_entered,
        )
        self.btn_set_time_sync.grid(row=0, column=col, rowspan=2, padx=(6, 2))
        col += 1

        # Synced indicator: a checkmark next to the button rather than trying to
        # recolor the button itself, since ttk buttons don't reliably support
        # custom background colors under Windows themes. Shown by
        # _show_time_sync_indicator once an anchor is actually set; empty
        # (invisible) until then.
        self.time_sync_indicator = ttk.Label(
            real_time_frame, text="", foreground="#008000", font=("Segoe UI", 12, "bold")
        )
        self.time_sync_indicator.grid(row=0, column=col, rowspan=2, padx=(2, 0))

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

        # ---- Shared Frame/Video Time/Actual Time readout ----
        # By the scrub bars (right below both panes), not up in the toolbar -
        # one shared readout since it's a single anchor, not duplicated per pane.
        self.time_readout_label = ttk.Label(self.root, text="", anchor="center")
        self.time_readout_label.grid(row=3, column=0, sticky="ew", pady=(2, 0))

    # -------------------------------------------------------------------------
    # Status bar
    # -------------------------------------------------------------------------

    def _build_statusbar(self):
        """Build the three-section status bar (left/mid/right labels).

        Returns:
            None
        """
        self.status = ttk.Frame(self.root, padding=(8, 6))
        self.status.grid(row=4, column=0, sticky="ew")
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

        Just handles the file dialog; the actual loading logic lives in
        `_load_left_video_from_path` so `on_open_project` can reuse it with
        a path read from a project file instead of a dialog.

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

        self._load_left_video_from_path(path)

    def _load_left_video_from_path(self, path):
        """Load the left video from an already-known path, updating UI state.

        Releases any previously open left capture, opens the given file,
        updates the header/slider/frame state, and re-renders. Dialog-free
        so it can be driven by either `on_load_left_video` (file picker)
        or `on_open_project` (a path stored in a project file).

        Args:
            path (str): Path to the left video file to open.

        Returns:
            bool: True if the video opened successfully, False otherwise
            (with an error dialog already shown).
        """
        # Close any previous capture so we do not leak file handles.
        if self.capL:
            self.capL.release()
            self.capL = None
            self.metaL = None

        # Open the new capture and read its metadata.
        cap, meta = self._open_video_capture(path)
        if cap is None:
            messagebox.showerror("Load Left Video", "Failed to open the selected video file.")
            return False

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

        return True

    def on_load_right_video(self):
        """Prompt for and load the right video, updating UI state.

        Just handles the file dialog; the actual loading logic lives in
        `_load_right_video_from_path` so `on_open_project` can reuse it
        with a path read from a project file instead of a dialog.

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

        self._load_right_video_from_path(path)

    def _load_right_video_from_path(self, path):
        """Load the right video from an already-known path, updating UI state.

        Mirrors `_load_left_video_from_path` for the right pane. Dialog-free
        so it can be driven by either `on_load_right_video` (file picker)
        or `on_open_project` (a path stored in a project file).

        Args:
            path (str): Path to the right video file to open.

        Returns:
            bool: True if the video opened successfully, False otherwise
            (with an error dialog already shown).
        """
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
            return False

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

        return True

    def on_load_calibration_folder(self):
        """Prompt for and load a stereo calibration folder.

        Just handles the file dialog; the actual loading logic lives in
        `_load_calibration_from_folder` so `on_open_project` can reuse it
        with a folder path read from a project file instead of a dialog.

        Note:
            Deliberately uses `filedialog.askopenfilename` (an *open
            file* dialog) rather than `filedialog.askdirectory`, even
            though what's actually wanted is a folder. Windows' native
            folder-picker dialog only shows folder names, never the files
            inside them — so a user comparing several candidate folders
            has no way to see which one actually contains the expected
            NPZ files before picking. Asking for "any file inside the
            calibration folder" instead, filtered to `calibration_*.npz`,
            means the picker's own file list does the job the dialog
            title alone couldn't: the four expected files are right there
            to look at. The containing folder is then just `dirname` of
            whichever one gets picked.

        Returns:
            None
        """
        sample_path = filedialog.askopenfilename(
            title="Select any file inside the calibration folder — expects "
            "calibration_intrinsics.npz, calibration_extrinsics.npz, "
            "calibration_rectification.npz, calibration_maps.npz",
            filetypes=[("Calibration NPZ", "calibration_*.npz"), ("All Files", "*.*")],
        )
        if not sample_path:
            return

        folder = os.path.dirname(sample_path)

        self._load_calibration_from_folder(folder)

    def _load_calibration_from_folder(self, folder):
        """Load a stereo calibration bundle from an already-known folder.

        Delegates the actual file loading/validation to
        `calibration_io.load_calibration_bundle` — this method just
        updates UI state from the result. On any failure, clears
        `self.cal`, disables rectified view, and shows a status message
        explaining why. Dialog-free so it can be driven by either
        `on_load_calibration_folder` (folder picker) or `on_open_project`
        (a path stored in a project file).

        Args:
            folder (str): Path to the calibration folder to load.

        Returns:
            bool: True if the calibration loaded successfully, False
            otherwise (with a status message already shown).
        """
        # Store the folder path for status display.
        self.calibration_folder = folder

        cal, err = calibration_io.load_calibration_bundle(folder, self.metaL, self.metaR)

        if err is not None:
            self.cal = None
            self.view_rectified.set(False)
            self._set_status_mid(err)
            self._refresh_status_left()
            return False

        self.cal = cal

        self._set_status_mid("Calibration loaded")
        self._refresh_status_left()

        # Trigger redraw so rectified mode can be enabled immediately.
        self._render_current_frames()

        # In real wiring, you will enable "Show Rectified" only after maps load.
        # For now, we leave it togglable to test UI.

        return True

    def _get_app_version(self):
        """Read this app's version string from `pyproject.toml`.

        Thin delegate to the module-level `get_app_version` (needed as a
        standalone function so the startup splash — shown before this
        app object is even constructed — can read the version too).

        Returns:
            str: The version string (e.g. "0.1.0"), or "unknown" if
            `pyproject.toml` can't be found or parsed for any reason.
        """
        return get_app_version()

    def on_save_project(self):
        """Prompt for a save location and write the current project state.

        Saves the left/right video paths, calibration folder, current
        resync offset, and rectified-view toggle state (whatever
        combination is currently set — any of them can be None/False if
        not loaded/enabled yet), the app version that created the file,
        the Measurement window's full Log content, and enough state to
        restore the most recently Recorded measurement, so this session
        can be reopened later via `on_open_project` without reselecting
        everything through file dialogs — or losing any recorded
        measurements — again.

        Returns:
            None
        """
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".json",
            filetypes=[("Sizeamatic Project", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return

        anchor_iso = self.real_time_anchor_dt.isoformat() if self.real_time_anchor_dt else None

        err = project_io.save_project(
            path,
            self.left_video_path,
            self.right_video_path,
            self.calibration_folder,
            self.lock_offset_frames,
            self.view_rectified.get(),
            self._get_app_version(),
            self.measurement_window.get_log_text(),
            self.last_recorded_snapshot,
            self.real_time_anchor_frame,
            anchor_iso,
            self.perform_calibration_window.capture_folder,
        )

        if err is not None:
            messagebox.showerror("Save Project", err)
            return

        recent_projects.add_recent_project(path)

        self.current_project_name = os.path.splitext(os.path.basename(path))[0]
        self._refresh_window_titles()

        self._set_status_mid("Project saved")

    def on_open_project(self):
        """Prompt for a project file and reload the saved video/calibration state.

        Just handles the file dialog — the actual load/restore logic is
        shared with the File > Recent Projects submenu via
        `_open_project_from_path`.

        Returns:
            None
        """
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Sizeamatic Project", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return

        self._open_project_from_path(path)

    def _open_project_from_path(self, path):
        """Load and apply a project manifest from an already-known path.

        Reads the project manifest via `project_io.load_project`, then
        reuses the same dialog-free loading helpers the file-picker menu
        items use (`_load_left_video_from_path`,
        `_load_right_video_from_path`, `_load_calibration_from_folder`)
        so a saved path that's since become invalid (moved/deleted file)
        surfaces the exact same error dialogs a manual reload would, one
        per stage, rather than failing the whole project load silently.
        Shared by `on_open_project` (after its file dialog) and the File >
        Recent Projects submenu (`on_open_recent_project`), so both go
        through identical load/restore logic.

        Args:
            path (str): Path to the project file to open.

        Returns:
            None
        """
        project, err = project_io.load_project(path)
        if err is not None:
            messagebox.showerror("Open Project", err)
            return

        if project["left_video_path"]:
            self._load_left_video_from_path(project["left_video_path"])

        if project["right_video_path"]:
            self._load_right_video_from_path(project["right_video_path"])

        if project["calibration_folder"]:
            self._load_calibration_from_folder(project["calibration_folder"])

        # Restore the resync offset after both videos are loaded, so it
        # doesn't get overwritten by anything the video loads above do.
        self.lock_offset_frames = int(project["lock_offset_frames"])
        self.offset_var.set(self.lock_offset_frames)

        # Restore the rectified-view toggle, after calibration is loaded - go
        # through the real toggle handler (not just the BooleanVar) so its
        # existing resolution-mismatch validation still applies, in case the
        # saved calibration folder no longer matches these videos.
        self.view_rectified.set(bool(project["view_rectified"]))
        self.on_toggle_view_rectified()

        # Restore the Measurement window's full recorded history.
        self.measurement_window.restore_log_text(project["measurement_log_text"])

        # Restore the real-world time anchor, if one was set - before
        # _update_frame_labels() below, which refreshes both the shared
        # readout and (now that an anchor exists again) the six entry
        # boxes themselves, live-tracking whatever frame we end up on.
        self.real_time_anchor_frame = project["real_time_anchor_frame"]
        anchor_iso = project["real_time_anchor_iso"]
        self.real_time_anchor_dt = datetime.datetime.fromisoformat(anchor_iso) if anchor_iso else None
        if self.real_time_anchor_dt is not None:
            self._show_time_sync_indicator()

        # Restore the Perform Calibration window's active capture folder,
        # if one was chosen this session before saving (ROADMAP.md
        # Phase 10) - if the window's already open, refresh its display
        # immediately too; otherwise the next ensure_window() call picks
        # this up on its own (see PerformCalibrationWindow.ensure_window).
        self.perform_calibration_window.capture_folder = project["perform_calibration_capture_folder"]
        if self.perform_calibration_window.win is not None:
            self.perform_calibration_window.folder_var.set(
                self.perform_calibration_window.capture_folder or "(no capture folder chosen yet)"
            )
            self.perform_calibration_window._refresh_pairs_listbox()

        # Jump back to "the very last place that was recorded" and show that
        # same measurement on screen again - last, since it depends on the
        # video/calibration/offset state above already being in place.
        snapshot = project["last_recorded_snapshot"]
        if snapshot:
            li = int(snapshot["left_frame_index"])
            ri = int(snapshot["right_frame_index"])

            self.left_frame_index.set(li)
            self.right_frame_index.set(ri)

            # Move the slider widgets to match without re-triggering their own
            # lock-offset jump logic.
            self._suppress_slider_callbacks = True
            try:
                self.left_slider.set(li)
                self.right_slider.set(ri)
            finally:
                self._suppress_slider_callbacks = False

            self.ptsL = [tuple(p) for p in snapshot["ptsL"]]
            self.ptsR = [tuple(p) for p in snapshot["ptsR"]]

            self._render_current_frames()
            self._update_measurement_status_stub()

        # Refresh the frame/video-time/actual-time readout regardless of
        # whether a snapshot was restored above, since the real-time anchor
        # (restored either way) affects it too.
        self._update_frame_labels()

        recent_projects.add_recent_project(path)

        self.current_project_name = os.path.splitext(os.path.basename(path))[0]
        self._refresh_window_titles()

    def on_open_recent_project(self, path):
        """Open a project path chosen from the File > Recent Projects submenu.

        Args:
            path (str): The project file path to open, as listed by
                `_refresh_recent_projects_menu`.

        Returns:
            None
        """
        self._open_project_from_path(path)

    def _refresh_recent_projects_menu(self):
        """Rebuild the File > Recent Projects submenu just before it's shown.

        Returns:
            None
        """
        self.recent_projects_menu.delete(0, "end")

        recent = recent_projects.load_recent_projects()
        if not recent:
            self.recent_projects_menu.add_command(label="(none yet)", state="disabled")
            return

        for path in recent:
            self.recent_projects_menu.add_command(
                label=self._short_path(path, max_len=60),
                command=lambda p=path: self.on_open_recent_project(p),
            )

        self._set_status_mid("Project loaded")
        self._refresh_status_left()

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

            # Keep the resync offset Spinbox showing the just-captured value.
            self.offset_var.set(self.lock_offset_frames)

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

    def on_offset_changed(self, _evt=None):
        """Handle a manually edited resync offset (the toolbar Spinbox).

        Reads the Spinbox's current value into `self.lock_offset_frames`
        and, if Lock is enabled with both videos loaded, immediately
        re-aligns the right timeline to the left's current position using
        the new offset — so a manual resync correction is visible right
        away instead of only taking effect on the next scrub.

        Args:
            _evt (tkinter.Event | None): The Spinbox's `<Return>`/
                `<FocusOut>` event, when triggered by one of those
                bindings rather than the Spinbox's own arrow-click
                `command` (unused either way).

        Returns:
            None
        """
        # Spinbox text can be temporarily empty while editing; ignore that rather
        # than raising.
        try:
            new_offset = int(self.offset_var.get())
        except (ValueError, tk.TclError):
            return

        self.lock_offset_frames = new_offset
        self._set_status_mid(f"Resync offset set to {new_offset:+d} frames")

        # Re-align immediately if locked, using the left timeline's current
        # position as the anchor — matches _jump_frames_locked_with_offset's own
        # "offset = R - L" convention.
        if self.lock_lr.get() and self._both_videos_loaded():
            li = int(self.left_frame_index.get())
            self._jump_frames_locked_with_offset("L", li)

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

            # Advance the master (left) timeline; the right timeline follows,
            # preserving the resync offset, via the same helper the slider and
            # step controls already use (this also renders internally, so
            # there's no separate render call for this branch below). See
            # FINDINGS.md #9 for why this replaced setting both sliders
            # directly: that skipped the resync offset entirely (always
            # setting right to the same index as left) and, worse, fired
            # each slider's `command` callback unsuppressed, which chained
            # into `_jump_frames_locked_with_offset` twice with
            # contradictory targets and could net-decrease the index every
            # tick — i.e. exactly the reported "Play runs backward" bug.
            self._jump_frames_locked_with_offset("L", nxt)
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

            # Render the new frames (the locked branch above already rendered
            # via _jump_frames_locked_with_offset).
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

        self._refresh_rectified_indicator()

    def _refresh_rectified_indicator(self):
        """Update the toolbar's RECTIFIED/NOT RECTIFIED label to match
        `self.view_rectified`.

        Returns:
            None
        """
        if self.view_rectified.get():
            self.rectified_indicator.config(text="RECTIFIED", foreground="#008000")
        else:
            self.rectified_indicator.config(text="NOT RECTIFIED", foreground="#cc0000")

    def _app_window_title(self):
        """Build the title text every app window should show.

        Returns:
            str: "Sizeamatic Pro vX.Y.Z", or "Sizeamatic Pro vX.Y.Z -
            <project name>" once a project has been saved/opened this
            session (`self.current_project_name`). Omits the version
            entirely if it couldn't be read (`_get_app_version()`
            returned "unknown"), rather than showing a literal
            "vunknown".
        """
        version = self._get_app_version()
        base = f"Sizeamatic Pro v{version}" if version != "unknown" else "Sizeamatic Pro"
        if self.current_project_name:
            return f"{base} - {self.current_project_name}"
        return base

    def _refresh_window_titles(self):
        """Apply the current project-aware title to every open window.

        Updates the main window plus the Measurement and Calibration
        Summary windows if they're currently open — a window not open
        yet picks up the right title when it's built, via
        `_app_window_title` being read directly in its own
        `ensure_window`.

        Returns:
            None
        """
        title = self._app_window_title()
        self.root.title(title)
        if self.measurement_window.win is not None:
            self.measurement_window.win.title(title)
        if self.cal_summary_window.win is not None:
            self.cal_summary_window.win.title(title)

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
        """Refresh the "Frame: i/max" labels under both sliders, the
        shared Frame/Video Time/Actual Time readout, and (if an anchor is
        set) the six real-time entry boxes themselves.

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

        # The shared readout is referenced to the left/master timeline, same as
        # the real-world time anchor itself.
        video_time = self._format_timestamp(li, self.metaL["fps"] if self.metaL else None)
        actual_time = self._format_actual_time(li)
        self.time_readout_label.config(
            text=f"Frame: {li}/{lmax} | Video: {video_time} | Actual: {actual_time}"
        )

        # Keep the entry boxes live-tracking the current frame's actual time,
        # once an anchor exists (a no-op before that, so typing a fresh anchor
        # isn't clobbered by this running on every frame change).
        self._refresh_real_time_entries(li)

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

        # Crop, then scale to the ON-SCREEN size this exact pixel range actually
        # covers — NOT unconditionally to the full (dw, dh) display rect.
        #
        # (rx0, ry0, rx1, ry1) can be smaller than the view's originally intended
        # region whenever that region reached past the image's edges (e.g. fully
        # zoomed out with even a tiny leftover pan offset from an earlier
        # off-center zoom — reachable at zoom_min itself, not just "way outside
        # the image"). This used to always resize the (possibly clamped) crop to
        # fill the entire (dw, dh) rect and draw it at (dx, dy) regardless — which
        # silently stretched a smaller-than-intended crop to fill the same space,
        # scaling the displayed image differently from the un-clamped scale
        # `_image_to_screen` uses for point overlays. Mapping the actual
        # (rx0, ry0, rx1, ry1) back through the same screen transform keeps the
        # displayed image and the overlay points using one consistent scale
        # regardless of clamping — and is a no-op change whenever nothing was
        # actually clamped (the common case), since then this reduces to exactly
        # (dw, dh) at (dx, dy) as before.
        screen_x0 = float(dx) + float(rx0) * S + off_x
        screen_y0 = float(dy) + float(ry0) * S + off_y
        out_w = max(1, int(round((rx1 - rx0) * S)))
        out_h = max(1, int(round((ry1 - ry0) * S)))

        crop = frame_bgr[ry0:ry1, rx0:rx1]
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

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

        # Draw the image at the screen position this exact (possibly clamped)
        # crop actually covers — see the comment above where screen_x0/screen_y0
        # are computed for why this isn't always just (dx, dy).
        canvas.create_image(int(round(screen_x0)), int(round(screen_y0)), anchor="nw", image=tk_img, tags=("frame",))

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

    def _redisplay_current_frames(self):
        """Redraw the already-decoded current frames without re-reading
        from the video captures.

        Used for interactions that only change the on-screen transform
        (currently just panning, `on_pan_drag`) rather than which video
        frame is showing. `self.current_frameL`/`self.current_frameR`
        already hold the correct pixel data (post-rectification, if
        enabled) — re-decoding via `_render_current_frames` on every
        mouse-drag event, which can fire dozens of times per second,
        would force a `cap.set()` keyframe seek on every single one (see
        `_read_frame_at`'s docstring), reintroducing the same seek-cost
        problem the Phase 5 sequential-playback fix solved, just through
        a different call path.

        Returns:
            None
        """
        if self.current_frameL is not None:
            self._display_bgr_on_canvas(self.video_overlay.left_canvas, self.current_frameL, "L")

        if self.current_frameR is not None:
            self._display_bgr_on_canvas(self.video_overlay.right_canvas, self.current_frameR, "R")

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


STARTUP_SPLASH_MIN_SECONDS = 4.0
"""Minimum time the startup splash (`_show_startup_splash`) stays on
screen, even if building the app finishes faster than that - so it's
actually readable rather than a barely-visible flash on a fast machine
or a source run. Within the project owner's requested 3-5 second
range."""

STARTUP_SPLASH_MAX_WIDTH_PX = 720
"""Cap the splash's on-screen width, scaling the source art down
proportionally if it's larger (assets/splash-pro.png is 1536px wide -
full native resolution filled most of the screen, and was noticeably
bigger than the size PyInstaller's own bootloader splash used to scale
it to before this app got its own Tk-based splash). Matches roughly
what that previous scaling already looked like."""


def _show_startup_splash(root):
    """Show a splash window while the rest of the app builds.

    A real Tkinter window (undecorated, via `overrideredirect`) rather
    than relying on PyInstaller's separate bootloader `--splash`
    feature, so it shows identically whether launched from source
    (`uv run python main.py`) or from a packaged `.exe` — the bootloader
    splash only exists inside a packaged build, and only covers a
    onefile build's self-extraction phase, before this function (or any
    of this app's own code) even runs.

    Args:
        root (tkinter.Tk): The (still-withdrawn) root window to parent
            the splash `Toplevel` to.

    Returns:
        tkinter.Toplevel: The splash window. Caller is responsible for
        destroying it once the real app window is ready to show.
    """
    img = prepare_splash_image.flatten_splash_image(resource_path("assets/splash-pro.png"))
    img = prepare_splash_image.add_version_text(img, f"v{get_app_version()}")

    if img.width > STARTUP_SPLASH_MAX_WIDTH_PX:
        scale = STARTUP_SPLASH_MAX_WIDTH_PX / img.width
        img = img.resize((STARTUP_SPLASH_MAX_WIDTH_PX, round(img.height * scale)), Image.LANCZOS)

    splash = tk.Toplevel(root)
    splash.configure(bg="black")

    photo = ImageTk.PhotoImage(img)
    label = tk.Label(splash, image=photo, bd=0, bg="black")
    label.image = photo  # Keep a reference - Tkinter doesn't hold its own.
    label.pack()

    # overrideredirect (no title bar/border) after the label exists, and
    # a forced update_idletasks before reading winfo_screen*, avoids a
    # Windows/Tcl-Tk quirk where an undecorated Toplevel can render
    # solid black instead of its actual content if made borderless and
    # topmost before it has anything to paint.
    splash.update_idletasks()
    splash.overrideredirect(True)

    # Center on the primary screen.
    sw = splash.winfo_screenwidth()
    sh = splash.winfo_screenheight()
    x = (sw - img.width) // 2
    y = (sh - img.height) // 2
    splash.geometry(f"{img.width}x{img.height}+{x}+{y}")

    splash.lift()
    splash.update()

    return splash


def main():
    """Entry point: build the Tk root window and run the application.

    Sets the Windows taskbar application identity (so the app groups under
    its own taskbar icon rather than a generic Python one), shows a
    startup splash while the rest of the window builds, applies the
    window icon if available, constructs `SizeamaticProApp`, wires up
    the close protocol, and starts the Tk event loop.

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

    # Keep the main window hidden until it's actually built and ready to
    # show - the splash below covers that gap instead.
    root.withdraw()

    splash_shown_at = time.monotonic()
    splash = _show_startup_splash(root)

    # Set the application window icon. assets/icon.ico is regenerated
    # from assets/default-icon.png on every build (create_app_icon.py),
    # so this stays best-effort: fall back to the default Tk icon rather
    # than crashing if it's ever missing/invalid.
    try:
        root.iconbitmap(resource_path("assets/icon.ico"))
    except tk.TclError:
        pass

    # ttk theme defaults are OK. If you want a darker theme later, we can style it.
    app = SizeamaticProApp(root)

    root.protocol("WM_DELETE_WINDOW", app.on_app_close)

    # Keep the splash up for a minimum duration even if building the app
    # above finished faster than that, so it's actually readable rather
    # than a barely-visible flash on a fast machine/source run. Pumps
    # the splash's own event loop while waiting (short update()+sleep()
    # steps) rather than a single blocking sleep, so Windows doesn't
    # mark the still-open splash window "Not Responding".
    remaining = STARTUP_SPLASH_MIN_SECONDS - (time.monotonic() - splash_shown_at)
    while remaining > 0:
        step = min(remaining, 0.05)
        splash.update()
        time.sleep(step)
        remaining -= step

    splash.destroy()
    root.deiconify()

    root.mainloop()


if __name__ == "__main__":
    main()
