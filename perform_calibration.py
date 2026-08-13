"""Perform Calibration window for Sizeamatic Pro.

This module creates and manages the Tkinter window used to build a new
stereo calibration from scratch (ROADMAP.md Phase 10), rather than only
ever loading an already-finished one (`calibration_io.py`). Step 1 of
that phase - the only part implemented so far - is frame-pair capture:
letting the project owner either manually scrub the already-loaded
left/right video pair to a frame showing a checkerboard/ChArUco
calibration target and capture it, or run an automatic scan that
samples through the video looking for frames where a board is actually
detected and captures those on its own. Either way, the result is the
same matched, numbered image pair on disk. Later Phase 10 steps will
extend this same window to actually run the calibration computation
(`cv2.calibrateCamera`/`stereoCalibrate`/`stereoRectify`) on the
captured pairs and show quality stats, rather than adding a separate
window for that.

Contents:
    - `PerformCalibrationWindow` — owns the window and its widgets.

Design notes:
    Follows the same "class instance owned by the main application"
    shape as `calibration_summary.CalibrationSummaryWindow` and
    `measurement_window.MeasurementWindow` - `self.win`/other widget
    references start `None` and only get built by `ensure_window`,
    which is safe to call repeatedly (it raises the existing window
    instead of building a second one).

    The capture folder is intentionally a *separate* piece of state
    from `main.py`'s `app.calibration_folder` (the folder an already
    *loaded* calibration came from, used for rectified-view display) -
    a brand new calibration can be captured into its own folder while a
    different, already-loaded calibration stays active for viewing the
    video in the meantime.

    The matched `left_####.png`/`right_####.png` naming convention this
    module writes (and re-reads, via `_scan_capture_folder`, so
    reopening a folder that already has captures in it just continues
    numbering rather than colliding with them) is deliberately the same
    one already implemented - and proven against this app's own
    calibration NPZ format - on the unmerged `feature-onlineCalibrations`
    branch's `perform_calibration.py`. Matching it here means the actual
    calibration-computation code that branch already has can be adapted
    for a later Phase 10 step with minimal changes, rather than also
    needing its file-discovery logic rewritten to match a different
    convention invented here instead.

    Alongside the image pairs themselves, a small `capture_metadata.json`
    in the capture folder records which left/right frame index each
    pair came from, keyed by pair ID. This is what lets double-clicking
    a pair in the list jump the main window's video back to the exact
    frame it was captured from (`on_pair_double_clicked`) - the image
    files alone don't carry that information. A pair captured or added
    to the folder some other way, with no matching metadata entry,
    still displays and can still be deleted; it just can't be jumped to.

    The auto-scan (`on_auto_scan_for_candidates`) runs on a background
    `threading.Thread`, not the main Tk thread, since decoding and
    running two board detectors across a whole video is too slow to do
    without freezing the UI. It opens its *own* fresh
    `cv2.VideoCapture`s on the video file paths rather than reusing
    `app.capL`/`app.capR` - those aren't safe to read from a second
    thread while the main thread might simultaneously be scrubbing the
    same capture object for display. Progress/results cross back to the
    main thread through a `queue.Queue`, drained by `_poll_scan_queue`
    via `Tk.after` polling - the standard safe pattern for a background
    worker that needs to update Tkinter widgets, since Tkinter itself
    isn't thread-safe to touch directly from a worker thread.

Assumptions:
    - The main application exposes: `app.root` (the Tk root window);
      `app.capL`/`app.capR` (the loaded `cv2.VideoCapture` objects);
      `app.left_video_path`/`app.right_video_path` (the same videos'
      file paths, for the auto-scan's own independent captures);
      `app.left_frame_index`/`app.right_frame_index` and
      `app.left_slider`/`app.right_slider` (each pane's current frame
      and its slider widget, as Tk variables/widgets);
      `app.left_frame_max` (the left video's highest valid frame
      index); `app.lock_offset_frames` (the current
      `right_index - left_index` sync offset, used to derive each
      scanned frame's right-side index from its left-side one);
      `app._suppress_slider_callbacks` (a flag `main.py` already uses
      to move both sliders without re-triggering their own lock-offset
      jump logic); `app._read_frame_at` (a raw, unrectified frame
      reader); `app._update_frame_labels`/`app._render_current_frames`
      (redraw hooks); and `app._app_window_title` (this app's
      project-aware window title, shared by every window). All defined
      on `main.py`'s `SizeamaticProApp`.
    - Capturing for calibration always wants the *raw* decoded frame,
      never a rectified one, even if the main window's rectified-view
      toggle happens to be on at the time - the whole point of a
      calibration run is estimating the distortion a raw frame has, so
      capturing an already-corrected frame would be circular.

Author:
    Isaac Travers

Date:
    2026-08-13
"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

PAIR_ID_DIGITS = 4
"""Zero-padding width for captured pair filenames (e.g. `left_0007.png`)
- wide enough that captured pairs still sort correctly by filename well
past what any single calibration run would realistically capture."""

CHECKERBOARD_INNER_CORNERS = (9, 6)
"""(columns, rows) of *inner* corners the auto-scan's checkerboard
detector looks for - a common default board size, not yet configurable
through any settings UI. A board printed with a different square count
won't be detected until this becomes adjustable."""

CHARUCO_SQUARES_X = 11
"""ChArUco board width in squares, matching
`create_charuco_calibration_target.py`'s own default board - so a board
already printed from that script is exactly what the auto-scan's
ChArUco detector expects, not a mismatched size."""

CHARUCO_SQUARES_Y = 8
"""ChArUco board height in squares. See `CHARUCO_SQUARES_X`."""

CHARUCO_SQUARE_SIZE_MM = 20.0
"""ChArUco square size in millimeters. See `CHARUCO_SQUARES_X`."""

CHARUCO_MARKER_SIZE_MM = 15.0
"""ChArUco marker size in millimeters. See `CHARUCO_SQUARES_X`."""

CHARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_1000
"""ArUco marker dictionary the auto-scan's ChArUco detector expects.
See `CHARUCO_SQUARES_X`."""

AUTO_SCAN_FRAME_STRIDE = 15
"""How many left-video frames the auto-scan advances between each
sampled frame it actually decodes and runs detection on. Sampling
rather than checking every single frame keeps a full-video scan fast;
15 is roughly a half-second step at a typical 30fps recording."""


def frame_has_checkerboard(gray_image):
    """Check whether a grayscale image contains a detectable checkerboard.

    Args:
        gray_image (numpy.ndarray): A single-channel (grayscale) image.

    Returns:
        bool: True if `cv2.findChessboardCorners` found a full
        `CHECKERBOARD_INNER_CORNERS`-sized grid of inner corners.
    """
    found, _corners = cv2.findChessboardCorners(
        gray_image,
        CHECKERBOARD_INNER_CORNERS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return bool(found)


def build_charuco_detector():
    """Build the ChArUco board/detector objects the auto-scan reuses.

    Built once per scan (not once per sampled frame) since constructing
    the dictionary/board/detector triplet repeatedly for every frame
    would be wasted work - the board geometry never changes mid-scan.

    Returns:
        cv2.aruco.CharucoDetector: A detector configured for the fixed
        `CHARUCO_SQUARES_X`/`CHARUCO_SQUARES_Y`/`CHARUCO_SQUARE_SIZE_MM`/
        `CHARUCO_MARKER_SIZE_MM`/`CHARUCO_DICTIONARY_ID` board geometry.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_SIZE_MM,
        CHARUCO_MARKER_SIZE_MM,
        dictionary,
    )
    return cv2.aruco.CharucoDetector(board)


def frame_has_charuco_board(gray_image, detector):
    """Check whether a grayscale image contains a detectable ChArUco board.

    Args:
        gray_image (numpy.ndarray): A single-channel (grayscale) image.
        detector (cv2.aruco.CharucoDetector): A detector built by
            `build_charuco_detector`.

    Returns:
        bool: True if at least 4 interpolated ChArUco corners were
        found - 4 is the minimum OpenCV itself needs to consider a
        ChArUco detection usable at all.
    """
    charuco_corners, _charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray_image)
    return charuco_corners is not None and len(charuco_corners) >= 4


def frame_has_calibration_board(bgr_image, charuco_detector):
    """Check whether a frame has a detectable checkerboard or ChArUco board.

    Tries checkerboard detection first, then ChArUco, so either board
    type present in the video gets picked up without the project owner
    needing to specify which one is actually in use.

    Args:
        bgr_image (numpy.ndarray): A decoded BGR video frame.
        charuco_detector (cv2.aruco.CharucoDetector): A detector built
            by `build_charuco_detector`.

    Returns:
        bool: True if either detector found a board.
    """
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return frame_has_checkerboard(gray_image) or frame_has_charuco_board(gray_image, charuco_detector)


class PerformCalibrationWindow:
    """Owns the Perform Calibration Toplevel window and its widgets.

    Currently implements only frame-pair capture (ROADMAP.md Phase 10,
    Step 1). One instance lives on the main application
    (`app.perform_calibration_window`), created once and reused for the
    lifetime of the app, matching `CalibrationSummaryWindow`'s pattern.
    """

    def __init__(self, app):
        """Store the owning app and initialize widget/session state to None.

        Args:
            app: The main application object - used for the Tk root
                window that owns this Toplevel, the currently loaded
                left/right video captures and frame indices, and the
                raw-frame reader (`app._read_frame_at`). See this
                module's docstring for the exact attributes assumed to
                exist on it.

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The Perform Calibration Toplevel window, or None if it
        hasn't been opened yet (or was closed)."""

        self.capture_folder = None
        """The folder captured calibration frame pairs are saved into,
        or None until the project owner chooses/creates one via
        `on_choose_capture_folder`. Survives this window being closed
        and reopened (`_on_close` deliberately doesn't clear it, only
        the widget references that get rebuilt by `ensure_window`), and
        is saved/restored through the project file too
        (`project_io.py`'s `perform_calibration_capture_folder`,
        `main.py`'s `on_save_project`/`_open_project_from_path`), so an
        in-progress capture session survives a save/reload the same way
        the rest of the app's session state does."""

        self.folder_var = None
        """`tk.StringVar` mirroring `self.capture_folder` for display in
        the window - created in `ensure_window` (needs a Tk root to
        exist first), read by nothing else."""

        self.status_var = None
        """`tk.StringVar` holding the window's status/error line, e.g.
        "Captured pair 0007 (7 total)". Created in `ensure_window`."""

        self.pairs_listbox = None
        """The `tkinter.Listbox` showing every captured pair currently
        in `self.capture_folder`, or None until `ensure_window` builds
        it."""

        self.auto_scan_button = None
        """The button that starts/stops the auto-scan
        (`on_auto_scan_for_candidates`), or None until `ensure_window`
        builds it - its label is toggled between "Auto-Scan for
        Candidates" and "Stop Scan" depending on `self._scan_thread`."""

        self._scan_thread = None
        """The background `threading.Thread` currently running
        `_run_auto_scan`, or None if no scan is in progress. Used both
        to detect "a scan is already running" and as the signal
        `_poll_scan_queue` checks to know when to stop rescheduling
        itself."""

        self._scan_queue = None
        """The `queue.Queue` the running scan thread posts progress/
        result messages onto, drained by `_poll_scan_queue`. None
        whenever `self._scan_thread` is None."""

        self._scan_stop_requested = False
        """Set True by `on_auto_scan_for_candidates` when the project
        owner clicks "Stop Scan" while a scan is running - checked by
        `_run_auto_scan` between sampled frames so it can exit early
        rather than always running to the end of the video."""

    def _on_close(self):
        """Handle the user manually closing the window.

        Destroys the Tkinter window and clears the stored widget
        references (matching `CalibrationSummaryWindow._on_close`'s
        reasoning) - the next `ensure_window` call needs to know the
        widgets no longer exist and must rebuild them, rather than
        holding onto references to already-destroyed widgets. Also
        unbinds the Space capture shortcut from the main app window (so
        Space goes back to doing nothing there once this window isn't
        open), and requests any running auto-scan stop - `self.win`
        becomes None right after this, and `_poll_scan_queue` refuses to
        touch a None window, so a scan left running with no window to
        report progress to would otherwise just keep going invisibly
        until the video ends.

        Returns:
            None
        """
        self.app.root.unbind("<space>")
        self._scan_stop_requested = True

        self.win.destroy()
        self.win = None
        self.folder_var = None
        self.status_var = None
        self.pairs_listbox = None
        self.auto_scan_button = None

    def _metadata_path(self):
        """Build the path to this capture folder's metadata JSON file.

        Returns:
            str: The path to `capture_metadata.json` inside
            `self.capture_folder`.
        """
        return os.path.join(self.capture_folder, "capture_metadata.json")

    def _load_metadata(self):
        """Load per-pair capture metadata (frame indices) from disk.

        Returns:
            dict: Mapping of pair ID (as a string, matching how
            `json.dump`/`json.load` round-trip dict keys) to a dict with
            keys `"left_frame_index"`/`"right_frame_index"`. Empty if no
            capture folder is chosen yet, the metadata file doesn't
            exist, or it can't be parsed - a missing/corrupt metadata
            file just means "no jump-to-frame info available yet", not
            an error to surface to the user.
        """
        if not self.capture_folder:
            return {}

        metadata_path = self._metadata_path()
        if not os.path.isfile(metadata_path):
            return {}

        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_metadata(self, metadata):
        """Write per-pair capture metadata (frame indices) to disk.

        Args:
            metadata (dict): The full metadata dict to write, keyed by
                pair ID (see `_load_metadata`) - always the complete
                dict, not just a single new entry, since this
                overwrites the file rather than merging into it.

        Returns:
            None
        """
        with open(self._metadata_path(), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def ensure_window(self):
        """Create the Perform Calibration window, or raise it if it already exists.

        Only builds the UI widgets - `self.capture_folder` (if already
        chosen from an earlier call this session) is re-displayed, and
        the pairs list is populated from whatever's actually in that
        folder on disk.

        Returns:
            None
        """

        # If the window already exists, bring it to the front and reuse it
        # instead of creating a duplicate window.
        if self.win is not None:
            try:
                self.win.lift()
                return

            # If the stored window reference is stale, clear it so a new
            # window can be created below.
            except Exception:
                self.win = None

        # Create a separate top level window owned by the main application root.
        win = tk.Toplevel(self.app.root)

        # Match the main window's project-aware title, same as the other
        # sub-windows.
        win.title(self.app._app_window_title())

        # Give the window an initial size large enough for the folder row,
        # capture button, and pairs list without feeling cramped.
        win.geometry("560x480")

        # Use the cleanup callback when the user closes this window.
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create one padded outer frame to hold all of this window's content.
        outer = ttk.Frame(win, padding=(10, 10))
        outer.grid(row=0, column=0, sticky="nsew")

        # Let the outer frame expand with the window.
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        # ---- Heading ----
        ttk.Label(
            outer,
            text="Perform Calibration — capture frame pairs",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # ---- Capture folder row ----
        # Shows the currently chosen capture folder (or a placeholder if
        # none has been chosen yet), plus the button to choose/create one.
        folder_row = ttk.Frame(outer)
        folder_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        folder_row.grid_columnconfigure(0, weight=1)

        self.folder_var = tk.StringVar(value=self.capture_folder or "(no capture folder chosen yet)")
        ttk.Label(folder_row, textvariable=self.folder_var, anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Button(
            folder_row,
            text="Choose Folder…",
            command=self.on_choose_capture_folder,
        ).grid(row=0, column=1, padx=(8, 0))

        # ---- Capture/delete buttons ----
        # Deliberately don't disable themselves based on whether a folder is
        # chosen yet, videos are loaded, or a pair is selected -
        # on_capture_frame_pair/on_delete_selected_pair already report
        # exactly what's missing via a messagebox, which reads more clearly
        # than a greyed-out button with no explanation attached.
        button_row = ttk.Frame(outer)
        button_row.grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Button(
            button_row,
            text="Capture Frame Pair",
            command=self.on_capture_frame_pair,
        ).grid(row=0, column=0)
        ttk.Button(
            button_row,
            text="Delete Selected Pair",
            command=self.on_delete_selected_pair,
        ).grid(row=0, column=1, padx=(8, 0))

        # Single button that toggles between starting and stopping the
        # auto-scan - see on_auto_scan_for_candidates/on_stop_scan.
        self.auto_scan_button = ttk.Button(
            button_row,
            text="Auto-Scan for Candidates",
            command=self.on_auto_scan_for_candidates,
        )
        self.auto_scan_button.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(
            outer,
            text=(
                "Tip: press Space to capture without reaching for the button. Double-click a pair "
                "below to jump the video back to it. Auto-Scan samples every "
                f"{AUTO_SCAN_FRAME_STRIDE} frames looking for a checkerboard/ChArUco board and "
                "captures whatever it finds - review the results below and delete any you don't want."
            ),
            foreground="#555555",
            wraplength=520,
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

        # ---- Captured pairs list ----
        list_frame = ttk.Frame(outer)
        list_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.pairs_listbox = tk.Listbox(list_frame)
        self.pairs_listbox.grid(row=0, column=0, sticky="nsew")
        self.pairs_listbox.bind("<Double-Button-1>", self.on_pair_double_clicked)

        # Attach a vertical scrollbar so a long capture session stays usable.
        pairs_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.pairs_listbox.yview)
        pairs_scrollbar.grid(row=0, column=1, sticky="ns")
        self.pairs_listbox.configure(yscrollcommand=pairs_scrollbar.set)

        # ---- Status line ----
        self.status_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )

        self.win = win

        # Bind the capture shortcut on both this window and the main app
        # window, so pressing Space captures a pair regardless of which of
        # the two currently has keyboard focus - the project owner's whole
        # point is scrubbing the *main* window's video with this window
        # merely open alongside it, not needing to click back and forth.
        # Unbound again in _on_close so Space stops doing anything once
        # this window isn't open.
        win.bind("<space>", self._on_space_key)
        self.app.root.bind("<space>", self._on_space_key)

        # Let the captured-pairs list get the extra vertical space when the
        # window resizes; everything above it stays a fixed height.
        outer.grid_rowconfigure(4, weight=1)

        # Populate the pairs list immediately in case a folder was already
        # chosen in an earlier call this session and already has captures.
        self._refresh_pairs_listbox()

    def on_choose_capture_folder(self):
        """Choose (or create) the folder captured frame pairs get saved into.

        Uses `filedialog.askdirectory`, whose native picker already
        supports creating a new folder from within it - covering both
        "start a brand new calibration capture session" and "add more
        frames to a folder from an earlier session" with the same
        dialog, no separate "New" vs. "Existing" choice needed. Refuses
        to change folders while a scan is running, since `_run_auto_scan`
        reads `self.capture_folder` live rather than a snapshot taken
        when it started - switching folders mid-scan would make it
        start writing into the new folder partway through.

        Returns:
            None
        """
        if self._scan_thread is not None:
            messagebox.showerror("Perform Calibration", "Stop the current scan before choosing a different folder.")
            return

        folder = filedialog.askdirectory(title="Choose Calibration Capture Folder")

        # A cancelled dialog returns an empty string, not None - treat both
        # as "no change" rather than clearing an already-chosen folder.
        if not folder:
            return

        self.capture_folder = folder
        self.folder_var.set(folder)

        # Show whatever's already captured in this folder, in case it's an
        # existing capture session being resumed rather than a brand new one.
        self._refresh_pairs_listbox()

    def _on_space_key(self, _event=None):
        """Handle the Space-bar capture shortcut.

        Bound to both this window and the main app window (see
        `ensure_window`) so scrubbing the main window's video with this
        window merely open alongside it still captures on Space,
        without needing to click back over to this window's button
        every time. Skips capturing if a text-entry-style widget
        currently has focus, so Space still just types a literal space
        while filling in one of the main window's own text boxes (e.g.
        the real-time-sync entries) rather than also capturing a pair.

        Args:
            _event (tkinter.Event | None): The key event, unused -
                required by Tkinter's bind signature.

        Returns:
            None
        """
        focused = self.app.root.focus_get()
        if isinstance(focused, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return

        self.on_capture_frame_pair()

    def on_capture_frame_pair(self):
        """Capture the currently displayed left/right frames as a calibration pair.

        Requires both videos to already be loaded and a capture folder
        to already be chosen - reports exactly which is missing via a
        messagebox rather than silently doing nothing. Reads the *raw*
        decoded frame at each pane's current index directly
        (`app._read_frame_at`), not `app.current_frameL`/
        `current_frameR`, since those may already be rectified if
        rectified view happens to be on - see this module's docstring
        for why that would be wrong for calibration specifically.
        Refused while an auto-scan is running - see
        `on_choose_capture_folder`'s docstring for why manual and
        automatic capture can't safely run at the same time.

        Returns:
            None
        """
        if self._scan_thread is not None:
            messagebox.showerror("Perform Calibration", "A scan is already running.")
            return

        app = self.app

        # Require both videos to actually be loaded before trying to read a
        # frame from either one.
        if not app.capL or not app.capR:
            messagebox.showerror("Perform Calibration", "Load both the left and right videos first.")
            return

        # Require a capture folder to already be chosen.
        if not self.capture_folder:
            messagebox.showerror("Perform Calibration", "Choose a capture folder first.")
            return

        # Read the raw (unrectified) frame at each pane's current index.
        left_index = int(app.left_frame_index.get())
        right_index = int(app.right_frame_index.get())
        frame_l = app._read_frame_at(app.capL, left_index)
        frame_r = app._read_frame_at(app.capR, right_index)

        # Report a decode failure rather than silently saving nothing.
        if frame_l is None or frame_r is None:
            messagebox.showerror(
                "Perform Calibration",
                "Could not read the current frame from one or both videos.",
            )
            return

        # Find the next unused pair ID by scanning the capture folder itself
        # (not an in-memory counter), so reopening this window - or resuming
        # an earlier session's folder - continues numbering forward instead
        # of colliding with or overwriting pairs already saved there.
        next_id = self._next_pair_id()
        left_filename = f"left_{next_id:0{PAIR_ID_DIGITS}d}.png"
        right_filename = f"right_{next_id:0{PAIR_ID_DIGITS}d}.png"

        # Save both frames as lossless PNGs - calibration corner detection
        # benefits from exact pixel data, not JPEG compression artifacts.
        cv2.imwrite(os.path.join(self.capture_folder, left_filename), frame_l)
        cv2.imwrite(os.path.join(self.capture_folder, right_filename), frame_r)

        # Record which frame index each side came from, so double-clicking
        # this pair later (on_pair_double_clicked) can jump the video back
        # to exactly this position - the image files alone don't carry that.
        metadata = self._load_metadata()
        metadata[str(next_id)] = {"left_frame_index": left_index, "right_frame_index": right_index}
        self._save_metadata(metadata)

        # Refresh the visible pairs list and report the capture.
        self._refresh_pairs_listbox()
        pair_count = len(self._scan_capture_folder())
        self.status_var.set(f"Captured pair {next_id:0{PAIR_ID_DIGITS}d} ({pair_count} total)")

    def on_pair_double_clicked(self, _event=None):
        """Jump the main window's video back to a double-clicked pair's frame.

        Reads the selected pair's frame indices from
        `capture_metadata.json` (see this module's docstring) and moves
        both panes to them, mirroring the exact slider-update pattern
        `main.py` already uses elsewhere (`_suppress_slider_callbacks`)
        to move both sliders without re-triggering their own lock-offset
        jump logic.

        Args:
            _event (tkinter.Event | None): The double-click event,
                unused - required by Tkinter's bind signature.

        Returns:
            None
        """
        selection = self.pairs_listbox.curselection()
        if not selection:
            return

        pairs = self._scan_capture_folder()
        selected_index = selection[0]
        if selected_index >= len(pairs):
            return
        pair_id, _left_filename, _right_filename = pairs[selected_index]

        entry = self._load_metadata().get(str(pair_id))
        if entry is None:
            messagebox.showerror(
                "Perform Calibration",
                "No saved frame position for this pair (it may have been added outside this window).",
            )
            return

        app = self.app
        left_index = int(entry["left_frame_index"])
        right_index = int(entry["right_frame_index"])

        app.left_frame_index.set(left_index)
        app.right_frame_index.set(right_index)

        # Move the slider widgets to match without re-triggering their own
        # lock-offset jump logic.
        app._suppress_slider_callbacks = True
        try:
            app.left_slider.set(left_index)
            app.right_slider.set(right_index)
        finally:
            app._suppress_slider_callbacks = False

        app._update_frame_labels()
        app._render_current_frames()

    def on_delete_selected_pair(self):
        """Delete the selected captured pair's image files and metadata.

        Asks for confirmation first (`messagebox.askyesno`), since this
        permanently removes files from disk rather than something
        recoverable within the app itself. Refused while an auto-scan
        is running - see `on_choose_capture_folder`'s docstring for why.

        Returns:
            None
        """
        if self._scan_thread is not None:
            messagebox.showerror("Perform Calibration", "Stop the current scan before deleting a pair.")
            return

        selection = self.pairs_listbox.curselection()
        if not selection:
            messagebox.showerror("Perform Calibration", "Select a captured pair to delete first.")
            return

        pairs = self._scan_capture_folder()
        selected_index = selection[0]
        if selected_index >= len(pairs):
            return
        pair_id, left_filename, right_filename = pairs[selected_index]

        confirmed = messagebox.askyesno(
            "Perform Calibration",
            f"Delete captured pair {pair_id:0{PAIR_ID_DIGITS}d}? This removes both image files "
            "and can't be undone.",
        )
        if not confirmed:
            return

        for filename in (left_filename, right_filename):
            file_path = os.path.join(self.capture_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        metadata = self._load_metadata()
        metadata.pop(str(pair_id), None)
        self._save_metadata(metadata)

        self._refresh_pairs_listbox()
        pair_count = len(self._scan_capture_folder())
        self.status_var.set(f"Deleted pair {pair_id:0{PAIR_ID_DIGITS}d} ({pair_count} total)")

    def on_auto_scan_for_candidates(self):
        """Start (or stop) a background scan for calibration candidate frames.

        Toggles based on whether a scan is already running
        (`self._scan_thread`) - clicking while idle starts one;
        clicking again while running requests it stop early
        (`on_stop_scan`) rather than starting a second one on top of
        it. Requires both videos loaded and a capture folder already
        chosen, same as a manual capture. Derives each sampled frame's
        right-side index from its left-side one using
        `app.lock_offset_frames` (`right_index = left_index +
        lock_offset_frames`) - the same relationship the app's own Lock
        feature already maintains, so this doesn't need a separate
        "how do these two videos line up" setting of its own.

        Returns:
            None
        """
        if self._scan_thread is not None:
            self.on_stop_scan()
            return

        app = self.app

        if not app.left_video_path or not app.right_video_path:
            messagebox.showerror("Perform Calibration", "Load both the left and right videos first.")
            return

        if not self.capture_folder:
            messagebox.showerror("Perform Calibration", "Choose a capture folder first.")
            return

        self._scan_stop_requested = False
        self._scan_queue = queue.Queue()
        self._scan_thread = threading.Thread(
            target=self._run_auto_scan,
            args=(
                app.left_video_path,
                app.right_video_path,
                int(app.lock_offset_frames),
                int(app.left_frame_max),
            ),
            daemon=True,
        )
        self._scan_thread.start()

        self.auto_scan_button.config(text="Stop Scan")
        self.status_var.set("Scanning…")
        self.win.after(100, self._poll_scan_queue)

    def on_stop_scan(self):
        """Request the running auto-scan stop early, at its next sampled frame.

        Doesn't stop it instantly - `_run_auto_scan` only checks
        `self._scan_stop_requested` between samples, so it can take up
        to one `AUTO_SCAN_FRAME_STRIDE` worth of decode/detect work to
        actually notice and exit.

        Returns:
            None
        """
        self._scan_stop_requested = True
        self.status_var.set("Stopping scan…")

    def _run_auto_scan(self, left_video_path, right_video_path, lock_offset_frames, left_frame_max):
        """Background worker: scan both videos for calibration candidate frames.

        Runs on a separate thread from the Tk main loop - see this
        module's docstring for why it opens its own fresh
        `cv2.VideoCapture`s instead of reusing `app.capL`/`app.capR`,
        and why it posts messages onto `self._scan_queue` rather than
        updating any Tkinter widget directly (`_poll_scan_queue` does
        that, on the main thread). Saves each detected candidate pair
        immediately (same file/metadata writes `on_capture_frame_pair`
        does) rather than collecting them all and saving at the end -
        see this module's docstring for why manual capture/deletion are
        refused while a scan owns the folder like this.

        Args:
            left_video_path (str): Path to the left video file.
            right_video_path (str): Path to the right video file.
            lock_offset_frames (int): The `right_index - left_index`
                sync offset to apply to every sampled left frame index.
            left_frame_max (int): The highest valid left-video frame
                index to scan up to.

        Returns:
            None
        """
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        charuco_detector = build_charuco_detector()

        found_count = 0

        try:
            for left_index in range(0, left_frame_max + 1, AUTO_SCAN_FRAME_STRIDE):
                if self._scan_stop_requested:
                    break

                # A negative offset can push the right index before frame 0
                # for early left frames - nothing valid to read there yet.
                right_index = left_index + lock_offset_frames
                if right_index < 0:
                    self._scan_queue.put(("progress", left_index, left_frame_max, found_count))
                    continue

                left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_index)
                left_ok, frame_l = left_cap.read()
                right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_index)
                right_ok, frame_r = right_cap.read()

                if (
                    left_ok
                    and right_ok
                    and frame_has_calibration_board(frame_l, charuco_detector)
                    and frame_has_calibration_board(frame_r, charuco_detector)
                ):
                    pair_id = self._next_pair_id()
                    left_filename = f"left_{pair_id:0{PAIR_ID_DIGITS}d}.png"
                    right_filename = f"right_{pair_id:0{PAIR_ID_DIGITS}d}.png"

                    cv2.imwrite(os.path.join(self.capture_folder, left_filename), frame_l)
                    cv2.imwrite(os.path.join(self.capture_folder, right_filename), frame_r)

                    metadata = self._load_metadata()
                    metadata[str(pair_id)] = {"left_frame_index": left_index, "right_frame_index": right_index}
                    self._save_metadata(metadata)

                    found_count += 1

                self._scan_queue.put(("progress", left_index, left_frame_max, found_count))
        finally:
            left_cap.release()
            right_cap.release()
            self._scan_queue.put(("done", found_count))

    def _poll_scan_queue(self):
        """Drain the scan progress queue and refresh the UI on the main thread.

        Rescheduled via `Tk.after` every 100ms while a scan is running -
        the standard safe way to let a background thread's results
        reach Tkinter widgets, since Tkinter itself must only be
        touched from the main thread. Stops rescheduling itself once
        the scan reports it's done, or if this window has been closed
        in the meantime (`self.win` is None).

        Returns:
            None
        """
        if self.win is None:
            return

        finished = False
        while True:
            try:
                message = self._scan_queue.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "progress":
                _kind, left_index, left_frame_max, found_count = message
                self.status_var.set(f"Scanning… frame {left_index}/{left_frame_max}, {found_count} found")
                self._refresh_pairs_listbox()
            elif kind == "done":
                _kind, found_count = message
                self.status_var.set(f"Scan finished — {found_count} candidate pair(s) found")
                finished = True

        if finished:
            self._scan_thread = None
            self._scan_queue = None
            self.auto_scan_button.config(text="Auto-Scan for Candidates")
            return

        self.win.after(100, self._poll_scan_queue)

    def _scan_capture_folder(self):
        """Scan the capture folder for existing matched left/right pairs.

        Only pairs where *both* the `left_####.png` and `right_####.png`
        file exist are counted as a pair - an unmatched file (e.g. one
        side manually deleted outside the app) is silently excluded
        rather than causing an error here.

        Returns:
            list[tuple[int, str, str]]: Sorted `(pair_id, left_filename,
            right_filename)` tuples for every matched pair found
            directly inside `self.capture_folder`. Empty if no folder
            has been chosen yet, the folder doesn't exist, or nothing
            matches.
        """
        if not self.capture_folder or not os.path.isdir(self.capture_folder):
            return []

        # Collect the numeric IDs present on each side separately, so only
        # IDs present on *both* sides get treated as a real pair below.
        left_ids = set()
        right_ids = set()
        for name in os.listdir(self.capture_folder):
            if name.startswith("left_") and name.endswith(".png"):
                digits = name[len("left_"):-len(".png")]
                if digits.isdigit():
                    left_ids.add(int(digits))
            elif name.startswith("right_") and name.endswith(".png"):
                digits = name[len("right_"):-len(".png")]
                if digits.isdigit():
                    right_ids.add(int(digits))

        matched_ids = sorted(left_ids & right_ids)
        return [
            (
                pair_id,
                f"left_{pair_id:0{PAIR_ID_DIGITS}d}.png",
                f"right_{pair_id:0{PAIR_ID_DIGITS}d}.png",
            )
            for pair_id in matched_ids
        ]

    def _next_pair_id(self):
        """Find the next unused numeric pair ID in the capture folder.

        Returns:
            int: One past the highest matched pair ID currently in
            `self.capture_folder`, or `1` if it's empty (or no folder
            has been chosen yet).
        """
        existing_pairs = self._scan_capture_folder()
        if not existing_pairs:
            return 1

        highest_id = existing_pairs[-1][0]
        return highest_id + 1

    def _refresh_pairs_listbox(self):
        """Repopulate the captured-pairs listbox from the capture folder.

        Re-scans the folder from disk every time (via
        `_scan_capture_folder`) rather than tracking captures in memory,
        so the list is always correct even for a folder that already
        had pairs in it before this window was opened.

        Returns:
            None
        """
        if self.pairs_listbox is None:
            return

        self.pairs_listbox.delete(0, "end")
        for _pair_id, left_filename, right_filename in self._scan_capture_folder():
            self.pairs_listbox.insert("end", f"{left_filename}   /   {right_filename}")
