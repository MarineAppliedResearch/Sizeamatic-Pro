"""Perform Calibration window for Sizeamatic Pro.

This module creates and manages the Tkinter window used to build a new
stereo calibration from scratch (ROADMAP.md Phase 10), rather than only
ever loading an already-finished one (`calibration_io.py`). Step 1 of
that phase is frame-pair capture: letting the project owner either
manually scrub the already-loaded left/right video pair to a frame
showing a checkerboard/ChArUco calibration target and capture it, or
run an automatic scan that samples through the video looking for
frames where a board is actually detected and captures those on its
own. Step 2 is running the actual calibration computation
(`cv2.calibrateCamera`/`stereoCalibrate`/`stereoRectify`) on those
captured pairs, saving the four calibration NPZ files
`calibration_io.py` already knows how to load, directly into the same
capture folder the source images live in - one button in this same
window, rather than a separate one.

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

    `on_run_calibration` (Step 2) needs the project owner to pick one
    board type up front (`self.board_type_var`, "Checkerboard" - the
    priority - or "ChArUco") rather than auto-detecting it per pair the
    way the auto-scan does - `cv2.calibrateCamera`/`stereoCalibrate`
    need every view's object points to describe the *same physical
    board*, so mixing checkerboard- and ChArUco-derived points into one
    run isn't valid the way "detect whichever's present" is for the
    auto-scan's much simpler yes/no question. Both board types have
    fully user-set dimensions - checkerboard's squares/square size
    (`self.checkerboard_squares_x_var`/`_y_var`/
    `checkerboard_square_size_var`) and ChArUco's squares/square size/
    marker size (`self.charuco_squares_x_var`/`_y_var`/
    `charuco_square_size_var`/`charuco_marker_size_var`) - defaulting to
    `create_checkerboard_calibration_target.py`'s/
    `create_charuco_calibration_target.py`'s own default printable
    boards respectively, so a board printed with either generator
    script's defaults is exactly what a freshly opened Perform
    Calibration window expects. A board printed at a different size, or
    with a different square/row count, wouldn't be detected (or would
    calibrate in the wrong real-world units) against the wrong settings
    - the same reasoning applies to ChArUco here as it already did to
    checkerboard, since ChArUco's ArUco markers encode each corner's
    *ID*, not the board's overall physical dimensions - the detector
    still has to be built with the right square/marker sizes for
    `cv2.calibrateCamera` to produce real-world units. Only the ArUco
    *dictionary* (`CHARUCO_DICTIONARY_ID`) stays a fixed module
    constant, not user-set - unlike squares/sizes, picking the wrong
    dictionary makes detection fail outright rather than merely
    calibrating in the wrong units, and this app only ever needs one.

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
import numpy as np

PAIR_ID_DIGITS = 4
"""Zero-padding width for captured pair filenames (e.g. `left_0007.png`)
- wide enough that captured pairs still sort correctly by filename well
past what any single calibration run would realistically capture."""

DEFAULT_CHECKERBOARD_SQUARES_X = 10
"""Default checkerboard width in *squares* (not inner corners), matching
`create_checkerboard_calibration_target.py`'s own default printable
board - so a board printed with that script's defaults is exactly what
the calibration UI's own default settings expect. The project owner can
override this per capture folder (`self.checkerboard_squares_x_var`) for
a different physical board - a plain checkerboard has no way to encode
its own dimensions the way a ChArUco board's markers encode corner IDs,
so there's no single "right" fixed size to hardcode instead."""

DEFAULT_CHECKERBOARD_SQUARES_Y = 7
"""Default checkerboard height in squares. See
`DEFAULT_CHECKERBOARD_SQUARES_X`."""

DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM = 25.0
"""Default checkerboard square size in millimeters. See
`DEFAULT_CHECKERBOARD_SQUARES_X`."""

DEFAULT_CHARUCO_SQUARES_X = 11
"""Default ChArUco board width in squares, matching
`create_charuco_calibration_target.py`'s own default printable board -
so a board printed with that script's defaults is exactly what this
window's own default settings expect. The project owner can override
this (`self.charuco_squares_x_var`) for a different physical ChArUco
board, same as checkerboard's own squares settings."""

DEFAULT_CHARUCO_SQUARES_Y = 8
"""Default ChArUco board height in squares. See
`DEFAULT_CHARUCO_SQUARES_X`."""

DEFAULT_CHARUCO_SQUARE_SIZE_MM = 20.0
"""Default ChArUco square size in millimeters. See
`DEFAULT_CHARUCO_SQUARES_X`."""

DEFAULT_CHARUCO_MARKER_SIZE_MM = 15.0
"""Default ChArUco marker size in millimeters. See
`DEFAULT_CHARUCO_SQUARES_X`."""

CHARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_1000
"""ArUco marker dictionary every ChArUco detector/generator in this app
uses - unlike squares/square size/marker size, this stays a fixed
module constant rather than a user-set field. See this module's
docstring for why."""

AUTO_SCAN_FRAME_STRIDE = 15
"""How many left-video frames the auto-scan advances between each
sampled frame it actually decodes and runs detection on. Sampling
rather than checking every single frame keeps a full-video scan fast;
15 is roughly a half-second step at a typical 30fps recording."""


def frame_has_checkerboard(gray_image, inner_corners):
    """Check whether a grayscale image contains a detectable checkerboard.

    Args:
        gray_image (numpy.ndarray): A single-channel (grayscale) image.
        inner_corners (tuple[int, int]): The `(columns, rows)` of
            *inner* corners to look for - one less than the board's
            actual square count in each direction (e.g. a 10x7-square
            board has 9x6 inner corners).

    Returns:
        bool: True if `cv2.findChessboardCorners` found a full
        `inner_corners`-sized grid.
    """
    found, _corners = cv2.findChessboardCorners(
        gray_image,
        inner_corners,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return bool(found)


def build_charuco_detector(squares_x, squares_y, square_size_mm, marker_size_mm):
    """Build the ChArUco board/detector objects the auto-scan/calibration reuse.

    Built once per scan or calibration run (not once per sampled
    frame/pair) since constructing the dictionary/board/detector triplet
    repeatedly would be wasted work - the board geometry never changes
    mid-run. Unlike the ArUco dictionary (`CHARUCO_DICTIONARY_ID`,
    always fixed), the board's physical geometry is user-set - see this
    module's docstring for why.

    Args:
        squares_x (int): Number of chessboard squares along the X axis.
        squares_y (int): Number of chessboard squares along the Y axis.
        square_size_mm (float): Physical size of each chessboard square,
            in millimeters.
        marker_size_mm (float): Physical size of each ArUco marker, in
            millimeters.

    Returns:
        cv2.aruco.CharucoDetector: A detector configured for the given
        board geometry, using the fixed `CHARUCO_DICTIONARY_ID`.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_mm,
        marker_size_mm,
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


def frame_has_calibration_board(bgr_image, charuco_detector, checkerboard_inner_corners):
    """Check whether a frame has a detectable checkerboard or ChArUco board.

    Tries checkerboard detection first, then ChArUco, so either board
    type present in the video gets picked up without the project owner
    needing to specify which one is actually in use during the
    auto-scan (unlike an actual calibration run - see this module's
    docstring for why that needs one committed-to type).

    Args:
        bgr_image (numpy.ndarray): A decoded BGR video frame.
        charuco_detector (cv2.aruco.CharucoDetector): A detector built
            by `build_charuco_detector`.
        checkerboard_inner_corners (tuple[int, int]): The `(columns,
            rows)` of checkerboard inner corners to look for - see
            `frame_has_checkerboard`.

    Returns:
        bool: True if either detector found a board.
    """
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return frame_has_checkerboard(gray_image, checkerboard_inner_corners) or frame_has_charuco_board(
        gray_image, charuco_detector
    )


def detect_checkerboard_points(gray_image, inner_corners):
    """Detect and subpixel-refine checkerboard corners in an image.

    Unlike `frame_has_checkerboard` (a plain yes/no check for the
    auto-scan), this returns the actual corner locations needed to
    build a real calibration - refined via `cv2.cornerSubPix` for
    accuracy, since the raw corners `findChessboardCorners` returns are
    only approximate.

    Args:
        gray_image (numpy.ndarray): A single-channel (grayscale) image.
        inner_corners (tuple[int, int]): The `(columns, rows)` of inner
            corners to look for - see `frame_has_checkerboard`.

    Returns:
        numpy.ndarray | None: The refined `(N, 2)` float32 corner array
        (this OpenCV build's `findChessboardCorners`/`cornerSubPix`
        return that shape rather than the `(N, 1, 2)` some older
        examples show - `cv2.calibrateCamera` accepts either) if a full
        `inner_corners`-sized grid was found, else None.
    """
    found, corners = cv2.findChessboardCorners(
        gray_image,
        inner_corners,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None

    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), refine_criteria)


def build_checkerboard_object_points(square_size_mm, inner_corners):
    """Build the real-world object-point grid for one checkerboard view.

    Every checkerboard view of the *same* physical board shares this
    exact same object-point array - only the detected image points
    (where those corners actually landed in a given photo) differ
    between views.

    Args:
        square_size_mm (float): The checkerboard's real-world square
            size, in millimeters.
        inner_corners (tuple[int, int]): The `(columns, rows)` of inner
            corners - see `frame_has_checkerboard`.

    Returns:
        numpy.ndarray: A `(rows * columns, 3)` float32 array of
        `(x, y, 0)` points in checkerboard order, scaled to real-world
        millimeters.
    """
    inner_columns, inner_rows = inner_corners
    object_points = np.zeros((inner_rows * inner_columns, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:inner_columns, 0:inner_rows].T.reshape(-1, 2)
    object_points *= square_size_mm
    return object_points


def detect_charuco_points(gray_image, detector):
    """Detect ChArUco corners and their board IDs in an image.

    Args:
        gray_image (numpy.ndarray): A single-channel (grayscale) image.
        detector (cv2.aruco.CharucoDetector): A detector built by
            `build_charuco_detector`.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray] | tuple[None, None]: The
        `(charuco_corners, charuco_ids)` pair if at least 4 corners
        were found, else `(None, None)`.
    """
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray_image)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None, None
    return charuco_corners, charuco_ids


def match_charuco_points_for_pair(left_corners, left_ids, right_corners, right_ids, charuco_board):
    """Match a stereo pair's ChArUco detections down to their shared corner IDs.

    A ChArUco corner ID detected in only one of the two images can't
    contribute a stereo point - occlusion or a bad viewing angle on one
    side is common even when the other side sees the board cleanly.
    This keeps only the IDs seen on *both* sides, in a consistent
    matching order, and looks up each one's real-world object point
    directly from the board geometry (`board.getChessboardCorners()`
    is indexed by corner ID, matching how the reference
    `feature-onlineCalibrations` branch this was adapted from does it).

    Args:
        left_corners (numpy.ndarray): Left-image ChArUco corners, from
            `detect_charuco_points`.
        left_ids (numpy.ndarray): Left-image ChArUco corner IDs,
            matching `left_corners`.
        right_corners (numpy.ndarray): Right-image ChArUco corners.
        right_ids (numpy.ndarray): Right-image ChArUco corner IDs.
        charuco_board (cv2.aruco.CharucoBoard): The board geometry
            (`detector.getBoard()`) both sides were detected against.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None:
        `(object_points, left_image_points, right_image_points)` for
        the shared IDs, or None if fewer than 4 IDs are shared.
    """
    left_corner_by_id = {int(cid): left_corners[i].reshape(2) for i, cid in enumerate(left_ids.reshape(-1))}
    right_corner_by_id = {int(cid): right_corners[i].reshape(2) for i, cid in enumerate(right_ids.reshape(-1))}

    shared_ids = sorted(set(left_corner_by_id) & set(right_corner_by_id))
    if len(shared_ids) < 4:
        return None

    chessboard_corners = charuco_board.getChessboardCorners()
    object_points = np.array([chessboard_corners[cid] for cid in shared_ids], dtype=np.float32)
    left_image_points = np.array([left_corner_by_id[cid] for cid in shared_ids], dtype=np.float32).reshape(-1, 1, 2)
    right_image_points = np.array([right_corner_by_id[cid] for cid in shared_ids], dtype=np.float32).reshape(-1, 1, 2)

    return object_points, left_image_points, right_image_points


def run_stereo_calibration(object_points_list, left_image_points_list, right_image_points_list, image_size, output_folder):
    """Run OpenCV stereo calibration and save the four calibration NPZ files.

    Calibrates each camera's intrinsics independently, then the
    stereo extrinsics between them (holding intrinsics fixed, via
    `cv2.CALIB_FIX_INTRINSIC`), then the rectification transforms/remap
    tables - the same sequence, and the same output file names/keys,
    `calibration_io.py` already expects (matching the reference
    `feature-onlineCalibrations` branch's own `_run_stereo_calibration`,
    which was written against this exact loader format).

    Args:
        object_points_list (list[numpy.ndarray]): One real-world
            object-point array per valid stereo pair.
        left_image_points_list (list[numpy.ndarray]): One left-image
            point array per valid stereo pair, same order/length as
            `object_points_list`.
        right_image_points_list (list[numpy.ndarray]): One right-image
            point array per valid stereo pair, same order/length as
            `object_points_list`.
        image_size (tuple[int, int]): The `(width, height)` of the
            images calibration was run against, in pixels.
        output_folder (str): Folder to write the four calibration NPZ
            files into.

    Returns:
        dict: Keys `"left_rms"`, `"right_rms"`, `"stereo_rms"` (OpenCV's
        own reprojection-error quality numbers, lower is better), and
        `"valid_pair_count"`.

    Raises:
        ValueError: If fewer than 3 valid stereo pairs are given -
            OpenCV's stereo calibration isn't numerically meaningful
            with less data than that.
    """
    if len(object_points_list) < 3:
        raise ValueError("At least 3 valid stereo pairs are required for calibration.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # Calibrate each camera's own intrinsics/distortion independently first.
    left_rms, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        object_points_list, left_image_points_list, image_size, None, None
    )
    right_rms, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        object_points_list, right_image_points_list, image_size, None, None
    )

    # Calibrate the stereo extrinsics between the two cameras, keeping the
    # intrinsics just computed above fixed rather than re-solving them.
    stereo_rms, mtx_l, dist_l, mtx_r, dist_r, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points_list,
        left_image_points_list,
        right_image_points_list,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    # Calculate stereo rectification transforms and projection matrices.
    rect_l, rect_r, proj_l, proj_r, disparity_to_depth, roi_l, roi_r = cv2.stereoRectify(
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    # Build the undistortion/rectification remap tables for each camera.
    map_lx, map_ly = cv2.initUndistortRectifyMap(mtx_l, dist_l, rect_l, proj_l, image_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(mtx_r, dist_r, rect_r, proj_r, image_size, cv2.CV_32FC1)

    # Save intrinsics, in the format calibration_io.py's loader expects.
    np.savez(
        os.path.join(output_folder, "calibration_intrinsics.npz"),
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        mtxL=mtx_l,
        distL=dist_l,
        mtxR=mtx_r,
        distR=dist_r,
        left_rms=left_rms,
        right_rms=right_rms,
        rvecsL=np.array(rvecs_l, dtype=object),
        tvecsL=np.array(tvecs_l, dtype=object),
        rvecsR=np.array(rvecs_r, dtype=object),
        tvecsR=np.array(tvecs_r, dtype=object),
    )

    # Save stereo extrinsics.
    np.savez(
        os.path.join(output_folder, "calibration_extrinsics.npz"),
        stereo_rms=stereo_rms,
        R=rotation,
        T=translation,
        E=essential,
        F=fundamental,
    )

    # Save stereo rectification data.
    np.savez(
        os.path.join(output_folder, "calibration_rectification.npz"),
        RL=rect_l,
        RR=rect_r,
        PL=proj_l,
        PR=proj_r,
        Q=disparity_to_depth,
        roiL=np.array(roi_l),
        roiR=np.array(roi_r),
    )

    # Save the rectification remap tables.
    np.savez(
        os.path.join(output_folder, "calibration_maps.npz"),
        mapLx=map_lx,
        mapLy=map_ly,
        mapRx=map_rx,
        mapRy=map_ry,
    )

    return {
        "left_rms": float(left_rms),
        "right_rms": float(right_rms),
        "stereo_rms": float(stereo_rms),
        "valid_pair_count": len(object_points_list),
    }


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

        self.board_type_var = None
        """`tk.StringVar` holding which board type `on_run_calibration`
        (Step 2) should treat every captured pair as - `"Checkerboard"`
        or `"ChArUco"`. Created in `ensure_window`. See this module's
        docstring for why a calibration run needs one consistent board
        type rather than auto-detecting it per pair the way the
        auto-scan does."""

        self.checkerboard_square_size_var = None
        """`tk.StringVar` holding the checkerboard's real-world square
        size in millimeters, read by `on_run_calibration` only when
        `self.board_type_var` is `"Checkerboard"`. Created in
        `ensure_window`."""

        self.checkerboard_squares_x_var = None
        """`tk.StringVar` holding the checkerboard's width in *squares*
        (not inner corners - see `frame_has_checkerboard`), read by both
        `on_run_calibration` and the auto-scan
        (`on_auto_scan_for_candidates`) whenever checkerboard detection
        is relevant. A plain checkerboard has no marker IDs to encode
        its own size, so there's no single fixed default that works for
        every physical board - this is user-set instead, defaulting to
        `DEFAULT_CHECKERBOARD_SQUARES_X` (matching
        `create_checkerboard_calibration_target.py`'s own default
        printable board, so the two line up out of the box). Created in
        `ensure_window`."""

        self.checkerboard_squares_y_var = None
        """`tk.StringVar` holding the checkerboard's height in squares.
        See `self.checkerboard_squares_x_var`."""

        self.charuco_squares_x_var = None
        """`tk.StringVar` holding the ChArUco board's width in squares,
        read by both `on_run_calibration` and the auto-scan whenever
        ChArUco detection is relevant, defaulting to
        `DEFAULT_CHARUCO_SQUARES_X` (matching
        `create_charuco_calibration_target.py`'s own default printable
        board). Created in `ensure_window`."""

        self.charuco_squares_y_var = None
        """`tk.StringVar` holding the ChArUco board's height in squares.
        See `self.charuco_squares_x_var`."""

        self.charuco_square_size_var = None
        """`tk.StringVar` holding the ChArUco board's real-world square
        size in millimeters. See `self.charuco_squares_x_var`."""

        self.charuco_marker_size_var = None
        """`tk.StringVar` holding the ChArUco board's real-world ArUco
        marker size in millimeters - must be smaller than the square
        size (`_parse_charuco_settings` enforces this). See
        `self.charuco_squares_x_var`."""

        self.checkerboard_row = None
        """The `ttk.Frame` holding the checkerboard-specific settings -
        shown only while `self.board_type_var` is `"Checkerboard"`.
        Created in `ensure_window`."""

        self.charuco_row = None
        """The `ttk.Frame` holding the ChArUco-specific settings - shown
        only while `self.board_type_var` is `"ChArUco"`. Created in
        `ensure_window`."""

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
        self.board_type_var = None
        self.checkerboard_square_size_var = None
        self.checkerboard_squares_x_var = None
        self.checkerboard_squares_y_var = None
        self.charuco_squares_x_var = None
        self.charuco_squares_y_var = None
        self.charuco_square_size_var = None
        self.charuco_marker_size_var = None
        self.checkerboard_row = None
        self.charuco_row = None

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
        # capture/calibration buttons, and pairs list without feeling
        # cramped.
        win.geometry("600x560")

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

        # ---- Calibration row ----
        # Board type (a run needs one consistent type - see this module's
        # docstring) plus the button that actually runs the calibration
        # math (Step 2). Choosing a board type here shows that type's own
        # size/dimension fields below (checkerboard_row/charuco_row) and
        # hides the other - see _on_board_type_changed.
        calibration_row = ttk.Frame(outer)
        calibration_row.grid(row=4, column=0, sticky="w", pady=(10, 0))

        ttk.Label(calibration_row, text="Board type:").grid(row=0, column=0)
        self.board_type_var = tk.StringVar(value="Checkerboard")
        board_type_combo = ttk.Combobox(
            calibration_row,
            textvariable=self.board_type_var,
            values=["Checkerboard", "ChArUco"],
            width=12,
            state="readonly",
        )
        board_type_combo.grid(row=0, column=1, padx=(4, 12))
        board_type_combo.bind("<<ComboboxSelected>>", lambda event: self._on_board_type_changed())

        ttk.Button(
            calibration_row,
            text="Run Calibration",
            command=self.on_run_calibration,
        ).grid(row=0, column=2)

        # A plain checkerboard has no marker IDs to encode its own
        # dimensions, so its size has to be set here, matching whatever
        # physical board is actually in use. Gridded onto the same row as
        # charuco_row below - only one of the two is ever visible at a
        # time, toggled by _on_board_type_changed.
        self.checkerboard_row = ttk.Frame(outer)
        self.checkerboard_row.grid(row=5, column=0, sticky="w", pady=(6, 0))

        ttk.Label(self.checkerboard_row, text="Checkerboard squares (columns x rows):").grid(row=0, column=0)
        self.checkerboard_squares_x_var = tk.StringVar(value=str(DEFAULT_CHECKERBOARD_SQUARES_X))
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_squares_x_var, width=4).grid(
            row=0, column=1, padx=(4, 2)
        )
        ttk.Label(self.checkerboard_row, text="x").grid(row=0, column=2)
        self.checkerboard_squares_y_var = tk.StringVar(value=str(DEFAULT_CHECKERBOARD_SQUARES_Y))
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_squares_y_var, width=4).grid(
            row=0, column=3, padx=(2, 12)
        )

        ttk.Label(self.checkerboard_row, text="Square size (mm):").grid(row=0, column=4)
        self.checkerboard_square_size_var = tk.StringVar(value=str(DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM))
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_square_size_var, width=8).grid(
            row=0, column=5, padx=(4, 0)
        )

        # ChArUco's own size/dimension/marker fields - see this module's
        # docstring for why ChArUco's geometry is user-set too, not fixed.
        self.charuco_row = ttk.Frame(outer)
        self.charuco_row.grid(row=5, column=0, sticky="w", pady=(6, 0))

        ttk.Label(self.charuco_row, text="ChArUco squares (columns x rows):").grid(row=0, column=0)
        self.charuco_squares_x_var = tk.StringVar(value=str(DEFAULT_CHARUCO_SQUARES_X))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_squares_x_var, width=4).grid(
            row=0, column=1, padx=(4, 2)
        )
        ttk.Label(self.charuco_row, text="x").grid(row=0, column=2)
        self.charuco_squares_y_var = tk.StringVar(value=str(DEFAULT_CHARUCO_SQUARES_Y))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_squares_y_var, width=4).grid(
            row=0, column=3, padx=(2, 12)
        )

        ttk.Label(self.charuco_row, text="Square (mm):").grid(row=0, column=4)
        self.charuco_square_size_var = tk.StringVar(value=str(DEFAULT_CHARUCO_SQUARE_SIZE_MM))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_square_size_var, width=6).grid(
            row=0, column=5, padx=(4, 12)
        )

        ttk.Label(self.charuco_row, text="Marker (mm):").grid(row=0, column=6)
        self.charuco_marker_size_var = tk.StringVar(value=str(DEFAULT_CHARUCO_MARKER_SIZE_MM))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_marker_size_var, width=6).grid(
            row=0, column=7, padx=(4, 0)
        )

        # ---- Captured pairs list ----
        list_frame = ttk.Frame(outer)
        list_frame.grid(row=6, column=0, sticky="nsew", pady=(10, 0))
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
            row=7, column=0, sticky="w", pady=(6, 0)
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
        outer.grid_rowconfigure(6, weight=1)

        # Show only the settings row matching the current (default)
        # board type.
        self._on_board_type_changed()

        # Populate the pairs list immediately in case a folder was already
        # chosen in an earlier call this session and already has captures.
        self._refresh_pairs_listbox()

    def _on_board_type_changed(self):
        """Show the settings row matching the selected board type.

        `self.checkerboard_row` and `self.charuco_row` are gridded onto
        the same row - only one is ever visible at a time. Called when
        the board type combobox changes, and once from `ensure_window`
        to set the correct initial visibility.

        Returns:
            None
        """
        if self.board_type_var.get() == "Checkerboard":
            self.charuco_row.grid_remove()
            self.checkerboard_row.grid()
        else:
            self.checkerboard_row.grid_remove()
            self.charuco_row.grid()

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

        # The scan always tries both checkerboard and ChArUco detection
        # regardless of self.board_type_var (see
        # frame_has_calibration_board), so it needs valid settings for
        # *both* board types even though only one is actually selected
        # for the eventual calibration run.
        parsed_checkerboard_settings = self._parse_checkerboard_settings()
        if parsed_checkerboard_settings is None:
            return
        checkerboard_inner_corners, _checkerboard_square_size_mm = parsed_checkerboard_settings

        parsed_charuco_settings = self._parse_charuco_settings()
        if parsed_charuco_settings is None:
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
                checkerboard_inner_corners,
                parsed_charuco_settings,
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

    def _run_auto_scan(
        self,
        left_video_path,
        right_video_path,
        lock_offset_frames,
        left_frame_max,
        checkerboard_inner_corners,
        charuco_settings,
    ):
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
            checkerboard_inner_corners (tuple[int, int]): The `(columns,
                rows)` of checkerboard inner corners to look for - see
                `frame_has_checkerboard`. Passed in from
                `on_auto_scan_for_candidates` (already parsed/validated
                there) rather than read from `self` directly, since Tk
                variables aren't safe to read from a background thread.
            charuco_settings (tuple[int, int, float, float]): The
                `(squares_x, squares_y, square_size_mm, marker_size_mm)`
                ChArUco board geometry to look for - see
                `_parse_charuco_settings`. Also passed in from
                `on_auto_scan_for_candidates` for the same reason as
                `checkerboard_inner_corners`.

        Returns:
            None
        """
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        charuco_detector = build_charuco_detector(*charuco_settings)

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
                    and frame_has_calibration_board(frame_l, charuco_detector, checkerboard_inner_corners)
                    and frame_has_calibration_board(frame_r, charuco_detector, checkerboard_inner_corners)
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

    def _parse_checkerboard_settings(self):
        """Parse and validate the checkerboard squares/size fields.

        Shared by `on_run_calibration` (needs the real-world size too)
        and `on_auto_scan_for_candidates` (only needs the inner-corner
        count, to know what to look for - checkerboard is one of two
        detectors the scan always tries, regardless of
        `self.board_type_var`, so it needs valid squares even when a
        ChArUco run is what's actually selected).

        Returns:
            tuple[tuple[int, int], float] | None: `((inner_columns,
            inner_rows), square_size_mm)` if every field is valid. If
            any field isn't, an error messagebox is shown and this
            returns None - callers just need to return early in that
            case, not show their own error too.
        """
        try:
            squares_x = int(self.checkerboard_squares_x_var.get())
            squares_y = int(self.checkerboard_squares_y_var.get())
        except ValueError:
            messagebox.showerror("Perform Calibration", "Enter valid whole numbers for the checkerboard's squares.")
            return None

        if squares_x < 2 or squares_y < 2:
            messagebox.showerror("Perform Calibration", "Checkerboard squares must be at least 2 in each direction.")
            return None

        try:
            square_size_mm = float(self.checkerboard_square_size_var.get())
        except ValueError:
            messagebox.showerror("Perform Calibration", "Enter a valid checkerboard square size in millimeters.")
            return None

        if square_size_mm <= 0:
            messagebox.showerror("Perform Calibration", "Checkerboard square size must be greater than zero.")
            return None

        # findChessboardCorners/calibrateCamera work in inner corners, one
        # fewer than the square count in each direction - see
        # frame_has_checkerboard's docstring.
        inner_corners = (squares_x - 1, squares_y - 1)
        return inner_corners, square_size_mm

    def _parse_charuco_settings(self):
        """Parse and validate the ChArUco squares/square size/marker size fields.

        Shared by `on_run_calibration` and `on_auto_scan_for_candidates`
        (ChArUco is one of two detectors the scan always tries,
        regardless of `self.board_type_var`, so it needs valid settings
        even when a checkerboard run is what's actually selected) -
        mirrors `_parse_checkerboard_settings`.

        Returns:
            tuple[int, int, float, float] | None: `(squares_x, squares_y,
            square_size_mm, marker_size_mm)` if every field is valid. If
            any field isn't, an error messagebox is shown and this
            returns None - callers just need to return early in that
            case, not show their own error too.
        """
        try:
            squares_x = int(self.charuco_squares_x_var.get())
            squares_y = int(self.charuco_squares_y_var.get())
        except ValueError:
            messagebox.showerror("Perform Calibration", "Enter valid whole numbers for the ChArUco board's squares.")
            return None

        if squares_x < 2 or squares_y < 2:
            messagebox.showerror("Perform Calibration", "ChArUco squares must be at least 2 in each direction.")
            return None

        try:
            square_size_mm = float(self.charuco_square_size_var.get())
            marker_size_mm = float(self.charuco_marker_size_var.get())
        except ValueError:
            messagebox.showerror(
                "Perform Calibration", "Enter valid ChArUco square/marker sizes in millimeters."
            )
            return None

        if square_size_mm <= 0 or marker_size_mm <= 0:
            messagebox.showerror("Perform Calibration", "ChArUco square/marker sizes must be greater than zero.")
            return None

        if marker_size_mm >= square_size_mm:
            messagebox.showerror("Perform Calibration", "ChArUco marker size must be smaller than the square size.")
            return None

        return squares_x, squares_y, square_size_mm, marker_size_mm

    def on_run_calibration(self):
        """Run stereo calibration on every captured pair and save the result.

        Detects board points in every pair currently in the capture
        folder, using whichever single board type is selected
        (`self.board_type_var`) - not auto-detected per pair, see this
        module's docstring for why a run needs one consistent type.
        Pairs where detection fails on either side (or, for ChArUco,
        where fewer than 4 corner IDs are shared between the two sides)
        are skipped and counted rather than aborting the whole run.
        Runs synchronously on the main thread rather than a background
        one like the auto-scan - detecting points in an already-small
        set of still images is fast enough not to freeze the UI
        noticeably, unlike scanning a whole video.

        Returns:
            None
        """
        if self._scan_thread is not None:
            messagebox.showerror("Perform Calibration", "Stop the current scan before running calibration.")
            return

        if not self.capture_folder:
            messagebox.showerror("Perform Calibration", "Choose a capture folder first.")
            return

        pairs = self._scan_capture_folder()
        if not pairs:
            messagebox.showerror("Perform Calibration", "No captured pairs to calibrate from.")
            return

        board_type = self.board_type_var.get()

        if board_type == "Checkerboard":
            parsed_settings = self._parse_checkerboard_settings()
            if parsed_settings is None:
                return
            inner_corners, square_size_mm = parsed_settings
            checkerboard_object_points = build_checkerboard_object_points(square_size_mm, inner_corners)
            charuco_detector = None
            charuco_board = None
        else:
            parsed_charuco_settings = self._parse_charuco_settings()
            if parsed_charuco_settings is None:
                return
            inner_corners = None
            checkerboard_object_points = None
            charuco_detector = build_charuco_detector(*parsed_charuco_settings)
            charuco_board = charuco_detector.getBoard()

        object_points_list = []
        left_image_points_list = []
        right_image_points_list = []
        image_size = None
        skipped_pair_count = 0

        for pair_id, left_filename, right_filename in pairs:
            left_image = cv2.imread(os.path.join(self.capture_folder, left_filename))
            right_image = cv2.imread(os.path.join(self.capture_folder, right_filename))
            if left_image is None or right_image is None:
                skipped_pair_count += 1
                continue

            # All pairs are assumed to share one image size - the first
            # successfully-read pair sets it for cv2.calibrateCamera below.
            if image_size is None:
                image_size = (left_image.shape[1], left_image.shape[0])

            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            if board_type == "Checkerboard":
                left_points = detect_checkerboard_points(left_gray, inner_corners)
                right_points = detect_checkerboard_points(right_gray, inner_corners)
                if left_points is None or right_points is None:
                    skipped_pair_count += 1
                    continue

                object_points_list.append(checkerboard_object_points)
                left_image_points_list.append(left_points)
                right_image_points_list.append(right_points)
            else:
                left_corners, left_ids = detect_charuco_points(left_gray, charuco_detector)
                right_corners, right_ids = detect_charuco_points(right_gray, charuco_detector)
                if left_corners is None or right_corners is None:
                    skipped_pair_count += 1
                    continue

                matched = match_charuco_points_for_pair(left_corners, left_ids, right_corners, right_ids, charuco_board)
                if matched is None:
                    skipped_pair_count += 1
                    continue

                object_points, left_points, right_points = matched
                object_points_list.append(object_points)
                left_image_points_list.append(left_points)
                right_image_points_list.append(right_points)

        if len(object_points_list) < 3:
            messagebox.showerror(
                "Perform Calibration",
                f"Only {len(object_points_list)} usable pair(s) out of {len(pairs)} - at least 3 are "
                "needed. Capture more pairs with the board clearly visible in both frames.",
            )
            return

        try:
            result = run_stereo_calibration(
                object_points_list, left_image_points_list, right_image_points_list, image_size, self.capture_folder
            )
        except ValueError as e:
            messagebox.showerror("Perform Calibration", str(e))
            return

        skipped_note = f", {skipped_pair_count} skipped" if skipped_pair_count else ""
        self.status_var.set(
            f"Calibration saved — stereo RMS {result['stereo_rms']:.3f} "
            f"(left {result['left_rms']:.3f}, right {result['right_rms']:.3f}), "
            f"{result['valid_pair_count']} pair(s) used{skipped_note}"
        )
        messagebox.showinfo(
            "Perform Calibration",
            f"Calibration saved to:\n{self.capture_folder}\n\n"
            f"Stereo RMS: {result['stereo_rms']:.4f}\n"
            f"Left RMS: {result['left_rms']:.4f}\n"
            f"Right RMS: {result['right_rms']:.4f}\n"
            f"Pairs used: {result['valid_pair_count']} / {len(pairs)}"
            + (f"\nSkipped: {skipped_pair_count}" if skipped_pair_count else ""),
        )

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
