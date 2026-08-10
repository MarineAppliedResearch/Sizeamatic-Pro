"""Calibration file loading and validation for Sizeamatic Pro.

Pulled out of `main.py`'s `on_load_calibration_folder` (Phase 5 in
`ROADMAP.md`) so the loading/validation logic is testable and reusable
without going through a Tkinter directory-chooser dialog first.
"""

import os

import numpy as np


def load_calibration_bundle(folder, meta_left=None, meta_right=None):
    """Load and validate a stereo calibration bundle from a folder.

    Verifies the four expected NPZ files exist, loads them, and (if
    given) validates the calibrated resolution against already-loaded
    video metadata. Returns a `(value, error)` pair rather than raising,
    matching the convention already used elsewhere in this codebase (e.g.
    `stereo_matching.triangulate_point_pair`).

    Args:
        folder (str): Path to the calibration folder, expected to contain
            `calibration_intrinsics.npz`, `calibration_extrinsics.npz`,
            `calibration_rectification.npz`, and `calibration_maps.npz`.
        meta_left (dict | None): Left video metadata (with "width"/
            "height" keys), or None if no left video is loaded yet.
        meta_right (dict | None): Right video metadata, or None if no
            right video is loaded yet.

    Returns:
        tuple[dict, None] | tuple[None, str]: `(cal, None)` on success,
        where `cal` has the same keys `SizeamaticProApp.cal` has always
        had ("w"/"h", intrinsics, extrinsics, rectification, remap
        arrays — see `main.py`'s `self.cal` docstring); or
        `(None, error_message)` if the files are missing, fail to load,
        or their resolution doesn't match the given video metadata.
    """

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

    # Report the specific missing files rather than failing generically.
    if missing:
        return None, f"Missing calibration files: {', '.join(missing)}"

    try:
        intr = np.load(intr_path)
        rect = np.load(rect_path)
        maps = np.load(maps_path)
        extr = np.load(extr_path)

        # Pull required matrices/maps.
        PL = rect["PL"]
        PR = rect["PR"]
        Q = rect["Q"]

        mapLx = maps["mapLx"]
        mapLy = maps["mapLy"]
        mapRx = maps["mapRx"]
        mapRy = maps["mapRy"]

        # Intrinsics
        mtxL = intr["mtxL"]
        distL = intr["distL"]
        mtxR = intr["mtxR"]
        distR = intr["distR"]

        # Extrinsics
        R = extr["R"]
        T = extr["T"]
        E = extr["E"]
        F = extr["F"]
        stereo_rms = float(extr["stereo_rms"]) if "stereo_rms" in extr.files else None

        # Rectification
        RL = rect["RL"] if "RL" in rect.files else None
        RR = rect["RR"] if "RR" in rect.files else None
        roiL = rect["roiL"] if "roiL" in rect.files else None
        roiR = rect["roiR"] if "roiR" in rect.files else None

        # Intrinsics file stores expected calibration resolution.
        cal_w = int(intr["image_width"])
        cal_h = int(intr["image_height"])

    except Exception as e:
        return None, f"Failed to load calibration: {e}"

    # If a video is already loaded, enforce resolution match now.
    # Rectification maps must match the decoded frame size.
    if meta_left:
        if meta_left["width"] != cal_w or meta_left["height"] != cal_h:
            return None, "Calibration resolution does not match LEFT video"

    if meta_right:
        if meta_right["width"] != cal_w or meta_right["height"] != cal_h:
            return None, "Calibration resolution does not match RIGHT video"

    # Build the calibration bundle.
    cal = {
        # Sizes
        "w": cal_w,
        "h": cal_h,

        # Intrinsics
        "mtxL": mtxL,
        "distL": distL,
        "mtxR": mtxR,
        "distR": distR,

        # Extrinsics
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "stereo_rms": stereo_rms,

        # Rectification
        "RL": RL,
        "RR": RR,
        "PL": PL,
        "PR": PR,
        "Q": Q,
        "roiL": roiL,
        "roiR": roiR,

        # Maps
        "mapLx": mapLx,
        "mapLy": mapLy,
        "mapRx": mapRx,
        "mapRy": mapRy,
    }

    return cal, None
