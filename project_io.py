"""Project file save/load for Sizeamatic Pro.

Lets a session's video paths, calibration folder, resync offset,
rectified-view state, and measurement log/last-recorded state be saved to
and reloaded from a small JSON manifest, instead of reselecting
everything through file dialogs (and losing all recorded measurements)
every time (ROADMAP.md Phase 7). Follows the same dialog-free, testable
pattern as `calibration_io.py`, pulled out of `main.py` so the read/write
logic can be tested without going through Tkinter file dialogs first.

Contents:
    - `save_project` — write a project manifest to a JSON file.
    - `load_project` — read and validate a project manifest from a JSON
      file.

Assumptions:
    - This module does not validate that the referenced video/calibration
      paths still exist or are still valid — `main.py`'s existing
      `_load_left_video_from_path`/`_load_right_video_from_path`/
      `_load_calibration_from_folder` already handle that (and its
      user-facing error messages) when `on_open_project` actually opens
      them.
"""

import json


def save_project(
    path,
    left_video_path,
    right_video_path,
    calibration_folder,
    lock_offset_frames,
    view_rectified,
    app_version,
    measurement_log_text,
    last_recorded_snapshot,
):
    """Save a project manifest to a JSON file.

    Args:
        path (str): Path to write the project file to.
        left_video_path (str | None): Path to the left video, or None if
            not loaded.
        right_video_path (str | None): Path to the right video, or None
            if not loaded.
        calibration_folder (str | None): Path to the calibration folder,
            or None if not loaded.
        lock_offset_frames (int): The current resync offset
            (`right_index - left_index`), saved so a resynced pair
            doesn't need re-correcting on every reopen.
        view_rectified (bool): Whether rectified view was enabled, saved
            so reopening the project puts the viewer back the way it was
            rather than always defaulting to raw view.
        app_version (str): The Sizeamatic Pro version string that created
            this project file (`main.py`'s `_get_app_version`) —
            informational only, not used for any compatibility check.
        measurement_log_text (str): The Measurement window's Log content
            verbatim (`measurement_window.py`'s `get_log_text`) — saved
            as exactly the text it is, since the Log is a plain editable
            `tk.Text` with no separate structured backing store to save
            instead.
        last_recorded_snapshot (dict | None): Enough state to restore the
            most recently *Recorded* measurement on reopen — keys
            "left_frame_index", "right_frame_index", "ptsL", "ptsR" — or
            None if nothing's been recorded yet this session.

    Returns:
        str | None: An error message if the file couldn't be written, or
        None on success. Unlike `load_project`, there's no separate value
        to return alongside success, so this returns a plain optional
        string rather than a `(value, error)` pair.
    """
    project = {
        "app_version": app_version,
        "left_video_path": left_video_path,
        "right_video_path": right_video_path,
        "calibration_folder": calibration_folder,
        "lock_offset_frames": int(lock_offset_frames),
        "view_rectified": bool(view_rectified),
        "measurement_log_text": measurement_log_text,
        "last_recorded_snapshot": last_recorded_snapshot,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2)
    except OSError as e:
        return f"Failed to save project: {e}"

    return None


def load_project(path):
    """Load and validate a project manifest from a JSON file.

    Args:
        path (str): Path to the project file to read.

    Returns:
        tuple[dict, None] | tuple[None, str]: `(project, None)` on
        success, where `project` has the keys "app_version",
        "left_video_path", "right_video_path", "calibration_folder",
        "lock_offset_frames", "view_rectified", "measurement_log_text",
        and "last_recorded_snapshot"; or `(None, error_message)` if the
        file can't be read, isn't valid JSON, or is missing required
        fields.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            project = json.load(f)
    except OSError as e:
        return None, f"Failed to open project file: {e}"
    except json.JSONDecodeError as e:
        return None, f"Project file is not valid JSON: {e}"

    required_keys = [
        "app_version",
        "left_video_path",
        "right_video_path",
        "calibration_folder",
        "lock_offset_frames",
        "view_rectified",
        "measurement_log_text",
        "last_recorded_snapshot",
    ]

    missing = [k for k in required_keys if k not in project]
    if missing:
        return None, f"Project file is missing required fields: {', '.join(missing)}"

    return project, None
