"""Tests for project_io.py's save_project/load_project.

Dialog-free, so these exercise the actual read/write mechanics directly
against tmp_path files rather than through Tkinter file dialogs — same
rationale as test_calibration_io.py.
"""

import os

import project_io


def test_save_then_load_project_round_trips(tmp_path):
    """Saving a project and loading it back should return exactly what
    was saved."""
    path = str(tmp_path / "project.json")

    snapshot = {"left_frame_index": 42, "right_frame_index": 45, "ptsL": [[1.0, 2.0]], "ptsR": [[3.0, 4.0]]}

    err = project_io.save_project(
        path,
        left_video_path="left.mp4",
        right_video_path="right.mp4",
        calibration_folder="misc/AprilCalibration1",
        lock_offset_frames=-4,
        view_rectified=True,
        app_version="0.1.0",
        measurement_log_text="Video\tFrame\n...",
        last_recorded_snapshot=snapshot,
    )
    assert err is None

    project, load_err = project_io.load_project(path)
    assert load_err is None
    assert project["left_video_path"] == "left.mp4"
    assert project["right_video_path"] == "right.mp4"
    assert project["calibration_folder"] == "misc/AprilCalibration1"
    assert project["lock_offset_frames"] == -4
    assert project["view_rectified"] is True
    assert project["app_version"] == "0.1.0"
    assert project["measurement_log_text"] == "Video\tFrame\n..."
    assert project["last_recorded_snapshot"] == snapshot


def test_save_project_allows_none_fields(tmp_path):
    """A project saved before anything is loaded yet (all None) should
    still save and load successfully — the field just carries no value
    rather than being omitted."""
    path = str(tmp_path / "project.json")

    err = project_io.save_project(
        path,
        left_video_path=None,
        right_video_path=None,
        calibration_folder=None,
        lock_offset_frames=0,
        view_rectified=False,
        app_version="0.1.0",
        measurement_log_text="",
        last_recorded_snapshot=None,
    )
    assert err is None

    project, load_err = project_io.load_project(path)
    assert load_err is None
    assert project["left_video_path"] is None
    assert project["calibration_folder"] is None


def test_load_project_rejects_invalid_json(tmp_path):
    """A file that isn't valid JSON should report a clear error rather
    than raising."""
    path = tmp_path / "project.json"
    path.write_text("not valid json {{{", encoding="utf-8")

    project, err = project_io.load_project(str(path))
    assert project is None
    assert "not valid JSON" in err


def test_load_project_rejects_missing_fields(tmp_path):
    """A JSON file missing required project fields should name what's
    missing rather than crashing with a KeyError later."""
    path = tmp_path / "project.json"
    path.write_text('{"left_video_path": "left.mp4"}', encoding="utf-8")

    project, err = project_io.load_project(str(path))
    assert project is None
    assert "right_video_path" in err
    assert "calibration_folder" in err
    assert "lock_offset_frames" in err
    assert "view_rectified" in err
    assert "app_version" in err
    assert "measurement_log_text" in err
    assert "last_recorded_snapshot" in err


def test_load_project_reports_missing_file():
    """Pointing at a project file that doesn't exist should report a
    clear error rather than raising."""
    project, err = project_io.load_project("this/path/does/not/exist.json")
    assert project is None
    assert err is not None
    assert not os.path.isfile("this/path/does/not/exist.json")
