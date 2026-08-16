"""Tests for main.py's SizeamaticProApp, for behavior that's easier to
verify against the real GUI than through a FakeApp stand-in.
"""

import datetime
import os

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

import main
import measurement_window
import project_io
import recent_projects

LEFT_VIDEO = "examples/left_20260309_171631.mp4"
RIGHT_VIDEO = "examples/right_20260309_171631.mp4"
APRIL_CALIBRATION_DIR = "misc/AprilCalibration1"


def test_save_project_writes_current_app_state(sizeamatic_app, monkeypatch, tmp_path):
    """Save Project should write whatever the app's current left/right
    video paths, calibration folder, resync offset, rectified-view
    state, Measurement Log, and last-recorded snapshot are — no real
    video/calibration needs to be loaded to exercise the write path
    itself. The Log/snapshot are populated by directly driving Record,
    rather than a full triangulation setup, since that's already covered
    elsewhere (test_measurement_chain_produces_point_segment_and_total_rows)."""

    app = sizeamatic_app
    app.left_video_path = "left.mp4"
    app.right_video_path = "right.mp4"
    app.calibration_folder = "some/cal/folder"
    app.lock_offset_frames = 9
    app.view_rectified.set(True)
    app.left_frame_index = 7
    app.ptsL = [(1.0, 2.0)]
    app.ptsR = [(3.0, 4.0)]

    fake_row = ("left.mp4", "7", "00:00:00.233", "", "", "Point", "0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "7.5", "8.0", "9.0")
    app.measurement_window.update_window([fake_row], None)
    app.measurement_window.record_current_measurement()

    save_path = str(tmp_path / "project.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))

    app.on_save_project()

    project, err = project_io.load_project(save_path)
    assert err is None
    assert project["left_video_path"] == "left.mp4"
    assert project["right_video_path"] == "right.mp4"
    assert project["calibration_folder"] == "some/cal/folder"
    assert project["lock_offset_frames"] == 9
    assert project["view_rectified"] is True
    assert project["app_version"] == main.get_app_version()
    assert "left.mp4" in project["measurement_log_text"]
    assert project["last_recorded_snapshot"]["left_frame_index"] == 7
    assert project["last_recorded_snapshot"]["ptsL"] == [[1.0, 2.0]]
    assert project["last_recorded_snapshot"]["ptsR"] == [[3.0, 4.0]]


def test_rectified_indicator_reflects_view_rectified_state(sizeamatic_app):
    """The toolbar's RECTIFIED/NOT RECTIFIED label should read
    not-rectified/off by default and flip to rectified/on once
    rectified view is actually toggled on (ROADMAP.md Phase 8's
    rectified/not-rectified indicator item) - measurements and clicked
    points are only real-world-accurate once calibration is loaded and
    rectified view is on. The label's color itself comes from a QSS
    rule keyed off its "state" property (see DARK_QSS in main.py), so
    this checks that property rather than a literal color."""

    app = sizeamatic_app

    assert app.rectified_indicator.text().strip() == "NOT RECTIFIED"
    assert app.rectified_indicator.property("state") == "not_rectified"

    # Turning rectified view on without calibration loaded should be
    # refused (existing on_toggle_view_rectified validation), so the
    # indicator should stay showing NOT RECTIFIED.
    app.on_toggle_view_rectified(True)
    assert app.view_rectified.get() is False
    assert app.rectified_indicator.text().strip() == "NOT RECTIFIED"

    # With calibration loaded, turning it on should actually take effect.
    app.cal = {"w": 640, "h": 480}
    app.on_toggle_view_rectified(True)
    assert app.view_rectified.get() is True
    assert app.rectified_indicator.text().strip() == "RECTIFIED"
    assert app.rectified_indicator.property("state") == "rectified"


def test_on_save_project_records_it_in_recent_projects(sizeamatic_app, monkeypatch, tmp_path):
    """Saving a project should add it to the persistent recent-projects
    list, so it shows up in the File > Recent Projects submenu next
    time without needing a file dialog (ROADMAP.md Phase 8)."""

    app = sizeamatic_app
    save_path = str(tmp_path / "project.json")
    recent_path = str(tmp_path / "recent_projects.json")

    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: recent_path)

    app.on_save_project()

    assert recent_projects.load_recent_projects(recent_path) == [save_path]


def test_on_open_project_records_it_in_recent_projects(sizeamatic_app, monkeypatch, tmp_path):
    """Opening a project should also add it to the recent-projects list -
    not just Save Project - so reopening the same file later doesn't
    require re-navigating to it manually. Uses a project with no
    video/calibration paths set, so the load path doesn't need the real
    example assets to exercise the recording behavior."""

    app = sizeamatic_app
    open_path = str(tmp_path / "project.json")
    recent_path = str(tmp_path / "recent_projects.json")

    err = project_io.save_project(
        open_path,
        left_video_path=None,
        right_video_path=None,
        calibration_folder=None,
        lock_offset_frames=0,
        view_rectified=False,
        app_version="0.1.0",
        measurement_log_text="",
        last_recorded_snapshot=None,
        real_time_anchor_frame=None,
        real_time_anchor_iso=None,
        perform_calibration_capture_folder=None,
    )
    assert err is None

    monkeypatch.setattr("main.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (open_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: recent_path)

    app.on_open_project()

    assert recent_projects.load_recent_projects(recent_path) == [open_path]


def test_refresh_recent_projects_menu_lists_entries_and_a_placeholder_when_empty(
    sizeamatic_app, monkeypatch, tmp_path
):
    """The File > Recent Projects submenu should show a disabled
    placeholder when nothing's been saved/opened yet, and one entry per
    recorded project (most-recent first) once something has."""

    app = sizeamatic_app
    recent_path = str(tmp_path / "recent_projects.json")
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: recent_path)

    app._refresh_recent_projects_menu()
    actions = app.recent_projects_menu.actions()
    assert len(actions) == 1
    assert actions[0].text() == "(none yet)"
    assert actions[0].isEnabled() is False

    project_a = str(tmp_path / "a.json")
    project_b = str(tmp_path / "b.json")
    open(project_a, "w").close()
    open(project_b, "w").close()
    recent_projects.add_recent_project(project_a, recent_path)
    recent_projects.add_recent_project(project_b, recent_path)

    app._refresh_recent_projects_menu()
    actions = app.recent_projects_menu.actions()
    assert len(actions) == 2
    assert actions[0].text() == app._short_path(project_b, max_len=60)
    assert actions[1].text() == app._short_path(project_a, max_len=60)


def test_on_open_recent_project_delegates_to_open_project_from_path(sizeamatic_app, monkeypatch):
    """Choosing a Recent Projects submenu entry should go through the
    exact same load/restore logic as Open Project's file dialog, just
    skipping the dialog itself."""

    app = sizeamatic_app
    calls = []
    monkeypatch.setattr(app, "_open_project_from_path", lambda p: calls.append(p))

    app.on_open_recent_project("some/project.json")

    assert calls == ["some/project.json"]


def test_saving_a_project_never_touches_the_real_appdata_recent_projects_file(
    sizeamatic_app, monkeypatch, tmp_path
):
    """Regression test for a real bug: `on_save_project`/`on_open_project`
    used to silently read and write the actual per-user
    `%APPDATA%\\SizeamaticPro\\recent_projects.json` whenever a test
    exercised them without its own explicit monkeypatch for
    `recent_projects.get_recent_projects_path` - clobbering the project
    owner's real Recent Projects list every time the test suite ran
    (discovered when the project owner reported their real recent
    projects had been replaced by pytest tmp-path entries). conftest.py's
    autouse `_isolate_recent_projects_file` fixture now redirects every
    test automatically - this test deliberately adds no monkeypatch of
    its own for that path, to prove the protection holds without it."""

    app = sizeamatic_app
    save_path = str(tmp_path / "project.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))

    real_path = os.path.join(os.environ.get("APPDATA", ""), "SizeamaticPro", "recent_projects.json")
    mtime_before = os.path.getmtime(real_path) if os.path.isfile(real_path) else None

    app.on_save_project()

    mtime_after = os.path.getmtime(real_path) if os.path.isfile(real_path) else None
    assert mtime_after == mtime_before


def test_app_window_title_reflects_no_project_then_a_saved_one(sizeamatic_app, monkeypatch, tmp_path):
    """The main window's title should start as plain "Sizeamatic Pro
    vX.Y.Z" and switch to "Sizeamatic Pro vX.Y.Z - <project name>" once
    a project has been saved this session (ROADMAP.md Phase 8's
    window-title item, extended in Phase 9 to include the version) -
    the project name is the file's base name, without its directory or
    ".json" extension."""

    app = sizeamatic_app
    base_title = f"Sizeamatic Pro v{main.get_app_version()}"
    assert app.windowTitle() == base_title

    save_path = str(tmp_path / "MySurveyDive.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: str(tmp_path / "recent.json"))

    app.on_save_project()

    assert app.current_project_name == "MySurveyDive"
    assert app.windowTitle() == f"{base_title} - MySurveyDive"


def test_app_window_title_updates_on_open_project_too(sizeamatic_app, monkeypatch, tmp_path):
    """Opening a project should update the window title the same way
    saving one does, not just on Save Project."""

    app = sizeamatic_app
    open_path = str(tmp_path / "ReefTransect3.json")
    err = project_io.save_project(
        open_path,
        left_video_path=None,
        right_video_path=None,
        calibration_folder=None,
        lock_offset_frames=0,
        view_rectified=False,
        app_version="0.1.0",
        measurement_log_text="",
        last_recorded_snapshot=None,
        real_time_anchor_frame=None,
        real_time_anchor_iso=None,
        perform_calibration_capture_folder=None,
    )
    assert err is None

    monkeypatch.setattr("main.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (open_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: str(tmp_path / "recent.json"))

    app.on_open_project()

    assert app.current_project_name == "ReefTransect3"
    assert app.windowTitle() == f"Sizeamatic Pro v{main.get_app_version()} - ReefTransect3"


def test_already_open_measurement_window_retitles_when_project_saved(sizeamatic_app, monkeypatch, tmp_path):
    """A Measurement window opened *before* a project is saved should
    still pick up the project name in its title afterward - not just
    windows opened for the first time after the project is loaded."""

    app = sizeamatic_app
    app.measurement_window.ensure_window()
    base_title = f"Sizeamatic Pro v{main.get_app_version()}"
    assert app.measurement_window.win.windowTitle() == base_title

    save_path = str(tmp_path / "MySurveyDive.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: str(tmp_path / "recent.json"))

    app.on_save_project()

    assert app.measurement_window.win.windowTitle() == f"{base_title} - MySurveyDive"

    app.measurement_window._on_close()


def test_measurement_window_opened_after_project_loaded_shows_project_title(
    sizeamatic_app, monkeypatch, tmp_path
):
    """A Measurement window opened for the first time *after* a project
    is already loaded should get the project-aware title immediately,
    not the plain default."""

    app = sizeamatic_app

    save_path = str(tmp_path / "MySurveyDive.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: str(tmp_path / "recent.json"))
    app.on_save_project()

    app.measurement_window.ensure_window()
    assert app.measurement_window.win.windowTitle() == f"Sizeamatic Pro v{main.get_app_version()} - MySurveyDive"

    app.measurement_window._on_close()


def test_calibration_summary_window_also_shows_project_title(sizeamatic_app, monkeypatch, tmp_path):
    """The Calibration Summary window should follow the same
    project-aware title as the main window and the Measurement window -
    "app name and then - project name" applies to every window, not
    just the main one."""

    app = sizeamatic_app

    save_path = str(tmp_path / "MySurveyDive.json")
    monkeypatch.setattr("main.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))
    monkeypatch.setattr("recent_projects.get_recent_projects_path", lambda: str(tmp_path / "recent.json"))
    app.on_save_project()

    app.cal_summary_window.ensure_window()
    assert app.cal_summary_window.win.windowTitle() == f"Sizeamatic Pro v{main.get_app_version()} - MySurveyDive"

    app.cal_summary_window._on_close()


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_open_project_restores_video_calibration_and_offset(
    sizeamatic_app, monkeypatch, tmp_path
):
    """Opening a project file should reload the saved left/right videos
    and calibration folder, and restore the resync offset — the whole
    point of the project file (ROADMAP.md Phase 7) is skipping a manual
    reselect of all three through file dialogs."""

    app = sizeamatic_app

    project_path = str(tmp_path / "project.json")
    err = project_io.save_project(
        project_path,
        left_video_path=LEFT_VIDEO,
        right_video_path=RIGHT_VIDEO,
        calibration_folder=APRIL_CALIBRATION_DIR,
        lock_offset_frames=3,
        view_rectified=True,
        app_version="0.1.0",
        measurement_log_text="",
        last_recorded_snapshot=None,
        real_time_anchor_frame=None,
        real_time_anchor_iso=None,
        perform_calibration_capture_folder=None,
    )
    assert err is None

    monkeypatch.setattr("main.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (project_path, "")))

    app.on_open_project()

    assert app.capL is not None
    assert app.capR is not None
    assert app.left_video_path == LEFT_VIDEO
    assert app.right_video_path == RIGHT_VIDEO
    assert app.cal is not None
    assert app.lock_offset_frames == 3
    assert app.offset_spin.value() == 3
    # The calibration's resolution matches these real videos, so the saved
    # rectified-view state should have restored successfully rather than
    # being forced back off by on_toggle_view_rectified's validation.
    assert app.view_rectified.get() is True


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_open_project_restores_last_recorded_frame_points_and_log(
    sizeamatic_app, monkeypatch, tmp_path
):
    """Opening a project file should jump the timelines back to "the very
    last place that was recorded", restore the exact clicked points from
    that moment (so the measurement is visibly back on screen, not just
    a historical number in the Log), and restore the full Log text."""

    app = sizeamatic_app

    project_path = str(tmp_path / "project.json")
    snapshot = {
        "left_frame_index": 40,
        "right_frame_index": 43,
        "ptsL": [[100.0, 50.0]],
        "ptsR": [[95.0, 50.0]],
    }
    header_line = "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)
    recorded_row = ("left.mp4", "40", "00:00:01.333", "2026-08-12 14:32:05.000", "1", "Point", "0", "100.0", "50.0", "0.0", "0.0", "5.0", "0.0", "0.5", "0.3", "1.0", "2.0")
    log_text = header_line + "\n" + "\t".join(recorded_row)

    err = project_io.save_project(
        project_path,
        left_video_path=LEFT_VIDEO,
        right_video_path=RIGHT_VIDEO,
        calibration_folder=APRIL_CALIBRATION_DIR,
        lock_offset_frames=3,
        view_rectified=False,
        app_version="0.1.0",
        measurement_log_text=log_text,
        last_recorded_snapshot=snapshot,
        real_time_anchor_frame=40,
        real_time_anchor_iso="2026-08-12T14:32:05",
        perform_calibration_capture_folder=None,
    )
    assert err is None

    monkeypatch.setattr("main.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (project_path, "")))

    app.on_open_project()

    assert int(app.left_frame_index) == 40
    assert int(app.right_frame_index) == 43
    assert int(app.left_slider.value()) == 40
    assert int(app.right_slider.value()) == 43
    assert app.ptsL == [(100.0, 50.0)]
    assert app.ptsR == [(95.0, 50.0)]
    assert app.measurement_window.get_log_text() == log_text
    # A later Record click should continue numbering after the restored log's
    # highest measurement ID (1 here), not restart at 1 and collide with it.
    assert app.measurement_window._next_measurement_id == 2
    # The real-time anchor should also be restored, and the shared readout
    # should reflect it immediately (frame 40 == the anchor frame itself, so
    # actual time should equal the anchor exactly, no elapsed-time math).
    assert app.real_time_anchor_frame == 40
    assert app.real_time_anchor_dt == datetime.datetime(2026, 8, 12, 14, 32, 5)
    assert "2026-08-12 14:32:05" in app.time_readout_label.text()
    # The anchor entry boxes should also reflect the restored anchor, not
    # whatever they defaulted to at app startup.
    assert int(app.real_time_year_edit.text()) == 2026
    assert int(app.real_time_month_edit.text()) == 8
    assert int(app.real_time_day_edit.text()) == 12
    assert int(app.real_time_hour_edit.text()) == 14
    assert int(app.real_time_minute_edit.text()) == 32
    assert int(app.real_time_second_edit.text()) == 5


def _set_real_time_boxes(app, dt):
    """Set the six real-time anchor entry boxes to match a datetime.

    Args:
        app (main.SizeamaticProApp): The app under test.
        dt (datetime.datetime): The date/time to fill the boxes with.

    Returns:
        None
    """
    app.real_time_year_edit.setText(f"{dt.year:04d}")
    app.real_time_month_edit.setText(f"{dt.month:02d}")
    app.real_time_day_edit.setText(f"{dt.day:02d}")
    app.real_time_hour_edit.setText(f"{dt.hour:02d}")
    app.real_time_minute_edit.setText(f"{dt.minute:02d}")
    app.real_time_second_edit.setText(f"{dt.second:02d}")


def test_format_actual_time_returns_not_set_without_anchor(sizeamatic_app):
    """With no real-time anchor set yet, the calculated actual time
    should read as "(not set)" rather than raising or showing a bogus
    value."""

    app = sizeamatic_app
    assert app._format_actual_time(0) == "(not set)"


def test_on_real_time_entered_requires_video_loaded(sizeamatic_app):
    """Typing a valid date+time before any video is loaded should be
    rejected with a status message, not crash trying to read fps off a
    None metaL."""

    app = sizeamatic_app
    _set_real_time_boxes(app, datetime.datetime(2026, 8, 12, 14, 32, 5))

    app.on_real_time_entered()

    assert app.real_time_anchor_frame is None
    assert app.real_time_anchor_dt is None


def test_on_real_time_entered_rejects_invalid_date(sizeamatic_app, monkeypatch):
    """A day/month combination that doesn't form a real date (e.g. day 31
    in a 30-day month) should show an error dialog and leave any existing
    anchor untouched, not raise."""

    app = sizeamatic_app
    errors = []
    monkeypatch.setattr(
        "main.QMessageBox.critical", staticmethod(lambda parent, title, msg: errors.append((title, msg)))
    )

    app.real_time_year_edit.setText("2026")
    app.real_time_month_edit.setText("4")  # April has 30 days
    app.real_time_day_edit.setText("31")
    app.real_time_hour_edit.setText("0")
    app.real_time_minute_edit.setText("0")
    app.real_time_second_edit.setText("0")
    app.on_real_time_entered()

    assert len(errors) == 1
    assert app.real_time_anchor_frame is None
    assert app.real_time_anchor_dt is None


def test_on_real_time_entered_sets_anchor_and_updates_readout(sizeamatic_app):
    """Setting the anchor entry boxes with a video loaded should anchor to
    the current left frame and immediately refresh the shared readout."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 25.0, "frame_count": 100}
    app.left_frame_index = 10

    _set_real_time_boxes(app, datetime.datetime(2026, 8, 12, 14, 32, 5))
    app.on_real_time_entered()

    assert app.real_time_anchor_frame == 10
    assert app.real_time_anchor_dt == datetime.datetime(2026, 8, 12, 14, 32, 5)

    readout = app.time_readout_label.text()
    assert "Frame: 10" in readout
    assert "2026-08-12 14:32:05" in readout


def test_format_actual_time_calculates_forward_and_backward_from_anchor(sizeamatic_app):
    """Once anchored, the actual time at other frames should be
    calculated correctly both forward and backward from the anchor
    frame, using the left video's fps."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 25.0, "frame_count": 100}
    app.left_frame_index = 10
    _set_real_time_boxes(app, datetime.datetime(2026, 8, 12, 14, 32, 5))
    app.on_real_time_entered()

    # 25 frames forward at 25fps = exactly 1.0 second (25 whole frames) later,
    # landing back on frame-in-second 0.
    assert app._format_actual_time(10 + 25) == "2026-08-12 14:32:06:00"
    # 25 frames backward = exactly 1.0 second earlier, also frame-in-second 0.
    assert app._format_actual_time(10 - 25) == "2026-08-12 14:32:04:00"
    # Exactly at the anchor frame itself.
    assert app._format_actual_time(10) == "2026-08-12 14:32:05:00"
    # A partial second forward: 3 frames in, not a fraction of a second.
    assert app._format_actual_time(10 + 3) == "2026-08-12 14:32:05:03"
    # A partial second backward: 3 frames before the anchor lands on the
    # *previous* second's frame 22 (25 - 3), not frame "-3".
    assert app._format_actual_time(10 - 3) == "2026-08-12 14:32:04:22"


def test_offset_changed_updates_lock_offset_frames_without_video(sizeamatic_app):
    """Editing the resync offset spin box should update
    `self.lock_offset_frames` even with no video loaded — it should just
    skip the realignment step (`_both_videos_loaded()` is False) rather
    than crash trying to re-render a frame that doesn't exist. Setting
    the QSpinBox's value fires `on_offset_changed` itself, via the same
    `valueChanged` signal connection the real UI uses - no separate
    manual handler call needed, unlike the original Tkinter version
    (whose Spinbox `command=` never fired on a programmatic `.set()`)."""

    app = sizeamatic_app
    app.offset_spin.setValue(7)

    assert app.lock_offset_frames == 7


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_toggle_lock_syncs_offset_var_when_capturing_new_offset(sizeamatic_app):
    """Enabling Lock while the two timelines are scrubbed apart captures
    that gap as `self.lock_offset_frames` (pre-existing behavior) — the
    resync offset spin box should reflect that just-captured value
    immediately, not keep showing whatever it displayed before
    (ROADMAP.md Phase 7's resync control)."""

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    app.capR, app.metaR = app._open_video_capture(RIGHT_VIDEO)
    app._update_slider_ranges()

    app.on_toggle_lock(False)
    app.left_frame_index = 10
    app.right_frame_index = 13
    app.offset_spin.setValue(0)

    app.on_toggle_lock(True)

    assert app.lock_offset_frames == 3
    assert app.offset_spin.value() == 3


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_offset_changed_realigns_right_timeline_when_locked(sizeamatic_app):
    """With Lock enabled, manually editing the resync offset should
    immediately move the right timeline to match the left timeline's
    current position plus the new offset — the whole point of exposing
    the offset as a directly-editable control (ROADMAP.md Phase 7)."""

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    app.capR, app.metaR = app._open_video_capture(RIGHT_VIDEO)
    app._update_slider_ranges()

    app.on_toggle_lock(True)
    app.left_frame_index = 20
    app.right_frame_index = 20
    app.lock_offset_frames = 0

    app.offset_spin.setValue(5)

    assert app.lock_offset_frames == 5
    assert int(app.left_frame_index) == 20
    assert int(app.right_frame_index) == 25


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_playback_advances_forward_with_nonzero_lock_offset(sizeamatic_app):
    """Regression test for a reported "pressing Play runs the video
    backward" bug, reproducible whenever Lock is on with a nonzero resync
    offset (FINDINGS.md #9).

    `_playback_tick`'s locked branch used to set both sliders directly
    without suppressing their `command` callbacks, which chained into
    `_jump_frames_locked_with_offset` twice per tick with contradictory
    targets ("L" driving, then "R" driving) and could net-decrease the
    left index every tick. Simulates several playback ticks and checks
    both timelines actually move forward, with the offset preserved.
    """

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    app.capR, app.metaR = app._open_video_capture(RIGHT_VIDEO)
    app._update_slider_ranges()

    app.left_frame_index = 50
    app.right_frame_index = 53
    app.on_toggle_lock(True)  # captures offset = 53 - 50 = +3
    assert app.lock_offset_frames == 3

    app.speed_combo.setCurrentText("1x")
    app.is_playing = True

    for expected_left in range(51, 57):
        app._playback_tick()
        assert int(app.left_frame_index) == expected_left
        assert int(app.right_frame_index) == expected_left + 3

    app.is_playing = False
    app.playback_timer.stop()


def test_measurement_rows_include_actual_time_once_anchored(
    sizeamatic_app, synthetic_cal, known_point_pixels
):
    """Once a real-time anchor is set, every measurement row should carry
    the calculated actual time in its own column - not just the elapsed
    "video time" - so a recorded measurement still means something once
    it's sitting in a spreadsheet, per the project owner's request."""

    app = sizeamatic_app
    app.view_rectified.set(True)
    app.cal = dict(synthetic_cal)
    app.ptsL = [(known_point_pixels["xL"], known_point_pixels["yL"])]
    app.ptsR = [(known_point_pixels["xR"], known_point_pixels["yR"])]

    app.left_video_path = "some/path/lefty_test.mp4"
    app.metaL = {"fps": 25.0}
    app.left_frame_index = 10
    _set_real_time_boxes(app, datetime.datetime(2026, 8, 12, 14, 32, 5))
    app.on_real_time_entered()

    # Move forward exactly 1 second (25 frames at 25fps) from the anchor.
    app.left_frame_index = 35
    app._update_measurement_status_stub()

    rows = app.measurement_window._last_rows
    assert len(rows) == 1
    assert rows[0][3] == "2026-08-12 14:32:06:00"


def test_measurement_chain_produces_point_segment_and_total_rows(
    sizeamatic_app, synthetic_cal, known_chain_pixels
):
    """A 3-point connected chain should produce 3 "Point" rows, 2
    "Segment" rows, and one "Total" row summing the chain's segment
    lengths — each row carrying the same video/frame/timestamp context.

    Regression/feature test for ROADMAP.md Phase 7's measurement output
    item: this is the first test to exercise more than 2 points through
    `_update_measurement_status_stub` at all (the point cap made it
    impossible before), and the first to check the Total row exists and
    is correct, since summing connected segments is new.
    """

    app = sizeamatic_app
    app.view_rectified.set(True)
    app.cal = dict(synthetic_cal)

    points = known_chain_pixels["points"]
    app.ptsL = [(p["xL"], p["yL"]) for p in points]
    app.ptsR = [(p["xR"], p["yR"]) for p in points]

    app.left_video_path = "some/path/lefty_test.mp4"
    app.metaL = {"fps": 25.0}
    app.left_frame_index = 125  # 125 / 25fps = exactly 5.0s

    app._update_measurement_status_stub()

    rows = app.measurement_window._last_rows
    types = [row[5] for row in rows]
    assert types == ["Point", "Point", "Point", "Segment", "Segment", "Total"]

    # Every row shares the same video/frame/timestamp context (columns 0-2),
    # carries no actual time (column 3, no real-time anchor set in this
    # test), and carries no measurement ID yet (column 4) - that's only
    # stamped in once actually Recorded, not for the live/current display.
    for row in rows:
        assert row[0] == "lefty_test.mp4"
        assert row[1] == "125"
        assert row[2] == "00:00:05:00"
        assert row[3] == ""
        assert row[4] == ""

    total_row = rows[-1]
    assert total_row[6] == ""  # no label on the Total row
    assert float(total_row[10]) == pytest.approx(known_chain_pixels["total_length_mm"], abs=0.05)


class _FakeMouseEvent:
    """Minimal stand-in for a PySide6 `QMouseEvent`, exposing only
    `.position()`/`.button()` - the only methods `VideoPane`'s mouse
    handlers actually call on a real one."""

    def __init__(self, x, y, button=Qt.MouseButton.MiddleButton):
        """Store the event's position and button.

        Args:
            x (float): Local X coordinate, in pane screen pixels.
            y (float): Local Y coordinate, in pane screen pixels.
            button (Qt.MouseButton): Which button this event is for.

        Returns:
            None
        """
        self._pos = QPointF(x, y)
        self._button = button

    def position(self):
        """Return this event's local position.

        Returns:
            QPointF: The stored position.
        """
        return self._pos

    def button(self):
        """Return this event's button.

        Returns:
            Qt.MouseButton: The stored button.
        """
        return self._button


def test_middle_drag_pans_without_redecoding(sizeamatic_app):
    """Middle-mouse-button drag should nudge the pane's pan offset by the
    on-screen distance moved, and redraw using the already-decoded
    current frame (ROADMAP.md Phase 7's pan feature) rather than
    re-reading from the video capture — there's no video loaded in this
    test at all, so a working pan that didn't crash confirms
    `redisplay_current_frames` really doesn't touch `capL`/`capR`. Each
    pane owns its own independent pan/zoom state (`self.view`), unlike
    the original's single app-level `viewL`/`viewR` dict pair, so a drag
    on one pane leaving the other's view untouched is now just a
    consequence of that per-pane ownership rather than something
    `on_pan_drag` had to check `which` for.
    """

    app = sizeamatic_app
    app.current_frameL = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

    app.pane_left.mousePressEvent(_FakeMouseEvent(100, 50))
    assert app.pane_left.pan_active is True

    app.pane_left.mouseMoveEvent(_FakeMouseEvent(130, 65))
    assert app.pane_left.view["off_x"] == 30.0
    assert app.pane_left.view["off_y"] == 15.0

    # A drag event from the other pane should be independent - its own
    # pan isn't active, so this is a no-op for it.
    app.pane_right.mouseMoveEvent(_FakeMouseEvent(999, 999))
    assert app.pane_right.view["off_x"] == 0.0

    app.pane_left.mouseReleaseEvent(_FakeMouseEvent(130, 65))
    assert app.pane_left.pan_active is False
    assert app.pane_left.pan_last_pos is None


def test_set_frame_renders_without_error(sizeamatic_app):
    """Regression test for the Phase 5 Pillow render-path fix, ported to
    this app's PySide6 pane.

    Displaying a frame should succeed and cache a real `QImage`. This
    doesn't assert anything about the old PNG/base64 path directly (it's
    long gone), but exercises the same "hand a freshly decoded frame to
    the display widget" call site that used to be the "unacceptably
    slow" bottleneck — see ROADMAP.md Phase 5.
    """

    sizeamatic_app.metaL = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 10}
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

    sizeamatic_app.pane_left.set_frame(frame)

    assert sizeamatic_app.pane_left.current_qimage is not None


@pytest.mark.skipif(
    not os.path.isfile(LEFT_VIDEO),
    reason="Real example video is gitignored/local-only, not present here.",
)
def test_read_frame_at_correct_for_sequential_and_random_access(sizeamatic_app):
    """Regression test for the seek-skip optimization in _read_frame_at
    (ROADMAP.md Phase 5).

    _read_frame_at skips the explicit seek when the capture's own
    reported position already matches the requested index, purely for
    speed. This test verifies it still returns the actually-correct frame
    under three access patterns: pure random access (forces a seek every
    time), pure sequential access (exercises the seek-skip path
    specifically), and a mix that jumps backward then resumes sequential
    — the exact shape of access pattern that would expose a naive "assume
    position is last-read-plus-one" optimization as wrong, which is
    exactly why this checks the capture's real position instead.
    """

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    assert app.capL is not None

    # A second, independent capture used only to establish ground-truth
    # frame content, so reading via it never disturbs app.capL's position.
    ground_truth_cap = cv2.VideoCapture(LEFT_VIDEO)

    def ground_truth_frame(index):
        ground_truth_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = ground_truth_cap.read()
        assert ok
        return frame

    access_patterns = [
        [20, 5, 21, 6, 100, 0],           # pure random access
        [30, 31, 32, 33, 34],             # pure sequential access
        [10, 11, 40, 41, 42],             # backward jump, then resume sequential
        [50, 52, 54, 56, 58],             # small forward skips (2x playback's step)
        [70, 74, 78, 82],                 # small forward skips (4x playback's step)
    ]

    for indices in access_patterns:
        for index in indices:
            actual = app._read_frame_at(app.capL, index)
            expected = ground_truth_frame(index)
            assert np.array_equal(actual, expected), f"Mismatch at frame {index}"

    ground_truth_cap.release()


class _SeekSpyCapture:
    """Thin wrapper around a real `cv2.VideoCapture` that records every
    `set(CAP_PROP_POS_FRAMES, ...)` call, forwarding everything else
    unchanged.

    `cv2.VideoCapture` is a C extension type - its methods can't be
    monkeypatched directly on an instance, so this stands in for it
    instead; `_read_frame_at` only ever calls `get`/`set`/`read`, all
    of which this forwards to the real capture.
    """

    def __init__(self, real_cap):
        self._real_cap = real_cap
        self.seek_calls = []

    def get(self, prop):
        return self._real_cap.get(prop)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seek_calls.append(value)
        return self._real_cap.set(prop, value)

    def read(self):
        return self._real_cap.read()


@pytest.mark.skipif(
    not os.path.isfile(LEFT_VIDEO),
    reason="Real example video is gitignored/local-only, not present here.",
)
def test_read_frame_at_skips_seeking_for_small_forward_gaps_only(sizeamatic_app):
    """Regression test: 2x/4x playback used to be *slower* than 1x,
    because requesting a small forward skip (e.g. +2 or +4 frames, the
    exact step 2x/4x speed produces every tick) fell through to a full
    `cap.set()` seek every single tick — neither "same index" nor
    "exactly one frame ahead" (the two cases the original seek-skip
    optimization actually covered). Profiling traced the reported "2x
    goes even slower than 1x" bug to seeking costing ~54-105ms/frame
    against ~3-10ms/frame to decode-and-discard the same gap.

    Verifies the fix by counting real `cap.set()` calls (via
    `_SeekSpyCapture`, not just checking the returned frame content —
    the prior test above already covers correctness): a small forward
    gap should decode-and-discard (no seek at all), while a gap larger
    than `MAX_SEQUENTIAL_SKIP_FRAMES` or a backward jump should still
    seek.
    """

    app = sizeamatic_app
    real_cap, app.metaL = app._open_video_capture(LEFT_VIDEO)
    assert real_cap is not None
    cap = _SeekSpyCapture(real_cap)

    # Establish a starting position with one real read.
    app._read_frame_at(cap, 20)
    cap.seek_calls.clear()

    # A gap of exactly MAX_SEQUENTIAL_SKIP_FRAMES (the inclusive boundary)
    # should decode-and-discard rather than seek. Computed from the
    # capture's own reported position, not assumed, so this doesn't
    # silently drift off-by-one if the read-and-discard path itself
    # changes how far the position advances.
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    app._read_frame_at(cap, pos + main.MAX_SEQUENTIAL_SKIP_FRAMES)
    assert cap.seek_calls == [], "gap exactly at the threshold should not seek"

    # A gap one frame past the threshold should fall back to a real seek.
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.seek_calls.clear()
    app._read_frame_at(cap, pos + main.MAX_SEQUENTIAL_SKIP_FRAMES + 1)
    assert len(cap.seek_calls) == 1, "gap beyond the threshold should seek"

    # A backward jump should always seek, regardless of distance.
    cap.seek_calls.clear()
    app._read_frame_at(cap, 5)
    assert len(cap.seek_calls) == 1, "backward jump should seek"

    real_cap.release()


def test_compute_playback_timing_uses_the_videos_real_fps():
    """Regression test: "1x" playback used to always tick at a fixed
    40ms, derived from an assumed ~25fps — correct only by coincidence
    for footage actually shot near 25fps, and visibly wrong (too slow
    or too fast) at any other real fps.

    `_compute_playback_timing` should derive the tick delay from the
    video's actual fps instead. 25fps is checked first specifically
    because it must reproduce the old hardcoded table's exact values —
    this is a behavior change for every other fps, not for existing
    25fps footage.
    """
    # Matches the old hardcoded SPEED_TABLE exactly at 25fps (the
    # assumption it used to hardcode for every video regardless of
    # actual fps).
    assert main._compute_playback_timing(25.0, "0.25x") == (1, 160)
    assert main._compute_playback_timing(25.0, "0.5x") == (1, 80)
    assert main._compute_playback_timing(25.0, "1x") == (1, 40)
    assert main._compute_playback_timing(25.0, "2x") == (2, 40)
    assert main._compute_playback_timing(25.0, "4x") == (4, 40)

    # 30fps footage should tick faster than the old hardcoded 40ms.
    assert main._compute_playback_timing(30.0, "1x") == (1, 33)
    assert main._compute_playback_timing(30.0, "2x") == (2, 33)

    # A video reporting a nonsensical fps (corrupt metadata) falls back
    # to DEFAULT_PLAYBACK_FPS rather than dividing by zero.
    assert main._compute_playback_timing(0.0, "1x") == (1, 40)
    assert main._compute_playback_timing(-1.0, "1x") == (1, 40)
