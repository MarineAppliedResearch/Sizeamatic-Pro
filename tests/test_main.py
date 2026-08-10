"""Tests for main.py's SizeamaticProApp, for behavior that's easier to
verify against the real GUI than through a FakeApp stand-in.
"""

import os

import cv2
import numpy as np
import pytest

import project_io

LEFT_VIDEO = "examples/left_20260309_171631.mp4"
RIGHT_VIDEO = "examples/right_20260309_171631.mp4"
APRIL_CALIBRATION_DIR = "misc/AprilCalibration1"


def test_save_project_writes_current_app_state(sizeamatic_app, monkeypatch, tmp_path):
    """Save Project should write whatever the app's current left/right
    video paths, calibration folder, and resync offset are — no real
    video/calibration needs to be loaded to exercise the write path
    itself."""

    app = sizeamatic_app
    app.left_video_path = "left.mp4"
    app.right_video_path = "right.mp4"
    app.calibration_folder = "some/cal/folder"
    app.lock_offset_frames = 9

    save_path = str(tmp_path / "project.json")
    monkeypatch.setattr(
        "main.filedialog.asksaveasfilename", lambda **_kwargs: save_path
    )

    app.on_save_project()

    project, err = project_io.load_project(save_path)
    assert err is None
    assert project["left_video_path"] == "left.mp4"
    assert project["right_video_path"] == "right.mp4"
    assert project["calibration_folder"] == "some/cal/folder"
    assert project["lock_offset_frames"] == 9


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
    )
    assert err is None

    monkeypatch.setattr(
        "main.filedialog.askopenfilename", lambda **_kwargs: project_path
    )

    app.on_open_project()

    assert app.capL is not None
    assert app.capR is not None
    assert app.left_video_path == LEFT_VIDEO
    assert app.right_video_path == RIGHT_VIDEO
    assert app.cal is not None
    assert app.lock_offset_frames == 3
    assert app.offset_var.get() == 3


def test_offset_changed_updates_lock_offset_frames_without_video(sizeamatic_app):
    """Editing the resync offset Spinbox should update
    `self.lock_offset_frames` even with no video loaded — it should just
    skip the realignment step (`_both_videos_loaded()` is False) rather
    than crash trying to re-render a frame that doesn't exist."""

    app = sizeamatic_app
    app.offset_var.set(7)

    app.on_offset_changed()

    assert app.lock_offset_frames == 7


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_toggle_lock_syncs_offset_var_when_capturing_new_offset(sizeamatic_app):
    """Enabling Lock while the two timelines are scrubbed apart captures
    that gap as `self.lock_offset_frames` (pre-existing behavior) — the
    resync offset Spinbox (`self.offset_var`) should reflect that
    just-captured value immediately, not keep showing whatever it
    displayed before (ROADMAP.md Phase 7's resync control)."""

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    app.capR, app.metaR = app._open_video_capture(RIGHT_VIDEO)
    app._update_slider_ranges()

    app.lock_lr.set(False)
    app.left_frame_index.set(10)
    app.right_frame_index.set(13)
    app.offset_var.set(0)

    app.lock_lr.set(True)
    app.on_toggle_lock()

    assert app.lock_offset_frames == 3
    assert app.offset_var.get() == 3


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

    app.lock_lr.set(True)
    app.left_frame_index.set(20)
    app.right_frame_index.set(20)
    app.lock_offset_frames = 0

    app.offset_var.set(5)
    app.on_offset_changed()

    assert app.lock_offset_frames == 5
    assert int(app.left_frame_index.get()) == 20
    assert int(app.right_frame_index.get()) == 25


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

    app.left_frame_index.set(50)
    app.right_frame_index.set(53)
    app.lock_lr.set(True)
    app.on_toggle_lock()  # captures offset = 53 - 50 = +3
    assert app.lock_offset_frames == 3

    app.speed_var.set("1x")
    app.is_playing = True

    for expected_left in range(51, 57):
        app._playback_tick()
        assert int(app.left_frame_index.get()) == expected_left
        assert int(app.right_frame_index.get()) == expected_left + 3

    app.is_playing = False
    if app.play_after_id is not None:
        app.root.after_cancel(app.play_after_id)
        app.play_after_id = None


def test_middle_drag_pans_without_redecoding(sizeamatic_app):
    """Middle-mouse-button drag should nudge the pane's pan offset by the
    on-screen distance moved, and redraw using the already-decoded
    current frame (ROADMAP.md Phase 7's pan feature) rather than
    re-reading from the video capture — there's no video loaded in this
    test at all, so a working pan that didn't crash confirms
    `_redisplay_current_frames` really doesn't touch `capL`/`capR`.
    """

    app = sizeamatic_app
    app.current_frameL = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

    from types import SimpleNamespace

    def event(x, y):
        return SimpleNamespace(x=x, y=y)

    app.on_pan_down("L", event(100, 50))
    assert app.pan_active is True
    assert app.pan_which == "L"

    app.on_pan_drag("L", event(130, 65))
    assert app.viewL["off_x"] == 30.0
    assert app.viewL["off_y"] == 15.0

    # A drag event from the other pane should be ignored.
    app.on_pan_drag("R", event(999, 999))
    assert app.viewR["off_x"] == 0.0

    app.on_pan_up("L", event(130, 65))
    assert app.pan_active is False
    assert app.pan_which is None


def test_display_bgr_on_canvas_renders_without_error(sizeamatic_app):
    """Regression test for the Phase 5 Pillow render-path fix.

    Displaying a frame should succeed and store a real PhotoImage. This
    doesn't assert anything about the old PNG/base64 path directly (it's
    gone), but exercises the same call site that used to be the
    "unacceptably slow" bottleneck — see ROADMAP.md Phase 5.
    """

    sizeamatic_app.metaL = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 10}
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

    sizeamatic_app._display_bgr_on_canvas(sizeamatic_app.video_overlay.left_canvas, frame, "L")

    assert sizeamatic_app.tkimg_left is not None


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
    ]

    for indices in access_patterns:
        for index in indices:
            actual = app._read_frame_at(app.capL, index)
            expected = ground_truth_frame(index)
            assert np.array_equal(actual, expected), f"Mismatch at frame {index}"

    ground_truth_cap.release()
