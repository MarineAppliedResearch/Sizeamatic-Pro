"""Tests for tutorial_fixtures.py's synthetic video+calibration generator.

Verifies the generated fixtures are genuinely loadable by this app's
existing loaders, not just "some files exist" - the whole point of this
module is producing real, workable tutorial content, not a mock of one.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

import calibration_io
import tutorial_fixtures


def test_generate_tutorial_calibration_produces_a_loadable_bundle(tmp_path):
    """The generated calibration folder should load cleanly through the
    real `calibration_io.load_calibration_bundle` - the exact function a
    real user's Calibration > Load Calibration… goes through."""

    folder = str(tmp_path / "calibration")
    tutorial_fixtures.generate_tutorial_calibration(folder)

    cal, error = calibration_io.load_calibration_bundle(folder)

    assert error is None
    assert cal is not None
    assert cal["w"] == tutorial_fixtures.TUTORIAL_VIDEO_WIDTH
    assert cal["h"] == tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT


def test_generate_tutorial_calibration_matches_a_given_video_resolution(tmp_path):
    """Loading the generated calibration alongside video metadata at the
    matching resolution shouldn't trip the existing resolution-mismatch
    check - confirms the fixture's own declared width/height genuinely
    line up with the videos it also generates."""

    folder = str(tmp_path / "calibration")
    tutorial_fixtures.generate_tutorial_calibration(folder)

    meta = {"width": tutorial_fixtures.TUTORIAL_VIDEO_WIDTH, "height": tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT}
    cal, error = calibration_io.load_calibration_bundle(folder, meta, meta)

    assert error is None
    assert cal is not None


def test_generate_tutorial_calibration_maps_crop_in_toward_center(tmp_path):
    """The rectification maps should crop in slightly toward the image
    center (not a plain identity grid) - see this module's "Design
    notes" for why: a pure identity grid would make "Show Rectified"
    a visual no-op, unlike a real calibration's rectified view, which
    always crops in a little after undistorting."""

    folder = str(tmp_path / "calibration")
    tutorial_fixtures.generate_tutorial_calibration(folder)
    cal, _error = calibration_io.load_calibration_bundle(folder)

    w, h = tutorial_fixtures.TUTORIAL_VIDEO_WIDTH, tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT
    cx, cy = w / 2.0, h / 2.0

    # The corner should sample a source point pulled toward center by
    # exactly TUTORIAL_RECTIFICATION_ZOOM, not the corner itself.
    expected_x = cx + (0.0 - cx) * tutorial_fixtures.TUTORIAL_RECTIFICATION_ZOOM
    expected_y = cy + (0.0 - cy) * tutorial_fixtures.TUTORIAL_RECTIFICATION_ZOOM
    assert cal["mapLx"][0, 0] == pytest.approx(expected_x, abs=0.5)
    assert cal["mapLy"][0, 0] == pytest.approx(expected_y, abs=0.5)

    # The exact center pixel is unaffected by a scale-from-center crop.
    assert cal["mapLx"][int(cy), int(cx)] == pytest.approx(cx, abs=0.5)
    assert cal["mapLy"][int(cy), int(cx)] == pytest.approx(cy, abs=0.5)

    frame = np.random.randint(0, 255, (h, w, 3)).astype("uint8")
    remapped = cv2.remap(frame, cal["mapLx"], cal["mapLy"], interpolation=cv2.INTER_LINEAR)
    assert not (remapped == frame).all()


def test_generate_tutorial_videos_produces_playable_correctly_shaped_videos(tmp_path):
    """The generated left/right videos should actually open, report the
    expected resolution/fps/frame count, and decode real frames - not
    just exist as files on disk."""

    left_path = str(tmp_path / "left.mp4")
    right_path = str(tmp_path / "right.mp4")
    tutorial_fixtures.generate_tutorial_videos(left_path, right_path)

    expected_frame_count = int(tutorial_fixtures.TUTORIAL_VIDEO_DURATION_SECONDS * tutorial_fixtures.TUTORIAL_VIDEO_FPS)

    for path in (left_path, right_path):
        cap = cv2.VideoCapture(path)
        try:
            assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == tutorial_fixtures.TUTORIAL_VIDEO_WIDTH
            assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frame_count

            ok, frame = cap.read()
            assert ok
            assert frame.shape == (tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT, tutorial_fixtures.TUTORIAL_VIDEO_WIDTH, 3)
        finally:
            cap.release()


def test_generate_tutorial_fixtures_returns_paths_that_all_exist(tmp_path, monkeypatch):
    """The single entry point should hand back paths to real files it
    just created, in a fresh directory each call."""

    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix=None: str(tmp_path / "run"))
    os.makedirs(str(tmp_path / "run"), exist_ok=True)

    paths = tutorial_fixtures.generate_tutorial_fixtures()

    assert os.path.isfile(paths["left_video_path"])
    assert os.path.isfile(paths["right_video_path"])
    assert os.path.isdir(paths["calibration_folder"])
    assert os.path.isfile(os.path.join(paths["calibration_folder"], "calibration_intrinsics.npz"))


def test_generate_tutorial_fixtures_video_and_calibration_agree_on_resolution():
    """An end-to-end sanity check with no mocking: generate real
    fixtures in a real temp directory and confirm the calibration
    successfully validates against the videos' own real metadata -
    exactly what `main.py`'s `_load_calibration_from_folder` checks
    against already-loaded video metadata."""

    paths = tutorial_fixtures.generate_tutorial_fixtures()

    cap = cv2.VideoCapture(paths["left_video_path"])
    try:
        meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()

    cal, error = calibration_io.load_calibration_bundle(paths["calibration_folder"], meta, meta)

    assert error is None
    assert cal is not None
