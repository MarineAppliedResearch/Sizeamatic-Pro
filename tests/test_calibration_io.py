"""Tests for calibration_io.py's load_calibration_bundle.

Uses the real calibration fixture in misc/AprilCalibration1/ rather than a
synthetic one, since this function is specifically about the NPZ
loading/validation mechanics (missing files, bad resolution) rather than
the calibration math itself — there's no "known correct answer" needed
here, just "does it load, and does it reject what it should reject."
"""

import calibration_io

APRIL_CALIBRATION_DIR = "misc/AprilCalibration1"


def test_load_calibration_bundle_success():
    """A valid calibration folder should load successfully with no error,
    and the returned dict should carry the calibrated resolution."""
    cal, err = calibration_io.load_calibration_bundle(APRIL_CALIBRATION_DIR)
    assert err is None
    assert cal is not None
    assert cal["w"] == 1920
    assert cal["h"] == 1080
    assert "PL" in cal and "PR" in cal


def test_load_calibration_bundle_reports_missing_files(tmp_path):
    """An empty folder should report which specific files are missing,
    rather than failing generically."""
    cal, err = calibration_io.load_calibration_bundle(str(tmp_path))
    assert cal is None
    assert "calibration_intrinsics.npz" in err


def test_load_calibration_bundle_rejects_left_resolution_mismatch():
    """A left video with a different resolution than the calibration
    should be rejected with a message naming the left video specifically."""
    wrong_meta = {"width": 640, "height": 480}
    cal, err = calibration_io.load_calibration_bundle(
        APRIL_CALIBRATION_DIR, meta_left=wrong_meta
    )
    assert cal is None
    assert "LEFT" in err


def test_load_calibration_bundle_rejects_right_resolution_mismatch():
    """A right video with a different resolution than the calibration
    should be rejected with a message naming the right video specifically."""
    wrong_meta = {"width": 640, "height": 480}
    cal, err = calibration_io.load_calibration_bundle(
        APRIL_CALIBRATION_DIR, meta_right=wrong_meta
    )
    assert cal is None
    assert "RIGHT" in err


def test_load_calibration_bundle_accepts_matching_resolution():
    """Video metadata matching the calibration's resolution should not be
    rejected."""
    matching_meta = {"width": 1920, "height": 1080}
    cal, err = calibration_io.load_calibration_bundle(
        APRIL_CALIBRATION_DIR, meta_left=matching_meta, meta_right=matching_meta
    )
    assert err is None
    assert cal is not None
