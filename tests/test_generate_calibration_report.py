"""Tests for generate_calibration_report.py's pure calibration-quality math.

These functions take plain numbers/arrays and have no `app` dependency,
so they're tested directly against hand-computed expected values.
"""

import math

import numpy as np
import pytest

import generate_calibration_report as gcr


def test_clamp():
    """clamp() should pass values through unchanged inside the range, and
    saturate to the bound when outside it."""
    assert gcr.clamp(5, 0, 10) == 5
    assert gcr.clamp(-5, 0, 10) == 0
    assert gcr.clamp(15, 0, 10) == 10


def test_rad_to_deg():
    """rad_to_deg() should convert pi radians to 180 degrees."""
    assert gcr.rad_to_deg(math.pi) == pytest.approx(180.0)
    assert gcr.rad_to_deg(0.0) == pytest.approx(0.0)


def test_compute_fov_degrees_known_formula():
    """compute_fov_degrees() should match the pinhole FOV formula directly."""
    # FOV = 2*atan((size/2)/f)
    fov = gcr.compute_fov_degrees(f_pixels=800.0, size_pixels=640.0)
    expected = math.degrees(2.0 * math.atan(320.0 / 800.0))
    assert fov == pytest.approx(expected)


def test_compute_fov_degrees_rejects_nonpositive_focal_length():
    """A zero or negative focal length has no valid FOV, so the function
    should return NaN rather than raising or dividing by zero."""
    assert math.isnan(gcr.compute_fov_degrees(f_pixels=0.0, size_pixels=640.0))
    assert math.isnan(gcr.compute_fov_degrees(f_pixels=-1.0, size_pixels=640.0))


def test_compute_diag_fov_degrees_known_formula():
    """compute_diag_fov_degrees() should match the pinhole diagonal-FOV
    formula directly."""
    fov = gcr.compute_diag_fov_degrees(f_pixels=800.0, width_pixels=640.0, height_pixels=480.0)
    half_diag = math.sqrt((640.0 / 2.0) ** 2 + (480.0 / 2.0) ** 2)
    expected = math.degrees(2.0 * math.atan(half_diag / 800.0))
    assert fov == pytest.approx(expected)


def test_compute_rotation_angle_degrees_identity_is_zero():
    """An identity rotation matrix (no relative rotation between cameras)
    should report a 0 degree angle."""
    identity = np.eye(3)
    assert gcr.compute_rotation_angle_degrees(identity) == pytest.approx(0.0, abs=1e-6)


def test_compute_rotation_angle_degrees_known_90_degree_rotation():
    """A known 90 degree rotation matrix should report exactly 90 degrees."""
    # 90 degree rotation about the Z axis.
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert gcr.compute_rotation_angle_degrees(R) == pytest.approx(90.0, abs=1e-6)


def test_compute_baseline_units():
    """compute_baseline_units() should return the Euclidean norm of the
    translation vector."""
    T = np.array([100.0, 0.0, 0.0])
    assert gcr.compute_baseline_units(T) == pytest.approx(100.0)

    T2 = np.array([3.0, 4.0, 0.0])
    assert gcr.compute_baseline_units(T2) == pytest.approx(5.0)


def test_units_to_meters_and_back():
    """units_to_meters() and meters_to_units() should be inverses of each
    other for a given mm-per-unit scale."""
    mm_per_unit = 1.0
    meters = gcr.units_to_meters(1000.0, mm_per_unit)
    assert meters == pytest.approx(1.0)

    units = gcr.meters_to_units(1.0, mm_per_unit)
    assert units == pytest.approx(1000.0)


def test_compute_depth_error_curves_disparity_matches_pinhole_formula():
    """The disparity curve should match d = f*B/Z at every sampled depth."""
    depth_m = np.array([1.0, 2.0, 10.0])
    f_rect_px = 800.0
    baseline_units = 100.0
    mm_per_unit = 1.0
    sigma_disp_px = 0.25

    result = gcr.compute_depth_error_curves(
        depth_m, f_rect_px, baseline_units, mm_per_unit, sigma_disp_px
    )

    baseline_m = 0.1
    expected_disparity = (f_rect_px * baseline_m) / depth_m
    np.testing.assert_allclose(result["disparity_px"], expected_disparity)


def test_compute_length_error_curves_mm_per_px_matches_formula():
    """The mm-per-pixel curve should match Z/f (converted to mm) at every
    sampled depth, and the example length should pass through unchanged."""
    depth_m = np.array([1.0, 2.0])
    f_rect_px = 800.0

    result = gcr.compute_length_error_curves(
        depth_m, f_rect_px, sigma_len_px=1.0, example_length_mm=300.0
    )

    expected_mm_per_px = (depth_m / f_rect_px) * 1000.0
    np.testing.assert_allclose(result["mm_per_px"], expected_mm_per_px)
    assert result["example_length_mm"] == 300.0


def test_compute_depth_bands_reports_max_depth_within_threshold():
    """Given relative error that grows with depth, the reported max depth
    for a threshold should be the deepest sample that still satisfies it."""
    depth_m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Relative error grows with depth.
    rel_sigma = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    bands = gcr.compute_depth_bands(depth_m, rel_sigma, thresholds=[3.0])
    assert len(bands) == 1
    assert bands[0]["threshold_percent"] == 3.0
    # Depths 1,2,3 satisfy rel_sigma <= 3%; max qualifying depth is 3.0.
    assert bands[0]["max_depth_m"] == pytest.approx(3.0)


def test_compute_depth_bands_reports_none_when_no_depth_qualifies():
    """If no sampled depth satisfies a threshold, the band should report
    None rather than a misleading numeric value."""
    depth_m = np.array([1.0, 2.0])
    rel_sigma = np.array([0.10, 0.20])

    bands = gcr.compute_depth_bands(depth_m, rel_sigma, thresholds=[1.0])
    assert bands[0]["max_depth_m"] is None
    assert bands[0]["range_m"] == [None, None]


def test_compute_point_estimates_interpolates():
    """A requested distance between two sampled depths should be linearly
    interpolated from their curve values."""
    depth_m = np.array([1.0, 2.0, 3.0])
    disparity = np.array([100.0, 50.0, 33.3])
    sigma_z = np.array([0.01, 0.02, 0.03])
    rel = np.array([0.01, 0.01, 0.01])

    curve_pack = {
        "depth_m": depth_m,
        "disparity_px": disparity,
        "sigma_z_m": sigma_z,
        "rel_sigma": rel,
    }

    rows = gcr.compute_point_estimates([1.5], curve_pack)
    assert len(rows) == 1
    # Linear interpolation between depth 1 (disp 100) and depth 2 (disp 50).
    assert rows[0]["disparity_px"] == pytest.approx(75.0)


def test_compute_point_estimates_clamps_out_of_range_distances():
    """A requested distance beyond the sampled depth range should clamp
    to the nearest edge of that range rather than extrapolating."""
    depth_m = np.array([1.0, 2.0])
    disparity = np.array([100.0, 50.0])
    sigma_z = np.array([0.01, 0.02])
    rel = np.array([0.01, 0.01])

    curve_pack = {
        "depth_m": depth_m,
        "disparity_px": disparity,
        "sigma_z_m": sigma_z,
        "rel_sigma": rel,
    }

    rows = gcr.compute_point_estimates([100.0], curve_pack)
    assert rows[0]["depth_m"] == pytest.approx(2.0)


def test_parse_thresholds_csv():
    """parse_thresholds_csv() should parse comma-separated numbers,
    tolerating surrounding whitespace and a single value with no commas."""
    assert gcr.parse_thresholds_csv("1,3,10") == [1.0, 3.0, 10.0]
    assert gcr.parse_thresholds_csv(" 1 , 3 ,10 ") == [1.0, 3.0, 10.0]
    assert gcr.parse_thresholds_csv("5") == [5.0]
