"""Tests for stereo_matching.py's triangulation, reprojection, and
uncertainty-estimation math.

Uses the synthetic rectified stereo rig from conftest.py, which has a
known-correct expected triangulation result — real captured calibration
data has no such "known correct answer" to assert against.
"""

import math

import numpy as np
import pytest

import stereo_matching


def test_triangulate_from_pixels_recovers_known_point(synthetic_cal, known_point_pixels):
    """Projecting a 3D point through PL/PR and triangulating the resulting
    pixels back should recover the original point (a round-trip check)."""

    X, Y, Z, xL, yL, xR, yR = (
        known_point_pixels["X"],
        known_point_pixels["Y"],
        known_point_pixels["Z"],
        known_point_pixels["xL"],
        known_point_pixels["yL"],
        known_point_pixels["xR"],
        known_point_pixels["yR"],
    )

    app = type("App", (), {"cal": synthetic_cal})()
    result = stereo_matching.triangulate_from_pixels(app, xL, yL, xR, yR)

    assert result is not None
    rX, rY, rZ = result
    assert rX == pytest.approx(X, abs=1e-3)
    assert rY == pytest.approx(Y, abs=1e-3)
    assert rZ == pytest.approx(Z, abs=1e-3)


def test_triangulate_from_pixels_rejects_degenerate_w(synthetic_cal):
    """A homogeneous result with W too close to zero should return None
    rather than dividing by (near) zero."""

    app = type("App", (), {"cal": synthetic_cal})()

    # Project matching image-plane-at-infinity-style pixels that make the
    # triangulated W degenerate: identical left/right pixels at the
    # principal point produce a point at infinity along the optical axis.
    result = stereo_matching.triangulate_from_pixels(
        app, xL=1e12, yL=1e12, xR=1e12, yR=1e12
    )
    assert result is None


def test_triangulate_point_pair_success(make_fake_app, synthetic_cal, known_point_pixels):
    """A valid rectified point pair should triangulate successfully."""

    app = make_fake_app(
        view_rectified=_TrueVar(),
        cal=synthetic_cal,
        ptsL=[(known_point_pixels["xL"], known_point_pixels["yL"])],
        ptsR=[(known_point_pixels["xR"], known_point_pixels["yR"])],
    )

    point, err = stereo_matching.triangulate_point_pair(app, 0)
    assert err is None
    assert point[2] == pytest.approx(known_point_pixels["Z"], abs=1e-3)


def test_triangulate_point_pair_requires_rectified_view(make_fake_app, synthetic_cal):
    """Triangulation should refuse to run while showing raw (non-rectified)
    frames, since the projection matrices only apply to rectified pixels."""
    app = make_fake_app(
        view_rectified=_FalseVar(),
        cal=synthetic_cal,
        ptsL=[(0, 0)],
        ptsR=[(0, 0)],
    )
    point, err = stereo_matching.triangulate_point_pair(app, 0)
    assert point is None
    assert "rectified" in err.lower()


def test_triangulate_point_pair_requires_calibration(make_fake_app):
    """Triangulation should refuse to run with no calibration loaded."""
    app = make_fake_app(view_rectified=_TrueVar(), cal=None, ptsL=[(0, 0)], ptsR=[(0, 0)])
    point, err = stereo_matching.triangulate_point_pair(app, 0)
    assert point is None
    assert "calibration" in err.lower()


def test_triangulate_point_pair_requires_pl_pr(make_fake_app):
    """A calibration dict missing the rectified projection matrices
    should report that specific reason rather than crashing on a
    KeyError."""
    app = make_fake_app(view_rectified=_TrueVar(), cal={}, ptsL=[(0, 0)], ptsR=[(0, 0)])
    point, err = stereo_matching.triangulate_point_pair(app, 0)
    assert point is None
    assert "PL/PR" in err


def test_triangulate_point_pair_rejects_negative_index(make_fake_app, synthetic_cal):
    """A negative point index should be rejected before touching the
    point lists at all."""
    app = make_fake_app(view_rectified=_TrueVar(), cal=synthetic_cal, ptsL=[], ptsR=[])
    point, err = stereo_matching.triangulate_point_pair(app, -1)
    assert point is None
    assert "Invalid point index" in err


def test_triangulate_point_pair_rejects_incomplete_pair(make_fake_app, synthetic_cal):
    """A point that exists in the left list but not the matching right
    list should be reported as an incomplete pair."""
    app = make_fake_app(
        view_rectified=_TrueVar(), cal=synthetic_cal, ptsL=[(0, 0)], ptsR=[]
    )
    point, err = stereo_matching.triangulate_point_pair(app, 0)
    assert point is None
    assert "incomplete" in err.lower()


def test_project_point_round_trips_with_triangulate(synthetic_cal, known_point_pixels):
    """project_point should recover the same pixels used to triangulate."""

    app = type("App", (), {"cal": synthetic_cal})()
    point3d = stereo_matching.triangulate_from_pixels(
        app,
        known_point_pixels["xL"],
        known_point_pixels["yL"],
        known_point_pixels["xR"],
        known_point_pixels["yR"],
    )
    u, v = stereo_matching.project_point(synthetic_cal["PL"], *point3d)
    assert u == pytest.approx(known_point_pixels["xL"], abs=1e-3)
    assert v == pytest.approx(known_point_pixels["yL"], abs=1e-3)


def test_reprojection_rms_px_near_zero_for_exact_match(
    make_fake_app, synthetic_cal, known_point_pixels
):
    """Reprojecting a point triangulated from exact pixels should land
    back on those same pixels, giving ~0 RMS error."""

    app = make_fake_app(
        view_rectified=_TrueVar(),
        cal=synthetic_cal,
        ptsL=[(known_point_pixels["xL"], known_point_pixels["yL"])],
        ptsR=[(known_point_pixels["xR"], known_point_pixels["yR"])],
    )
    erms = stereo_matching.reprojection_rms_px(app, 0)
    assert erms == pytest.approx(0.0, abs=1e-3)


def test_estimate_point_sigma_mm_returns_positive_finite_values(
    make_fake_app, synthetic_cal, known_point_pixels
):
    """A well-conditioned point pair should produce a positive, finite
    depth/range uncertainty estimate rather than None, NaN, or infinity."""
    app = make_fake_app(
        cal=synthetic_cal,
        ptsL=[(known_point_pixels["xL"], known_point_pixels["yL"])],
        ptsR=[(known_point_pixels["xR"], known_point_pixels["yR"])],
    )
    result = stereo_matching.estimate_point_sigma_mm(app, 0, sigma_px=1.0)
    assert result is not None
    sZ, sR = result
    assert sZ > 0 and math.isfinite(sZ)
    assert sR > 0 and math.isfinite(sR)


def test_estimate_point_sigma_mm_grows_with_click_uncertainty(
    make_fake_app, synthetic_cal, known_point_pixels
):
    """A larger assumed click uncertainty should produce a larger (or
    equal) depth sigma estimate — sanity check on the direction of the
    relationship, not an exact value."""

    app = make_fake_app(
        cal=synthetic_cal,
        ptsL=[(known_point_pixels["xL"], known_point_pixels["yL"])],
        ptsR=[(known_point_pixels["xR"], known_point_pixels["yR"])],
    )
    small_sigma, _ = stereo_matching.estimate_point_sigma_mm(app, 0, sigma_px=0.5)
    large_sigma, _ = stereo_matching.estimate_point_sigma_mm(app, 0, sigma_px=5.0)
    assert large_sigma > small_sigma


def test_estimate_segment_sigma_len_mm_matches_known_baseline_length(
    make_fake_app, synthetic_cal
):
    """Two known 3D points a known distance apart should triangulate to
    (approximately) that same segment length."""

    X0, Y0, Z0 = 0.0, 0.0, 2000.0
    X1, Y1, Z1 = 100.0, 0.0, 2000.0
    expected_length = 100.0

    xL0, yL0 = _project(synthetic_cal["PL"], X0, Y0, Z0)
    xR0, yR0 = _project(synthetic_cal["PR"], X0, Y0, Z0)
    xL1, yL1 = _project(synthetic_cal["PL"], X1, Y1, Z1)
    xR1, yR1 = _project(synthetic_cal["PR"], X1, Y1, Z1)

    app = make_fake_app(
        cal=synthetic_cal,
        ptsL=[(xL0, yL0), (xL1, yL1)],
        ptsR=[(xR0, yR0), (xR1, yR1)],
    )

    result = stereo_matching.estimate_segment_sigma_len_mm(app, 0, 1, sigma_px=1.0)
    assert result is not None
    length, sigma_length = result
    assert length == pytest.approx(expected_length, abs=1e-2)
    assert sigma_length > 0


def test_guess_mate_point_on_scanline_finds_known_shift(make_fake_app, synthetic_cal):
    """A synthetic feature shifted by a known number of pixels between the
    left and right frames should be found at (approximately) that shift."""

    height, width = 60, 300
    base = np.full((height, width, 3), 50, dtype=np.uint8)

    left = base.copy()
    right = base.copy()

    # Place a distinct bright patch at x=100 in the left frame, and the
    # same patch shifted 15px left (positive disparity) in the right frame.
    src_x, src_y = 100, 30
    disparity = 15
    _draw_patch(left, src_x, src_y)
    _draw_patch(right, src_x - disparity, src_y)

    app = make_fake_app(
        view_rectified=_TrueVar(),
        cal=synthetic_cal,
        current_frameL=left,
        current_frameR=right,
    )

    result = stereo_matching.guess_mate_point_on_scanline(app, "L", src_x, src_y)
    assert result is not None
    found_x, found_y = result
    assert found_x == pytest.approx(src_x - disparity, abs=1.0)
    assert found_y == pytest.approx(src_y, abs=1e-6)


def test_guess_mate_point_on_scanline_requires_rectified_view(make_fake_app):
    """The scanline matcher is only valid in rectified view; it should
    refuse to guess a mate point otherwise."""
    app = make_fake_app(view_rectified=_FalseVar(), cal={}, current_frameL=None, current_frameR=None)
    assert stereo_matching.guess_mate_point_on_scanline(app, "L", 10, 10) is None


def test_guess_mate_point_on_scanline_requires_calibration(make_fake_app):
    """The scanline matcher should refuse to guess a mate point with no
    calibration loaded."""
    app = make_fake_app(view_rectified=_TrueVar(), cal=None, current_frameL=None, current_frameR=None)
    assert stereo_matching.guess_mate_point_on_scanline(app, "L", 10, 10) is None


# --- Small local helpers (deliberately not shared with conftest.py's
# project_through, to keep this file's expectations independent of that
# helper's correctness) ---


class _TrueVar:
    """Minimal stand-in for a Tkinter BooleanVar that's always True."""

    def get(self):
        """Return True.

        Returns:
            bool: Always True.
        """
        return True


class _FalseVar:
    """Minimal stand-in for a Tkinter BooleanVar that's always False."""

    def get(self):
        """Return False.

        Returns:
            bool: Always False.
        """
        return False


def _project(P, X, Y, Z):
    """Project a 3D point through a projection matrix (test helper only).

    Args:
        P (numpy.ndarray): 3x4 projection matrix.
        X (float): 3D point X coordinate.
        Y (float): 3D point Y coordinate.
        Z (float): 3D point Z coordinate.

    Returns:
        tuple[float, float]: The projected (u, v) pixel coordinates.
    """
    p = P @ np.array([X, Y, Z, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def _draw_patch(img, cx, cy, half=5, value=220):
    """Draw a solid bright square patch onto an image in place.

    Args:
        img (numpy.ndarray): The BGR image to draw on (mutated in place).
        cx (int): Patch center X coordinate.
        cy (int): Patch center Y coordinate.
        half (int): Half-width of the square patch, in pixels.
        value (int): Pixel value to fill the patch with.

    Returns:
        None
    """
    img[cy - half : cy + half, cx - half : cx + half] = value
