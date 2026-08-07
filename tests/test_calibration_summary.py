"""Tests for calibration_summary.py's pure remap-quality math.

update_calibration_window itself is UI-building code exercised indirectly
via tests/test_regressions.py; this file covers the one genuinely pure
function, map_oob_percent.
"""

import numpy as np
import pytest

import calibration_summary


def test_map_oob_percent_all_in_bounds():
    """An identity remap (each pixel maps to itself) should report the
    out-of-bounds percentage implied by the function's own edge rule, not
    a naive 0%."""
    w, h = 10, 10
    mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pct = calibration_summary.map_oob_percent(mapx, mapy, w, h)
    # The last row/column (index w-1 / h-1) is treated as out of bounds by
    # the ">= (w - 1)" check, so a plain identity map isn't 0% — this
    # pins down that documented edge behavior rather than assuming 0%.
    expected_oob = w * h - (w - 1) * (h - 1)
    expected_pct = 100.0 * expected_oob / (w * h)
    assert pct == pytest.approx(expected_pct)


def test_map_oob_percent_all_out_of_bounds():
    """A remap that samples entirely outside the source image should
    report 100%."""
    w, h = 10, 10
    mapx = np.full((h, w), -5.0, dtype=np.float32)
    mapy = np.full((h, w), -5.0, dtype=np.float32)
    pct = calibration_summary.map_oob_percent(mapx, mapy, w, h)
    assert pct == pytest.approx(100.0)


def test_map_oob_percent_known_fraction():
    """A remap with a hand-picked number of out-of-bounds samples should
    report exactly that fraction as a percentage."""
    w, h = 4, 3
    mapx = np.zeros((h, w), dtype=np.float32)
    mapy = np.zeros((h, w), dtype=np.float32)

    # Push exactly 3 of the 12 samples out of bounds.
    mapx[0, 0] = -1.0
    mapx[0, 1] = float(w)
    mapy[1, 0] = -1.0

    pct = calibration_summary.map_oob_percent(mapx, mapy, w, h)
    assert pct == pytest.approx(100.0 * 3 / 12)
