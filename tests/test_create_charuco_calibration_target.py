"""Tests for create_charuco_calibration_target.py."""

import cv2
import numpy as np
import pytest

import create_charuco_calibration_target as charuco


def test_mm_to_points_known_conversion():
    """25.4mm is exactly 1 inch, which is exactly 72 PDF points."""
    assert charuco.mm_to_points(25.4) == pytest.approx(72.0)
    assert charuco.mm_to_points(0.0) == pytest.approx(0.0)


def test_build_charuco_image_has_expected_pixel_dimensions():
    """The rendered board image's pixel size should match the requested
    physical size converted to pixels at the requested DPI."""
    dpi = 300
    squares_x, squares_y = 4, 3
    square_size_mm = 20.0

    img = charuco.build_charuco_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_mm=square_size_mm,
        marker_size_mm=15.0,
        dictionary_id=cv2.aruco.DICT_4X4_50,
        dpi=dpi,
        margin_mm=0,
    )

    expected_w = int((squares_x * square_size_mm / 25.4) * dpi)
    expected_h = int((squares_y * square_size_mm / 25.4) * dpi)

    assert img.shape[1] == expected_w
    assert img.shape[0] == expected_h
    assert img.dtype == np.uint8


def test_write_pdf_letter_landscape_rejects_oversized_board(tmp_path):
    """A board too large for LETTER landscape with the given margin should
    fail loudly with RuntimeError rather than silently clipping or
    producing a broken PDF."""
    dummy_img = np.zeros((10, 10), dtype=np.uint8)
    out_path = tmp_path / "too_big.pdf"

    with pytest.raises(RuntimeError, match="does not fit"):
        charuco.write_pdf_letter_landscape(
            out_pdf_path=str(out_path),
            board_img_gray=dummy_img,
            board_w_mm=10000.0,
            board_h_mm=10000.0,
            margin_in=0.5,
        )


def test_write_pdf_letter_landscape_writes_a_file(tmp_path):
    """A board that fits the page should produce a non-empty PDF file at
    the requested output path."""
    dummy_img = np.zeros((50, 50), dtype=np.uint8)
    out_path = tmp_path / "small_board.pdf"

    charuco.write_pdf_letter_landscape(
        out_pdf_path=str(out_path),
        board_img_gray=dummy_img,
        board_w_mm=50.0,
        board_h_mm=50.0,
        margin_in=0.5,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
