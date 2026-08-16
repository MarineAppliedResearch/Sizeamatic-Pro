"""Tests for create_checkerboard_calibration_target.py."""

import numpy as np
import pytest

import create_checkerboard_calibration_target as checkerboard


def test_mm_to_points_known_conversion():
    """25.4mm is exactly 1 inch, which is exactly 72 PDF points."""
    assert checkerboard.mm_to_points(25.4) == pytest.approx(72.0)
    assert checkerboard.mm_to_points(0.0) == pytest.approx(0.0)


def test_build_checkerboard_image_has_expected_pixel_dimensions():
    """The rendered board image's pixel size should match the requested
    physical size converted to pixels at the requested DPI."""
    dpi = 300
    squares_x, squares_y = 4, 3
    square_size_mm = 20.0

    img = checkerboard.build_checkerboard_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_mm=square_size_mm,
        dpi=dpi,
    )

    expected_w = int((squares_x * square_size_mm / 25.4) * dpi)
    expected_h = int((squares_y * square_size_mm / 25.4) * dpi)

    assert img.shape[1] == expected_w
    assert img.shape[0] == expected_h
    assert img.dtype == np.uint8


def test_build_checkerboard_image_alternates_squares():
    """Adjacent squares should have opposite colors (black/white), the
    same alternating pattern cv2.findChessboardCorners expects — not a
    uniform fill or a stripe pattern."""
    dpi = 100
    squares_x, squares_y = 3, 3
    square_size_mm = 25.0

    img = checkerboard.build_checkerboard_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_mm=square_size_mm,
        dpi=dpi,
    )

    square_size_px = int((square_size_mm / 25.4) * dpi)

    # Sample the center pixel of each of the first two squares in the top
    # row — they must be opposite colors for the board to be detectable.
    top_left_px = img[square_size_px // 2, square_size_px // 2]
    top_right_px = img[square_size_px // 2, square_size_px + square_size_px // 2]

    assert top_left_px != top_right_px


def test_build_checkerboard_info_lines_reports_actual_numbers():
    """The printed-on-page text should reflect the actual board settings
    passed in, not a hardcoded example - so a printout says exactly what
    board it is, including the inner-corner count derived from squares."""
    lines = checkerboard.build_checkerboard_info_lines(squares_x=10, squares_y=7, square_size_mm=25.0)

    joined = " ".join(lines)
    assert "10 x 7" in joined
    assert "9 x 6" in joined
    assert "25.0 mm" in joined


def test_write_pdf_letter_landscape_rejects_oversized_board(tmp_path):
    """A board too large for LETTER landscape with the given margin should
    fail loudly with RuntimeError rather than silently clipping or
    producing a broken PDF."""
    dummy_img = np.zeros((10, 10), dtype=np.uint8)
    out_path = tmp_path / "too_big.pdf"

    with pytest.raises(RuntimeError, match="does not fit"):
        checkerboard.write_pdf_letter_landscape(
            out_pdf_path=str(out_path),
            board_img_gray=dummy_img,
            board_w_mm=10000.0,
            board_h_mm=10000.0,
            margin_in=0.5,
            info_lines=[],
        )


def test_write_pdf_letter_landscape_writes_a_file(tmp_path):
    """A board that fits the page should produce a non-empty PDF file at
    the requested output path."""
    dummy_img = np.zeros((50, 50), dtype=np.uint8)
    out_path = tmp_path / "small_board.pdf"

    checkerboard.write_pdf_letter_landscape(
        out_pdf_path=str(out_path),
        board_img_gray=dummy_img,
        board_w_mm=50.0,
        board_h_mm=50.0,
        margin_in=0.5,
        info_lines=checkerboard.build_checkerboard_info_lines(4, 3, 25.0),
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
