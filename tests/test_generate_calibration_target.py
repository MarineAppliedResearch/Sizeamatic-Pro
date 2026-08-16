"""Tests for generate_calibration_target.py.

Covers the pure logic (settings parsing, board-image building, and the
save flow) plus a real-widget test of `ensure_window`/board-type
switching - the live preview label's actual pixel rendering isn't
covered here, matching this suite's existing convention of not testing
rendering pixel-by-pixel.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit

import generate_calibration_target
import perform_calibration


def _make_window(qapp):
    """Build a `GenerateCalibrationTargetWindow` with real, standalone
    Qt widgets standing in for the ones `ensure_window` would normally
    build.

    Bypasses `ensure_window` (which builds a whole dialog) - sets
    exactly the widget-state attributes the functions under test
    actually read, matching `perform_calibration.py`'s own test
    pattern for `_parse_checkerboard_settings`. A real `QLineEdit`/
    `QComboBox` works perfectly well unshown and outside any layout, so
    this needs the `qapp` fixture (a `QApplication` must already exist
    to construct any `QWidget`) but not a full dialog.

    Args:
        qapp (QApplication): The shared test-session `QApplication`.

    Returns:
        generate_calibration_target.GenerateCalibrationTargetWindow: A
        window instance with its settings widgets set to standalone
        stand-ins, `win=None`.
    """
    win = generate_calibration_target.GenerateCalibrationTargetWindow(app=None)
    win.board_type_combo = QComboBox()
    win.board_type_combo.addItems(["Checkerboard", "ChArUco"])
    win.checkerboard_squares_x_edit = QLineEdit(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X))
    win.checkerboard_squares_y_edit = QLineEdit(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y))
    win.checkerboard_square_size_edit = QLineEdit(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM))
    win.charuco_squares_x_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_X))
    win.charuco_squares_y_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_Y))
    win.charuco_square_size_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM))
    win.charuco_marker_size_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM))
    win.status_label = QLabel("")
    return win


def test_parse_checkerboard_settings_returns_defaults(qapp):
    """The default field values should parse to the same defaults
    perform_calibration.py's Perform Calibration window uses, so a
    freshly opened Generate Calibration Target window matches a freshly
    opened Perform Calibration window out of the box."""
    win = _make_window(qapp)

    result = win._parse_checkerboard_settings()

    assert result == (
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X,
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y,
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM,
    )


def test_parse_checkerboard_settings_rejects_non_numeric_squares(qapp):
    """A non-numeric squares field should fail to parse rather than
    raising."""
    win = _make_window(qapp)
    win.checkerboard_squares_x_edit.setText("abc")

    assert win._parse_checkerboard_settings() is None


def test_parse_checkerboard_settings_rejects_too_few_squares(qapp):
    """Fewer than 2 squares in either direction can't form a detectable
    checkerboard."""
    win = _make_window(qapp)
    win.checkerboard_squares_x_edit.setText("1")

    assert win._parse_checkerboard_settings() is None


def test_parse_checkerboard_settings_rejects_non_positive_square_size(qapp):
    """A square size of zero (or negative) is physically meaningless."""
    win = _make_window(qapp)
    win.checkerboard_square_size_edit.setText("0")

    assert win._parse_checkerboard_settings() is None


def test_parse_charuco_settings_returns_defaults(qapp):
    """The default ChArUco field values should parse to the same
    defaults perform_calibration.py's Perform Calibration window uses."""
    win = _make_window(qapp)

    result = win._parse_charuco_settings()

    assert result == (
        perform_calibration.DEFAULT_CHARUCO_SQUARES_X,
        perform_calibration.DEFAULT_CHARUCO_SQUARES_Y,
        perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM,
        perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM,
    )


def test_parse_charuco_settings_rejects_too_few_squares(qapp):
    """Fewer than 2 squares in either direction can't form a valid
    ChArUco board."""
    win = _make_window(qapp)
    win.charuco_squares_x_edit.setText("1")

    assert win._parse_charuco_settings() is None


def test_parse_charuco_settings_rejects_marker_not_smaller_than_square(qapp):
    """A marker size that isn't smaller than the square size is
    physically invalid for a ChArUco board."""
    win = _make_window(qapp)
    win.charuco_marker_size_edit.setText(win.charuco_square_size_edit.text())

    assert win._parse_charuco_settings() is None


def test_build_current_board_image_for_checkerboard_matches_settings(qapp):
    """Building a checkerboard image should use the parsed settings and
    report the correct physical board size and info text."""
    win = _make_window(qapp)
    win.checkerboard_squares_x_edit.setText("6")
    win.checkerboard_squares_y_edit.setText("4")
    win.checkerboard_square_size_edit.setText("10.0")

    board_img_gray, board_w_mm, board_h_mm, info_lines = win._build_current_board_image(dpi=100)

    assert isinstance(board_img_gray, np.ndarray)
    assert board_img_gray.dtype == np.uint8
    assert board_w_mm == pytest.approx(60.0)
    assert board_h_mm == pytest.approx(40.0)
    assert "6 x 4" in " ".join(info_lines)


def test_build_current_board_image_returns_none_for_invalid_checkerboard_settings(qapp):
    """An unparseable checkerboard setting should propagate as None
    rather than raising deep inside board rendering."""
    win = _make_window(qapp)
    win.checkerboard_squares_x_edit.setText("not a number")

    assert win._build_current_board_image(dpi=100) is None


def test_build_current_board_image_for_charuco_uses_charuco_settings(qapp):
    """ChArUco generation should use the parsed ChArUco settings (now
    editable, mirroring checkerboard), not the checkerboard fields."""
    win = _make_window(qapp)
    win.board_type_combo.setCurrentText("ChArUco")
    win.charuco_squares_x_edit.setText("6")
    win.charuco_squares_y_edit.setText("5")
    win.charuco_square_size_edit.setText("20.0")
    win.charuco_marker_size_edit.setText("15.0")

    board_img_gray, board_w_mm, board_h_mm, info_lines = win._build_current_board_image(dpi=100)

    assert isinstance(board_img_gray, np.ndarray)
    assert board_w_mm == pytest.approx(120.0)
    assert board_h_mm == pytest.approx(100.0)
    assert "6 x 5" in " ".join(info_lines)


def test_build_current_board_image_returns_none_for_invalid_charuco_settings(qapp):
    """An unparseable ChArUco setting should propagate as None rather
    than raising deep inside board rendering."""
    win = _make_window(qapp)
    win.board_type_combo.setCurrentText("ChArUco")
    win.charuco_squares_x_edit.setText("not a number")

    assert win._build_current_board_image(dpi=100) is None


def test_on_save_printable_board_rejects_invalid_checkerboard_settings(qapp, monkeypatch):
    """An invalid checkerboard setting should surface a message box and
    never reach the save dialog."""
    win = _make_window(qapp)
    win.checkerboard_squares_x_edit.setText("1")

    errors = []
    monkeypatch.setattr(
        "generate_calibration_target.QMessageBox.critical",
        staticmethod(lambda *args: errors.append(args[-1])),
    )
    monkeypatch.setattr(
        "generate_calibration_target.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: pytest.fail("save dialog should not open for invalid settings")),
    )

    win.on_save_printable_board()

    assert len(errors) == 1
    assert win.status_label.text() == ""


def test_on_save_printable_board_writes_a_checkerboard_pdf(qapp, tmp_path, monkeypatch):
    """Saving with valid checkerboard settings should write a real PDF
    file and update the status line."""
    win = _make_window(qapp)
    out_path = tmp_path / "checkerboard.pdf"

    monkeypatch.setattr(
        "generate_calibration_target.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )

    win.on_save_printable_board()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert str(out_path) in win.status_label.text()


def test_on_save_printable_board_writes_a_charuco_pdf(qapp, tmp_path, monkeypatch):
    """Saving with ChArUco selected should write a real PDF file using
    the default ChArUco board settings."""
    win = _make_window(qapp)
    win.board_type_combo.setCurrentText("ChArUco")
    out_path = tmp_path / "charuco.pdf"

    monkeypatch.setattr(
        "generate_calibration_target.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )

    win.on_save_printable_board()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert str(out_path) in win.status_label.text()


def test_on_save_printable_board_does_nothing_when_dialog_is_cancelled(qapp, tmp_path, monkeypatch):
    """Cancelling the save dialog (empty path) should leave the status
    line untouched rather than trying to write to an empty path."""
    win = _make_window(qapp)

    monkeypatch.setattr(
        "generate_calibration_target.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )

    win.on_save_printable_board()

    assert win.status_label.text() == ""


def test_ensure_window_builds_all_widgets_and_switches_board_type(qapp, make_fake_app):
    """`ensure_window` should build without raising, and closing it
    should clear every widget reference back to None - matching
    `PerformCalibrationWindow`'s own close-then-reopen guarantee."""
    app = make_fake_app()
    win = generate_calibration_target.GenerateCalibrationTargetWindow(app)

    win.ensure_window()

    assert win.win is not None
    assert win.board_type_combo.currentText() == "Checkerboard"
    assert win.checkerboard_row.isVisible()
    assert not win.charuco_row.isVisible()

    # Switching board type should swap which settings row is visible -
    # the combo's own currentTextChanged signal drives this, no manual
    # re-invocation needed (unlike the original StringVar, which had no
    # signal of its own).
    win.board_type_combo.setCurrentText("ChArUco")
    assert not win.checkerboard_row.isVisible()
    assert win.charuco_row.isVisible()

    win._on_close()

    assert win.win is None
    assert win.board_type_combo is None
    assert win.checkerboard_row is None
    assert win.charuco_row is None
    assert win.preview_label is None
