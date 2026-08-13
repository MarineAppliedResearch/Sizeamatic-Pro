"""Tests for generate_calibration_target.py.

Only the pure logic (settings parsing, board-image building, and the
save flow) is exercised here - the live preview canvas
(`_redraw_preview`) needs a real Tk widget to measure/draw onto and
isn't covered, matching this suite's existing convention of not testing
pure-Tkinter widget rendering directly.
"""

import numpy as np
import pytest

import generate_calibration_target
import perform_calibration


class _FakeVar:
    """Minimal `.get()`/`.set()` stand-in for a Tk `StringVar`, so these
    tests don't need a real Tk root."""

    def __init__(self, value=""):
        """Store the initial value this stand-in should report.

        Args:
            value: The initial value `.get()` should return.

        Returns:
            None
        """
        self._value = value

    def set(self, value):
        """Store a new value.

        Args:
            value: The new value to store.

        Returns:
            None
        """
        self._value = value

    def get(self):
        """Return the stored value.

        Returns:
            The most recently stored value.
        """
        return self._value


def _make_window():
    """Build a `GenerateCalibrationTargetWindow` with fake settings vars.

    Bypasses `ensure_window` (which needs a real Tk root) - sets exactly
    the widget-state attributes the functions under test actually read,
    matching `perform_calibration.py`'s own test pattern for
    `_parse_checkerboard_settings`.

    Returns:
        generate_calibration_target.GenerateCalibrationTargetWindow: A
        window instance with `board_type_var`/checkerboard settings
        vars/`status_var` set to fakes, `win=None`.
    """
    win = generate_calibration_target.GenerateCalibrationTargetWindow(app=None)
    win.board_type_var = _FakeVar("Checkerboard")
    win.checkerboard_squares_x_var = _FakeVar(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X))
    win.checkerboard_squares_y_var = _FakeVar(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y))
    win.checkerboard_square_size_var = _FakeVar(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM))
    win.charuco_squares_x_var = _FakeVar(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_X))
    win.charuco_squares_y_var = _FakeVar(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_Y))
    win.charuco_square_size_var = _FakeVar(str(perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM))
    win.charuco_marker_size_var = _FakeVar(str(perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM))
    win.status_var = _FakeVar("")
    return win


def test_parse_checkerboard_settings_returns_defaults():
    """The default field values should parse to the same defaults
    perform_calibration.py's Perform Calibration window uses, so a
    freshly opened Generate Calibration Target window matches a freshly
    opened Perform Calibration window out of the box."""
    win = _make_window()

    result = win._parse_checkerboard_settings()

    assert result == (
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X,
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y,
        perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM,
    )


def test_parse_checkerboard_settings_rejects_non_numeric_squares():
    """A non-numeric squares field should fail to parse rather than
    raising."""
    win = _make_window()
    win.checkerboard_squares_x_var.set("abc")

    assert win._parse_checkerboard_settings() is None


def test_parse_checkerboard_settings_rejects_too_few_squares():
    """Fewer than 2 squares in either direction can't form a detectable
    checkerboard."""
    win = _make_window()
    win.checkerboard_squares_x_var.set("1")

    assert win._parse_checkerboard_settings() is None


def test_parse_checkerboard_settings_rejects_non_positive_square_size():
    """A square size of zero (or negative) is physically meaningless."""
    win = _make_window()
    win.checkerboard_square_size_var.set("0")

    assert win._parse_checkerboard_settings() is None


def test_parse_charuco_settings_returns_defaults():
    """The default ChArUco field values should parse to the same
    defaults perform_calibration.py's Perform Calibration window uses."""
    win = _make_window()

    result = win._parse_charuco_settings()

    assert result == (
        perform_calibration.DEFAULT_CHARUCO_SQUARES_X,
        perform_calibration.DEFAULT_CHARUCO_SQUARES_Y,
        perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM,
        perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM,
    )


def test_parse_charuco_settings_rejects_too_few_squares():
    """Fewer than 2 squares in either direction can't form a valid
    ChArUco board."""
    win = _make_window()
    win.charuco_squares_x_var.set("1")

    assert win._parse_charuco_settings() is None


def test_parse_charuco_settings_rejects_marker_not_smaller_than_square():
    """A marker size that isn't smaller than the square size is
    physically invalid for a ChArUco board."""
    win = _make_window()
    win.charuco_marker_size_var.set(win.charuco_square_size_var.get())

    assert win._parse_charuco_settings() is None


def test_build_current_board_image_for_checkerboard_matches_settings():
    """Building a checkerboard image should use the parsed settings and
    report the correct physical board size and info text."""
    win = _make_window()
    win.checkerboard_squares_x_var.set("6")
    win.checkerboard_squares_y_var.set("4")
    win.checkerboard_square_size_var.set("10.0")

    board_img_gray, board_w_mm, board_h_mm, info_lines = win._build_current_board_image(dpi=100)

    assert isinstance(board_img_gray, np.ndarray)
    assert board_img_gray.dtype == np.uint8
    assert board_w_mm == pytest.approx(60.0)
    assert board_h_mm == pytest.approx(40.0)
    assert "6 x 4" in " ".join(info_lines)


def test_build_current_board_image_returns_none_for_invalid_checkerboard_settings():
    """An unparseable checkerboard setting should propagate as None
    rather than raising deep inside board rendering."""
    win = _make_window()
    win.checkerboard_squares_x_var.set("not a number")

    assert win._build_current_board_image(dpi=100) is None


def test_build_current_board_image_for_charuco_uses_charuco_settings():
    """ChArUco generation should use the parsed ChArUco settings (now
    editable, mirroring checkerboard), not the checkerboard fields."""
    win = _make_window()
    win.board_type_var.set("ChArUco")
    win.charuco_squares_x_var.set("6")
    win.charuco_squares_y_var.set("5")
    win.charuco_square_size_var.set("20.0")
    win.charuco_marker_size_var.set("15.0")

    board_img_gray, board_w_mm, board_h_mm, info_lines = win._build_current_board_image(dpi=100)

    assert isinstance(board_img_gray, np.ndarray)
    assert board_w_mm == pytest.approx(120.0)
    assert board_h_mm == pytest.approx(100.0)
    assert "6 x 5" in " ".join(info_lines)


def test_build_current_board_image_returns_none_for_invalid_charuco_settings():
    """An unparseable ChArUco setting should propagate as None rather
    than raising deep inside board rendering."""
    win = _make_window()
    win.board_type_var.set("ChArUco")
    win.charuco_squares_x_var.set("not a number")

    assert win._build_current_board_image(dpi=100) is None


def test_on_save_printable_board_rejects_invalid_checkerboard_settings(monkeypatch):
    """An invalid checkerboard setting should surface a messagebox error
    and never reach the save dialog."""
    win = _make_window()
    win.checkerboard_squares_x_var.set("1")

    errors = []
    monkeypatch.setattr(
        "generate_calibration_target.messagebox.showerror", lambda title, msg: errors.append(msg)
    )
    monkeypatch.setattr(
        "generate_calibration_target.filedialog.asksaveasfilename",
        lambda **_kwargs: pytest.fail("save dialog should not open for invalid settings"),
    )

    win.on_save_printable_board()

    assert len(errors) == 1
    assert win.status_var.get() == ""


def test_on_save_printable_board_writes_a_checkerboard_pdf(tmp_path, monkeypatch):
    """Saving with valid checkerboard settings should write a real PDF
    file and update the status line."""
    win = _make_window()
    out_path = tmp_path / "checkerboard.pdf"

    monkeypatch.setattr("generate_calibration_target.filedialog.asksaveasfilename", lambda **_kwargs: str(out_path))

    win.on_save_printable_board()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert str(out_path) in win.status_var.get()


def test_on_save_printable_board_writes_a_charuco_pdf(tmp_path, monkeypatch):
    """Saving with ChArUco selected should write a real PDF file using
    the default ChArUco board settings."""
    win = _make_window()
    win.board_type_var.set("ChArUco")
    out_path = tmp_path / "charuco.pdf"

    monkeypatch.setattr("generate_calibration_target.filedialog.asksaveasfilename", lambda **_kwargs: str(out_path))

    win.on_save_printable_board()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert str(out_path) in win.status_var.get()


def test_on_save_printable_board_does_nothing_when_dialog_is_cancelled(tmp_path, monkeypatch):
    """Cancelling the save dialog (empty path) should leave the status
    line untouched rather than trying to write to an empty path."""
    win = _make_window()

    monkeypatch.setattr("generate_calibration_target.filedialog.asksaveasfilename", lambda **_kwargs: "")

    win.on_save_printable_board()

    assert win.status_var.get() == ""


def test_ensure_window_builds_all_widgets_on_a_real_tk_root(hidden_tk_root, make_fake_app):
    """`ensure_window` should build without raising against a real Tk
    root, and closing it should clear every widget reference back to
    None - matching `PerformCalibrationWindow`'s own close-then-reopen
    guarantee. A `_FakeVar`-based unit test can't catch a real grid
    conflict or a bad widget reference the way building actual Tkinter
    widgets can."""
    app = make_fake_app(root=hidden_tk_root)
    win = generate_calibration_target.GenerateCalibrationTargetWindow(app)

    win.ensure_window()

    assert win.win is not None
    assert win.board_type_var.get() == "Checkerboard"
    # grid_info() is empty once a widget's been grid_remove()'d - checking
    # it (rather than winfo_ismapped(), which needs an actual mapped
    # window/display) reflects the geometry manager's own visibility state
    # even against a withdrawn Toplevel.
    assert win.checkerboard_row.grid_info()
    assert not win.charuco_row.grid_info()

    # Switching board type should swap which settings row is visible.
    win.board_type_var.set("ChArUco")
    win._on_settings_changed()
    assert not win.checkerboard_row.grid_info()
    assert win.charuco_row.grid_info()

    win._on_close()

    assert win.win is None
    assert win.board_type_var is None
    assert win.checkerboard_row is None
    assert win.charuco_row is None
    assert win.preview_canvas is None
