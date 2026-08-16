"""PySide6 Generate Calibration Target window for Sizeamatic Pro.

Lets the project owner generate a printable checkerboard or ChArUco
calibration target from inside the app itself (ROADMAP.md Phase 10),
rather than needing to run a separate command-line script.

This is a PySide6 port of the original Tkinter module (ROADMAP.md Phase
11) - see `main.py`'s module docstring for why the app switched
frameworks. The board-image rendering/PDF layout are unchanged
(untouched Tk-free library calls); only the settings form + live
preview + save dialog widgets changed.

Contents:
    - `GenerateCalibrationTargetWindow` — owns the window and its
      widgets.

Design notes:
    Follows the same "class instance owned by the main application"
    shape as `calibration_summary.CalibrationSummaryWindow` and
    `perform_calibration.PerformCalibrationWindow` - `self.win`/other
    widget references start `None` and only get built by
    `ensure_window`, which is safe to call repeatedly.

    The actual board-image rendering and PDF page layout are NOT
    duplicated here - this window is a thin UI shell around
    `create_checkerboard_calibration_target.py`'s and
    `create_charuco_calibration_target.py`'s existing
    `build_checkerboard_image`/`build_charuco_image`/
    `write_pdf_letter_landscape` functions (those two modules stay as
    small, independently-testable/importable libraries; the CLI
    `main()` each one still has is a low-level dev convenience, not the
    intended way a project owner generates a target day to day - this
    window is).

    Both checkerboard settings (squares x squares, square size in mm)
    and ChArUco settings (squares x squares, square size in mm, marker
    size in mm) are editable, defaulting to `perform_calibration.py`'s
    `DEFAULT_CHECKERBOARD_SQUARES_X`/`_Y`/
    `DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM` and `DEFAULT_CHARUCO_SQUARES_X`/
    `_Y`/`DEFAULT_CHARUCO_SQUARE_SIZE_MM`/`DEFAULT_CHARUCO_MARKER_SIZE_MM`
    respectively - importing those constants directly (rather than
    duplicating a third copy of the same literal values here) keeps
    this window's defaults, `perform_calibration.py`'s own defaults, and
    whatever board actually gets printed all guaranteed to agree. Both
    board types are editable here because `perform_calibration.py`'s
    detection is *also* fully configurable for both (ROADMAP.md Phase
    10) - a board printed with different settings here is still
    detectable there as long as the same numbers get entered into both
    windows. Only the ArUco *dictionary* (`CHARUCO_DICTIONARY_ID`) stays
    a fixed module constant on both sides, not user-set - see
    `perform_calibration.py`'s docstring for why.

    Every printed board also has its own actual settings written on the
    page itself as small text (`build_checkerboard_info_lines`/
    `build_charuco_info_lines`, passed as `write_pdf_letter_landscape`'s
    `info_lines`) - a physical printout should state its own exact
    numbers rather than relying on whoever printed it to remember them.

    The live preview renders at a low, fixed `PREVIEW_DPI` (fast enough
    to redraw on every keystroke) into a `QPixmap` via
    `qt_helpers.pil_image_to_qpixmap`.

Assumptions:
    - The main application exposes `app._app_window_title` (this app's
      project-aware window title, shared by every window). Defined on
      `main.py`'s `SizeamaticProApp`.

Author:
    Isaac Travers

Date:
    2026-08-13
"""

import cv2
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import create_charuco_calibration_target
import create_checkerboard_calibration_target
import perform_calibration
from qt_helpers import ClosableDialog, move_to_same_screen_as, pil_image_to_qpixmap

PREVIEW_DPI = 96
"""Render resolution used only for this window's on-screen preview -
low enough to rebuild the board image instantly on every settings
change, unlike the 300 DPI `on_save_printable_board` renders the actual
printable PDF at."""

PDF_PAGE_MARGIN_IN = 0.5
"""PDF page margin (inches) passed to `write_pdf_letter_landscape` for
a saved board - matches both generator scripts' own `main()` default."""


class _PreviewLabel(QLabel):
    """The live preview `QLabel`, redrawing itself whenever resized.

    A plain `QLabel` doesn't emit anything on resize - this override is
    the Qt equivalent of the original's `<Configure>` binding on its
    preview Canvas, which redrew the preview to fit its new size.
    """

    def __init__(self, on_resize):
        """Store the resize callback.

        Args:
            on_resize (Callable[[], None]): Called (with no arguments)
                after this label is resized.

        Returns:
            None
        """
        super().__init__()
        self._on_resize = on_resize

    def resizeEvent(self, event):
        """Call the stored resize callback after the base resize handling.

        Args:
            event (QResizeEvent): The resize event.

        Returns:
            None
        """
        super().resizeEvent(event)
        self._on_resize()


class GenerateCalibrationTargetWindow:
    """Owns the Generate Calibration Target dialog and its widgets.

    One instance lives on the main application
    (`app.generate_calibration_target_window`), created once and reused
    for the lifetime of the app, matching `PerformCalibrationWindow`'s
    pattern.
    """

    def __init__(self, app):
        """Store the owning app and initialize widget state to None.

        Args:
            app: The main application object - see this module's
                docstring for the exact attributes assumed to exist on
                it.

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The Generate Calibration Target dialog, or None if it hasn't
        been opened yet (or was closed)."""

        self.board_type_combo = None
        """`QComboBox` holding which board type is currently selected -
        `"Checkerboard"` or `"ChArUco"`. Created in `ensure_window`."""

        self.checkerboard_squares_x_edit = None
        """`QLineEdit` holding the checkerboard's width in squares,
        defaulting to `perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X`.
        Created in `ensure_window`."""

        self.checkerboard_squares_y_edit = None
        """`QLineEdit` holding the checkerboard's height in squares. See
        `self.checkerboard_squares_x_edit`."""

        self.checkerboard_square_size_edit = None
        """`QLineEdit` holding the checkerboard's real-world square size
        in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM`.
        Created in `ensure_window`."""

        self.charuco_squares_x_edit = None
        """`QLineEdit` holding the ChArUco board's width in squares,
        defaulting to `perform_calibration.DEFAULT_CHARUCO_SQUARES_X`.
        Created in `ensure_window`."""

        self.charuco_squares_y_edit = None
        """`QLineEdit` holding the ChArUco board's height in squares. See
        `self.charuco_squares_x_edit`."""

        self.charuco_square_size_edit = None
        """`QLineEdit` holding the ChArUco board's real-world square size
        in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM`. Created in
        `ensure_window`."""

        self.charuco_marker_size_edit = None
        """`QLineEdit` holding the ChArUco board's real-world ArUco
        marker size in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM` - must be
        smaller than `self.charuco_square_size_edit`
        (`_parse_charuco_settings` enforces this). Created in
        `ensure_window`."""

        self.checkerboard_row = None
        """The `QWidget` holding the checkerboard-specific settings -
        shown only while `self.board_type_combo` reads "Checkerboard".
        Created in `ensure_window`."""

        self.charuco_row = None
        """The `QWidget` holding the ChArUco-specific settings - shown
        only while `self.board_type_combo` reads "ChArUco". Created in
        `ensure_window`."""

        self.preview_label = None
        """The `_PreviewLabel` the current board settings are rendered
        onto. Created in `ensure_window`."""

        self.status_label = None
        """`QLabel` holding the window's status line, e.g. "Saved:
        C:/.../checkerboard_letter_landscape.pdf". Created in
        `ensure_window`."""

    def _on_close(self):
        """Handle the user manually closing the window.

        Clears the stored widget references, matching
        `PerformCalibrationWindow._on_close`'s reasoning - the next
        `ensure_window` call needs to rebuild them rather than holding
        onto references to already-destroyed widgets.

        Returns:
            None
        """
        self.win = None
        self.board_type_combo = None
        self.checkerboard_squares_x_edit = None
        self.checkerboard_squares_y_edit = None
        self.checkerboard_square_size_edit = None
        self.charuco_squares_x_edit = None
        self.charuco_squares_y_edit = None
        self.charuco_square_size_edit = None
        self.charuco_marker_size_edit = None
        self.checkerboard_row = None
        self.charuco_row = None
        self.preview_label = None
        self.status_label = None

    def ensure_window(self):
        """Create the Generate Calibration Target window, or raise it if it exists.

        Returns:
            None
        """
        if self.win is not None:
            self.win.show()
            self.win.raise_()
            self.win.activateWindow()
            return

        win = ClosableDialog(self._on_close)
        win.setWindowTitle(self.app._app_window_title())
        win.resize(560, 600)

        outer = QVBoxLayout(win)

        heading = QLabel("Generate Calibration Target")
        heading.setStyleSheet("font-weight: bold;")
        outer.addWidget(heading)

        # ---- Board type row ----
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Board type:"))
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(["Checkerboard", "ChArUco"])
        self.board_type_combo.currentTextChanged.connect(lambda _text: self._on_settings_changed())
        type_row.addWidget(self.board_type_combo)
        type_row.addStretch(1)
        outer.addLayout(type_row)

        # ---- Checkerboard settings row ----
        # Editable - see this module's docstring for why checkerboard
        # (unlike ChArUco) is safe to let the project owner customize.
        self.checkerboard_row = QWidget()
        checkerboard_layout = QHBoxLayout(self.checkerboard_row)
        checkerboard_layout.setContentsMargins(0, 0, 0, 0)

        checkerboard_layout.addWidget(QLabel("Squares (columns x rows):"))
        self.checkerboard_squares_x_edit = QLineEdit(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X))
        self.checkerboard_squares_x_edit.setFixedWidth(45)
        checkerboard_layout.addWidget(self.checkerboard_squares_x_edit)
        checkerboard_layout.addWidget(QLabel("x"))
        self.checkerboard_squares_y_edit = QLineEdit(str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y))
        self.checkerboard_squares_y_edit.setFixedWidth(45)
        checkerboard_layout.addWidget(self.checkerboard_squares_y_edit)

        checkerboard_layout.addWidget(QLabel("Square size (mm):"))
        self.checkerboard_square_size_edit = QLineEdit(
            str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM)
        )
        self.checkerboard_square_size_edit.setFixedWidth(70)
        checkerboard_layout.addWidget(self.checkerboard_square_size_edit)
        checkerboard_layout.addStretch(1)

        # Redraw the preview whenever any checkerboard setting changes.
        for edit in (
            self.checkerboard_squares_x_edit,
            self.checkerboard_squares_y_edit,
            self.checkerboard_square_size_edit,
        ):
            edit.textChanged.connect(lambda _text: self._redraw_preview())

        outer.addWidget(self.checkerboard_row)

        # ---- ChArUco settings row ----
        # Editable, same as checkerboard - see this module's docstring
        # for why ChArUco's geometry is user-set too, not fixed.
        self.charuco_row = QWidget()
        charuco_layout = QHBoxLayout(self.charuco_row)
        charuco_layout.setContentsMargins(0, 0, 0, 0)

        charuco_layout.addWidget(QLabel("Squares (columns x rows):"))
        self.charuco_squares_x_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_X))
        self.charuco_squares_x_edit.setFixedWidth(45)
        charuco_layout.addWidget(self.charuco_squares_x_edit)
        charuco_layout.addWidget(QLabel("x"))
        self.charuco_squares_y_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARES_Y))
        self.charuco_squares_y_edit.setFixedWidth(45)
        charuco_layout.addWidget(self.charuco_squares_y_edit)

        charuco_layout.addWidget(QLabel("Square (mm):"))
        self.charuco_square_size_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM))
        self.charuco_square_size_edit.setFixedWidth(55)
        charuco_layout.addWidget(self.charuco_square_size_edit)

        charuco_layout.addWidget(QLabel("Marker (mm):"))
        self.charuco_marker_size_edit = QLineEdit(str(perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM))
        self.charuco_marker_size_edit.setFixedWidth(55)
        charuco_layout.addWidget(self.charuco_marker_size_edit)
        charuco_layout.addStretch(1)

        # Redraw the preview whenever any ChArUco setting changes - same
        # pattern as the checkerboard fields above.
        for edit in (
            self.charuco_squares_x_edit,
            self.charuco_squares_y_edit,
            self.charuco_square_size_edit,
            self.charuco_marker_size_edit,
        ):
            edit.textChanged.connect(lambda _text: self._redraw_preview())

        outer.addWidget(self.charuco_row)

        # ---- Live preview ----
        self.preview_label = _PreviewLabel(self._redraw_preview)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #dddddd; border: 1px solid #999999;")
        self.preview_label.setMinimumHeight(200)
        outer.addWidget(self.preview_label, stretch=1)

        # ---- Save button ----
        button_row = QHBoxLayout()
        save_button = QPushButton("Save Printable Board…")
        save_button.clicked.connect(self.on_save_printable_board)
        button_row.addWidget(save_button)
        button_row.addStretch(1)
        outer.addLayout(button_row)

        # ---- Status line ----
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8ea2c6;")
        outer.addWidget(self.status_label)

        self.win = win
        win.show()
        # Positioned only after show(), which finalizes the window's real
        # layout-driven size rather than the initial resize() hint.
        move_to_same_screen_as(win, self.app)

        # Show only the settings row matching the current board type, and
        # draw the first preview.
        self._on_settings_changed()

    def _on_settings_changed(self):
        """Show the settings row matching the selected board type and redraw.

        Called whenever `self.board_type_combo` changes - swaps which of
        `self.checkerboard_row`/`self.charuco_row` is visible, then
        redraws the preview for the newly selected board type.

        Returns:
            None
        """
        is_checkerboard = self.board_type_combo.currentText() == "Checkerboard"
        self.checkerboard_row.setVisible(is_checkerboard)
        self.charuco_row.setVisible(not is_checkerboard)

        self._redraw_preview()

    def _parse_checkerboard_settings(self):
        """Parse and validate the checkerboard squares/size fields.

        Deliberately returns `None` on any problem instead of showing a
        message box - this is called on every keystroke to refresh the
        live preview, and popping an error dialog while the project
        owner is still mid-edit (e.g. squares_x is temporarily empty
        between typing "1" and "10") would be disruptive.
        `on_save_printable_board` is the one place that treats `None`
        as an error worth surfacing to the user.

        Returns:
            tuple[int, int, float] | None: `(squares_x, squares_y,
            square_size_mm)` if all three fields are valid, else None.
        """
        try:
            squares_x = int(self.checkerboard_squares_x_edit.text())
            squares_y = int(self.checkerboard_squares_y_edit.text())
        except ValueError:
            return None
        if squares_x < 2 or squares_y < 2:
            return None

        try:
            square_size_mm = float(self.checkerboard_square_size_edit.text())
        except ValueError:
            return None
        if square_size_mm <= 0:
            return None

        return squares_x, squares_y, square_size_mm

    def _parse_charuco_settings(self):
        """Parse and validate the ChArUco squares/square size/marker size fields.

        Mirrors `_parse_checkerboard_settings` - deliberately returns
        `None` on any problem instead of showing a message box, since
        this is also called on every keystroke to refresh the live
        preview.

        Returns:
            tuple[int, int, float, float] | None: `(squares_x, squares_y,
            square_size_mm, marker_size_mm)` if every field is valid,
            else None.
        """
        try:
            squares_x = int(self.charuco_squares_x_edit.text())
            squares_y = int(self.charuco_squares_y_edit.text())
        except ValueError:
            return None
        if squares_x < 2 or squares_y < 2:
            return None

        try:
            square_size_mm = float(self.charuco_square_size_edit.text())
            marker_size_mm = float(self.charuco_marker_size_edit.text())
        except ValueError:
            return None
        if square_size_mm <= 0 or marker_size_mm <= 0:
            return None
        if marker_size_mm >= square_size_mm:
            return None

        return squares_x, squares_y, square_size_mm, marker_size_mm

    def _build_current_board_image(self, dpi):
        """Render the currently selected board type/settings to an image.

        Args:
            dpi (int): Render resolution, in dots per inch - `PREVIEW_DPI`
                for the on-screen preview, or 300 (matching both
                generator scripts' own `main()`) when actually saving.

        Returns:
            tuple[numpy.ndarray, float, float, list] | None:
            `(board_img_gray, board_w_mm, board_h_mm, info_lines)` if
            the current settings are valid, else None.
        """
        if self.board_type_combo.currentText() == "Checkerboard":
            settings = self._parse_checkerboard_settings()
            if settings is None:
                return None
            squares_x, squares_y, square_size_mm = settings

            board_img_gray = create_checkerboard_calibration_target.build_checkerboard_image(
                squares_x=squares_x,
                squares_y=squares_y,
                square_size_mm=square_size_mm,
                dpi=dpi,
            )
            info_lines = create_checkerboard_calibration_target.build_checkerboard_info_lines(
                squares_x, squares_y, square_size_mm
            )
            return board_img_gray, squares_x * square_size_mm, squares_y * square_size_mm, info_lines

        settings = self._parse_charuco_settings()
        if settings is None:
            return None
        squares_x, squares_y, square_size_mm, marker_size_mm = settings

        board_img_gray = create_charuco_calibration_target.build_charuco_image(
            squares_x=squares_x,
            squares_y=squares_y,
            square_size_mm=square_size_mm,
            marker_size_mm=marker_size_mm,
            dictionary_id=perform_calibration.CHARUCO_DICTIONARY_ID,
            dpi=dpi,
            margin_mm=0,
        )
        info_lines = create_charuco_calibration_target.build_charuco_info_lines(
            squares_x, squares_y, square_size_mm, marker_size_mm
        )
        return board_img_gray, squares_x * square_size_mm, squares_y * square_size_mm, info_lines

    def _redraw_preview(self):
        """Rebuild and redraw the board preview from the current settings.

        Renders at `PREVIEW_DPI`, scales the result to fit
        `self.preview_label`'s current size, and draws it centered.
        Shows a plain text message instead if the current settings
        don't parse, or if the label has no usable size yet (e.g. the
        very first call, before Qt has laid out the window).

        Returns:
            None
        """
        if self.preview_label is None:
            return

        label_width = self.preview_label.width()
        label_height = self.preview_label.height()
        if label_width < 2 or label_height < 2:
            return

        built = self._build_current_board_image(dpi=PREVIEW_DPI)
        if built is None:
            self.preview_label.clear()
            self.preview_label.setText(
                "Enter valid squares (>= 2 in each direction) and a positive "
                "square size - and, for ChArUco, a marker size smaller than "
                "the square size."
            )
            self.preview_label.setWordWrap(True)
            self.preview_label.setStyleSheet(
                "background-color: #dddddd; border: 1px solid #999999; color: #555555; padding: 10px;"
            )
            return

        board_img_gray, _board_w_mm, _board_h_mm, _info_lines = built

        # Convert to RGB for Pillow, matching video_overlay.py's own
        # convention for showing OpenCV/NumPy image data via Qt.
        board_img_rgb = cv2.cvtColor(board_img_gray, cv2.COLOR_GRAY2RGB)
        board_h_px, board_w_px = board_img_rgb.shape[:2]

        preview_margin_px = 10
        scale = min(
            (label_width - preview_margin_px * 2) / board_w_px,
            (label_height - preview_margin_px * 2) / board_h_px,
        )
        scale = max(scale, 0.01)
        display_w_px = max(1, int(board_w_px * scale))
        display_h_px = max(1, int(board_h_px * scale))

        pil_image = Image.fromarray(board_img_rgb).resize((display_w_px, display_h_px), Image.NEAREST)

        self.preview_label.setStyleSheet("background-color: #dddddd; border: 1px solid #999999;")
        self.preview_label.setText("")
        self.preview_label.setPixmap(pil_image_to_qpixmap(pil_image))

    def on_save_printable_board(self):
        """Save the currently selected board as a printable LETTER landscape PDF.

        Renders at 300 DPI (matching both generator scripts' own
        `main()`), then reuses `write_pdf_letter_landscape` to place it
        on the page with a 100mm scale bar and print instructions - see
        this module's docstring for why the rendering itself isn't
        duplicated here.

        Returns:
            None
        """
        board_type = self.board_type_combo.currentText()

        built = self._build_current_board_image(dpi=300)
        if built is None:
            QMessageBox.critical(
                self.win,
                "Generate Calibration Target",
                "Enter valid squares (at least 2 in each direction) and a positive "
                "square size in millimeters - and, for ChArUco, a marker size "
                "smaller than the square size.",
            )
            return
        board_img_gray, board_w_mm, board_h_mm, info_lines = built

        default_name = "checkerboard_letter_landscape.pdf" if board_type == "Checkerboard" else "charuco_letter_landscape.pdf"

        file_path, _filter = QFileDialog.getSaveFileName(
            self.win, "Save Printable Calibration Board", default_name, "PDF Files (*.pdf);;All Files (*)"
        )
        if not file_path:
            return

        # Both generator modules' write_pdf_letter_landscape functions are
        # identical (board-agnostic page layout) - use whichever module
        # matches the board type actually being saved, purely so the call
        # site reads naturally rather than saving a ChArUco board "through"
        # the checkerboard module.
        writer = (
            create_checkerboard_calibration_target
            if board_type == "Checkerboard"
            else create_charuco_calibration_target
        )

        try:
            writer.write_pdf_letter_landscape(
                out_pdf_path=file_path,
                board_img_gray=board_img_gray,
                board_w_mm=board_w_mm,
                board_h_mm=board_h_mm,
                margin_in=PDF_PAGE_MARGIN_IN,
                info_lines=info_lines,
            )
        except RuntimeError as error:
            QMessageBox.critical(self.win, "Generate Calibration Target", str(error))
            return

        self.status_label.setText(f"Saved: {file_path}")
