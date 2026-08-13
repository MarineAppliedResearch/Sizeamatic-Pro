"""Generate Calibration Target window for Sizeamatic Pro.

Lets the project owner generate a printable checkerboard or ChArUco
calibration target from inside the app itself (ROADMAP.md Phase 10),
rather than needing to run a separate command-line script - matching a
UI precedent found on the unmerged `origin/feature-onlineCalibrations`
branch, which built board generation directly into its
`perform_calibration.py` window (settings + a live on-screen preview +
a "Generate Printable Board" save dialog), never as a standalone script.

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

    The live preview canvas renders at a low, fixed `PREVIEW_DPI`
    (fast enough to redraw on every keystroke) using Pillow's
    `Image.fromarray`/`ImageTk.PhotoImage`, the same approach
    `main.py`'s `_display_bgr_on_canvas` already settled on for
    rendering OpenCV/NumPy image data onto a Tk canvas - and NOT the
    `tkinter.PhotoImage(data=base64_png)` round-trip
    `origin/feature-onlineCalibrations`'s equivalent preview used,
    which `main.py`'s own docstring already flags as the exact thing
    that made an earlier version of this app's video rendering
    "unacceptably slow" (ROADMAP.md Phase 5) before it was replaced.
    `self.preview_photo_image` holds a live reference to the current
    `ImageTk.PhotoImage` for the same reason `main.py` keeps one too -
    Tkinter doesn't keep its own reference, so a locally-scoped
    `PhotoImage` would get garbage collected and the canvas would show
    nothing.

Assumptions:
    - The main application exposes `app.root` (the Tk root window that
      owns this Toplevel) and `app._app_window_title` (this app's
      project-aware window title, shared by every window). Both defined
      on `main.py`'s `SizeamaticProApp`.

Author:
    Isaac Travers

Date:
    2026-08-13
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import create_charuco_calibration_target
import create_checkerboard_calibration_target
import perform_calibration

PREVIEW_DPI = 96
"""Render resolution used only for this window's on-screen preview -
low enough to rebuild the board image instantly on every settings
change, unlike the 300 DPI `on_save_printable_board` renders the actual
printable PDF at."""

PDF_PAGE_MARGIN_IN = 0.5
"""PDF page margin (inches) passed to `write_pdf_letter_landscape` for
a saved board - matches both generator scripts' own `main()` default."""


class GenerateCalibrationTargetWindow:
    """Owns the Generate Calibration Target Toplevel window and its widgets.

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
        """The Generate Calibration Target Toplevel window, or None if
        it hasn't been opened yet (or was closed)."""

        self.board_type_var = None
        """`tk.StringVar` holding which board type is currently
        selected - `"Checkerboard"` or `"ChArUco"`. Created in
        `ensure_window`."""

        self.checkerboard_squares_x_var = None
        """`tk.StringVar` holding the checkerboard's width in squares,
        defaulting to `perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X`.
        Created in `ensure_window`."""

        self.checkerboard_squares_y_var = None
        """`tk.StringVar` holding the checkerboard's height in squares.
        See `self.checkerboard_squares_x_var`."""

        self.checkerboard_square_size_var = None
        """`tk.StringVar` holding the checkerboard's real-world square
        size in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM`.
        Created in `ensure_window`."""

        self.charuco_squares_x_var = None
        """`tk.StringVar` holding the ChArUco board's width in squares,
        defaulting to `perform_calibration.DEFAULT_CHARUCO_SQUARES_X`.
        Created in `ensure_window`."""

        self.charuco_squares_y_var = None
        """`tk.StringVar` holding the ChArUco board's height in squares.
        See `self.charuco_squares_x_var`."""

        self.charuco_square_size_var = None
        """`tk.StringVar` holding the ChArUco board's real-world square
        size in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM`. Created in
        `ensure_window`."""

        self.charuco_marker_size_var = None
        """`tk.StringVar` holding the ChArUco board's real-world ArUco
        marker size in millimeters, defaulting to
        `perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM` - must be
        smaller than `self.charuco_square_size_var`
        (`_parse_charuco_settings` enforces this). Created in
        `ensure_window`."""

        self.checkerboard_row = None
        """The `ttk.Frame` holding the checkerboard-specific settings -
        shown only while `self.board_type_var` is `"Checkerboard"`.
        Created in `ensure_window`."""

        self.charuco_row = None
        """The `ttk.Frame` holding the ChArUco-specific settings - shown
        only while `self.board_type_var` is `"ChArUco"`. Created in
        `ensure_window`."""

        self.preview_canvas = None
        """The `tkinter.Canvas` the current board settings are rendered
        onto. Created in `ensure_window`."""

        self.preview_photo_image = None
        """The `PIL.ImageTk.PhotoImage` currently drawn on
        `self.preview_canvas` - see this module's docstring for why
        this reference must be kept alive."""

        self.status_var = None
        """`tk.StringVar` holding the window's status line, e.g. "Saved:
        C:/.../checkerboard_letter_landscape.pdf". Created in
        `ensure_window`."""

    def _on_close(self):
        """Handle the user manually closing the window.

        Destroys the Tkinter window and clears the stored widget
        references, matching `PerformCalibrationWindow._on_close`'s
        reasoning - the next `ensure_window` call needs to rebuild them
        rather than holding onto references to already-destroyed
        widgets.

        Returns:
            None
        """
        self.win.destroy()
        self.win = None
        self.board_type_var = None
        self.checkerboard_squares_x_var = None
        self.checkerboard_squares_y_var = None
        self.checkerboard_square_size_var = None
        self.charuco_squares_x_var = None
        self.charuco_squares_y_var = None
        self.charuco_square_size_var = None
        self.charuco_marker_size_var = None
        self.checkerboard_row = None
        self.charuco_row = None
        self.preview_canvas = None
        self.preview_photo_image = None
        self.status_var = None

    def ensure_window(self):
        """Create the Generate Calibration Target window, or raise it if it exists.

        Returns:
            None
        """
        # If the window already exists, bring it to the front and reuse it
        # instead of creating a duplicate window.
        if self.win is not None:
            try:
                self.win.lift()
                return

            # If the stored window reference is stale, clear it so a new
            # window can be created below.
            except Exception:
                self.win = None

        win = tk.Toplevel(self.app.root)
        win.title(self.app._app_window_title())
        win.geometry("560x600")
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = ttk.Frame(win, padding=(10, 10))
        outer.grid(row=0, column=0, sticky="nsew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        # ---- Heading ----
        ttk.Label(
            outer,
            text="Generate Calibration Target",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # ---- Board type row ----
        type_row = ttk.Frame(outer)
        type_row.grid(row=1, column=0, sticky="w", pady=(10, 0))

        ttk.Label(type_row, text="Board type:").grid(row=0, column=0)
        self.board_type_var = tk.StringVar(value="Checkerboard")
        board_type_combo = ttk.Combobox(
            type_row,
            textvariable=self.board_type_var,
            values=["Checkerboard", "ChArUco"],
            width=12,
            state="readonly",
        )
        board_type_combo.grid(row=0, column=1, padx=(4, 0))
        board_type_combo.bind("<<ComboboxSelected>>", lambda event: self._on_settings_changed())

        # ---- Checkerboard settings row ----
        # Editable - see this module's docstring for why checkerboard
        # (unlike ChArUco) is safe to let the project owner customize.
        self.checkerboard_row = ttk.Frame(outer)
        self.checkerboard_row.grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Label(self.checkerboard_row, text="Squares (columns x rows):").grid(row=0, column=0)
        self.checkerboard_squares_x_var = tk.StringVar(
            value=str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_X)
        )
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_squares_x_var, width=4).grid(
            row=0, column=1, padx=(4, 2)
        )
        ttk.Label(self.checkerboard_row, text="x").grid(row=0, column=2)
        self.checkerboard_squares_y_var = tk.StringVar(
            value=str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARES_Y)
        )
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_squares_y_var, width=4).grid(
            row=0, column=3, padx=(2, 12)
        )

        ttk.Label(self.checkerboard_row, text="Square size (mm):").grid(row=0, column=4)
        self.checkerboard_square_size_var = tk.StringVar(
            value=str(perform_calibration.DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM)
        )
        ttk.Entry(self.checkerboard_row, textvariable=self.checkerboard_square_size_var, width=8).grid(
            row=0, column=5, padx=(4, 0)
        )

        # Redraw the preview whenever any checkerboard setting changes -
        # matching the write-trace pattern
        # `origin/feature-onlineCalibrations`'s own live preview used.
        for settings_var in (
            self.checkerboard_squares_x_var,
            self.checkerboard_squares_y_var,
            self.checkerboard_square_size_var,
        ):
            settings_var.trace_add("write", lambda *args: self._redraw_preview())

        # ---- ChArUco settings row ----
        # Editable, same as checkerboard - see this module's docstring
        # for why ChArUco's geometry is user-set too, not fixed.
        self.charuco_row = ttk.Frame(outer)
        self.charuco_row.grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Label(self.charuco_row, text="Squares (columns x rows):").grid(row=0, column=0)
        self.charuco_squares_x_var = tk.StringVar(value=str(perform_calibration.DEFAULT_CHARUCO_SQUARES_X))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_squares_x_var, width=4).grid(
            row=0, column=1, padx=(4, 2)
        )
        ttk.Label(self.charuco_row, text="x").grid(row=0, column=2)
        self.charuco_squares_y_var = tk.StringVar(value=str(perform_calibration.DEFAULT_CHARUCO_SQUARES_Y))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_squares_y_var, width=4).grid(
            row=0, column=3, padx=(2, 12)
        )

        ttk.Label(self.charuco_row, text="Square (mm):").grid(row=0, column=4)
        self.charuco_square_size_var = tk.StringVar(value=str(perform_calibration.DEFAULT_CHARUCO_SQUARE_SIZE_MM))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_square_size_var, width=6).grid(
            row=0, column=5, padx=(4, 12)
        )

        ttk.Label(self.charuco_row, text="Marker (mm):").grid(row=0, column=6)
        self.charuco_marker_size_var = tk.StringVar(value=str(perform_calibration.DEFAULT_CHARUCO_MARKER_SIZE_MM))
        ttk.Entry(self.charuco_row, textvariable=self.charuco_marker_size_var, width=6).grid(
            row=0, column=7, padx=(4, 0)
        )

        # Redraw the preview whenever any ChArUco setting changes -
        # same write-trace pattern as the checkerboard fields above.
        for settings_var in (
            self.charuco_squares_x_var,
            self.charuco_squares_y_var,
            self.charuco_square_size_var,
            self.charuco_marker_size_var,
        ):
            settings_var.trace_add("write", lambda *args: self._redraw_preview())

        # ---- Live preview canvas ----
        preview_frame = ttk.Frame(outer)
        preview_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_frame,
            background="#dddddd",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda event: self._redraw_preview())

        # ---- Save button ----
        button_row = ttk.Frame(outer)
        button_row.grid(row=4, column=0, sticky="w", pady=(10, 0))

        ttk.Button(
            button_row,
            text="Save Printable Board…",
            command=self.on_save_printable_board,
        ).grid(row=0, column=0)

        # ---- Status line ----
        self.status_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )

        # Let the preview canvas get the extra space when the window
        # resizes; everything above/below it stays a fixed height.
        outer.grid_rowconfigure(3, weight=1)

        self.win = win

        # Show only the settings row matching the current board type, and
        # draw the first preview.
        self._on_settings_changed()

    def _on_settings_changed(self):
        """Show the settings row matching the selected board type and redraw.

        Called whenever `self.board_type_var` changes - swaps which of
        `self.checkerboard_row`/`self.charuco_row` is visible (both are
        gridded onto the same row, so only one is ever showing), then
        redraws the preview for the newly selected board type.

        Returns:
            None
        """
        if self.board_type_var.get() == "Checkerboard":
            self.charuco_row.grid_remove()
            self.checkerboard_row.grid()
        else:
            self.checkerboard_row.grid_remove()
            self.charuco_row.grid()

        self._redraw_preview()

    def _parse_checkerboard_settings(self):
        """Parse and validate the checkerboard squares/size fields.

        Deliberately returns `None` on any problem instead of showing a
        `messagebox` - this is called on every keystroke to refresh the
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
            squares_x = int(self.checkerboard_squares_x_var.get())
            squares_y = int(self.checkerboard_squares_y_var.get())
        except ValueError:
            return None
        if squares_x < 2 or squares_y < 2:
            return None

        try:
            square_size_mm = float(self.checkerboard_square_size_var.get())
        except ValueError:
            return None
        if square_size_mm <= 0:
            return None

        return squares_x, squares_y, square_size_mm

    def _parse_charuco_settings(self):
        """Parse and validate the ChArUco squares/square size/marker size fields.

        Mirrors `_parse_checkerboard_settings` - deliberately returns
        `None` on any problem instead of showing a `messagebox`, since
        this is also called on every keystroke to refresh the live
        preview.

        Returns:
            tuple[int, int, float, float] | None: `(squares_x, squares_y,
            square_size_mm, marker_size_mm)` if every field is valid,
            else None.
        """
        try:
            squares_x = int(self.charuco_squares_x_var.get())
            squares_y = int(self.charuco_squares_y_var.get())
        except ValueError:
            return None
        if squares_x < 2 or squares_y < 2:
            return None

        try:
            square_size_mm = float(self.charuco_square_size_var.get())
            marker_size_mm = float(self.charuco_marker_size_var.get())
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
        if self.board_type_var.get() == "Checkerboard":
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
        `self.preview_canvas`'s current size, and draws it centered.
        Shows a plain text message instead if the current settings
        don't parse, or if the canvas has no usable size yet (e.g. the
        very first call, before Tkinter has laid out the window).

        Returns:
            None
        """
        if self.preview_canvas is None:
            return

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            return

        self.preview_canvas.delete("all")

        built = self._build_current_board_image(dpi=PREVIEW_DPI)
        if built is None:
            self.preview_canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text=(
                    "Enter valid squares (>= 2 in each direction) and a positive "
                    "square size - and, for ChArUco, a marker size smaller than "
                    "the square size."
                ),
                fill="#555555",
                width=canvas_width - 20,
                justify="center",
            )
            return

        board_img_gray, _board_w_mm, _board_h_mm, _info_lines = built

        # Convert to RGB for Pillow, matching main.py's own
        # `_display_bgr_on_canvas` convention for showing OpenCV/NumPy
        # image data on a Tk canvas.
        board_img_rgb = cv2.cvtColor(board_img_gray, cv2.COLOR_GRAY2RGB)
        board_h_px, board_w_px = board_img_rgb.shape[:2]

        preview_margin_px = 10
        scale = min(
            (canvas_width - preview_margin_px * 2) / board_w_px,
            (canvas_height - preview_margin_px * 2) / board_h_px,
        )
        scale = max(scale, 0.01)
        display_w_px = max(1, int(board_w_px * scale))
        display_h_px = max(1, int(board_h_px * scale))

        pil_image = Image.fromarray(board_img_rgb).resize((display_w_px, display_h_px), Image.NEAREST)
        self.preview_photo_image = ImageTk.PhotoImage(pil_image)

        self.preview_canvas.create_image(
            canvas_width / 2,
            canvas_height / 2,
            image=self.preview_photo_image,
            anchor="center",
        )

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
        board_type = self.board_type_var.get()

        built = self._build_current_board_image(dpi=300)
        if built is None:
            messagebox.showerror(
                "Generate Calibration Target",
                "Enter valid squares (at least 2 in each direction) and a positive "
                "square size in millimeters - and, for ChArUco, a marker size "
                "smaller than the square size.",
            )
            return
        board_img_gray, board_w_mm, board_h_mm, info_lines = built

        default_name = "checkerboard_letter_landscape.pdf" if board_type == "Checkerboard" else "charuco_letter_landscape.pdf"

        file_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="Save Printable Calibration Board",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
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
            messagebox.showerror("Generate Calibration Target", str(error))
            return

        self.status_var.set(f"Saved: {file_path}")
