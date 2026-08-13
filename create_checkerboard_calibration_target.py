"""Checkerboard calibration target generator.

Generates a plain checkerboard calibration target as a LETTER
landscape PDF at true scale (ROADMAP.md Phase 10) - the checkerboard
counterpart to `create_charuco_calibration_target.py`, restoring the
board-generation capability the unmerged `feature-onlineCalibrations`
branch had for both board types, not just ChArUco. Includes a 100 mm
scale bar so you can verify the print came out correctly.

Default board settings here (`squares_x`/`squares_y`/`square_size_mm`
in `main()`) match `perform_calibration.py`'s
`DEFAULT_CHECKERBOARD_SQUARES_X`/`DEFAULT_CHECKERBOARD_SQUARES_Y`/
`DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM` - printing with this script's
defaults and calibrating with the Perform Calibration window's own
defaults line up without needing to change either side, the same
relationship `create_charuco_calibration_target.py`'s defaults already
have with `perform_calibration.py`'s `DEFAULT_CHARUCO_SQUARES_X`/etc.

Author:
    Isaac Travers

Date:
    2026-08-13
"""

import io
import os

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def mm_to_points(mm: float) -> float:
    """Convert millimeters to PDF points.

    Args:
        mm (float): Length in millimeters.

    Returns:
        float: Length in PDF points (1 inch = 72 points).
    """

    # 25.4 mm per inch, 72 points per inch.
    return (mm / 25.4) * 72.0


def build_checkerboard_image(squares_x: int,
                             squares_y: int,
                             square_size_mm: float,
                             dpi: int) -> np.ndarray:
    """Build a high resolution checkerboard image suitable for printing.

    Args:
        squares_x (int): Number of checkerboard squares along the X axis.
        squares_y (int): Number of checkerboard squares along the Y axis.
        square_size_mm (float): Physical size of each square, in
            millimeters.
        dpi (int): Render resolution, in dots per inch.

    Returns:
        numpy.ndarray: A grayscale uint8 image (0..255) of the rendered
        board - alternating black/white squares, no margin (the caller
        places it within the PDF page margin instead).
    """

    # Convert one square's physical size to pixels at the requested DPI.
    square_size_px = int((square_size_mm / 25.4) * dpi)

    # Compute the full board size in pixels.
    board_w_px = squares_x * square_size_px
    board_h_px = squares_y * square_size_px

    # Start with an all-black image, then paint every other square white -
    # the same alternating pattern cv2.findChessboardCorners expects.
    img = np.zeros((board_h_px, board_w_px), dtype=np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y0 = row * square_size_px
                x0 = col * square_size_px
                img[y0:y0 + square_size_px, x0:x0 + square_size_px] = 255

    return img


def build_checkerboard_info_lines(squares_x: int, squares_y: int, square_size_mm: float) -> list:
    """Build the small-text lines describing an actual printed checkerboard.

    Written directly onto the printed page (`write_pdf_letter_landscape`'s
    `info_lines`) so a physical printout - found later, or handed to
    someone else - states its own exact settings rather than relying on
    whoever printed it to remember or re-derive them (the same problem
    this phase's checkerboard-size-configurability work was created to
    fix in the first place - see `perform_calibration.py`'s docstring).

    Args:
        squares_x (int): Number of checkerboard squares along the X axis.
        squares_y (int): Number of checkerboard squares along the Y axis.
        square_size_mm (float): Physical size of each square, in
            millimeters.

    Returns:
        list[str]: One or more lines of small print-on-page text.
    """
    return [
        f"Checkerboard: {squares_x} x {squares_y} squares "
        f"({squares_x - 1} x {squares_y - 1} inside corners), "
        f"{square_size_mm:.1f} mm per square"
    ]


def write_pdf_letter_landscape(out_pdf_path: str,
                               board_img_gray: np.ndarray,
                               board_w_mm: float,
                               board_h_mm: float,
                               margin_in: float,
                               info_lines: list) -> None:
    """Write a board image into a LETTER landscape PDF at true physical size.

    Centers the board within the printable area and adds a 100 mm scale
    bar plus a printing-instructions note for print verification.
    Board-agnostic - identical to
    `create_charuco_calibration_target.py`'s function of the same name,
    since placing an already-rendered board image on the page doesn't
    depend on what kind of board it is.

    Args:
        out_pdf_path (str): Destination PDF file path.
        board_img_gray (numpy.ndarray): Grayscale board image, as
            returned by `build_checkerboard_image`.
        board_w_mm (float): Physical board width, in millimeters.
        board_h_mm (float): Physical board height, in millimeters.
        margin_in (float): PDF page margin around the board, in inches.
        info_lines (list[str]): Small-text lines describing the actual
            board settings used (see `build_checkerboard_info_lines`),
            printed on the page itself rather than left implicit.

    Raises:
        RuntimeError: If the board does not fit within the usable page
            area given `margin_in`, or if the board image fails to encode
            as PNG.

    Returns:
        None
    """

    # Use a fixed page size: US Letter in landscape orientation.
    page_w_pt, page_h_pt = landscape(letter)

    # Create a PDF canvas.
    c = canvas.Canvas(out_pdf_path, pagesize=(page_w_pt, page_h_pt))

    # Define page margins in points.
    margin_pt = margin_in * inch

    # Convert board physical size to points.
    board_w_pt = mm_to_points(board_w_mm)
    board_h_pt = mm_to_points(board_h_mm)

    # Compute a centered placement within the printable region.
    usable_w_pt = page_w_pt - 2.0 * margin_pt
    usable_h_pt = page_h_pt - 2.0 * margin_pt

    # If the board does not fit, fail loudly.
    if board_w_pt > usable_w_pt or board_h_pt > usable_h_pt:
        raise RuntimeError(
            f"Board does not fit on LETTER landscape with margin {margin_in} in. "
            f"Board {board_w_mm:.1f}x{board_h_mm:.1f} mm exceeds usable area."
        )

    # Place the board centered in the usable area.
    x_pt = margin_pt + (usable_w_pt - board_w_pt) * 0.5
    y_pt = margin_pt + (usable_h_pt - board_h_pt) * 0.5

    # Convert the OpenCV image to PNG bytes so ReportLab can embed it.
    ok, png = cv2.imencode(".png", board_img_gray)
    if not ok:
        raise RuntimeError("Failed to encode board image as PNG.")

    img_reader = ImageReader(io.BytesIO(png.tobytes()))

    # Draw the board image at exact physical size on the PDF.
    c.drawImage(img_reader, x_pt, y_pt, width=board_w_pt, height=board_h_pt, mask="auto")

    # Draw a 100 mm scale bar near the bottom left for print verification.
    scale_mm = 100.0
    scale_pt = mm_to_points(scale_mm)

    bar_x0 = margin_pt
    bar_y0 = margin_pt * 0.6

    c.setLineWidth(2)

    c.line(bar_x0, bar_y0, bar_x0 + scale_pt, bar_y0)

    c.setFont("Helvetica", 10)

    c.drawString(bar_x0, bar_y0 + 10, "Scale check: 100 mm (measure this line after printing)")

    # Add a short note about print settings.
    c.setFont("Helvetica", 9)

    c.drawString(margin_pt, page_h_pt - margin_pt * 0.7, "Print at 100% / Actual size. Disable Fit to page if possible.")

    # Write the actual board settings used, as small text on the page
    # itself, right below the print-settings note.
    c.setFont("Helvetica", 8)

    info_line_y = page_h_pt - margin_pt * 0.7 - 11
    for info_line in info_lines:
        c.drawString(margin_pt, info_line_y, info_line)
        info_line_y -= 10

    # Finalize the PDF.
    c.showPage()
    c.save()


def main() -> None:
    """Build the checkerboard and write checkerboard_letter_landscape.pdf.

    Uses a fixed 10x7 square board layout sized to fit LETTER landscape
    with margins, writes the output PDF into the gitignored `output/`
    folder (creating it if needed), and prints the output path. These
    defaults (10x7 squares, 25mm each) match
    `perform_calibration.py`'s `DEFAULT_CHECKERBOARD_SQUARES_X`/
    `DEFAULT_CHECKERBOARD_SQUARES_Y`/`DEFAULT_CHECKERBOARD_SQUARE_SIZE_MM`
    - see this module's docstring for why that matters.

    Returns:
        None
    """

    # Board layout in squares (not inner corners).
    squares_x = 10
    squares_y = 7

    # Physical square size in mm.
    # NOTE: chosen to fit LETTER landscape with margins.
    square_size_mm = 25.0

    # Render resolution.
    dpi = 300

    # PDF page margin around the board, in inches.
    margin_in = 0.5

    # Output file. Generated artifacts live in output/, which is gitignored.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    out_pdf_path = os.path.join(output_dir, "checkerboard_letter_landscape.pdf")

    # Build the board image.
    board_img = build_checkerboard_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_mm=square_size_mm,
        dpi=dpi
    )

    # Compute board physical size for placement.
    board_w_mm = squares_x * square_size_mm
    board_h_mm = squares_y * square_size_mm

    # Write the PDF at true physical size, with the actual board settings
    # printed on the page itself.
    write_pdf_letter_landscape(
        out_pdf_path=out_pdf_path,
        board_img_gray=board_img,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        margin_in=margin_in,
        info_lines=build_checkerboard_info_lines(squares_x, squares_y, square_size_mm)
    )

    print("Wrote:", out_pdf_path)

    print("After printing, measure the 100 mm line.")

    print(
        f"Board is {squares_x}x{squares_y} squares "
        f"({squares_x - 1}x{squares_y - 1} inner corners) - "
        "enter these same numbers in Perform Calibration's checkerboard settings."
    )


if __name__ == "__main__":
    main()
