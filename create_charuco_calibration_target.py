# =============================================================================
# make_charuco_pdf_letter_landscape.py
#
# Generates a ChArUco calibration target as a LETTER landscape PDF at true scale.
# Includes a 100 mm scale bar so you can verify the print came out correctly.
#
# Author: Isaac Travers
# Date: 2026-03-01
# =============================================================================

import io

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------------------------------------------------------
# mm_to_points
#
# Convert millimeters to PDF points (1 inch = 72 points).
# -----------------------------------------------------------------------------
def mm_to_points(mm: float) -> float:

    # 25.4 mm per inch, 72 points per inch.
    return (mm / 25.4) * 72.0


# -----------------------------------------------------------------------------
# build_charuco_image
#
# Builds a high resolution ChArUco board image suitable for printing.
# Returns a grayscale uint8 image (0..255).
# -----------------------------------------------------------------------------
def build_charuco_image(squares_x: int,
                        squares_y: int,
                        square_size_mm: float,
                        marker_size_mm: float,
                        dictionary_id: int,
                        dpi: int,
                        margin_mm: float) -> np.ndarray:

    # Create the ArUco dictionary that defines the marker family.
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # Define the ChArUco board geometry in real units (mm).
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_mm,
        marker_size_mm,
        dictionary
    )

    # Compute physical board size in mm (board area only, excluding margins).
    board_w_mm = squares_x * square_size_mm
    board_h_mm = squares_y * square_size_mm

    # Convert physical size to pixels at the requested DPI.
    board_w_px = int((board_w_mm / 25.4) * dpi)
    board_h_px = int((board_h_mm / 25.4) * dpi)

    # Convert margin from mm to pixels.
    margin_px = int((margin_mm / 25.4) * dpi)

    # Render the board to an image.
    img = board.generateImage(
        (board_w_px, board_h_px),
        marginSize=margin_px,
        borderBits=1
    )

    return img


# -----------------------------------------------------------------------------
# write_pdf_letter_landscape
#
# Writes the given board image into a LETTER landscape PDF at true physical size.
# Adds a 100 mm scale bar for print verification.
# -----------------------------------------------------------------------------
def write_pdf_letter_landscape(out_pdf_path: str,
                               board_img_gray: np.ndarray,
                               board_w_mm: float,
                               board_h_mm: float,
                               margin_in: float) -> None:

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

    # Finalize the PDF.
    c.showPage()
    c.save()


# -----------------------------------------------------------------------------
# main
#
# Generates charuco_letter_landscape.pdf for printing.
# -----------------------------------------------------------------------------
def main() -> None:

    # Board layout in squares (not corners).
    squares_x = 11
    squares_y = 8

    # Physical sizes in mm.
    # NOTE: These values are chosen to fit LETTER landscape with margins.
    square_size_mm = 20.0
    marker_size_mm = 15.0

    # Marker dictionary.
    dictionary_id = cv2.aruco.DICT_4X4_1000

    # Render resolution.
    dpi = 300

    # White margin inside the rendered image, in mm.
    margin_mm = 0

    # PDF page margin around the board, in inches.
    margin_in = 0.5

    # Output file.
    out_pdf_path = "charuco_letter_landscape.pdf"

    # Build the board image.
    board_img = build_charuco_image(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size_mm=square_size_mm,
        marker_size_mm=marker_size_mm,
        dictionary_id=dictionary_id,
        dpi=dpi,
        margin_mm=margin_mm
    )

    # Compute board physical size for placement.
    board_w_mm = squares_x * square_size_mm
    board_h_mm = squares_y * square_size_mm

    # Write the PDF at true physical size.
    write_pdf_letter_landscape(
        out_pdf_path=out_pdf_path,
        board_img_gray=board_img,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        margin_in=margin_in
    )

    print("Wrote:", out_pdf_path)

    print("After printing, measure the 100 mm line.")


if __name__ == "__main__":
    main()