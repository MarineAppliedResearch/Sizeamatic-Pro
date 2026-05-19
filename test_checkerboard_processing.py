# =============================================================================
# File: test_checkerboard_processing.py
# Calibration Test 1
# Date: 2026-05-19
# Author: Isaac Travers
#
# Purpose:
#   Test process a folder of stereo checkerboard calibration images.
#   Each image is loaded with OpenCV, checkerboard corners are detected, the
#   detected pattern is drawn onto the image, and a processed copy is written
#   into a test_processed subfolder.
# =============================================================================

from pathlib import Path
import cv2
from tkinter import filedialog, messagebox


# Checkerboard geometry.
CHECKERBOARD = (9, 6)      # Number of interior corners, columns by rows.
SQUARE_SIZE_MM = 23.5      # Physical square size in millimeters.


# =============================================================================
# draw_text_line
#
# Draws one readable metadata line onto an OpenCV image.
#
# Inputs:
#   image: OpenCV BGR image
#   text: text to draw
#   x: left position in pixels
#   y: baseline position in pixels
# =============================================================================
def draw_text_line(image, text, x, y):

    # Use a compact OpenCV font that stays readable on saved test images.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    # Measure the text so we can draw a dark backing rectangle behind it.
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw a dark rectangle behind the text for contrast.
    cv2.rectangle(
        image,
        (x - 4, y - text_size[1] - 4),
        (x + text_size[0] + 4, y + baseline + 4),
        (0, 0, 0),
        -1
    )

    # Draw the actual text in white.
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =============================================================================
# test_process_checkerboard_folder
#
# Asks the user to select a folder of checkerboard calibration images, processes
# each image with OpenCV checkerboard detection, and saves annotated copies into
# a test_processed subfolder.
#
# Inputs:
#   parent: optional Tkinter parent window
#
# Returns:
#   output_folder path if processing completed
#   None if the user canceled or no images were processed
# =============================================================================
def test_process_checkerboard_folder(parent=None):

    # Ask the user to select the folder containing calibration images.
    folder = filedialog.askdirectory(
        parent=parent,
        title="Select stereo checkerboard calibration image folder"
    )

    # Stop cleanly if the user cancels the folder picker.
    if not folder:
        return None

    # Convert the selected folder into a pathlib Path.
    input_folder = Path(folder)

    # Create the output folder inside the selected folder.
    output_folder = input_folder / "test_processed"
    output_folder.mkdir(exist_ok=True)

    # Accept common image file formats.
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    # Find all image files directly inside the selected folder.
    image_paths = [
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    # Stop if the folder does not contain usable image files.
    if not image_paths:
        messagebox.showwarning(
            "No Images Found",
            "No image files were found in the selected folder."
        )
        return None

    # Track processing results for the final message.
    processed_count = 0
    detected_count = 0
    failed_count = 0

    # Process images in name order so left/right stereo pairs stay grouped.
    for image_path in sorted(image_paths):

        # Load the image using OpenCV.
        image = cv2.imread(str(image_path))

        # Skip unreadable files instead of crashing the whole batch.
        if image is None:
            failed_count += 1
            continue

        # Get useful image information for the overlay.
        height, width = image.shape[:2]

        # Convert to grayscale for checkerboard detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use the newer OpenCV checkerboard detector when available.
        if hasattr(cv2, "findChessboardCornersSB"):

            # Detect checkerboard corners with the more robust SB detector.
            found, corners = cv2.findChessboardCornersSB(
                gray,
                CHECKERBOARD,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )

        else:

            # Use the classic detector as a fallback.
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                    + cv2.CALIB_CB_FAST_CHECK
                )
            )

            # Refine classic detector corners to subpixel accuracy.
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001
                    )
                )

        # Make a copy that we can safely draw onto.
        annotated = image.copy()

        # Draw OpenCV's detected checkerboard corner overlay.
        if found:
            cv2.drawChessboardCorners(
                annotated,
                CHECKERBOARD,
                corners,
                found
            )

            detected_count += 1

        else:

            # Mark failed detections clearly on the image.
            cv2.rectangle(
                annotated,
                (0, 0),
                (width - 1, height - 1),
                (0, 0, 255),
                6
            )

        # Build the text overlay.
        status_text = "DETECTED" if found else "NOT DETECTED"
        corner_count = len(corners) if found and corners is not None else 0

        # Draw OpenCV and calibration metadata along the bottom.
        y0 = height - 82
        line_gap = 24

        draw_text_line(
            annotated,
            f"Marine Applied Research & Exploration - MoorSea Project | OpenCV {cv2.__version__}",
            12,
            y0
        )

        draw_text_line(
            annotated,
            f"Checkerboard inner corners: {CHECKERBOARD[0]} x {CHECKERBOARD[1]} | Square size: {SQUARE_SIZE_MM} mm | Image: {width} x {height}",
            12,
            y0 + line_gap
        )

        draw_text_line(
            annotated,
            f"File: {image_path.name} | Detection: {status_text} | Corners found: {corner_count}",
            12,
            y0 + (line_gap * 2)
        )

        # Build the output filename.
        output_path = output_folder / f"{image_path.stem}_processed{image_path.suffix}"

        # Save the annotated image.
        if cv2.imwrite(str(output_path), annotated):
            processed_count += 1
        else:
            failed_count += 1

    # Show a compact completion message.
    messagebox.showinfo(
        "Checkerboard Test Complete",
        (
            f"Processed images: {processed_count}\n"
            f"Checkerboards detected: {detected_count}\n"
            f"Failed or skipped files: {failed_count}\n\n"
            f"Output folder:\n{output_folder}"
        )
    )

    return output_folder