# -----------------------------------------------------------------------------
# perform_calibration.py
#
# Calibration workflow window for Sizeamatic Pro.
# -----------------------------------------------------------------------------

# Open saved printable board files with the operating system default handler.
import os

# Escape text that is written into generated SVG output.
import html

# Encode OpenCV-generated preview images so Tkinter can display them.
import base64

# Build the calibration workflow window using Tkinter.
import tkinter as tk

# Create bold fonts for calibration window labels and buttons.
import tkinter.font as tkfont

# Use themed Tkinter widgets for the calibration UI.
from tkinter import ttk

# Open folder and save file dialogs.
from tkinter import filedialog

# Show temporary status and error popups.
from tkinter import messagebox

# Import OpenCV
import cv2

# Import NumPy  because OpenCV image buffers are NumPy arrays.
import numpy as np

# Work with calibration image folders and generated run folders.
from pathlib import Path

# Match left_xxxx and right_xxxx stereo calibration filenames.
import re

# Save readable calibration run metadata for reports and future debugging.
import json

# Run OpenCV calibration processing without blocking the Tkinter UI.
import threading

# Pass calibration progress messages safely from worker threads to the GUI.
import queue

# Generate the HTML calibration report after calibration succeeds.
from generate_calibration_report import generate_calibration_report

# Open generated HTML calibration reports in the user's default browser.
import webbrowser


# -----------------------------------------------------------------------------
# ToolTip
#
# Creates a small popup help label for a Tkinter widget.
# -----------------------------------------------------------------------------

class ToolTip:

    # -------------------------------------------------------------------------
    # __init__
    #
    # Attaches tooltip behavior to a widget.
    # -------------------------------------------------------------------------

    def __init__(self, widget, text):

        # Store the widget this tooltip belongs to.
        self.widget = widget

        # Store the text shown inside the tooltip.
        self.text = text

        # Store the active tooltip window.
        self.tooltip_window = None

        # Show the tooltip when the mouse enters the widget.
        self.widget.bind("<Enter>", self._show_tooltip)

        # Hide the tooltip when the mouse leaves the widget.
        self.widget.bind("<Leave>", self._hide_tooltip)


    # -------------------------------------------------------------------------
    # _show_tooltip
    #
    # Displays the tooltip near the current widget.
    # -------------------------------------------------------------------------

    def _show_tooltip(self, event=None):

        # Do not create a second tooltip if one is already visible.
        if self.tooltip_window is not None:
            return

        # Get the widget position on screen.
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        # Create a small floating tooltip window.
        self.tooltip_window = tk.Toplevel(self.widget)

        # Remove the normal window border.
        self.tooltip_window.wm_overrideredirect(True)

        # Place the tooltip near the widget.
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create the tooltip label.
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=4,
            wraplength=360,
        )

        # Pack the tooltip label into the floating window.
        label.pack()


    # -------------------------------------------------------------------------
    # _hide_tooltip
    #
    # Hides the tooltip window.
    # -------------------------------------------------------------------------

    def _hide_tooltip(self, event=None):

        # Stop if there is no tooltip to destroy.
        if self.tooltip_window is None:
            return

        # Destroy the tooltip window.
        self.tooltip_window.destroy()

        # Clear the stored tooltip window reference.
        self.tooltip_window = None


# -----------------------------------------------------------------------------
# PerformCalibrationWindow
#
# Creates and manages the calibration workflow window.
# -----------------------------------------------------------------------------

class PerformCalibrationWindow:

    # -------------------------------------------------------------------------
    # __init__
    #
    # Initializes the calibration window and builds its controls.
    # -------------------------------------------------------------------------

    def __init__(self, parent, app=None):

        # Store the main Tk window.
        self.parent = parent

        # Store the main application object so this window can call shared app
        # functions when needed.
        self.app = app

        # Store the selected calibration board type.
        self.board_type = tk.StringVar(value="checkerboard")

        # Store checkerboard board settings.
        self.checkerboard_paper_size = tk.StringVar(value="Letter")
        self.checkerboard_orientation = tk.StringVar(value="Landscape")
        self.checkerboard_inner_rows = tk.IntVar(value=6)
        self.checkerboard_inner_columns = tk.IntVar(value=9)
        self.checkerboard_square_size_mm = tk.DoubleVar(value=23.5)

        # Store circle grid settings.
        self.circle_grid_rows = tk.IntVar(value=7)
        self.circle_grid_columns = tk.IntVar(value=10)
        self.circle_grid_spacing_mm = tk.DoubleVar(value=25.0)
        self.circle_grid_layout = tk.StringVar(value="symmetric")

         # Store Charuco board settings.
        self.charuco_paper_size = tk.StringVar(value="Letter")
        self.charuco_orientation = tk.StringVar(value="Landscape")
        self.charuco_detail = tk.StringVar(value="Medium")
        self.charuco_dictionary = tk.StringVar(value="DICT_4X4_50")

        # Store the selected stereo calibration image folder.
        self.image_folder = tk.StringVar(value="")

        # Create the top level calibration window.
        self.window = tk.Toplevel(self.parent)

        # Set the calibration window title.
        self.window.title("Sizeamatic Pro - Perform Calibration")

        # Set a reasonable starting size for the calibration window.
        self.window.geometry("850x520")

        # Keep the calibration window above the parent window when first opened.
        self.window.transient(self.parent)

        # Configure bold fonts for this window.
        self._configure_styles()

        # Build the window layout and controls.
        self._build_ui()

        # Draw the first board preview after Tkinter has calculated widget sizes.
        self.window.after(100, self._draw_board_preview)


    # -------------------------------------------------------------------------
    # _configure_styles
    #
    # Configures bold ttk styles used by the calibration window.
    # -------------------------------------------------------------------------

    def _configure_styles(self):

        # Copy the default Tk font so the size stays native to the system.
        self.bold_font = tkfont.nametofont("TkDefaultFont").copy()

        # Make the copied font bold.
        self.bold_font.configure(weight="bold")

        # Create a ttk style object for this window.
        self.style = ttk.Style(self.window)

        # Make ttk labels bold.
        self.style.configure("Bold.TLabel", font=self.bold_font)

        # Make ttk buttons bold.
        self.style.configure("Bold.TButton", font=self.bold_font)

        # Make ttk radio buttons bold.
        self.style.configure("Bold.TRadiobutton", font=self.bold_font)

        # Make ttk label frame titles bold.
        self.style.configure("Bold.TLabelframe.Label", font=self.bold_font)


    # -------------------------------------------------------------------------
    # _build_ui
    #
    # Builds the calibration workflow window.
    # -------------------------------------------------------------------------

    def _build_ui(self):

        # Create the main container for the calibration window.
        main_frame = ttk.Frame(self.window, padding=12)

        # Allow the main frame to fill the full calibration window.
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Allow the left settings area and right preview area to resize.
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Allow the bottom area to stay compact.
        main_frame.rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # Board setup section
        # ---------------------------------------------------------------------

        # Create the board setup frame.
        self.board_frame = ttk.LabelFrame(
            main_frame,
            text="Calibration Board",
            padding=10,
            style="Bold.TLabelframe",
        )

        # Place the board setup frame on the left side.
        self.board_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        # Let the settings area expand horizontally.
        self.board_frame.columnconfigure(0, weight=1)

        # Add the checkerboard radio button.
        checkerboard_radio = ttk.Radiobutton(
            self.board_frame,
            text="Checkerboard",
            variable=self.board_type,
            value="checkerboard",
            command=self._on_board_type_changed,
            style="Bold.TRadiobutton",
        )

        # Place the checkerboard radio button.
        checkerboard_radio.grid(row=0, column=0, sticky="w", pady=2)

        # Add tooltip help for checkerboard calibration.
        ToolTip(
            checkerboard_radio,
            "Checkerboard calibration is the standard choice. It is easy to print "
            "and works well when the board is flat and the inner corners are clear.",
        )

        # Add the circle grid radio button.
        circle_grid_radio = ttk.Radiobutton(
            self.board_frame,
            text="Circle Grid",
            variable=self.board_type,
            value="circle_grid",
            command=self._on_board_type_changed,
            style="Bold.TRadiobutton",
        )

        # Place the circle grid radio button.
        #circle_grid_radio.grid(row=1, column=0, sticky="w", pady=2)

        # Add tooltip help for circle grid calibration.
        ToolTip(
            circle_grid_radio,
            "Circle grids can be easier to detect in some blurry or lower contrast "
            "images. They usually need clean printed circles and correct spacing.",
        )

        # Add the Charuco radio button.
        charuco_radio = ttk.Radiobutton(
            self.board_frame,
            text="Charuco Board",
            variable=self.board_type,
            value="charuco",
            command=self._on_board_type_changed,
            style="Bold.TRadiobutton",
        )

        # Place the Charuco radio button.
        charuco_radio.grid(row=2, column=0, sticky="w", pady=2)

        # Add tooltip help for Charuco calibration.
        ToolTip(
            charuco_radio,
            "Charuco boards combine checkerboard corners with ArUco markers. They "
            "can work well with partial board views, but require OpenCV ArUco support.",
        )

        # Create a frame that will be rebuilt when the board type changes.
        self.board_settings_frame = ttk.Frame(self.board_frame)

        # Place the dynamic board settings frame.
        self.board_settings_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        # Build the first board settings panel.
        self._rebuild_board_settings()


        # ---------------------------------------------------------------------
        # Board preview section
        # ---------------------------------------------------------------------

        # Create the preview frame.
        preview_frame = ttk.LabelFrame(
            main_frame,
            text="Board Preview",
            padding=10,
            style="Bold.TLabelframe",
        )

        # Place the preview frame on the right side.
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))

        # Allow the preview frame to expand.
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # Create a canvas for the simplified board preview.
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=300,
            height=260,
            background="white",
            highlightthickness=1,
            highlightbackground="#999999",
            cursor="hand2",
        )

        # Place the board preview canvas.
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        # Redraw the preview whenever the preview canvas changes size.
        self.preview_canvas.bind("<Configure>", lambda event: self._draw_board_preview())

        # Allow the preview image itself to generate the printable board.
        self.preview_canvas.bind("<Button-1>", lambda event: self.on_generate_printable_board())

        # Add tooltip help for the preview canvas.
        ToolTip(
            self.preview_canvas,
            "Click the preview to generate the printable calibration board file.",
        )

        # Add a button below the preview to generate the printable board.
        generate_button = ttk.Button(
            preview_frame,
            text="Generate Printable Board",
            command=self.on_generate_printable_board,
            style="Bold.TButton",
        )

        # Place the generate button.
        generate_button.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Add tooltip help for the generate button.
        ToolTip(
            generate_button,
            "Creates a printable board file using the current board type and settings.",
        )


        # ---------------------------------------------------------------------
        # Calibration image folder section
        # ---------------------------------------------------------------------

        # Create the image folder area.
        image_frame = ttk.LabelFrame(
            main_frame,
            text="Stereo Calibration Images",
            padding=10,
            style="Bold.TLabelframe",
        )

        # Place the image folder area below the board controls.
        image_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Allow the folder path display to resize.
        image_frame.columnconfigure(0, weight=1)

        # Add a read only display for the selected image folder.
        image_folder_entry = ttk.Entry(
            image_frame,
            textvariable=self.image_folder,
            state="readonly",
            font=self.bold_font,
        )

        # Place the image folder display.
        image_folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        # Add the button for selecting the stereo calibration image folder.
        load_folder_button = ttk.Button(
            image_frame,
            text="Load Image Folder...",
            command=self.on_load_image_folder,
            style="Bold.TButton",
        )

                # Place the folder loading button.
        load_folder_button.grid(row=0, column=1, sticky="e", padx=(0, 8))

        # Add tooltip help for the expected calibration image folder contents.
        ToolTip(
            load_folder_button,
            "Select a folder containing matching stereo image pairs. Expected names "
            "look like left_3133.jpg and right_3133.jpg, where the number matches "
            "the left and right images.",
        )

        # Add the button for starting the calibration process.
        perform_calibration_button = ttk.Button(
            image_frame,
            text="Perform Stereo Calibration",
            command=self.on_run_calibration,
            style="Bold.TButton",
        )

        # Place the calibration button to the right of the image folder button.
        perform_calibration_button.grid(row=0, column=2, sticky="e")

        # Add tooltip help for the calibration button.
        ToolTip(
            perform_calibration_button,
            "This begins the calibration.",
        )


    # -------------------------------------------------------------------------
    # _rebuild_board_settings
    #
    # Rebuilds the settings controls for the selected board type.
    # -------------------------------------------------------------------------

    def _rebuild_board_settings(self):

        # Remove any previous board settings controls.
        for child in self.board_settings_frame.winfo_children():
            child.destroy()

        # Get the selected board type.
        board_type = self.board_type.get()

        # Build checkerboard settings when checkerboard is selected.
        if board_type == "checkerboard":
            self._add_combo_setting(
                row=0,
                label_text="Paper size",
                variable=self.checkerboard_paper_size,
                values=("Letter", "A4", "Custom"),
                tooltip_text=(
                    "Physical paper size used for the printable checkerboard. "
                    "Letter is 8.5 x 11 inches. A4 is 210 x 297 mm. Custom can "
                    "be expanded later if a non-standard print size is needed."
                ),
            )

            self._add_combo_setting(
                row=1,
                label_text="Orientation",
                variable=self.checkerboard_orientation,
                values=("Landscape", "Portrait"),
                tooltip_text=(
                    "Direction the board is placed on the page. Landscape usually "
                    "fits wider stereo calibration boards better."
                ),
            )

            self._add_number_setting(
                row=2,
                label_text="Inner corner rows",
                variable=self.checkerboard_inner_rows,
                tooltip_text=(
                    "Number of inside checkerboard corners OpenCV should detect "
                    "vertically. This is one less than the number of printed square "
                    "rows."
                ),
                minimum=2,
                maximum=80,
            )

            self._add_number_setting(
                row=3,
                label_text="Inner corner columns",
                variable=self.checkerboard_inner_columns,
                tooltip_text=(
                    "Number of inside checkerboard corners OpenCV should detect "
                    "horizontally. This is one less than the number of printed "
                    "square columns."
                ),
                minimum=2,
                maximum=80,
            )

            self._add_number_setting(
                row=4,
                label_text="Square size mm",
                variable=self.checkerboard_square_size_mm,
                tooltip_text=(
                    "Real-world size of one printed checker square in millimeters. "
                    "This value must match the printed board when calibration is run."
                ),
                minimum=1.0,
                maximum=500.0,
                increment=0.5,
            )

        # Build circle grid settings when circle grid is selected.
        elif board_type == "circle_grid":
            self._add_number_setting(
                row=0,
                label_text="Circle rows",
                variable=self.circle_grid_rows,
                tooltip_text="Number of circle center rows OpenCV should detect.",
                minimum=2,
                maximum=80,
            )

            self._add_number_setting(
                row=1,
                label_text="Circle columns",
                variable=self.circle_grid_columns,
                tooltip_text="Number of circle center columns OpenCV should detect.",
                minimum=2,
                maximum=80,
            )

            self._add_number_setting(
                row=2,
                label_text="Center spacing mm",
                variable=self.circle_grid_spacing_mm,
                tooltip_text="Real world distance from one circle center to the next circle center.",
                minimum=1.0,
                maximum=500.0,
                increment=0.5,
            )

            self._add_combo_setting(
                row=3,
                label_text="Grid layout",
                variable=self.circle_grid_layout,
                values=("symmetric", "asymmetric"),
                tooltip_text="Symmetric grids are straight rows and columns. Asymmetric grids offset every other row.",
            )

         # Build Charuco settings when Charuco is selected.
        elif board_type == "charuco":
            self._add_combo_setting(
                row=0,
                label_text="Paper size",
                variable=self.charuco_paper_size,
                values=("Letter", "A4", "Custom"),
                tooltip_text=(
                    "Physical paper size used for the printable Charuco board. "
                    "Letter is 8.5 x 11 inches. A4 is 210 x 297 mm. Custom can "
                    "be expanded later if a non-standard print size is needed."
                ),
            )

            self._add_combo_setting(
                row=1,
                label_text="Orientation",
                variable=self.charuco_orientation,
                values=("Landscape", "Portrait"),
                tooltip_text=(
                    "Direction the board is placed on the page. Landscape usually "
                    "fits wider stereo calibration boards better."
                ),
            )

            self._add_combo_setting(
                row=2,
                label_text="Board detail",
                variable=self.charuco_detail,
                values=(
                    "Large markers, easier detection",
                    "Medium",
                    "More markers, more calibration points",
                ),
                tooltip_text=(
                    "Controls how dense the printed board is. Large markers are "
                    "usually easier to detect. More markers can provide more "
                    "calibration points, but may be harder to detect if the image "
                    "is blurry or low resolution."
                ),
            )

            self._add_combo_setting(
                row=3,
                label_text="Dictionary",
                variable=self.charuco_dictionary,
                values=("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"),
                tooltip_text=(
                    "ArUco dictionary used for the printed markers. The same "
                    "dictionary must be used later when detecting the board during "
                    "calibration."
                ),
            )


    # -------------------------------------------------------------------------
    # _add_number_setting
    #
    # Adds a labeled numeric setting row to the board settings frame.
    # -------------------------------------------------------------------------

    def _add_number_setting(self, row, label_text, variable, tooltip_text, minimum, maximum, increment=1):

        # Create the setting label.
        label = ttk.Label(
            self.board_settings_frame,
            text=label_text,
            style="Bold.TLabel",
        )

        # Place the setting label.
        label.grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))

        # Add tooltip help to the label.
        ToolTip(label, tooltip_text)

        # Create the numeric input.
        spinbox = ttk.Spinbox(
            self.board_settings_frame,
            from_=minimum,
            to=maximum,
            increment=increment,
            textvariable=variable,
            width=12,
            font=self.bold_font,
            command=self._draw_board_preview,
        )

        # Place the numeric input.
        spinbox.grid(row=row, column=1, sticky="ew", pady=3)

        # Add tooltip help to the input.
        ToolTip(spinbox, tooltip_text)

        # Redraw the preview when the typed value changes.
        variable.trace_add("write", lambda *args: self._draw_board_preview())

        # Allow the value column to resize.
        self.board_settings_frame.columnconfigure(1, weight=1)


    # -------------------------------------------------------------------------
    # _add_combo_setting
    #
    # Adds a labeled combo setting row to the board settings frame.
    # -------------------------------------------------------------------------

    def _add_combo_setting(self, row, label_text, variable, values, tooltip_text):

        # Create the setting label.
        label = ttk.Label(
            self.board_settings_frame,
            text=label_text,
            style="Bold.TLabel",
        )

        # Place the setting label.
        label.grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))

        # Add tooltip help to the label.
        ToolTip(label, tooltip_text)

        # Create the combo box input.
        combo = ttk.Combobox(
            self.board_settings_frame,
            textvariable=variable,
            values=values,
            state="readonly",
            font=self.bold_font,
            width=16,
        )

        # Place the combo box input.
        combo.grid(row=row, column=1, sticky="ew", pady=3)

        # Redraw the preview when the selection changes.
        combo.bind("<<ComboboxSelected>>", lambda event: self._draw_board_preview())

        # Add tooltip help to the combo box.
        ToolTip(combo, tooltip_text)

        # Allow the value column to resize.
        self.board_settings_frame.columnconfigure(1, weight=1)


    # -------------------------------------------------------------------------
    # _on_board_type_changed
    #
    # Handles changes to the selected calibration board type.
    # -------------------------------------------------------------------------

    def _on_board_type_changed(self):

        # Rebuild the settings panel for the selected board type.
        self._rebuild_board_settings()

        # Redraw the preview for the selected board type.
        self._draw_board_preview()


    # -------------------------------------------------------------------------
    # _draw_board_preview
    #
    # Draws a simplified preview of the selected calibration board.
    # -------------------------------------------------------------------------

    def _draw_board_preview(self):

        # Stop if the preview canvas has not been created yet.
        if not hasattr(self, "preview_canvas"):
            return

        # Clear the current preview.
        self.preview_canvas.delete("all")

        # Get the current preview canvas size.
        width = max(self.preview_canvas.winfo_width(), 300)
        height = max(self.preview_canvas.winfo_height(), 260)

        # Draw the selected board preview.
        if self.board_type.get() == "checkerboard":
            self._draw_checkerboard_preview(width, height)

        elif self.board_type.get() == "circle_grid":
            self._draw_circle_grid_preview(width, height)

        elif self.board_type.get() == "charuco":
            self._draw_charuco_preview(width, height)


    # -------------------------------------------------------------------------
    # _draw_checkerboard_preview
    #
    # Draws the checkerboard preview using the calculated printable page layout.
    #
    # This uses the same paper size, orientation, inner corner count, square
    # size, and board position settings that are used when generating the final
    # printable checkerboard image.
    # -------------------------------------------------------------------------

    def _draw_checkerboard_preview(self, width, height):

        # Calculate the checkerboard page and board settings from the current UI.
        settings = self._get_checkerboard_board_settings()

        # Get the calculated page size.
        paper_width_mm = float(settings["paper_width_mm"])
        paper_height_mm = float(settings["paper_height_mm"])

        # Get the calculated checkerboard square layout.
        square_rows = int(settings["square_rows"])
        square_columns = int(settings["square_columns"])

        # Get the physical square size.
        square_size_mm = float(settings["square_size_mm"])

        # Get the calculated checkerboard position on the page.
        board_x_mm = float(settings["board_x_mm"])
        board_y_mm = float(settings["board_y_mm"])

        # Leave a small visual margin inside the preview canvas.
        preview_margin_px = 10

        # Calculate the scale needed to fit the full page into the preview.
        scale = min(
            (width - preview_margin_px * 2) / paper_width_mm,
            (height - preview_margin_px * 2) / paper_height_mm,
        )

        # Calculate the preview page size.
        page_width_px = paper_width_mm * scale
        page_height_px = paper_height_mm * scale

        # Center the page inside the preview canvas.
        page_x0 = (width - page_width_px) / 2.0
        page_y0 = (height - page_height_px) / 2.0
        page_x1 = page_x0 + page_width_px
        page_y1 = page_y0 + page_height_px

        # Draw the white paper area.
        self.preview_canvas.create_rectangle(
            page_x0,
            page_y0,
            page_x1,
            page_y1,
            fill="white",
            outline="#999999",
        )

        # Convert board position and square size to preview pixels.
        board_x_px = page_x0 + board_x_mm * scale
        board_y_px = page_y0 + board_y_mm * scale
        square_size_px = square_size_mm * scale

        # Draw each checkerboard square.
        for row in range(square_rows):
            for column in range(square_columns):

                # Only draw black squares onto the white page.
                if (row + column) % 2 == 0:

                    # Calculate this square position.
                    x1 = board_x_px + column * square_size_px
                    y1 = board_y_px + row * square_size_px
                    x2 = x1 + square_size_px
                    y2 = y1 + square_size_px

                    # Draw the black checker square.
                    self.preview_canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="black",
                        outline="black",
                    )

        # Build the small preview metadata line.
        metadata_text = (
            f"Inner corners: {settings['inner_columns']} x {settings['inner_rows']}  |  "
            f"Square: {square_size_mm:.2f} mm"
        )

        # Draw the metadata preview near the bottom of the page.
        self.preview_canvas.create_text(
            page_x0 + 8,
            page_y1 - 10,
            text=metadata_text,
            anchor="sw",
            fill="black",
            font=self.bold_font,
        )


     # -------------------------------------------------------------------------
    # _draw_charuco_preview
    #
    # Draws the Charuco preview using OpenCV's real Charuco board generator.
    #
    # This keeps the on-screen preview consistent with the printable PNG output
    # by rendering the same valid ArUco marker patterns that OpenCV uses for the
    # saved calibration board.
    # -------------------------------------------------------------------------

    def _draw_charuco_preview(self, width, height):

        # Import OpenCV here so the window can still load even if OpenCV has a
        # setup issue.
        import cv2

        # Map the UI dictionary names to OpenCV ArUco dictionary constants.
        dictionary_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }

        # Calculate the OpenCV board settings from the current Charuco UI values.
        settings = self._get_charuco_board_settings()

        # Get the selected dictionary name.
        dictionary_name = settings["dictionary"]

        # Stop if the selected dictionary is not supported.
        if dictionary_name not in dictionary_map:
            self.preview_canvas.create_text(
                width / 2,
                height / 2,
                text="Unsupported Charuco dictionary",
                anchor="center",
            )
            return

        # Get the OpenCV ArUco dictionary object.
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_map[dictionary_name])

        # Stop if the calculated board needs more markers than the dictionary has.
        if settings["marker_count"] > settings["dictionary_marker_limit"]:
            self.preview_canvas.create_text(
                width / 2,
                height / 2,
                text=(
                    f"{settings['dictionary']} is too small for this board.\n\n"
                    f"Board needs about {settings['marker_count']} markers.\n"
                    f"Dictionary has {settings['dictionary_marker_limit']} markers.\n\n"
                    "Choose a larger dictionary or lower the board detail."
                ),
                anchor="center",
                justify=tk.CENTER,
                width=width - 30,
            )
            return

        # Get the calculated board dimensions.
        squares_x = int(settings["squares_x"])
        squares_y = int(settings["squares_y"])

        # Get the calculated physical square and marker sizes.
        square_size_mm = float(settings["square_size_mm"])
        marker_size_mm = float(settings["marker_size_mm"])

        # Stop if the marker is not smaller than the square.
        if marker_size_mm >= square_size_mm:
            self.preview_canvas.create_text(
                width / 2,
                height / 2,
                text="Marker size must be smaller than square size",
                anchor="center",
            )
            return

        # Use the same raster scale used by printable board generation.
        pixels_per_mm = int(settings["pixels_per_mm"])

        # Convert physical board dimensions into image pixels.
        board_width_px = int(squares_x * square_size_mm * pixels_per_mm)
        board_height_px = int(squares_y * square_size_mm * pixels_per_mm)

        # Convert physical square and marker sizes into OpenCV drawing units.
        square_size_px = square_size_mm * pixels_per_mm
        marker_size_px = marker_size_mm * pixels_per_mm

        # Create the OpenCV Charuco board object.
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size_px,
            marker_size_px,
            aruco_dictionary,
        )

        # Render the real Charuco board image.
        board_image = board.generateImage(
            (board_width_px, board_height_px),
            marginSize=int(settings["margin_mm"] * pixels_per_mm),
            borderBits=1,
        )

        # Leave a small visual margin inside the preview canvas.
        preview_width = max(20, width - 20)
        preview_height = max(20, height - 20)

        # Calculate the scale needed to fit the rendered board into the preview.
        scale = min(
            preview_width / board_width_px,
            preview_height / board_height_px,
        )

        # Calculate the resized preview image dimensions.
        display_width = max(1, int(board_width_px * scale))
        display_height = max(1, int(board_height_px * scale))

        # Resize the OpenCV board image for the Tkinter preview.
        preview_image = cv2.resize(
            board_image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

        # Encode the resized image as PNG bytes.
        success, png_buffer = cv2.imencode(".png", preview_image)

        # Stop if OpenCV could not encode the preview image.
        if not success:
            self.preview_canvas.create_text(
                width / 2,
                height / 2,
                text="Could not render Charuco preview",
                anchor="center",
            )
            return

        # Convert the PNG bytes into base64 text for Tkinter PhotoImage.
        png_base64 = base64.b64encode(png_buffer).decode("ascii")

        # Create a Tkinter image from the rendered PNG.
        self.charuco_preview_image = tk.PhotoImage(data=png_base64)

        # Draw the preview image centered in the preview canvas.
        self.preview_canvas.create_image(
            width / 2,
            height / 2,
            image=self.charuco_preview_image,
            anchor="center",
        )


    # -------------------------------------------------------------------------
    # on_load_image_folder
    #
    # Handles the Load Image Folder button.
    # -------------------------------------------------------------------------

    def on_load_image_folder(self):

        # Ask the user to select a folder containing stereo calibration images.
        folder_path = filedialog.askdirectory(
            parent=self.window,
            title="Select Stereo Calibration Image Folder",
        )

        # Stop if the user cancelled the folder selection.
        if not folder_path:
            return

        # Store the selected image folder path.
        self.image_folder.set(folder_path)

        # Print the selected folder for early testing.
        print(f"Calibration image folder: {folder_path}")


    # -------------------------------------------------------------------------
    # on_generate_printable_board
    #
    # Saves a printable calibration board file for the selected board type.
    #
    # Checkerboard and Charuco boards are generated as PNG images so the page
    # layout, board graphics, and printed metadata can be rendered directly with
    # OpenCV. Circle grid boards are still saved as SVG files because they are
    # currently generated as simple geometric vector output.
    # -------------------------------------------------------------------------

    def on_generate_printable_board(self):

        # Get the selected calibration board type.
        board_type = self.board_type.get()

        # Check whether the selected board type uses OpenCV PNG output.
        uses_png_output = board_type in ("checkerboard", "charuco")

        # Use the correct default extension for the selected board type.
        default_extension = ".png" if uses_png_output else ".svg"

        # Use matching file type options for the selected board type.
        filetypes = [("PNG Files", "*.png"), ("All Files", "*.*")] if uses_png_output else [("SVG Files", "*.svg"), ("All Files", "*.*")]

        # Build a default filename that matches the selected output format.
        initial_file = f"{board_type}_calibration_board{default_extension}"

        # Ask the user where to save the printable board file.
        file_path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save Printable Calibration Board",
            defaultextension=default_extension,
            filetypes=filetypes,
            initialfile=initial_file,
        )

        # Stop if the user cancelled the save dialog.
        if not file_path:
            return

        # Generate the checkerboard as a printable PNG image.
        if board_type == "checkerboard":
            self._save_checkerboard_board_png(file_path)

        # Generate the Charuco board as a printable PNG image.
        elif board_type == "charuco":
            self._save_charuco_board_png(file_path)

        # Generate the circle grid as an SVG file.
        else:
            svg_text = self._build_board_svg()

            # Write the SVG file.
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(svg_text)

        # Open the saved file using the operating system default handler.
        os.startfile(file_path)

        # Print the saved file path for early testing.
        print(f"Printable board saved: {file_path}")
   

     # -------------------------------------------------------------------------
    # _save_checkerboard_board_png
    #
    # Generates a printable checkerboard calibration board and saves it as a PNG.
    #
    # This uses the calculated checkerboard page layout to draw the checkerboard
    # on the selected paper size, add Sizeamatic Pro calibration metadata in the
    # bottom margin, and write the printable image to disk.
    # -------------------------------------------------------------------------

    def _save_checkerboard_board_png(self, file_path):

        # Calculate the checkerboard page and board settings from the current UI.
        settings = self._get_checkerboard_board_settings()

        # Get the detected inner corner dimensions.
        inner_rows = int(settings["inner_rows"])
        inner_columns = int(settings["inner_columns"])

        # Get the printed checkerboard square dimensions.
        square_rows = int(settings["square_rows"])
        square_columns = int(settings["square_columns"])

        # Get the physical size of one printed checker square.
        square_size_mm = float(settings["square_size_mm"])

        # Get the raster output scale.
        pixels_per_mm = int(settings["pixels_per_mm"])

        # Convert the selected paper size into output image pixels.
        image_width_px = int(settings["paper_width_mm"] * pixels_per_mm)
        image_height_px = int(settings["paper_height_mm"] * pixels_per_mm)

        # Convert checkerboard position and square size into pixels.
        board_x_px = int(settings["board_x_mm"] * pixels_per_mm)
        board_y_px = int(settings["board_y_mm"] * pixels_per_mm)
        square_size_px = int(square_size_mm * pixels_per_mm)

        # Create a white printable page image.
        board_image = np.full(
            (image_height_px, image_width_px),
            255,
            dtype=np.uint8,
        )

        # Draw each checkerboard square.
        for row in range(square_rows):
            for column in range(square_columns):

                # Only draw the black squares onto the white page.
                if (row + column) % 2 == 0:

                    # Calculate the top left corner of this square.
                    x1 = board_x_px + column * square_size_px
                    y1 = board_y_px + row * square_size_px

                    # Calculate the bottom right corner of this square.
                    x2 = x1 + square_size_px
                    y2 = y1 + square_size_px

                    # Draw the black checker square.
                    cv2.rectangle(
                        board_image,
                        (x1, y1),
                        (x2, y2),
                        0,
                        thickness=-1,
                    )

        # Build the first printed settings line.
        settings_line_1 = (
            "Sizeamatic Pro  |  OpenCV Checkerboard Calibration Board"
        )

        # Build the second printed settings line.
        settings_line_2 = (
            f"Paper: {self.checkerboard_paper_size.get()}  |  "
            f"Orientation: {self.checkerboard_orientation.get()}  |  "
            f"Inner corners: {inner_columns} x {inner_rows}  |  "
            f"Printed squares: {square_columns} x {square_rows}  |  "
            f"Square: {square_size_mm:.2f} mm"
        )

        # Set the text drawing scale.
        text_scale = 0.6

        # Set the text drawing thickness.
        text_thickness = 1

        # Set the left edge for the printed settings text.
        text_x = int(settings["margin_mm"] * pixels_per_mm)

        # Set the vertical positions inside the bottom page margin.
        line_1_y = image_height_px - int(8.0 * pixels_per_mm)
        line_2_y = image_height_px - int(4.0 * pixels_per_mm)

        # Draw the title line into the bottom margin.
        cv2.putText(
            board_image,
            settings_line_1,
            (text_x, line_1_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            0,
            text_thickness,
            cv2.LINE_AA,
        )

        # Draw the OpenCV settings line into the bottom margin.
        cv2.putText(
            board_image,
            settings_line_2,
            (text_x, line_2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            0,
            text_thickness,
            cv2.LINE_AA,
        )

        # Save the rendered checkerboard image.
        cv2.imwrite(file_path, board_image)


    # -------------------------------------------------------------------------
    # _get_checkerboard_board_settings
    #
    # Calculates the printable checkerboard layout from the UI settings.
    #
    # This converts paper size, page orientation, inner corner count, and square
    # size into the printed square count, board position, page size, print
    # margin, and raster scale needed to generate the checkerboard image.
    # -------------------------------------------------------------------------

    def _get_checkerboard_board_settings(self):

        # Define supported paper sizes in millimeters.
        paper_sizes_mm = {
            "Letter": (215.9, 279.4),
            "A4": (210.0, 297.0),
        }

        # Get the selected paper size.
        paper_size_name = self.checkerboard_paper_size.get()

        # Fall back to Letter if Custom is selected before custom sizing exists.
        if paper_size_name not in paper_sizes_mm:
            paper_size_name = "Letter"

        # Get the physical paper size in millimeters.
        paper_width_mm, paper_height_mm = paper_sizes_mm[paper_size_name]

        # Swap width and height when using landscape orientation.
        if self.checkerboard_orientation.get() == "Landscape":
            paper_width_mm, paper_height_mm = paper_height_mm, paper_width_mm

        # Keep the board away from the page edge and leave room for metadata.
        margin_mm = 15.0

        # Get the detected inner corner dimensions from the UI.
        inner_rows = int(self.checkerboard_inner_rows.get())
        inner_columns = int(self.checkerboard_inner_columns.get())

        # Convert detected inner corners to printed checkerboard square count.
        square_rows = inner_rows + 1
        square_columns = inner_columns + 1

        # Get the physical size of one printed checker square.
        square_size_mm = float(self.checkerboard_square_size_mm.get())

        # Calculate the physical board dimensions.
        board_width_mm = square_columns * square_size_mm
        board_height_mm = square_rows * square_size_mm

        # Center the checkerboard inside the printable page area.
        board_x_mm = (paper_width_mm - board_width_mm) / 2.0
        board_y_mm = (paper_height_mm - board_height_mm) / 2.0

        # Use a fixed raster scale for the generated printable image.
        pixels_per_mm = 10

        # Return all values needed for checkerboard generation and calibration.
        return {
            "paper_width_mm": paper_width_mm,
            "paper_height_mm": paper_height_mm,
            "margin_mm": margin_mm,
            "inner_rows": inner_rows,
            "inner_columns": inner_columns,
            "square_rows": square_rows,
            "square_columns": square_columns,
            "square_size_mm": square_size_mm,
            "board_width_mm": board_width_mm,
            "board_height_mm": board_height_mm,
            "board_x_mm": board_x_mm,
            "board_y_mm": board_y_mm,
            "pixels_per_mm": pixels_per_mm,
        }

     # -------------------------------------------------------------------------
    # _save_charuco_board_png
    #
    # Generates a real OpenCV Charuco board and saves it as a PNG image.
    #
    # This uses the calculated Charuco board settings to create an OpenCV
    # CharucoBoard object, render the board with valid ArUco marker patterns,
    # and write the printable image to disk.
    # -------------------------------------------------------------------------

    def _save_charuco_board_png(self, file_path):

        # Import OpenCV here so the calibration window can still open even if
        # OpenCV has a setup issue.
        import cv2

        # Stop if this OpenCV build does not include the ArUco module.
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "This OpenCV install does not include cv2.aruco. "
                "Install opencv-contrib-python."
            )

        # Map the UI dictionary names to OpenCV ArUco dictionary constants.
        dictionary_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }

        # Calculate the OpenCV board settings from the current Charuco UI values.
        settings = self._get_charuco_board_settings()

        # Get the selected dictionary name.
        dictionary_name = settings["dictionary"]

        # Stop if the selected dictionary is not supported.
        if dictionary_name not in dictionary_map:
            raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
        
         # Stop if the calculated board needs more marker IDs than the dictionary contains.
        if settings["marker_count"] > settings["dictionary_marker_limit"]:
            raise ValueError(
                f"{dictionary_name} is too small for this Charuco board. "
                f"The board needs about {settings['marker_count']} markers, "
                f"but the dictionary only has {settings['dictionary_marker_limit']}. "
                "Choose a larger dictionary or lower the board detail."
            )

        # Get the OpenCV dictionary object.
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_map[dictionary_name])

        # Get the calculated board dimensions.
        squares_x = int(settings["squares_x"])
        squares_y = int(settings["squares_y"])

        # Get the calculated physical square and marker sizes.
        square_size_mm = float(settings["square_size_mm"])
        marker_size_mm = float(settings["marker_size_mm"])

        # Stop if the marker is not smaller than the square.
        if marker_size_mm >= square_size_mm:
            raise ValueError("Charuco marker size must be smaller than square size.")

        # Get the raster output scale.
        pixels_per_mm = int(settings["pixels_per_mm"])

          # Convert the selected paper size into output image pixels.
        image_width_px = int(settings["paper_width_mm"] * pixels_per_mm)
        image_height_px = int(settings["paper_height_mm"] * pixels_per_mm)

        # Convert physical square and marker sizes into OpenCV drawing units.
        square_size_px = square_size_mm * pixels_per_mm
        marker_size_px = marker_size_mm * pixels_per_mm

        # Create the OpenCV Charuco board object.
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size_px,
            marker_size_px,
            aruco_dictionary,
        )

        # Render the Charuco board image with valid ArUco marker patterns.
        board_image = board.generateImage(
            (image_width_px, image_height_px),
            marginSize=int(settings["margin_mm"] * pixels_per_mm),
            borderBits=1,
        )

        # Build the first printed settings line.
        settings_line_1 = (
            "Sizeamatic Pro  |  OpenCV Charuco Calibration Board"
        )

         # Build the second printed settings line.
        settings_line_2 = (
            f"Paper: {self.charuco_paper_size.get()}  |  "
            f"Orientation: {self.charuco_orientation.get()}  |  "
            f"Dictionary: {dictionary_name}  |  "
            f"Squares: {squares_x} x {squares_y}  |  "
            f"Square: {square_size_mm:.2f} mm  |  "
            f"Marker: {marker_size_mm:.2f} mm"
        )

        # Set the text drawing scale.
        text_scale = 0.6

        # Set the text drawing thickness.
        text_thickness = 1

        # Set the left edge for the printed settings text.
        text_x = int(settings["margin_mm"] * pixels_per_mm)

        # Set the vertical positions inside the bottom page margin.
        line_1_y = image_height_px - int(8.0 * pixels_per_mm)
        line_2_y = image_height_px - int(4.0 * pixels_per_mm)

        # Draw the title line into the bottom margin.
        cv2.putText(
            board_image,
            settings_line_1,
            (text_x, line_1_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            0,
            text_thickness,
            cv2.LINE_AA,
        )

        # Draw the OpenCV settings line into the bottom margin.
        cv2.putText(
            board_image,
            settings_line_2,
            (text_x, line_2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            0,
            text_thickness,
            cv2.LINE_AA,
        )

        # Save the rendered Charuco board image.
        cv2.imwrite(file_path, board_image)


    # -------------------------------------------------------------------------
    # _build_board_svg
    #
    # Builds a simple printable SVG board file.
    # -------------------------------------------------------------------------

    def _build_board_svg(self):

        # Route SVG generation based on the selected board type.
        if self.board_type.get() == "checkerboard":
            return self._build_checkerboard_svg()

        if self.board_type.get() == "circle_grid":
            return self._build_circle_grid_svg()

        if self.board_type.get() == "charuco":
            return self._build_charuco_svg()

        # Return an empty SVG if the board type is unknown.
        return "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>"


    # -------------------------------------------------------------------------
    # _build_checkerboard_svg
    #
    # Builds a printable checkerboard SVG.
    # -------------------------------------------------------------------------

    def _build_checkerboard_svg(self):

        # Convert inner corner count to printed square count.
        rows = self.checkerboard_inner_rows.get() + 1
        columns = self.checkerboard_inner_columns.get() + 1

        # Get the printed square size.
        square = self.checkerboard_square_size_mm.get()

        # Calculate the board size.
        width = columns * square
        height = rows * square

        # Start the SVG document.
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">']

        # Draw every checker square.
        for row in range(rows):
            for column in range(columns):

                # Only black squares need to be drawn onto the white page.
                if (row + column) % 2 == 0:
                    x = column * square
                    y = row * square
                    parts.append(f'<rect x="{x}" y="{y}" width="{square}" height="{square}" fill="black"/>')

        # Close the SVG document.
        parts.append("</svg>")

        # Return the SVG text.
        return "\n".join(parts)


    # -------------------------------------------------------------------------
    # _build_circle_grid_svg
    #
    # Builds a printable circle grid SVG.
    # -------------------------------------------------------------------------

    def _build_circle_grid_svg(self):

        # Get the circle grid dimensions.
        rows = self.circle_grid_rows.get()
        columns = self.circle_grid_columns.get()

        # Get the circle center spacing.
        spacing = self.circle_grid_spacing_mm.get()

        # Use a circle diameter slightly smaller than the spacing.
        radius = spacing * 0.22

        # Add a margin around the board.
        margin = spacing

        # Calculate the SVG size.
        width = (columns - 1) * spacing + margin * 2
        height = (rows - 1) * spacing + margin * 2

        # Start the SVG document.
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">']

        # Draw every circle.
        for row in range(rows):
            for column in range(columns):

                # Offset every other row when using an asymmetric circle grid.
                offset = spacing * 0.5 if self.circle_grid_layout.get() == "asymmetric" and row % 2 == 1 else 0

                # Calculate the circle center.
                cx = margin + column * spacing + offset
                cy = margin + row * spacing

                # Draw the circle.
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="black"/>')

        # Close the SVG document.
        parts.append("</svg>")

        # Return the SVG text.
        return "\n".join(parts)


    # -------------------------------------------------------------------------
    # _build_charuco_svg
    #
    # Builds a simple printable Charuco style SVG.
    # -------------------------------------------------------------------------

    def _build_charuco_svg(self):

        # Get the Charuco board dimensions.
        rows = self.charuco_squares_y.get()
        columns = self.charuco_squares_x.get()

        # Get the Charuco square and marker sizes.
        square = self.charuco_square_size_mm.get()
        marker = self.charuco_marker_size_mm.get()

        # Calculate the board size.
        width = columns * square
        height = rows * square

        # Escape the dictionary name for safe SVG text.
        dictionary_name = html.escape(self.charuco_dictionary.get())

        # Start the SVG document.
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">']

        # Draw every Charuco square.
        for row in range(rows):
            for column in range(columns):

                # Calculate this square position.
                x = column * square
                y = row * square

                # Draw black chessboard squares.
                if (row + column) % 2 == 0:
                    parts.append(f'<rect x="{x}" y="{y}" width="{square}" height="{square}" fill="black"/>')

                # Draw simplified marker placeholders inside white squares.
                else:
                    marker_offset = (square - marker) / 2
                    parts.append(
                        f'<rect x="{x + marker_offset}" y="{y + marker_offset}" '
                        f'width="{marker}" height="{marker}" fill="black"/>'
                    )

        # Add a small note so this placeholder is not mistaken for final ArUco output.
        parts.append(
            f'<text x="2" y="{height - 2}" font-size="3" fill="red">'
            f'Placeholder Charuco preview, dictionary {dictionary_name}'
            f'</text>'
        )

        # Close the SVG document.
        parts.append("</svg>")

        # Return the SVG text.
        return "\n".join(parts)


    # -------------------------------------------------------------------------
    # on_run_calibration
    #
    # Starts calibration processing on a background thread.
    #
    # This validates the selected image folder, creates a thread safe progress
    # queue, starts the OpenCV calibration worker, and begins polling for preview
    # images, completion messages, and errors from the GUI thread.
    # -------------------------------------------------------------------------

    def on_run_calibration(self):

        # Get the selected calibration image folder.
        image_folder = self.image_folder.get()

        # Stop if the user has not selected an image folder yet.
        if not image_folder:
            messagebox.showwarning(
                "No Image Folder",
                "Please load a stereo calibration image folder first.",
                parent=self.window,
            )
            return

        # Stop if a calibration worker is already running.
        if (
            hasattr(self, "calibration_worker_thread")
            and self.calibration_worker_thread is not None
            and self.calibration_worker_thread.is_alive()
        ):
            messagebox.showwarning(
                "Calibration Already Running",
                "Calibration is already running.",
                parent=self.window,
            )
            return

        # Create a queue for worker thread messages.
        self.calibration_progress_queue = queue.Queue()

        # Create the background calibration worker thread.
        self.calibration_worker_thread = threading.Thread(
            target=self._run_calibration_worker,
            args=(image_folder,),
            daemon=True,
        )

        # Start the background calibration worker.
        self.calibration_worker_thread.start()

        # Start polling worker progress from the Tkinter GUI thread.
        self._poll_calibration_progress_queue()


    # -------------------------------------------------------------------------
    # _poll_calibration_progress_queue
    #
    # Applies calibration worker updates from the Tkinter GUI thread.
    #
    # This polls the worker message queue for processed preview images, status
    # messages, completion messages, and error messages. All Tkinter updates are
    # handled here so the background OpenCV thread never touches GUI widgets
    # directly.
    # -------------------------------------------------------------------------

    def _poll_calibration_progress_queue(self):

        # Process every message currently waiting in the queue.
        while not self.calibration_progress_queue.empty():

            # Get the next worker message.
            message = self.calibration_progress_queue.get()

            # Update the preview canvas with a processed calibration image.
            if message["type"] == "preview":
                self._show_processed_image_preview(message["image"])

            # Update the window title with lightweight status text.
            elif message["type"] == "status":
                self.window.title(message["text"])

            elif message["type"] == "complete":
                messagebox.showinfo(
                    "Calibration Processing Complete",
                    message["text"],
                    parent=self.window,
                )

                # Open the generated calibration report when one was created.
                if message.get("report_index_html"):
                    webbrowser.open(Path(message["report_index_html"]).resolve().as_uri())

                # Open the run folder so the user can inspect the results.
                os.startfile(message["run_folder"])

                # Restore the normal window title.
                self.window.title("Perform Calibration")

                # Stop polling after completion.
                return

            # Show worker errors without crashing the GUI.
            elif message["type"] == "error":
                messagebox.showerror(
                    "Calibration Processing Failed",
                    message["text"],
                    parent=self.window,
                )

                # Restore the normal window title.
                self.window.title("Perform Calibration")

                # Stop polling after an unrecoverable worker error.
                return

        # Keep polling while the worker thread is still alive.
        if (
            hasattr(self, "calibration_worker_thread")
            and self.calibration_worker_thread is not None
            and self.calibration_worker_thread.is_alive()
        ):
            self.window.after(100, self._poll_calibration_progress_queue)

        # Restore the normal title if the worker ended without sending a message.
        else:
            self.window.title("Perform Calibration")

    # -------------------------------------------------------------------------
    # _run_calibration_worker
    #
    # Processes calibration images and runs OpenCV calibration off the GUI thread.
    #
    # This finds stereo image pairs, creates a run folder, processes each left
    # and right image, sends processed preview images back to the GUI through the
    # progress queue, collects valid calibration points, runs stereo calibration,
    # writes summary and metadata files, and sends a final completion or error
    # message back to the Tkinter thread.
    # -------------------------------------------------------------------------

    def _run_calibration_worker(self, image_folder):

        try:
            # Find matching left/right stereo image pairs before creating output.
            stereo_pairs = self._find_stereo_calibration_pairs(image_folder)

            # Stop if no matching stereo pairs were found.
            if not stereo_pairs:
                self.calibration_progress_queue.put({
                    "type": "error",
                    "text": (
                        "No matching stereo calibration image pairs were found.\n\n"
                        "Expected filenames look like:\n"
                        "left_3133.jpg\n"
                        "right_3133.jpg"
                    ),
                })
                return

            # Create a new output folder for this calibration run.
            run_folder = self._create_calibration_run_folder(image_folder)

            # Tell the GUI that processing has started.
            self.calibration_progress_queue.put({
                "type": "status",
                "text": "Perform Calibration - Processing Images",
            })

            # Track how many individual images were processed.
            processed_count = 0

            # Track how many individual images had successful board detections.
            detected_count = 0

            # Track how many individual images failed detection or processing.
            failed_count = 0

            # Store per-image processing results for the summary file.
            image_results = []

            # Store one object point array for each usable stereo pair.
            object_points_list = []

            # Store detected left image points for each usable stereo pair.
            left_image_points_list = []

            # Store detected right image points for each usable stereo pair.
            right_image_points_list = []

            # Store the image size used by OpenCV calibration.
            image_size = None

            # Store calibration result details if calibration succeeds.
            calibration_result = None

            # Store calibration error details if calibration fails.
            calibration_error = None

            # Process each matched stereo pair in sorted order.
            for pair_index, pair in enumerate(stereo_pairs, start=1):

                # Update the GUI status with the current pair number.
                self.calibration_progress_queue.put({
                    "type": "status",
                    "text": f"Perform Calibration - Processing Pair {pair_index} of {len(stereo_pairs)}",
                })

                # Process the left image for this stereo pair.
                left_result = self._process_single_calibration_image(
                    pair["left_path"],
                    run_folder,
                )

                # Store the left image result.
                image_results.append(("left", pair["frame_id"], left_result))

                # Count the left image result.
                processed_count += 1
                detected_count += 1 if left_result["found"] else 0
                failed_count += 0 if left_result["found"] else 1

                # Process the right image for this stereo pair.
                right_result = self._process_single_calibration_image(
                    pair["right_path"],
                    run_folder,
                )

                # Store the right image result.
                image_results.append(("right", pair["frame_id"], right_result))

                # Count the right image result.
                processed_count += 1
                detected_count += 1 if right_result["found"] else 0
                failed_count += 0 if right_result["found"] else 1

                # Use this stereo pair for checkerboard calibration only if both
                # images had successful detections.
                if (
                    self.board_type.get() == "checkerboard"
                    and left_result["found"]
                    and right_result["found"]
                ):

                    # Store the image size from the first valid stereo pair.
                    if image_size is None:
                        image_size = left_result["image_size"]

                    # Add one matching object point array for this stereo pair.
                    object_points_list.append(self._build_checkerboard_object_points())

                    # Add the left image points for this stereo pair.
                    left_image_points_list.append(left_result["corners"].astype(np.float32))

                    # Add the right image points for this stereo pair.
                    right_image_points_list.append(right_result["corners"].astype(np.float32))

                # Use this stereo pair for Charuco calibration when enough matching
                # Charuco corner IDs were detected in both images.
                elif self.board_type.get() == "charuco":

                    # Build matched object and image points for this Charuco pair.
                    charuco_pair_points = self._build_charuco_stereo_points_for_pair(
                        left_result,
                        right_result,
                    )

                    # Store this pair only when enough shared Charuco corners exist.
                    if charuco_pair_points is not None:

                        # Store the image size from the first valid stereo pair.
                        if image_size is None:
                            image_size = left_result["image_size"]

                        # Add the matched object points for this stereo pair.
                        object_points_list.append(charuco_pair_points["object_points"])

                        # Add the matched left image points for this stereo pair.
                        left_image_points_list.append(charuco_pair_points["left_image_points"])

                        # Add the matched right image points for this stereo pair.
                        right_image_points_list.append(charuco_pair_points["right_image_points"])

            # Run stereo calibration when enough valid stereo pairs were collected.
            if self.board_type.get() in ("checkerboard", "charuco") and object_points_list:

                # Tell the GUI that calibration math has started.
                self.calibration_progress_queue.put({
                    "type": "status",
                    "text": "Perform Calibration - Running OpenCV Stereo Calibration",
                })

                # Try to run the OpenCV stereo calibration math.
                try:

                    # Run OpenCV stereo calibration and save the expected calibration files.
                    calibration_result = self._run_stereo_calibration(
                        object_points_list,
                        left_image_points_list,
                        right_image_points_list,
                        image_size,
                        run_folder,
                    )

                    # Save readable metadata for this completed calibration.
                    self._save_calibration_metadata(
                        run_folder,
                        image_folder,
                        image_size,
                        len(object_points_list),
                        calibration_result,
                    )

                    # Generate the HTML calibration report for this run.
                    report_paths = generate_calibration_report(
                        calib_dir=run_folder,
                        out_root=run_folder,
                    )

                    # Store the generated report path in the calibration result.
                    calibration_result["report_index_html"] = str(report_paths["index_html"])

                # Keep the worker alive if OpenCV calibration fails.
                except Exception as error:

                    # Store the calibration error so it can be written into the summary.
                    calibration_error = str(error)

                    # Leave calibration_result empty so the UI reports processing only.
                    calibration_result = None

            # Build the path to the run summary file.
            summary_path = run_folder / "calibration_run_summary.txt"

            # Write a simple text summary of the processing run.
            with open(summary_path, "w", encoding="utf-8") as summary_file:

                # Write the run header.
                summary_file.write("Sizeamatic Pro Calibration Processing Run\n")
                summary_file.write("========================================\n\n")

                # Write the selected board type.
                summary_file.write(f"Board type: {self.board_type.get()}\n")

                # Write the selected board settings.
                summary_file.write(f"Board settings: {self._get_board_settings_summary_text()}\n")

                # Write the source image folder.
                summary_file.write(f"Image folder: {image_folder}\n")

                # Write the output run folder.
                summary_file.write(f"Run folder: {run_folder}\n\n")

                # Write the aggregate counts.
                summary_file.write(f"Stereo pairs found: {len(stereo_pairs)}\n")
                summary_file.write(f"Images processed: {processed_count}\n")
                summary_file.write(f"Images detected: {detected_count}\n")
                summary_file.write(f"Images failed: {failed_count}\n")
                summary_file.write(f"Valid calibration pairs: {len(object_points_list)}\n\n")

                # Write calibration RMS results when calibration was run.
                if calibration_result is not None:
                    summary_file.write("Calibration results\n")
                    summary_file.write("-------------------\n")
                    summary_file.write(f"Valid stereo pairs: {calibration_result['valid_pair_count']}\n")
                    summary_file.write(f"Left RMS: {calibration_result['left_rms']}\n")
                    summary_file.write(f"Right RMS: {calibration_result['right_rms']}\n")
                    summary_file.write(f"Stereo RMS: {calibration_result['stereo_rms']}\n\n")

                    

                # Write calibration error details when calibration failed.
                if calibration_error is not None:
                    summary_file.write("Calibration error\n")
                    summary_file.write("-----------------\n")
                    summary_file.write(f"{calibration_error}\n\n")

                # Write the per-image results header.
                summary_file.write("Per-image results\n")
                summary_file.write("-----------------\n")

                # Write each processed image result.
                for side_name, frame_id, result in image_results:

                    # Build the detection status string.
                    status_text = "DETECTED" if result["found"] else "NOT DETECTED"

                    # Get the output path text.
                    output_path_text = str(result["output_path"]) if result["output_path"] else "None"

                    # Write this image result.
                    summary_file.write(
                        f"{frame_id} | {side_name} | "
                        f"{result['image_path'].name} | "
                        f"{status_text} | "
                        f"Points: {result['point_count']} | "
                        f"Output: {output_path_text}\n"
                    )

            # Build the completion message.
            completion_message = (
                f"Processed {processed_count} images.\n"
                f"Detected boards in {detected_count} images.\n"
                f"Failed detections: {failed_count}\n"
                f"Valid calibration pairs: {len(object_points_list)}\n"
            )

            # Add calibration RMS values when calibration was run.
            if calibration_result is not None:
                completion_message += (
                    f"\nCalibration completed.\n"
                    f"Left RMS: {calibration_result['left_rms']:.4f}\n"
                    f"Right RMS: {calibration_result['right_rms']:.4f}\n"
                    f"Stereo RMS: {calibration_result['stereo_rms']:.4f}\n"
                )

                # Add the generated report location when available.
                if "report_index_html" in calibration_result:
                    completion_message += (
                        f"\nReport:\n{calibration_result['report_index_html']}\n"
                    )

            # Add calibration error details when image processing completed but calibration failed.
            elif calibration_error is not None:
                completion_message += (
                    "\nImage processing completed, but calibration failed.\n"
                    f"{calibration_error}\n"
                )

            # Add the output folder location.
            completion_message += f"\nOutput folder:\n{run_folder}"

            # Send the completion message back to the GUI thread.
            self.calibration_progress_queue.put({
                "type": "complete",
                "text": completion_message,
                "run_folder": str(run_folder),
                "report_index_html": (
                    calibration_result.get("report_index_html")
                    if calibration_result is not None
                    else None
                ),
            })

        except Exception as error:

            # Send unrecoverable worker errors back to the GUI thread.
            self.calibration_progress_queue.put({
                "type": "error",
                "text": str(error),
            })


    # -------------------------------------------------------------------------
    # on_open_report
    #
    # Handles the Calibration Report button.
    # -------------------------------------------------------------------------

    def on_open_report(self):

        # Print a temporary message to confirm the report button is connected.
        print("Calibration Report selected")

        # Show a temporary message until report generation is connected.
        messagebox.showinfo(
            "Calibration Report",
            "Calibration report generation is not connected yet.",
            parent=self.window,
        )

    # -------------------------------------------------------------------------
    # _get_charuco_board_settings
    #
    # Calculates the real Charuco board dimensions from the simplified UI
    # settings.
    #
    # This converts paper size, page orientation, board detail, and dictionary
    # selection into the square counts, square size, marker size, print margins,
    # and raster scale needed to generate an OpenCV Charuco board image.
    # -------------------------------------------------------------------------

    def _get_charuco_board_settings(self):

        # Define supported paper sizes in millimeters.
        paper_sizes_mm = {
            "Letter": (215.9, 279.4),
            "A4": (210.0, 297.0),
        }

        # Get the selected paper size.
        paper_size_name = self.charuco_paper_size.get()

        # Fall back to Letter if Custom is selected before custom sizing exists.
        if paper_size_name not in paper_sizes_mm:
            paper_size_name = "Letter"

        # Get the physical paper size in millimeters.
        paper_width_mm, paper_height_mm = paper_sizes_mm[paper_size_name]

        # Swap width and height when using landscape orientation.
        if self.charuco_orientation.get() == "Landscape":
            paper_width_mm, paper_height_mm = paper_height_mm, paper_width_mm

        # Keep the board away from the edge of the printed page.
        margin_mm = 10.0

        # Get the selected board detail level.
        detail = self.charuco_detail.get()

        # Choose square size from the selected detail level.
        if detail == "Large markers, easier detection":
            square_size_mm = 32.0

        elif detail == "More markers, more calibration points":
            square_size_mm = 18.0

        else:
            square_size_mm = 25.0

        # Use markers that fill most of each white square while leaving a border.
        marker_size_mm = square_size_mm * 0.70

        # Calculate the usable printable area.
        usable_width_mm = paper_width_mm - margin_mm * 2.0
        usable_height_mm = paper_height_mm - margin_mm * 2.0

        # Calculate how many Charuco squares fit on the page.
        squares_x = max(2, int(usable_width_mm // square_size_mm))
        squares_y = max(2, int(usable_height_mm // square_size_mm))

        # Use a fixed raster resolution for OpenCV board image generation.
        pixels_per_mm = 10

                # Get the selected ArUco dictionary.
        dictionary_name = self.charuco_dictionary.get()

        # Define how many marker IDs each supported dictionary contains.
        dictionary_marker_limits = {
            "DICT_4X4_50": 50,
            "DICT_5X5_100": 100,
            "DICT_6X6_250": 250,
        }

        # Estimate how many ArUco markers the Charuco board will need.
        marker_count = (squares_x * squares_y) // 2

        # Get the marker limit for the selected dictionary.
        dictionary_marker_limit = dictionary_marker_limits.get(dictionary_name, 0)

        # Return all values needed for OpenCV board generation and calibration.
        return {
            "paper_width_mm": paper_width_mm,
            "paper_height_mm": paper_height_mm,
            "margin_mm": margin_mm,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_size_mm": square_size_mm,
            "marker_size_mm": marker_size_mm,
            "dictionary": dictionary_name,
            "dictionary_marker_limit": dictionary_marker_limit,
            "marker_count": marker_count,
            "pixels_per_mm": pixels_per_mm,
        }
    

    # -------------------------------------------------------------------------
    # _create_calibration_run_folder
    #
    # Creates a new output folder for one calibration processing run.
    #
    # This places each run inside the selected calibration image folder using a
    # numbered run folder name, such as run_001, run_002, and run_003, so the
    # processed detection images and run summary are kept separate from the raw
    # calibration images.
    # -------------------------------------------------------------------------

    def _create_calibration_run_folder(self, image_folder):

        # Convert the selected image folder into a Path object.
        image_folder = Path(image_folder)

        # Try numbered run folder names until an unused one is found.
        for run_number in range(1, 1000):

            # Build the candidate run folder path.
            run_folder = image_folder / f"run_{run_number:03d}"

            # Use the first run folder that does not already exist.
            if not run_folder.exists():

                # Create the run folder.
                run_folder.mkdir(parents=True, exist_ok=False)

                # Return the new run folder path.
                return run_folder

        # Stop if too many run folders already exist.
        raise RuntimeError("Could not create a new calibration run folder.")
    
   
    # -------------------------------------------------------------------------
    # _find_stereo_calibration_pairs
    #
    # Finds matching left and right calibration images in a folder.
    #
    # This searches the selected image folder for files named like left_3133.jpg
    # and right_3133.jpg, uses the shared numeric ID to match stereo pairs, and
    # returns the matched image paths in sorted order for calibration processing.
    # -------------------------------------------------------------------------

    def _find_stereo_calibration_pairs(self, image_folder):

        # Convert the selected image folder into a Path object.
        image_folder = Path(image_folder)

        # Accept common image file formats.
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }

        # Store left images by their shared stereo frame ID.
        left_images = {}

        # Store right images by their shared stereo frame ID.
        right_images = {}

        # Match filenames like left_3133.jpg or right_3133.png.
        filename_pattern = re.compile(r"^(left|right)_(.+)$", re.IGNORECASE)

        # Search every image file directly inside the selected folder.
        for image_path in image_folder.iterdir():

            # Skip folders and unsupported file types.
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue

            # Match the filename stem against the expected stereo naming pattern.
            match = filename_pattern.match(image_path.stem)

            # Skip files that do not look like left_xxxx or right_xxxx.
            if not match:
                continue

            # Get the side name and shared stereo frame ID from the filename.
            side_name = match.group(1).lower()
            frame_id = match.group(2)

            # Store left images by frame ID.
            if side_name == "left":
                left_images[frame_id] = image_path

            # Store right images by frame ID.
            elif side_name == "right":
                right_images[frame_id] = image_path

        # Find only frame IDs that exist on both the left and right side.
        matched_frame_ids = sorted(set(left_images.keys()) & set(right_images.keys()))

        # Build the matched stereo pair list.
        stereo_pairs = [
            {
                "frame_id": frame_id,
                "left_path": left_images[frame_id],
                "right_path": right_images[frame_id],
            }
            for frame_id in matched_frame_ids
        ]

        # Return the matched stereo pairs.
        return stereo_pairs
    

    # -------------------------------------------------------------------------
    # _show_processed_image_preview
    #
    # Displays a processed OpenCV image in the board preview canvas.
    #
    # This takes an annotated OpenCV image, scales it to fit inside the preview
    # canvas while preserving aspect ratio, converts it to a Tkinter image, and
    # draws it in the same preview area used for calibration board previews.
    # -------------------------------------------------------------------------

    def _show_processed_image_preview(self, image):

        # Stop if the preview canvas has not been created yet.
        if not hasattr(self, "preview_canvas"):
            return

        # Clear the current preview canvas.
        self.preview_canvas.delete("all")

        # Get the current preview canvas size.
        canvas_width = max(self.preview_canvas.winfo_width(), 300)
        canvas_height = max(self.preview_canvas.winfo_height(), 260)

        # Get the OpenCV image dimensions.
        image_height, image_width = image.shape[:2]

        # Stop if the image dimensions are invalid.
        if image_width <= 0 or image_height <= 0:
            return

        # Leave a small visual margin inside the preview canvas.
        preview_margin_px = 10

        # Calculate the scale needed to fit the image inside the preview canvas.
        scale = min(
            (canvas_width - preview_margin_px * 2) / image_width,
            (canvas_height - preview_margin_px * 2) / image_height,
        )

        # Calculate the displayed image size.
        display_width = max(1, int(image_width * scale))
        display_height = max(1, int(image_height * scale))

        # Resize the processed image for preview display.
        preview_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

        # Convert BGR OpenCV images to RGB for Tkinter display.
        if len(preview_image.shape) == 3:
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)

        # Encode the resized image as PNG bytes.
        success, png_buffer = cv2.imencode(".png", preview_image)

        # Stop if OpenCV could not encode the preview image.
        if not success:
            return

        # Convert the PNG bytes into base64 text for Tkinter PhotoImage.
        png_base64 = base64.b64encode(png_buffer).decode("ascii")

        # Store the image reference so Tkinter does not garbage collect it.
        self.processed_preview_image = tk.PhotoImage(data=png_base64)

        # Draw the processed image centered in the preview canvas.
        self.preview_canvas.create_image(
            canvas_width / 2,
            canvas_height / 2,
            image=self.processed_preview_image,
            anchor="center",
        )

        # Let Tkinter update the preview during long processing runs.
        self.window.update_idletasks()

   
    # -------------------------------------------------------------------------
    # _draw_processed_image_text_line
    #
    # Draws one readable metadata line onto a processed calibration image.
    #
    # This adds outlined text to an OpenCV image so detection results, board
    # settings, filenames, and image information remain readable over both dark
    # and light image backgrounds.
    # -------------------------------------------------------------------------

    def _draw_processed_image_text_line(self, image, text, x, y):

        # Draw a thick white outline behind the text for contrast.
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )

        # Draw the black foreground text over the white outline.
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


    # -------------------------------------------------------------------------
    # _get_board_settings_summary_text
    #
    # Builds a short settings summary for the selected calibration board type.
    #
    # This creates the board-specific metadata text that gets drawn onto
    # processed calibration images, so saved detection images show which OpenCV
    # settings were used for checkerboard, circle grid, or Charuco detection.
    # -------------------------------------------------------------------------

    def _get_board_settings_summary_text(self):

        # Get the currently selected board type.
        board_type = self.board_type.get()

        # Build the checkerboard settings summary.
        if board_type == "checkerboard":

            # Calculate the current checkerboard settings.
            settings = self._get_checkerboard_board_settings()

            # Return the checkerboard settings in OpenCV pattern size order.
            return (
                f"Checkerboard | "
                f"Inner corners: {settings['inner_columns']} x {settings['inner_rows']} | "
                f"Square: {settings['square_size_mm']:.2f} mm | "
                f"Paper: {self.checkerboard_paper_size.get()} | "
                f"Orientation: {self.checkerboard_orientation.get()}"
            )

        # Build the circle grid settings summary.
        if board_type == "circle_grid":

            # Return the circle grid settings in OpenCV pattern size order.
            return (
                f"Circle Grid | "
                f"Centers: {self.circle_grid_columns.get()} x {self.circle_grid_rows.get()} | "
                f"Spacing: {self.circle_grid_spacing_mm.get():.2f} mm | "
                f"Layout: {self.circle_grid_layout.get()} | "
                f"Paper: {self.circle_grid_paper_size.get()} | "
                f"Orientation: {self.circle_grid_orientation.get()}"
            )

        # Build the Charuco settings summary.
        if board_type == "charuco":

            # Calculate the current Charuco settings.
            settings = self._get_charuco_board_settings()

            # Return the Charuco settings in OpenCV board size order.
            return (
                f"Charuco | "
                f"Squares: {settings['squares_x']} x {settings['squares_y']} | "
                f"Square: {settings['square_size_mm']:.2f} mm | "
                f"Marker: {settings['marker_size_mm']:.2f} mm | "
                f"Dictionary: {settings['dictionary']} | "
                f"Paper: {self.charuco_paper_size.get()} | "
                f"Orientation: {self.charuco_orientation.get()}"
            )

        # Return a safe fallback for unknown board types.
        return f"Board type: {board_type}"
    

    # -------------------------------------------------------------------------
    # _detect_checkerboard
    #
    # Detects checkerboard inner corners in one calibration image.
    #
    # This uses the selected checkerboard settings to run OpenCV checkerboard
    # detection, refines the detected corner locations when needed, draws the
    # OpenCV corner overlay onto a copy of the image, and returns the detection
    # result for later calibration processing.
    # -------------------------------------------------------------------------

    def _detect_checkerboard(self, image):

        # Calculate the current checkerboard settings.
        settings = self._get_checkerboard_board_settings()

        # Build the OpenCV checkerboard pattern size in columns x rows order.
        pattern_size = (
            int(settings["inner_columns"]),
            int(settings["inner_rows"]),
        )

        # Convert the input image to grayscale for checkerboard detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use the newer OpenCV checkerboard detector when available.
        if hasattr(cv2, "findChessboardCornersSB"):

            # Detect checkerboard corners with the more robust SB detector.
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

        else:

            # Detect checkerboard corners with the classic OpenCV detector.
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                    + cv2.CALIB_CB_FAST_CHECK
                ),
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
                        0.001,
                    ),
                )

        # Make a copy of the image so drawing does not modify the original.
        annotated = image.copy()

        # Draw the detected checkerboard corners when detection succeeds.
        if found:
            cv2.drawChessboardCorners(
                annotated,
                pattern_size,
                corners,
                found,
            )

        # Return the detection result and annotated image.
        return {
            "found": bool(found),
            "corners": corners,
            "annotated": annotated,
            "point_count": len(corners) if found and corners is not None else 0,
            "pattern_size": pattern_size,
        }
    

    # -------------------------------------------------------------------------
    # _detect_charuco
    #
    # Detects Charuco corners in one calibration image.
    #
    # This rebuilds the same OpenCV CharucoBoard described by the current UI
    # settings, detects the ArUco markers and interpolated Charuco corners in the
    # image, draws the detected marker and corner overlay, and returns the
    # detected corner positions and IDs for later calibration processing.
    # -------------------------------------------------------------------------

    def _detect_charuco(self, image):

        # Calculate the current Charuco board settings.
        settings = self._get_charuco_board_settings()

        # Map the UI dictionary names to OpenCV ArUco dictionary constants.
        dictionary_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }

        # Get the selected dictionary name.
        dictionary_name = settings["dictionary"]

        # Make a copy of the image so drawing does not modify the original.
        annotated = image.copy()

        # Stop if the selected dictionary is not supported.
        if dictionary_name not in dictionary_map:
            return {
                "found": False,
                "corners": None,
                "ids": None,
                "annotated": annotated,
                "point_count": 0,
                "pattern_size": None,
                "message": f"Unsupported ArUco dictionary: {dictionary_name}",
            }

        # Stop if the calculated board needs more marker IDs than the dictionary has.
        if settings["marker_count"] > settings["dictionary_marker_limit"]:
            return {
                "found": False,
                "corners": None,
                "ids": None,
                "annotated": annotated,
                "point_count": 0,
                "pattern_size": None,
                "message": (
                    f"{dictionary_name} is too small for this board. "
                    f"Board needs {settings['marker_count']} markers, "
                    f"dictionary has {settings['dictionary_marker_limit']}."
                ),
            }

                # Get the calculated board dimensions.
        squares_x = int(settings["squares_x"])
        squares_y = int(settings["squares_y"])

        # Stop if the marker is not smaller than the square.
        if float(settings["marker_size_mm"]) >= float(settings["square_size_mm"]):
            return {
                "found": False,
                "corners": None,
                "ids": None,
                "annotated": annotated,
                "point_count": 0,
                "pattern_size": None,
                "message": "Charuco marker size must be smaller than square size.",
            }

        # Create the same OpenCV Charuco board used by printing and calibration.
        board = self._build_charuco_board()

        # Create the modern OpenCV Charuco detector.
        detector = cv2.aruco.CharucoDetector(board)

        # Detect ArUco marker corners and interpolated Charuco corners.
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)

        # Draw detected ArUco markers when any marker IDs were found.
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                marker_corners,
                marker_ids,
            )

        # Count detected Charuco corners.
        point_count = len(charuco_corners) if charuco_corners is not None else 0

        # Treat the board as found when enough Charuco corners were detected.
        found = charuco_ids is not None and point_count >= 4

        # Draw detected Charuco corners when enough corners were found.
        if found:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                charuco_corners,
                charuco_ids,
            )

        # Return the detection result and annotated image.
        return {
            "found": bool(found),
            "corners": charuco_corners,
            "ids": charuco_ids,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "annotated": annotated,
            "point_count": point_count,
            "pattern_size": (squares_x, squares_y),
        }
    

    # -------------------------------------------------------------------------
    # _detect_calibration_board
    #
    # Runs board detection for the selected calibration board type.
    #
    # This routes one OpenCV image through the checkerboard, circle grid, or
    # Charuco detector based on the current UI selection, returning a common
    # detection result format that the calibration processing loop can use for
    # overlays, previews, saved images, and later calibration point collection.
    # -------------------------------------------------------------------------

    def _detect_calibration_board(self, image):

        # Get the currently selected calibration board type.
        board_type = self.board_type.get()

        # Run checkerboard detection.
        if board_type == "checkerboard":
            return self._detect_checkerboard(image)

        # Return a temporary circle grid result until circle grid detection is added.
        if board_type == "circle_grid":

            # Make a copy of the image so the caller always gets an annotated image.
            annotated = image.copy()

            # Return a consistent no detection result.
            return {
                "found": False,
                "corners": None,
                "annotated": annotated,
                "point_count": 0,
                "pattern_size": None,
                "message": "Circle grid detection is not connected yet.",
            }

        # Run Charuco detection.
        if board_type == "charuco":
            return self._detect_charuco(image)

        # Make a copy of the image so the caller always gets an annotated image.
        annotated = image.copy()

        # Return a safe fallback result for unknown board types.
        return {
            "found": False,
            "corners": None,
            "annotated": annotated,
            "point_count": 0,
            "pattern_size": None,
            "message": f"Unknown board type: {board_type}",
        }
    

    # -------------------------------------------------------------------------
    # _process_single_calibration_image
    #
    # Processes one calibration image and saves an annotated output image.
    #
    # This loads a left or right calibration image, runs the selected OpenCV board
    # detector, draws detection status and calibration settings onto the image,
    # saves the annotated result into the current run folder, and displays the
    # processed image in the preview canvas.
    # -------------------------------------------------------------------------

    def _process_single_calibration_image(self, image_path, output_folder):

        # Convert the image path and output folder into Path objects.
        image_path = Path(image_path)
        output_folder = Path(output_folder)

        # Load the calibration image.
        image = cv2.imread(str(image_path))

        # Return a failed result if OpenCV could not read the image.
        if image is None:
            return {
                "image_path": image_path,
                "output_path": None,
                "found": False,
                "corners": None,
                "point_count": 0,
                "error": "OpenCV could not read image.",
            }

        # Get useful image information for the status overlay.
        height, width = image.shape[:2]

        # Run the selected board detector.
        result = self._detect_calibration_board(image)

        # Get the annotated image returned by the detector.
        annotated = result["annotated"]

        # Mark failed detections clearly with a red border.
        if not result["found"]:
            cv2.rectangle(
                annotated,
                (0, 0),
                (width - 1, height - 1),
                (0, 0, 255),
                6,
            )

        # Build the detection status text.
        status_text = "DETECTED" if result["found"] else "NOT DETECTED"

        # Get the number of detected points.
        point_count = int(result.get("point_count", 0))

        # Build the board settings text.
        board_settings_text = self._get_board_settings_summary_text()

        # Set the bottom text overlay position.
        y0 = height - 82
        line_gap = 24

        # Draw OpenCV and application metadata.
        self._draw_processed_image_text_line(
            annotated,
            f"Sizeamatic Pro | OpenCV {cv2.__version__}",
            12,
            y0,
        )

        # Draw the board settings metadata.
        self._draw_processed_image_text_line(
            annotated,
            f"{board_settings_text} | Image: {width} x {height}",
            12,
            y0 + line_gap,
        )

        # Draw the filename and detection status metadata.
        self._draw_processed_image_text_line(
            annotated,
            f"File: {image_path.name} | Detection: {status_text} | Points found: {point_count}",
            12,
            y0 + (line_gap * 2),
        )

        # Build the output filename.
        output_path = output_folder / f"{image_path.stem}_processed{image_path.suffix}"

        # Save the annotated image.
        saved_ok = cv2.imwrite(str(output_path), annotated)

         # Send the processed image to the GUI thread when running in worker mode.
        if hasattr(self, "calibration_progress_queue"):
            self.calibration_progress_queue.put({
                "type": "preview",
                "image": annotated,
            })

        # Update the preview directly when no worker queue exists.
        else:
            self._show_processed_image_preview(annotated)

         # Return the processed image result.
        return {
            "image_path": image_path,
            "output_path": output_path if saved_ok else None,
            "found": bool(result["found"]),
            "corners": result.get("corners"),
            "ids": result.get("ids"),
            "point_count": point_count,
            "image_size": (width, height),
            "error": None if saved_ok else "OpenCV could not save processed image.",
        }


    # -------------------------------------------------------------------------
    # _build_checkerboard_object_points
    #
    # Builds the real-world checkerboard corner coordinates for calibration.
    #
    # This creates one 3D object point array for the selected checkerboard
    # settings, using inner corner columns, inner corner rows, and square size.
    # The board is treated as a flat Z=0 plane, which is what OpenCV expects for
    # standard checkerboard camera calibration.
    # -------------------------------------------------------------------------

    def _build_checkerboard_object_points(self):

        # Calculate the current checkerboard settings.
        settings = self._get_checkerboard_board_settings()

        # Get the checkerboard inner corner dimensions.
        inner_columns = int(settings["inner_columns"])
        inner_rows = int(settings["inner_rows"])

        # Get the real-world square size in millimeters.
        square_size_mm = float(settings["square_size_mm"])

        # Create an empty object point array with one XYZ point per inner corner.
        object_points = np.zeros(
            (inner_rows * inner_columns, 3),
            np.float32,
        )

        # Fill the X and Y coordinates in checkerboard order.
        object_points[:, :2] = np.mgrid[
            0:inner_columns,
            0:inner_rows,
        ].T.reshape(-1, 2)

        # Scale the board coordinates into real-world millimeters.
        object_points *= square_size_mm

        # Return the object point array.
        return object_points
    

    # -------------------------------------------------------------------------
    # _run_stereo_calibration
    #
    # Runs OpenCV stereo calibration from collected checkerboard detections.
    #
    # This uses the matched object points, left image points, and right image
    # points gathered from valid stereo image pairs to calculate camera
    # intrinsics, stereo extrinsics, rectification transforms, and remap tables,
    # then saves the four calibration files expected by the rest of the program.
    # -------------------------------------------------------------------------

    def _run_stereo_calibration(
        self,
        object_points_list,
        left_image_points_list,
        right_image_points_list,
        image_size,
        run_folder,
    ):

        # Stop if there are not enough valid stereo pairs for calibration.
        if len(object_points_list) < 3:
            raise ValueError(
                "At least 3 valid stereo pairs are required for calibration."
            )

        # Convert the run folder into a Path object.
        run_folder = Path(run_folder)

        # Set OpenCV calibration termination criteria.
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-5,
        )

        # Calibrate the left camera intrinsics.
        left_rms, left_camera_matrix, left_dist_coeffs, left_rvecs, left_tvecs = cv2.calibrateCamera(
            object_points_list,
            left_image_points_list,
            image_size,
            None,
            None,
        )

        # Calibrate the right camera intrinsics.
        right_rms, right_camera_matrix, right_dist_coeffs, right_rvecs, right_tvecs = cv2.calibrateCamera(
            object_points_list,
            right_image_points_list,
            image_size,
            None,
            None,
        )

        # Keep intrinsics mostly fixed while solving stereo relationship.
        stereo_flags = cv2.CALIB_FIX_INTRINSIC

        # Calibrate the stereo extrinsics between the left and right cameras.
        stereo_rms, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, rotation, translation, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
            object_points_list,
            left_image_points_list,
            right_image_points_list,
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            image_size,
            criteria=criteria,
            flags=stereo_flags,
        )

        # Calculate stereo rectification transforms and projection matrices.
        left_rectification, right_rectification, left_projection, right_projection, disparity_to_depth, left_roi, right_roi = cv2.stereoRectify(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            image_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        # Build the left image remap tables.
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            left_camera_matrix,
            left_dist_coeffs,
            left_rectification,
            left_projection,
            image_size,
            cv2.CV_32FC1,
        )

        # Build the right image remap tables.
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            right_camera_matrix,
            right_dist_coeffs,
            right_rectification,
            right_projection,
            image_size,
            cv2.CV_32FC1,
        )

        # Save camera intrinsics and distortion coefficients.
        np.savez(
            run_folder / "calibration_intrinsics.npz",

            # Store the calibration image size in the format expected by the loader.
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),

            # Store left and right camera intrinsics in the format expected by the loader.
            mtxL=left_camera_matrix,
            distL=left_dist_coeffs,
            mtxR=right_camera_matrix,
            distR=right_dist_coeffs,

            # Store calibration quality values for reference.
            left_rms=left_rms,
            right_rms=right_rms,

            # Store per-camera pose estimates from individual calibration.
            rvecsL=np.array(left_rvecs, dtype=object),
            tvecsL=np.array(left_tvecs, dtype=object),
            rvecsR=np.array(right_rvecs, dtype=object),
            tvecsR=np.array(right_tvecs, dtype=object),
        )

        # Save stereo extrinsics in the format expected by the loader.
        np.savez(
            run_folder / "calibration_extrinsics.npz",
            stereo_rms=stereo_rms,
            R=rotation,
            T=translation,
            E=essential_matrix,
            F=fundamental_matrix,
        )

        # Save stereo rectification data in the format expected by the loader.
        np.savez(
            run_folder / "calibration_rectification.npz",
            RL=left_rectification,
            RR=right_rectification,
            PL=left_projection,
            PR=right_projection,
            Q=disparity_to_depth,
            roiL=np.array(left_roi),
            roiR=np.array(right_roi),
        )

        # Save rectification remap tables in the format expected by the loader.
        np.savez(
            run_folder / "calibration_maps.npz",
            mapLx=left_map_x,
            mapLy=left_map_y,
            mapRx=right_map_x,
            mapRy=right_map_y,
        )

        # Return calibration quality information for summaries and status messages.
        return {
            "left_rms": left_rms,
            "right_rms": right_rms,
            "stereo_rms": stereo_rms,
            "valid_pair_count": len(object_points_list),
        }
    
    # -------------------------------------------------------------------------
    # _save_calibration_metadata
    #
    # Saves readable metadata for a completed calibration run.
    #
    # This writes the selected board type, board settings, image folder, output
    # folder, valid pair count, image size, OpenCV version, and RMS results to a
    # JSON file so the calibration can be reviewed later without opening the
    # binary NumPy calibration files.
    # -------------------------------------------------------------------------

    def _save_calibration_metadata(
        self,
        run_folder,
        image_folder,
        image_size,
        valid_pair_count,
        calibration_result,
    ):

        # Convert the run folder into a Path object.
        run_folder = Path(run_folder)

        # Build the metadata dictionary.
        metadata = {
            "application": "Sizeamatic Pro",
            "opencv_version": cv2.__version__,
            "board_type": self.board_type.get(),
            "board_settings": self._get_board_settings_summary_text(),
            "image_folder": str(image_folder),
            "run_folder": str(run_folder),
            "image_width": int(image_size[0]) if image_size else None,
            "image_height": int(image_size[1]) if image_size else None,
            "valid_pair_count": int(valid_pair_count),
            "left_rms": float(calibration_result["left_rms"]),
            "right_rms": float(calibration_result["right_rms"]),
            "stereo_rms": float(calibration_result["stereo_rms"]),
        }

        # Build the metadata output path.
        metadata_path = run_folder / "calibration_metadata.json"

        # Write the metadata file.
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        # Return the saved metadata path.
        return metadata_path
    
    # -------------------------------------------------------------------------
    # _build_charuco_board
    #
    # Builds the OpenCV CharucoBoard object described by the current UI settings.
    #
    # This recreates the same board used for printing and detection, using the
    # selected square count, square size, marker size, and ArUco dictionary. The
    # returned board is used to map detected Charuco corner IDs back to real
    # world object points for camera calibration.
    # -------------------------------------------------------------------------

    def _build_charuco_board(self):

        # Calculate the current Charuco board settings.
        settings = self._get_charuco_board_settings()

        # Map the UI dictionary names to OpenCV ArUco dictionary constants.
        dictionary_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }

        # Get the selected dictionary name.
        dictionary_name = settings["dictionary"]

        # Stop if the selected dictionary is not supported.
        if dictionary_name not in dictionary_map:
            raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

        # Get the OpenCV ArUco dictionary object.
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            dictionary_map[dictionary_name]
        )

        # Create the OpenCV Charuco board object.
        board = cv2.aruco.CharucoBoard(
            (
                int(settings["squares_x"]),
                int(settings["squares_y"]),
            ),
            float(settings["square_size_mm"]),
            float(settings["marker_size_mm"]),
            aruco_dictionary,
        )

        # Return the board object.
        return board
    
    # -------------------------------------------------------------------------
    # _build_charuco_stereo_points_for_pair
    #
    # Builds matched object and image points for one Charuco stereo pair.
    #
    # This compares the detected Charuco corner IDs from the left and right
    # images, keeps only IDs found in both images, and returns matching object
    # points, left image points, and right image points in the same order for
    # OpenCV stereo calibration.
    # -------------------------------------------------------------------------

    def _build_charuco_stereo_points_for_pair(self, left_result, right_result):

        # Stop if either image failed Charuco detection.
        if not left_result["found"] or not right_result["found"]:
            return None

        # Stop if either image has no Charuco corner IDs.
        if left_result.get("ids") is None or right_result.get("ids") is None:
            return None

        # Get the detected Charuco corners and IDs.
        left_corners = left_result["corners"]
        left_ids = left_result["ids"].reshape(-1)

        # Get the detected Charuco corners and IDs.
        right_corners = right_result["corners"]
        right_ids = right_result["ids"].reshape(-1)

        # Build lookup tables from Charuco ID to detected corner.
        left_corner_by_id = {
            int(charuco_id): left_corners[index].reshape(2)
            for index, charuco_id in enumerate(left_ids)
        }

        # Build lookup tables from Charuco ID to detected corner.
        right_corner_by_id = {
            int(charuco_id): right_corners[index].reshape(2)
            for index, charuco_id in enumerate(right_ids)
        }

        # Find Charuco corner IDs detected in both left and right images.
        shared_ids = sorted(set(left_corner_by_id.keys()) & set(right_corner_by_id.keys()))

        # Stop if there are too few shared points to be useful.
        if len(shared_ids) < 4:
            return None

        # Recreate the OpenCV Charuco board.
        board = self._build_charuco_board()

        # Get the real world chessboard corner coordinates from the Charuco board.
        chessboard_corners = board.getChessboardCorners()

        # Build object points for the shared Charuco IDs.
        object_points = np.array(
            [
                chessboard_corners[charuco_id]
                for charuco_id in shared_ids
            ],
            dtype=np.float32,
        )

        # Build left image points in the same ID order.
        left_image_points = np.array(
            [
                left_corner_by_id[charuco_id]
                for charuco_id in shared_ids
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        # Build right image points in the same ID order.
        right_image_points = np.array(
            [
                right_corner_by_id[charuco_id]
                for charuco_id in shared_ids
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        # Return the matched calibration points for this stereo pair.
        return {
            "object_points": object_points,
            "left_image_points": left_image_points,
            "right_image_points": right_image_points,
            "shared_ids": shared_ids,
        }

# -----------------------------------------------------------------------------
# open_perform_calibration_window
#
# Opens the calibration workflow window.
# -----------------------------------------------------------------------------

def open_perform_calibration_window(parent, app=None):

    # Create and return the calibration window object.
    return PerformCalibrationWindow(parent, app=app)