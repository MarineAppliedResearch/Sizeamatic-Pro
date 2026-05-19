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
            text="Sizeamatic Pro - Stereo Calibration",
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
    # Handles the Perform Calibration button.
    # -------------------------------------------------------------------------

    def on_run_calibration(self):

        # Print the current calibration settings for early testing.
        print("Perform Calibration selected")
        print(f"Board type: {self.board_type.get()}")
        print(f"Image folder: {self.image_folder.get()}")

        # Show a temporary message until calibration processing is connected.
        messagebox.showinfo(
            "Perform Calibration",
            "Calibration processing is not connected yet.",
            parent=self.window,
        )


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


# -----------------------------------------------------------------------------
# open_perform_calibration_window
#
# Opens the calibration workflow window.
# -----------------------------------------------------------------------------

def open_perform_calibration_window(parent, app=None):

    # Create and return the calibration window object.
    return PerformCalibrationWindow(parent, app=app)