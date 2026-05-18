# -----------------------------------------------------------------------------
# measurement_window.py
#
# Author: Isaac Travers
# Created: 2026-05-18
# Project: Sizeamatic Pro
#
# Purpose:
#   Provides the Tkinter measurement results window used by Sizeamatic Pro.
#
#   This module creates and updates the measurement output window, including the
#   point diagnostics table, segment table, error/status line, and copyable text
#   output for spreadsheet use.
#
# Contents:
#   - Measurement window creation.
#   - Point result table setup.
#   - Segment result table setup.
#   - Measurement status/error display.
#   - Tab separated copy block generation.
#
# Design Notes:
#   Functions in this file receive the main application object so they can access
#   Tkinter root state, measurement widgets, and display settings owned by the
#   app. This keeps measurement display behavior grouped in one file while
#   preserving the current application state model.
#
# Assumptions:
#   - Measurement rows passed into this module are already computed and formatted.
#   - This module does not perform stereo triangulation or measurement math.
#   - Widget references are stored on the app object so later update calls can
#     reuse or rebuild the measurement window as needed.
#
# Dependencies:
#   - tkinter provides the Toplevel window and text variable support.
#   - tkinter.ttk provides themed frames, labels, and Treeview tables.
# -----------------------------------------------------------------------------

# tkinter provides the measurement results window widgets.
import tkinter as tk

# ttk provides themed Tkinter widgets such as Frame, Label, and Treeview.
from tkinter import ttk


# -----------------------------------------------------------------------------
# ensure_measurement_window
#
# Inputs: app provides the Tk root window and stores measurement window widget
# references used by later measurement display updates.
# Outputs: creates the measurement Toplevel window if needed and stores widget
# references on app; returns nothing.
#
# Creates the measurement results window, including the point diagnostics table,
# segment measurement table, error message line, and copyable text area. If the
# window already exists, the function exits without creating another one. This
# function only builds the UI widgets; measurement values are filled in later by
# update_measurement_window.
# -----------------------------------------------------------------------------
def ensure_measurement_window(app):

    # If the measurement window already exists, reuse it instead of creating a
    # duplicate Toplevel window.
    if app.meas_win is not None:
        return

    # Create a separate top level window owned by the main application root.
    win = tk.Toplevel(app.root)

    # Set the user visible title for the measurement results window.
    win.title("Measurement")

    # Give the window an initial size large enough for both result tables and the
    # copyable text box.
    win.geometry("620x520")

    # -------------------------------------------------------------------------
    # _on_close
    #
    # Inputs: none.
    # Outputs: destroys the measurement window and clears stored widget
    # references on the app object.
    #
    # Handles the user closing the measurement window manually. Clearing the app
    # references is important because the next measurement update needs to know
    # that the widgets no longer exist and must be rebuilt.
    # -------------------------------------------------------------------------
    def _on_close():

        # Destroy the Tkinter window.
        win.destroy()

        # Clear the stored measurement window reference.
        app.meas_win = None

        # Clear the stored table references because the widgets were destroyed.
        app.points_tree = None
        app.segs_tree = None

        # Clear the copy text widget reference because the widget was destroyed.
        app.meas_copy_text = None

        # Clear the error text variable reference because the window was destroyed.
        app.meas_error_var = None

    # Use the cleanup callback when the user closes the measurement window.
    win.protocol("WM_DELETE_WINDOW", _on_close)

    # Create one padded outer frame to hold all measurement window content.
    outer = ttk.Frame(win, padding=(10, 10))
    outer.grid(row=0, column=0, sticky="nsew")

    # Let the outer frame expand with the measurement window.
    win.grid_rowconfigure(0, weight=1)
    win.grid_columnconfigure(0, weight=1)

    # Let the table rows and main content column expand when the window resizes.
    outer.grid_rowconfigure(1, weight=1)
    outer.grid_rowconfigure(3, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    # Create the error/status line used for triangulation failures or other
    # measurement warnings.
    app.meas_error_var = tk.StringVar(value="")
    ttk.Label(
        outer,
        textvariable=app.meas_error_var,
        foreground="red",
    ).grid(row=0, column=0, sticky="w", pady=(0, 6))

    # -------------------------------------------------------------------------
    # Points table.
    # -------------------------------------------------------------------------

    # Add the section label for the per point measurement diagnostics.
    ttk.Label(
        outer,
        text="Points (mm)",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=1, column=0, sticky="w")

    # Define the point result columns shown in the table.
    points_cols = (
        "idx",
        "X",
        "Y",
        "Z",
        "Range",
        "Disp",
        "dY",
        "ReprojRMS",
        "sZ",
        "sRange",
    )

    # Create the points table widget using heading only columns.
    points_tree = ttk.Treeview(
        outer,
        columns=points_cols,
        show="headings",
        height=8,
    )
    points_tree.grid(row=2, column=0, sticky="nsew", pady=(4, 12))

    # Label each point table column.
    points_tree.heading("idx", text="#")
    points_tree.heading("X", text="X")
    points_tree.heading("Y", text="Y")
    points_tree.heading("Z", text="Z")
    points_tree.heading("Range", text="Range")
    points_tree.heading("Disp", text="Disp (px)")
    points_tree.heading("dY", text="dY (px)")
    points_tree.heading("ReprojRMS", text="Reproj RMS (px)")
    points_tree.heading("sZ", text="σZ")
    points_tree.heading("sRange", text="σRange")

    # Set point table column widths and alignment.
    points_tree.column("idx", width=40, anchor="center")
    points_tree.column("X", width=85, anchor="e")
    points_tree.column("Y", width=85, anchor="e")
    points_tree.column("Z", width=85, anchor="e")
    points_tree.column("Range", width=95, anchor="e")
    points_tree.column("Disp", width=85, anchor="e")
    points_tree.column("dY", width=75, anchor="e")
    points_tree.column("ReprojRMS", width=105, anchor="e")
    points_tree.column("sZ", width=80, anchor="e")
    points_tree.column("sRange", width=95, anchor="e")

    # -------------------------------------------------------------------------
    # Segments table.
    # -------------------------------------------------------------------------

    # Add the section label for segment measurements between point pairs.
    ttk.Label(
        outer,
        text="Segments (mm)",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=3, column=0, sticky="w")

    # Define the segment result columns shown in the table.
    seg_cols = ("seg", "dX", "dY", "dZ", "Len", "sLen")

    # Create the segments table widget using heading only columns.
    segs_tree = ttk.Treeview(
        outer,
        columns=seg_cols,
        show="headings",
        height=8,
    )
    segs_tree.grid(row=4, column=0, sticky="nsew", pady=(4, 12))

    # Label each segment table column.
    segs_tree.heading("seg", text="Seg")
    segs_tree.heading("dX", text="dX")
    segs_tree.heading("dY", text="dY")
    segs_tree.heading("dZ", text="dZ")
    segs_tree.heading("Len", text="Len")
    segs_tree.heading("sLen", text="σLen")

    # Set segment table column widths and alignment.
    segs_tree.column("seg", width=60, anchor="center")
    segs_tree.column("dX", width=120, anchor="e")
    segs_tree.column("dY", width=120, anchor="e")
    segs_tree.column("dZ", width=120, anchor="e")
    segs_tree.column("Len", width=140, anchor="e")
    segs_tree.column("sLen", width=110, anchor="e")

    # -------------------------------------------------------------------------
    # Copy box.
    # -------------------------------------------------------------------------

    # Add the section label for copyable measurement output.
    ttk.Label(
        outer,
        text="Copy",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=5, column=0, sticky="w")

    # Create a disabled text widget that later receives copyable measurement text.
    txt = tk.Text(outer, height=7, width=1, wrap="none")
    txt.grid(row=6, column=0, sticky="nsew")

    # Keep the copy box from consuming extra vertical stretch by default.
    outer.grid_rowconfigure(6, weight=0)

    # Start disabled so users do not accidentally edit generated measurement text.
    txt.configure(state="disabled")

    # Store the window and widgets on the app object for later update calls.
    app.meas_win = win
    app.points_tree = points_tree
    app.segs_tree = segs_tree
    app.meas_copy_text = txt


# -----------------------------------------------------------------------------
# update_measurement_window
#
# Inputs: app provides the measurement window widgets and click uncertainty
# setting; points_rows and seg_rows contain already formatted measurement table
# rows; error_msg contains an optional measurement error message.
# Outputs: updates the measurement window tables, status line, and copy text box;
# returns nothing.
#
# Refreshes the measurement results window using computed point and segment rows.
# The function clears any previous table contents, inserts the latest rows, and
# builds a tab separated copy block that can be pasted into a spreadsheet. This
# function only updates display widgets; it does not compute measurement values.
# -----------------------------------------------------------------------------
def update_measurement_window(app, points_rows, seg_rows, error_msg):

    # Make sure the measurement window and its child widgets exist before trying
    # to update table rows or copy text.
    ensure_measurement_window(app)

    # Show the measurement error message if one was provided.
    app.meas_error_var.set(error_msg if error_msg else "")

    # If there is no error, show the assumed click uncertainty used for the
    # uncertainty estimates.
    if not error_msg:
        app.meas_error_var.set(f"Assumed click σ = {app.click_sigma_px:.1f} px")

    # Clear all existing point rows from the previous measurement update.
    for item in app.points_tree.get_children():
        app.points_tree.delete(item)

    # Clear all existing segment rows from the previous measurement update.
    for item in app.segs_tree.get_children():
        app.segs_tree.delete(item)

    # Insert the latest formatted point rows into the points table.
    for row in points_rows:

        # Each row is expected to match the points table column order:
        # idx, X, Y, Z, Range, Disp, dY, ReprojRMS, sZ, sRange.
        app.points_tree.insert("", "end", values=row)

    # Insert the latest formatted segment rows into the segments table.
    for row in seg_rows:

        # Each row is expected to match the segments table column order:
        # seg, dX, dY, dZ, Len, sLen.
        app.segs_tree.insert("", "end", values=row)

    # Build a tab separated copy block so the results can be pasted directly into
    # Excel, LibreOffice Calc, Google Sheets, or a text file.
    lines = []

    # Add the points section title.
    lines.append("Points")

    # Add the point diagnostics header in the same order as the points table.
    lines.append(
        "idx\tX(mm)\tY(mm)\tZ(mm)\tRange(mm)\tDisp(px)\tdY(px)"
        "\tReprojRMS(px)\tSigmaZ(mm)\tSigmaRange(mm)"
    )

    # Copy each point row in the same order as the table.
    for idx, X, Y, Z, R, disp, dy, erms, sZ, sR in points_rows:
        lines.append(f"{idx}\t{X}\t{Y}\t{Z}\t{R}\t{disp}\t{dy}\t{erms}\t{sZ}\t{sR}")

    # Add the segments section only when segment rows exist.
    if seg_rows:

        # Separate point and segment sections with a blank line.
        lines.append("")

        # Add the segments section title and header.
        lines.append("Segments")
        lines.append("seg\tdX(mm)\tdY(mm)\tdZ(mm)\tLen(mm)\tSigmaLen(mm)")

        # Copy each segment row in the same order as the table.
        for seg, dX, dY, dZ, L, sL in seg_rows:
            lines.append(f"{seg}\t{dX}\t{dY}\t{dZ}\t{L}\t{sL}")

    # Join the output lines into one text block.
    copy_block = "\n".join(lines)

    # Temporarily enable the text widget so generated output can be replaced.
    app.meas_copy_text.configure(state="normal")

    # Clear the previous copy block.
    app.meas_copy_text.delete("1.0", "end")

    # Insert the latest copyable measurement output.
    app.meas_copy_text.insert("1.0", copy_block)

    # Disable editing again so users do not accidentally modify generated output.
    app.meas_copy_text.configure(state="disabled")