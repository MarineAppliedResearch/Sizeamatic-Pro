"""Tkinter measurement results window for Sizeamatic Pro.

This module creates and updates the measurement output window, including
the point diagnostics table, segment table, error/status line, and
copyable text output for spreadsheet use.

Contents:
    - `MeasurementWindow` — owns the measurement results window and its
      widgets.

Design notes:
    `MeasurementWindow` is a plain class instance owned by the main
    application (`app.measurement_window`), matching the same conversion
    already done for `calibration_summary.CalibrationSummaryWindow` — see
    that module's "Design notes" for why (removes the module-level-global
    fragility that caused `FINDINGS.md` #1).

Assumptions:
    - Measurement rows passed into `update_window` are already computed
      and formatted.
    - This module does not perform stereo triangulation or measurement
      math.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# tkinter provides the measurement results window widgets.
import tkinter as tk

# ttk provides themed Tkinter widgets such as Frame, Label, and Treeview.
from tkinter import ttk


class MeasurementWindow:
    """Owns the measurement results Toplevel window and its widgets.

    One instance lives on the main application (`app.measurement_window`),
    created once and reused for the lifetime of the app.
    """

    def __init__(self, app):
        """Store the owning app and initialize widget references to None.

        Args:
            app: The main application object, used for the Tk root window
                that owns the measurement Toplevel, and for the assumed
                click uncertainty setting (`app.click_sigma_px`) shown in
                the status line.

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The measurement results Toplevel window, or None if it hasn't
        been built yet (or was closed)."""

        self.points_tree = None
        """The per-point diagnostics Treeview table, or None if the
        window hasn't been built yet."""

        self.segs_tree = None
        """The segment measurements Treeview table, or None if the window
        hasn't been built yet."""

        self.copy_text = None
        """The copyable measurement results Text widget, or None if the
        window hasn't been built yet."""

        self.error_var = None
        """The `tkinter.StringVar` backing the error/status line, or None
        if the window hasn't been built yet."""

    def _on_close(self):
        """Handle the user manually closing the measurement window.

        Destroys the Tkinter window and clears the stored widget
        references. Clearing these references is important because the
        next `update_window` call needs to know the widgets no longer
        exist and must be rebuilt via `ensure_window` first.

        Returns:
            None
        """

        # Destroy the Tkinter window.
        self.win.destroy()

        # Clear the stored references because the widgets were destroyed.
        self.win = None
        self.points_tree = None
        self.segs_tree = None
        self.copy_text = None
        self.error_var = None

    def ensure_window(self):
        """Create the measurement results window if it doesn't already exist.

        Creates the measurement results window, including the point
        diagnostics table, segment measurement table, error message
        line, and copyable text area. If the window already exists, the
        function exits without creating another one. Only builds the UI
        widgets; measurement values are filled in later by
        `update_window`.

        Returns:
            None
        """

        # If the measurement window already exists, reuse it instead of
        # creating a duplicate Toplevel window.
        if self.win is not None:
            return

        # Create a separate top level window owned by the main application root.
        win = tk.Toplevel(self.app.root)

        # Set the user visible title for the measurement results window.
        win.title("Measurement")

        # Give the window an initial size large enough for both result tables and the
        # copyable text box.
        win.geometry("620x520")

        # Use the cleanup callback when the user closes the measurement window.
        win.protocol("WM_DELETE_WINDOW", self._on_close)

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
        error_var = tk.StringVar(value="")
        ttk.Label(
            outer,
            textvariable=error_var,
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

        # Store the window and widgets for later update calls.
        self.win = win
        self.points_tree = points_tree
        self.segs_tree = segs_tree
        self.copy_text = txt
        self.error_var = error_var

    def update_window(self, points_rows, seg_rows, error_msg):
        """Refresh the measurement window with the latest computed rows.

        Clears any previous table contents, inserts the latest rows, and
        builds a tab separated copy block that can be pasted into a
        spreadsheet. Only updates display widgets; does not compute
        measurement values.

        Args:
            points_rows (list[tuple]): Already-formatted point diagnostic
                rows, each `(idx, X, Y, Z, Range, Disp, dY, ReprojRMS, sZ,
                sRange)`, matching the points table column order.
            seg_rows (list[tuple]): Already-formatted segment rows, each
                `(seg, dX, dY, dZ, Len, sLen)`, matching the segments
                table column order.
            error_msg (str | None): Optional measurement error message to
                show in the status line instead of the assumed click
                uncertainty.

        Returns:
            None
        """

        # Make sure the measurement window and its child widgets exist before trying
        # to update table rows or copy text.
        self.ensure_window()

        # Show the measurement error message if one was provided.
        self.error_var.set(error_msg if error_msg else "")

        # If there is no error, show the assumed click uncertainty used for the
        # uncertainty estimates.
        if not error_msg:
            self.error_var.set(f"Assumed click σ = {self.app.click_sigma_px:.1f} px")

        # Clear all existing point rows from the previous measurement update.
        for item in self.points_tree.get_children():
            self.points_tree.delete(item)

        # Clear all existing segment rows from the previous measurement update.
        for item in self.segs_tree.get_children():
            self.segs_tree.delete(item)

        # Insert the latest formatted point rows into the points table.
        for row in points_rows:

            # Each row is expected to match the points table column order:
            # idx, X, Y, Z, Range, Disp, dY, ReprojRMS, sZ, sRange.
            self.points_tree.insert("", "end", values=row)

        # Insert the latest formatted segment rows into the segments table.
        for row in seg_rows:

            # Each row is expected to match the segments table column order:
            # seg, dX, dY, dZ, Len, sLen.
            self.segs_tree.insert("", "end", values=row)

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
        self.copy_text.configure(state="normal")

        # Clear the previous copy block.
        self.copy_text.delete("1.0", "end")

        # Insert the latest copyable measurement output.
        self.copy_text.insert("1.0", copy_block)

        # Disable editing again so users do not accidentally modify generated output.
        self.copy_text.configure(state="disabled")
