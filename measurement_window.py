"""Tkinter measurement results window for Sizeamatic Pro.

This module creates and updates the measurement output window: one
unified results table (points, segments, and a chain-total row, tagged by
a "Type" column), a copyable text block matching that table exactly, and
an accumulating "Log" that only grows when the user explicitly presses
Record — all in the same flat, spreadsheet-ready row format (leading
Video/Frame/Timestamp columns identify which frame each row came from).

Contents:
    - `MeasurementWindow` — owns the measurement results window and its
      widgets.
    - `RESULT_COLUMNS` / `RESULT_HEADERS` — the shared column order and
      display headers used by the results table, the copy block, and the
      log, so all three always agree.

Design notes:
    `MeasurementWindow` is a plain class instance owned by the main
    application (`app.measurement_window`), matching the same conversion
    already done for `calibration_summary.CalibrationSummaryWindow` — see
    that module's "Design notes" for why (removes the module-level-global
    fragility that caused `FINDINGS.md` #1).

    Points and segments used to render as two separate Treeview tables
    with two separate, differently-shaped copy formats. They're now one
    flat table with a "Type" column (see ROADMAP.md Phase 7's measurement
    output item) — the project owner wanted the *same* rows a user
    records over a session to land in one continuous, filterable block
    once pasted into a spreadsheet, rather than two separate shapes to
    juggle. Point-only columns (Disp/dY/ReprojRMS) and segment-only
    columns are simply blank on rows they don't apply to; "val_a"
    through "val_d" are intentionally generic (X/dX, Y/dY, Z/dZ,
    Range/Len) rather than named per row type, for the same reason.

    The chain "Total" row's sigma is the quadrature sum of each segment's
    independently-estimated sigma (`sqrt(sum(sigma_i**2))`) — the
    standard way to propagate uncertainty across a sum of independent
    measurements — computed in `main.py` alongside the rest of the
    measurement math, not in this display-only module.

    The Log is a plain, always-editable `tk.Text` widget, not a
    read-only display — the project owner wanted to be able to fix or
    remove a bad recorded measurement directly, and a shared
    "Measurement ID" per Record click (see `_next_measurement_id`) is
    what makes it possible to find every row from one bad click and
    delete them together. There's deliberately no structured undo/edit
    API beyond that: editing the plain tab-separated text directly is
    the whole mechanism, by request, not a stopgap.

Assumptions:
    - Rows passed into `update_window` are already computed and
      formatted, each matching `RESULT_COLUMNS`'s order exactly.
    - This module does not perform stereo triangulation or measurement
      math, and does not know what a "frame" or "video" is beyond the
      already-formatted strings it's given.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# tkinter provides the measurement results window widgets.
import tkinter as tk

# ttk provides themed Tkinter widgets such as Frame, Label, and Treeview.
from tkinter import ttk


RESULT_COLUMNS = (
    "video",
    "frame",
    "timestamp",
    "measurement_id",
    "type",
    "label",
    "val_a",
    "val_b",
    "val_c",
    "val_d",
    "disp",
    "dy_px",
    "reproj_rms",
    "sigma1",
    "sigma2",
)
"""Shared column order for the results table, the copy block, and the
log. Every row tuple passed into `MeasurementWindow.update_window` must
have exactly this many fields, in this order — there's no per-row-type
column mapping, so a "Point" row simply leaves the segment-only fields
("" in "val_c"/"val_d" isn't special-cased, it's just whatever the
caller passed) and vice versa.

"measurement_id" is always blank ("") coming out of `update_window` — the
live "current measurement" isn't part of the accumulated log, so it
doesn't need one yet. `record_current_measurement` stamps in the actual
ID (shared across every row from that one Record click, so a whole
recorded chain can be found/deleted together) only when a row actually
gets appended to the Log."""

RESULT_HEADERS = {
    "video": "Video",
    "frame": "Frame",
    "timestamp": "Time",
    "measurement_id": "Measurement ID",
    "type": "Type",
    "label": "#/Seg",
    "val_a": "X / dX (mm)",
    "val_b": "Y / dY (mm)",
    "val_c": "Z / dZ (mm)",
    "val_d": "Range / Len (mm)",
    "disp": "Disp (px)",
    "dy_px": "dY (px)",
    "reproj_rms": "Reproj RMS (px)",
    "sigma1": "σZ / σLen (mm)",
    "sigma2": "σRange (mm)",
}
"""Human-readable header text for each `RESULT_COLUMNS` entry, used for
both the Treeview column headings and the tab-separated header line in
the copy block/log."""


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

        self.results_tree = None
        """The single unified results Treeview (points, segments, and the
        chain total, tagged by a "Type" column), or None if the window
        hasn't been built yet."""

        self.copy_text = None
        """The copyable current-measurement Text widget, or None if the
        window hasn't been built yet."""

        self.log_text = None
        """The accumulating recorded-measurements Text widget, or None if
        the window hasn't been built yet. Only grows when the user
        presses Record (`record_current_measurement`) — never
        auto-populated by `update_window`, so it doesn't fill with
        in-progress drag states."""

        self.error_var = None
        """The `tkinter.StringVar` backing the error/status line, or None
        if the window hasn't been built yet."""

        self._last_rows = []
        """The most recently displayed measurement's rows (same shape
        `update_window` received), kept so `record_current_measurement`
        can append exactly what's currently shown without needing the
        caller to recompute or resend anything."""

        self._log_has_header = False
        """Whether the log Text widget already has the header line
        written. The header should appear exactly once at the top of the
        log, not once per Record action — repeating it would break a
        straight paste-into-spreadsheet workflow."""

        self._next_measurement_id = 1
        """The measurement ID `record_current_measurement` will stamp
        onto every row from its *next* call. Every row recorded from the
        same Record click shares one ID (so a whole bad chain can be
        found and deleted together in the Log), then this increments for
        the next click. Starts at 1, not 0, purely so an empty/default
        ID column reads unambiguously as "not recorded" rather than
        looking like a real ID."""

    def _on_close(self):
        """Handle the user manually closing the measurement window.

        Destroys the Tkinter window and clears the stored widget
        references. Clearing these references is important because the
        next `update_window` call needs to know the widgets no longer
        exist and must be rebuilt via `ensure_window` first. Recorded log
        content is lost when the window closes, same as the rest of its
        state — there's no separate persistence for it.

        Returns:
            None
        """

        # Destroy the Tkinter window.
        self.win.destroy()

        # Clear the stored references because the widgets were destroyed.
        self.win = None
        self.results_tree = None
        self.copy_text = None
        self.log_text = None
        self.error_var = None
        self._log_has_header = False
        self._next_measurement_id = 1

    def ensure_window(self):
        """Create the measurement results window if it doesn't already exist.

        Creates the measurement results window, including the unified
        results table, error message line, copyable text area, and the
        Record button + log. If the window already exists, the function
        exits without creating another one. Only builds the UI widgets;
        measurement values are filled in later by `update_window`.

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

        # Give the window an initial size large enough for the results table,
        # the copy box, and the log.
        win.geometry("900x760")

        # Use the cleanup callback when the user closes the measurement window.
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create one padded outer frame to hold all measurement window content.
        outer = ttk.Frame(win, padding=(10, 10))
        outer.grid(row=0, column=0, sticky="nsew")

        # Let the outer frame expand with the measurement window.
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        # Let the results table and the log both get a share of extra vertical
        # space when the window resizes; the copy box stays fixed-height.
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_rowconfigure(7, weight=1)
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
        # Unified results table.
        # -------------------------------------------------------------------------

        ttk.Label(
            outer,
            text="Results",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="w")

        results_tree = ttk.Treeview(
            outer,
            columns=RESULT_COLUMNS,
            show="headings",
            height=10,
        )
        results_tree.grid(row=2, column=0, sticky="nsew", pady=(4, 12))

        # Label and size each column from the shared RESULT_HEADERS mapping, so
        # the table, copy block, and log headers can never drift apart.
        for col in RESULT_COLUMNS:
            results_tree.heading(col, text=RESULT_HEADERS[col])

        # Narrow columns for short identifying fields, wider for the numeric
        # measurement columns.
        narrow_cols = {"video", "frame", "timestamp", "measurement_id", "type", "label"}
        for col in RESULT_COLUMNS:
            width = 70 if col not in narrow_cols else 90
            anchor = "center" if col in ("measurement_id", "type", "label") else ("w" if col == "video" else "e")
            results_tree.column(col, width=width, anchor=anchor)

        # -------------------------------------------------------------------------
        # Copy box (current measurement only).
        # -------------------------------------------------------------------------

        ttk.Label(
            outer,
            text="Copy (current measurement)",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=3, column=0, sticky="w")

        copy_txt = tk.Text(outer, height=6, width=1, wrap="none")
        copy_txt.grid(row=4, column=0, sticky="nsew", pady=(4, 12))
        outer.grid_rowconfigure(4, weight=0)
        copy_txt.configure(state="disabled")

        # -------------------------------------------------------------------------
        # Log (accumulated recorded measurements).
        # -------------------------------------------------------------------------

        log_header = ttk.Frame(outer)
        log_header.grid(row=5, column=0, sticky="ew")
        log_header.grid_columnconfigure(0, weight=1)

        ttk.Label(
            log_header,
            text="Log (recorded measurements — editable; select and delete lines to remove a bad one)",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # Explicit action, not auto-logged on every recalculation — see this
        # module's "Design notes" and ROADMAP.md Phase 7.
        ttk.Button(
            log_header,
            text="Record",
            command=self.record_current_measurement,
        ).grid(row=0, column=1, sticky="e")

        # Deliberately left editable (never disabled) — see this module's
        # "Design notes": fixing/removing a bad recorded measurement is done
        # by editing this text directly, not through a separate UI.
        log_txt = tk.Text(outer, height=10, width=1, wrap="none")
        log_txt.grid(row=7, column=0, sticky="nsew")

        # Store the window and widgets for later update/record calls.
        self.win = win
        self.results_tree = results_tree
        self.copy_text = copy_txt
        self.log_text = log_txt
        self.error_var = error_var

    def update_window(self, rows, error_msg):
        """Refresh the measurement window with the latest computed rows.

        Clears any previous table contents, inserts the latest rows into
        the unified results table, and rebuilds the "current measurement"
        copy block. Does not touch the Log — that only grows via an
        explicit `record_current_measurement` call. Only updates display
        widgets; does not compute measurement values.

        Args:
            rows (list[tuple]): Already-formatted result rows, each
                matching `RESULT_COLUMNS`'s order — a mix of "Point",
                "Segment", and (if there are 2+ points) one "Total" row,
                as built by `main.py`'s `_update_measurement_status_stub`.
            error_msg (str | None): Optional measurement error message to
                show in the status line instead of the assumed click
                uncertainty.

        Returns:
            None
        """

        # Make sure the measurement window and its child widgets exist before trying
        # to update table rows or copy text.
        self.ensure_window()

        # Remember these rows so a later Record click can use them without the
        # caller needing to resend anything.
        self._last_rows = list(rows)

        # Show the measurement error message if one was provided.
        self.error_var.set(error_msg if error_msg else "")

        # If there is no error, show the assumed click uncertainty used for the
        # uncertainty estimates.
        if not error_msg:
            self.error_var.set(f"Assumed click σ = {self.app.click_sigma_px:.1f} px")

        # Clear all existing rows from the previous measurement update.
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Insert the latest formatted rows into the unified results table.
        for row in rows:
            self.results_tree.insert("", "end", values=row)

        # Build the tab-separated copy block: one header line plus one line per
        # row, so it can be pasted directly into Excel, LibreOffice Calc, Google
        # Sheets, or a text file.
        lines = ["\t".join(RESULT_HEADERS[col] for col in RESULT_COLUMNS)]
        for row in rows:
            lines.append("\t".join(str(v) for v in row))
        copy_block = "\n".join(lines)

        # Temporarily enable the text widget so generated output can be replaced.
        self.copy_text.configure(state="normal")
        self.copy_text.delete("1.0", "end")
        self.copy_text.insert("1.0", copy_block)
        self.copy_text.configure(state="disabled")

    def record_current_measurement(self):
        """Append the currently displayed measurement to the Log.

        Writes the shared header line first if the Log is still empty,
        then appends one line per row from the most recent
        `update_window` call — each stamped with the same measurement ID
        (see `_next_measurement_id`), so every row from this one Record
        click can be found (and, since the Log is a plain editable Text
        widget, deleted) together later. Does nothing if there's no
        current measurement to record (e.g. the window was just opened,
        or the last update had zero valid rows).

        Notifies `self.app._on_measurement_recorded()` afterward so the
        app can snapshot enough state (which frame, which clicked points)
        to restore this exact measurement later from a project file — see
        ROADMAP.md Phase 7's project file item.

        Returns:
            None
        """
        if not self._last_rows:
            return

        if not self._log_has_header:
            header_line = "\t".join(RESULT_HEADERS[col] for col in RESULT_COLUMNS)
            self.log_text.insert("end", header_line + "\n")
            self._log_has_header = True

        id_index = RESULT_COLUMNS.index("measurement_id")
        measurement_id = self._next_measurement_id

        for row in self._last_rows:
            stamped = list(row)
            stamped[id_index] = str(measurement_id)
            self.log_text.insert("end", "\t".join(stamped) + "\n")

        self._next_measurement_id += 1

        self.app._on_measurement_recorded()

    def get_log_text(self):
        """Return the Log's exact current text content, for saving to a project file.

        Returns:
            str: The Log's full text content, or "" if the window/log
            hasn't been built yet.
        """
        if self.log_text is None:
            return ""
        return self.log_text.get("1.0", "end-1c")

    def restore_log_text(self, text):
        """Replace the Log's content with previously-saved text.

        Builds the window if it doesn't exist yet, then recomputes
        `_log_has_header`/`_next_measurement_id` from the restored
        content — by scanning every row's measurement_id column for the
        highest value seen — so a later Record click continues numbering
        correctly instead of restarting at 1 or duplicating the header.

        Args:
            text (str): Previously-saved Log text, as returned by
                `get_log_text`.

        Returns:
            None
        """
        self.ensure_window()

        self.log_text.delete("1.0", "end")

        if not text:
            self._log_has_header = False
            self._next_measurement_id = 1
            return

        self.log_text.insert("1.0", text)
        self._log_has_header = True

        id_index = RESULT_COLUMNS.index("measurement_id")
        max_id = 0
        for line in text.splitlines()[1:]:  # skip the header line
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) <= id_index:
                continue
            try:
                max_id = max(max_id, int(fields[id_index]))
            except ValueError:
                continue

        self._next_measurement_id = max_id + 1
