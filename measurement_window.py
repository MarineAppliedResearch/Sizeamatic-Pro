"""PySide6 measurement results window for Sizeamatic Pro.

This module creates and updates the measurement output window: one
unified results table (points, segments, and a chain-total row, tagged by
a "Type" column), a copyable text block matching that table exactly, and
an accumulating "Log" that only grows when the user explicitly presses
Record — all in the same flat, spreadsheet-ready row format (leading
Video/Frame/Timestamp columns identify which frame each row came from).

This is a PySide6 port of the original Tkinter module (ROADMAP.md Phase
11) - see `main.py`'s module docstring for why the app switched
frameworks. `RESULT_COLUMNS`/`RESULT_HEADERS` and the row-shape
assumptions below are unchanged from the original; only the widgets
(`ttk.Treeview` -> `QTableWidget`, `tk.Text` -> `QPlainTextEdit`,
`tk.Toplevel` -> `qt_helpers.ClosableDialog`) changed.

Contents:
    - `MeasurementWindow` — owns the measurement results window and its
      widgets.
    - `RESULT_COLUMNS` / `RESULT_HEADERS` — the shared column order and
      display headers used by the results table, the copy block, and the
      log, so all three always agree.

Design notes:
    Points and segments used to render as two separate Treeview tables
    with two separate, differently-shaped copy formats. They're now one
    flat table with a "Type" column (see ROADMAP.md Phase 7's measurement
    output item) — the project owner wanted the *same* rows a user
    records over a session to land in one continuous, filterable block
    once pasted into a spreadsheet, rather than two separate shapes to
    juggle. Point-only columns (Disp/dY/ReprojRMS/RayResidual) and
    segment-only columns are simply blank on rows they don't apply to;
    "val_a" through "val_d" are intentionally generic (X/dX, Y/dY, Z/dZ,
    Range/Len) rather than named per row type, for the same reason.

    `reproj_rms` (pixels) and `ray_residual_mm` (millimeters) are both
    point-quality diagnostics but deliberately distinct quantities, not
    two views of the same number: `reproj_rms` measures how far the
    triangulated (Y-averaged) 3D point reprojects from the original
    clicked pixels, in pixel units; `ray_residual_mm` measures the
    closest-approach distance between the two original, un-averaged
    left/right viewing rays, in the calibration's real-world units
    (normally millimeters) - see `stereo_matching.py`'s `ray_residual_mm`
    docstring. Keeping both, with distinct labeled units, avoids
    conflating them with SeaGIS EventMeasure's "RMS" (an object-space, mm
    quantity resembling `ray_residual_mm`, not `reproj_rms`).

    The chain "Total" row's sigma is the quadrature sum of each segment's
    independently-estimated sigma (`sqrt(sum(sigma_i**2))`) — the
    standard way to propagate uncertainty across a sum of independent
    measurements — computed in `main.py` alongside the rest of the
    measurement math, not in this display-only module.

    The Log is a plain, always-editable `QPlainTextEdit`, not a
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

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qt_helpers import ClosableDialog

RESULT_COLUMNS = (
    "video",
    "frame",
    "timestamp",
    "actual_time",
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
    "ray_residual_mm",
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
gets appended to the Log.

"actual_time" is "" whenever no real-time anchor has been set (see
ROADMAP.md Phase 8's video-time-sync item) — there's nothing calculated
to show, so it's left empty rather than filled with a placeholder string
that would otherwise pollute a spreadsheet column."""

RESULT_HEADERS = {
    "video": "Video",
    "frame": "Frame",
    "timestamp": "Time",
    "actual_time": "Actual Time",
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
    "ray_residual_mm": "Ray Residual (mm)",
    "sigma1": "σZ / σLen (mm)",
    "sigma2": "σRange (mm)",
}
"""Human-readable header text for each `RESULT_COLUMNS` entry, used for
both the table column headings and the tab-separated header line in the
copy block/log."""


class MeasurementWindow:
    """Owns the measurement results dialog and its widgets.

    One instance lives on the main application (`app.measurement_window`),
    created once and reused for the lifetime of the app.
    """

    def __init__(self, app):
        """Store the owning app and initialize widget references to None.

        Args:
            app: The main application object, used as the dialog's
                parent, for the app window title, and for the assumed
                click uncertainty setting (`app.click_sigma_px`) shown in
                the status line.

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The measurement results dialog, or None if it hasn't been
        built yet (or was closed)."""

        self.results_table = None
        """The single unified results `QTableWidget` (points, segments,
        and the chain total, tagged by a "Type" column), or None if the
        window hasn't been built yet."""

        self.copy_text = None
        """The copyable current-measurement `QPlainTextEdit`, or None if
        the window hasn't been built yet."""

        self.log_text = None
        """The accumulating recorded-measurements `QPlainTextEdit`, or
        None if the window hasn't been built yet. Only grows when the
        user presses Record (`record_current_measurement`) — never
        auto-populated by `update_window`, so it doesn't fill with
        in-progress drag states."""

        self.error_label = None
        """The `QLabel` backing the error/status line, or None if the
        window hasn't been built yet."""

        self._last_rows = []
        """The most recently displayed measurement's rows (same shape
        `update_window` received), kept so `record_current_measurement`
        can append exactly what's currently shown without needing the
        caller to recompute or resend anything."""

        self._log_has_header = False
        """Whether the log widget already has the header line written.
        The header should appear exactly once at the top of the log, not
        once per Record action — repeating it would break a straight
        paste-into-spreadsheet workflow."""

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

        Clears the stored widget references. Clearing these is
        important because the next `update_window` call needs to know
        the widgets no longer exist and must be rebuilt via
        `ensure_window` first. Recorded log content is lost when the
        window closes, same as the rest of its state — there's no
        separate persistence for it.

        Returns:
            None
        """
        self.win = None
        self.results_table = None
        self.copy_text = None
        self.log_text = None
        self.error_label = None
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
        if self.win is not None:
            return

        win = ClosableDialog(self._on_close)
        win.setWindowTitle(self.app._app_window_title())
        win.resize(900, 760)

        outer = QVBoxLayout(win)

        error_label = QLabel("")
        error_label.setStyleSheet("color: #ef5350;")
        outer.addWidget(error_label)

        # -------------------------------------------------------------------------
        # Unified results table.
        # -------------------------------------------------------------------------

        results_label = QLabel("Results")
        results_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(results_label)

        results_table = QTableWidget(0, len(RESULT_COLUMNS))
        results_table.setHorizontalHeaderLabels([RESULT_HEADERS[col] for col in RESULT_COLUMNS])
        results_table.verticalHeader().setVisible(False)
        results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        outer.addWidget(results_table, stretch=1)

        # -------------------------------------------------------------------------
        # Copy box (current measurement only).
        # -------------------------------------------------------------------------

        copy_label = QLabel("Copy (current measurement)")
        copy_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(copy_label)

        copy_text = QPlainTextEdit()
        copy_text.setReadOnly(True)
        copy_text.setFixedHeight(120)
        copy_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(copy_text)

        # -------------------------------------------------------------------------
        # Log (accumulated recorded measurements).
        # -------------------------------------------------------------------------

        log_header = QHBoxLayout()
        log_label = QLabel("Log (recorded measurements — editable; select and delete lines to remove a bad one)")
        log_label.setStyleSheet("font-weight: bold;")
        log_header.addWidget(log_label, stretch=1)

        # Explicit action, not auto-logged on every recalculation — see this
        # module's "Design notes" and ROADMAP.md Phase 7.
        record_button = QPushButton("Record")
        record_button.clicked.connect(self.record_current_measurement)
        log_header.addWidget(record_button)
        outer.addLayout(log_header)

        # Deliberately left editable (never read-only) — see this module's
        # "Design notes": fixing/removing a bad recorded measurement is done
        # by editing this text directly, not through a separate UI.
        log_text = QPlainTextEdit()
        log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(log_text, stretch=1)

        # Store the window and widgets for later update/record calls.
        self.win = win
        self.results_table = results_table
        self.copy_text = copy_text
        self.log_text = log_text
        self.error_label = error_label

        win.show()

        # This window auto-opens the first time a measurement becomes
        # available - while the project owner's hands are still on the
        # mouse, over the video panes - unlike the other three sub-
        # windows, which only ever open from an explicit menu click.
        # Anchoring it to the screen's right edge (rather than dead
        # center, which would land right on top of the video/cursor)
        # keeps it out of the way of what's actually being worked on.
        # Positioned only after show() (which finalizes the window's real
        # layout-driven size - e.g. the results table's 16 columns want
        # more than the initial `resize(900, 760)` hint) so this reads
        # `win.width()`/`win.height()` accurately rather than stale.
        screen = self.app.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            x = available.x() + max(0, available.width() - win.width() - 20)
            y = available.y() + max(0, (available.height() - win.height()) // 2)
            win.move(x, y)

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
        self.ensure_window()

        # Remember these rows so a later Record click can use them without the
        # caller needing to resend anything.
        self._last_rows = list(rows)

        # Show the measurement error message if one was provided; otherwise
        # show the assumed click uncertainty used for the uncertainty
        # estimates.
        if error_msg:
            self.error_label.setText(error_msg)
        else:
            self.error_label.setText(f"Assumed click σ = {self.app.click_sigma_px:.1f} px")

        # Replace all rows in the results table with the latest measurement.
        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.results_table.setItem(r, c, QTableWidgetItem(str(value)))

        # Build the tab-separated copy block: one header line plus one line per
        # row, so it can be pasted directly into Excel, LibreOffice Calc, Google
        # Sheets, or a text file.
        lines = ["\t".join(RESULT_HEADERS[col] for col in RESULT_COLUMNS)]
        for row in rows:
            lines.append("\t".join(str(v) for v in row))
        self.copy_text.setPlainText("\n".join(lines))

    def record_current_measurement(self):
        """Append the currently displayed measurement to the Log.

        Writes the shared header line first if the Log is still empty,
        then appends one line per row from the most recent
        `update_window` call — each stamped with the same measurement ID
        (see `_next_measurement_id`), so every row from this one Record
        click can be found (and, since the Log is a plain editable
        widget, deleted) together later. Does nothing if there's no
        current measurement to record (e.g. the window was just opened,
        or the last update had zero valid rows).

        Notifies `self.app._on_measurement_recorded()` afterward so the
        app can snapshot enough state (which frame, which clicked points)
        to restore this exact measurement later from a project file.

        Returns:
            None
        """
        if not self._last_rows:
            return

        if not self._log_has_header:
            header_line = "\t".join(RESULT_HEADERS[col] for col in RESULT_COLUMNS)
            self.log_text.appendPlainText(header_line)
            self._log_has_header = True

        id_index = RESULT_COLUMNS.index("measurement_id")
        measurement_id = self._next_measurement_id

        for row in self._last_rows:
            stamped = list(row)
            stamped[id_index] = str(measurement_id)
            self.log_text.appendPlainText("\t".join(stamped))

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
        return self.log_text.toPlainText()

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

        if not text:
            self.log_text.setPlainText("")
            self._log_has_header = False
            self._next_measurement_id = 1
            return

        self.log_text.setPlainText(text)
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
