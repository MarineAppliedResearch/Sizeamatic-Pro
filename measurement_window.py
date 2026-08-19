"""PySide6 measurement results window for Sizeamatic Pro.

This module creates and updates the measurement output window: one
unified results table (points, segments, and a chain-total row, tagged by
a "Type" column) and an accumulating, editable Log that only grows when
the user explicitly presses Record — both in the same flat,
spreadsheet-ready row format (leading Video/Frame/Timestamp columns
identify which frame each row came from), each with its own Copy to
Clipboard/Export to CSV buttons.

Contents:
    - `MeasurementWindow` — owns the measurement results window and its
      widgets.
    - `RESULT_COLUMNS` / `RESULT_HEADERS` / `RESULT_TOOLTIPS` — the shared
      column order, display headers, and header-hover explanations used
      by both tables, so they always agree.
    - `SIMPLE_VIEW_COLUMNS` — the subset shown by default; a "Show
      Advanced View" toggle reveals every other column on both tables.

Design notes:
    Points and segments render as one flat table with a "Type" column
    (see ROADMAP.md Phase 7's measurement output item) rather than two
    separately-shaped tables, so a session's recorded rows land in one
    continuous, filterable block once exported. Point-only columns
    (Disp/dY/ReprojRMS/RayResidual) and segment-only columns are simply
    blank on rows they don't apply to; "val_a" through "val_d" are
    intentionally generic (X/dX, Y/dY, Z/dZ, Range/Len) rather than named
    per row type, for the same reason - the newer `range`/`distance`/
    `length`/`error` columns below are NOT generic in this same way; each
    is its own dedicated column, populated per row type, blank otherwise.

    `reproj_rms` (pixels) and `ray_residual_mm` (millimeters) are both
    point-quality diagnostics but deliberately distinct quantities:
    `reproj_rms` measures how far the triangulated (Y-averaged) 3D point
    reprojects from the original clicked pixels, in pixel units;
    `ray_residual_mm` measures the closest-approach distance between the
    two original, un-averaged left/right viewing rays, in real-world
    units - see `stereo_matching.py`'s `ray_residual_mm` docstring.

    `sigma1_jac`/`sigma2_jac` (ROADMAP.md Phase 13) are a second,
    statistically more formal uncertainty estimate shown deliberately
    side by side with the original `sigma1`/`sigma2` rather than
    replacing them outright - see `stereo_matching.estimate_point_sigma_mm_jacobian`'s
    docstring for how the two estimators differ.

    ROADMAP.md Phase 16 (usability round 3, issue #17) added: a
    Simple/Advanced view toggle (`SIMPLE_VIEW_COLUMNS`, applied via
    column visibility only - the underlying data and column order never
    change); four new dedicated columns (`range`/`angle`/`length`/`error`
    - see `RESULT_COLUMNS`'s own docstring for why a fifth, `distance`,
    was added then cut before shipping; `angle` shipped always blank
    that phase, a placeholder for the real calculation Phase 20 added
    below; `error` is `ray_residual_mm` for Point rows, and the average
    of a Segment's two endpoints' `ray_residual_mm` for Segment/Total
    rows); the Log converted from a plain-text block to a real editable
    `QTableWidget` (still user-editable per-cell, rows deleted via
    Delete/right-click with a confirmation prompt, per the project
    owner's explicit request); and Copy to Clipboard/Export to CSV
    buttons replacing the old read-only "Copy (current measurement)"
    text panel entirely.

    `RESULT_TOOLTIPS`'s text for the four new columns received the
    project owner's scientific-accuracy sign-off 2026-08-19, matching the
    precedent from Phases 12/13/15 - the Error tooltip's original draft
    (referencing the RayResidual column) was revised per that review to
    instead point at the measurement methodology whitepaper, opened via
    the new Help menu item `main.py`'s `on_open_whitepaper` adds (a
    `QToolTip` can't contain a clickable link).

    ROADMAP.md Phase 20 populated the previously-always-blank `angle`
    column (and the previously Point-only `range` column, for Segment/
    Total rows) with real calculations, computed in `main.py`'s
    `_update_measurement_status_stub`:
    - Point `range` is unchanged - each point's own distance from the
      camera, `sqrt(X**2 + Y**2 + Z**2)`.
    - Segment `range` is the average of its two endpoints' `range`
      values - a Segment has two ends, not one distance from the camera
      the way a Point does.
    - Total `range` is the average of every point's `range` across the
      whole connected chain (not just the two outer endpoints) - the
      project owner's explicit choice.
    - Segment `angle` is the segment's own orientation relative to the
      camera's viewing axis, rotated only around the vertical Y axis:
      `degrees(atan2(abs(dZ), abs(dX)))`, using the segment's own
      already-computed `dX`/`dZ`. Confirmed against this app's own
      whitepaper (`docs/Sizeamatic_Pro_Stereo_Length_Measurement_Method.pdf`),
      which already lists "objects angled toward or away from the
      stereo pair" as a cause of higher length uncertainty - this
      operationalizes that concern as a reportable number. Always in
      [0, 90] degrees by construction (`abs()` on both components,
      never a signed full-range rotation): 0 degrees means broadside/
      perpendicular to the camera (dZ ~ 0, the most reliable
      presentation), 90 degrees means pointing straight at/away from
      the camera (dX ~ 0, the least reliable). A signed value was
      deliberately rejected - a segment's start/end order is arbitrary
      (click order, not a real "facing direction"), and this app only
      ever sees the side of an object facing the camera, so there's no
      physically meaningful angle beyond 90 degrees.
    - Total `angle` is the simple arithmetic mean of every segment's
      `angle` in the chain - safe with no circular-mean wraparound
      handling needed, since `angle` is always bounded to [0, 90].

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

import csv

import qtawesome as qta
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qt_helpers import ClosableDialog

ICON_COLOR = "#e8eefc"
"""Matches `main.py`'s own `ICON_COLOR` constant - duplicated here (not
imported) since `main.py` imports this module, and this module importing
back from `main.py` would be circular."""

RECORD_BUTTON_COLOR = "#7a2e2e"
"""A warm, distinctly reddish accent for the Record button (and its
hover state) - deliberately different from the app's normal cool-blue
accent, since this specific action commits data to the permanent Log,
unlike every other button in this window."""

TYPE_COLORS = {
    "Point": QColor("#6fa8ff"),
    "Segment": QColor("#2ec4b6"),
    "Total": QColor("#f2b134"),
}
"""Text color applied to every cell in a row based on its "type" value,
identically on both the Results and Log tables - lets a user tell
Point/Segment/Total rows apart at a glance across the whole row, not
just by reading the Type cell itself."""

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
    "sigma1_jac",
    "sigma2_jac",
    "range",
    "angle",
    "length",
    "error",
)
"""Shared column order for both tables. Every row tuple passed into
`MeasurementWindow.update_window` must have exactly this many fields, in
this order. The four trailing columns (ROADMAP.md Phase 16) are
appended, never inserted between existing ones - this is what keeps
`restore_log_text` trivially backward-compatible with a project file
saved before this phase (an old, shorter row just gets padded with
blanks for these fields, never a schema mismatch). (A fifth trailing
column, "distance", was added then removed again during this same
phase's manual proof-testing, before ever shipping - it turned out
redundant with "range" once "range" stayed Point-only rather than
gaining a Segment/Total meaning, so it was cut outright rather than left
as a permanently-blank dead column.)

"measurement_id" is always blank ("") coming out of `update_window` - the
live "current measurement" isn't part of the accumulated log yet.
`record_current_measurement` stamps in the actual ID (shared across every
row from one Record click) only when a row gets appended to the Log.

"actual_time" is "" whenever no real-time anchor has been set."""

RESULT_HEADERS = {
    "video": "Video Name",
    "frame": "Video Frame",
    "timestamp": "Video Time",
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
    "sigma1_jac": "σZ / σLen (Jacobian, mm)",
    "sigma2_jac": "σRange (Jacobian, mm)",
    "range": "Range (mm)",
    "angle": "Angle (°)",
    "length": "Length (mm)",
    "error": "Error (mm)",
}
"""Human-readable header text for each `RESULT_COLUMNS` entry."""

SIMPLE_VIEW_COLUMNS = (
    "type",
    "length",
    "range",
    "angle",
    "error",
    "actual_time",
    "timestamp",
    "frame",
    "video",
)
"""The columns shown by default ("Simple View") - the exact order the
project owner requested. "type" leads so a user knows how to read the
numbers that follow before they even get to them - without it, a Point's
and a Segment's numbers looked identical at a glance (a real gap found
during this phase's manual proof-testing). Every other `RESULT_COLUMNS`
entry is hidden until "Show Advanced View" is toggled on. This only
controls column *visibility*; the underlying column order
(`VISUAL_COLUMN_ORDER` below) and the data itself never change."""

VISUAL_COLUMN_ORDER = SIMPLE_VIEW_COLUMNS + tuple(c for c in RESULT_COLUMNS if c not in SIMPLE_VIEW_COLUMNS)
"""On-screen left-to-right column order for both tables: the Simple View
columns first (in their requested order), then everything else. Applied
once per table via `_apply_visual_column_order` - this reorders Qt's
*visual* column positions, never the *logical* ones `RESULT_COLUMNS.index(...)`
addresses, so row-building code elsewhere never needs to know about it."""

RESULT_TOOLTIPS = {
    "video": "The loaded left video's filename.",
    "frame": "The left video's frame number this measurement was taken on.",
    "timestamp": "Elapsed time into the left video (HH:MM:SS:FF).",
    "actual_time": "The calculated real-world date/time, once a real-time sync anchor has been set. Blank if no anchor is set.",
    "measurement_id": "Shared ID for every row recorded from the same Record click - lets a whole recorded chain be found (and deleted) together later.",
    "type": (
        "What kind of row this is - the whole row is color-coded by this value so it's "
        "easy to tell apart at a glance: a Point is one clicked location; a Segment is the "
        "straight-line distance between two consecutive points; a Total sums every segment "
        "in a connected chain into one overall length."
    ),
    "label": "The point's index (0, 1, 2, ...), or a segment's two connected point indices (e.g. \"0-1\").",
    "val_a": "X (Point) or the X difference between segment endpoints (Segment), in millimeters.",
    "val_b": "Y (Point) or the Y difference between segment endpoints (Segment), in millimeters.",
    "val_c": "Z (Point) or the Z difference between segment endpoints (Segment), in millimeters.",
    "val_d": "Range (Point: straight-line distance from the camera) or Len (Segment: this segment's 3D length), in millimeters.",
    "disp": "Horizontal pixel difference between the left and right clicks - drives the computed depth.",
    "dy_px": "Vertical pixel difference between the left and right clicks - near zero in a well-rectified pair.",
    "reproj_rms": (
        "How far your clicked pixels are from where the computed 3D point would land if "
        "projected back into the images, in pixels. Smaller is better; a large value means "
        "the clicked point doesn't fit the calibration well."
    ),
    "ray_residual_mm": (
        "How well your left and right clicks agree with each other, in millimeters - the two "
        "cameras' sightlines to your two clicks should meet at exactly the real point in "
        "space, and this is how far apart they actually pass instead. Smaller is better; a "
        "large value usually means one of the two clicks landed on the wrong spot."
    ),
    "sigma1": "Estimated uncertainty in Z (Point) or Length (Segment/Total), in millimeters - sample standard deviation of several slightly-perturbed re-measurements.",
    "sigma2": "Estimated uncertainty in Range, in millimeters - sample standard deviation of several slightly-perturbed re-measurements.",
    "sigma1_jac": "Same as σZ / σLen, but from a more formal statistical estimate (Jacobian/covariance propagation) instead of a sample standard deviation.",
    "sigma2_jac": "Same as σRange, but from a more formal statistical estimate (Jacobian/covariance propagation) instead of a sample standard deviation.",
    "range": (
        "Straight-line distance from the camera, in millimeters: for a Point, its own distance; "
        "for a Segment, the average of its two endpoints' distances; for a Total, the average "
        "across every point in the chain."
    ),
    "angle": (
        "How the measurement is oriented relative to the camera, in degrees: 0° means broadside/"
        "perpendicular to the camera (the most reliable presentation), 90° means pointing "
        "straight at or away from the camera (the least reliable - see the measurement "
        "methodology whitepaper (Help menu) for why). Blank on Point rows, which don't have an "
        "orientation of their own. On a Total row, this is the average of every segment's Angle "
        "in the chain."
    ),
    "length": "This segment's real-world length, or (on a Total row) the summed length of every connected segment in the chain, in millimeters. Blank on individual Point rows, which don't have a length of their own.",
    "error": (
        "How much you can trust this measurement, in millimeters: for a Point, how well its "
        "left and right clicks agree with each other; for a Segment or Total, the average of "
        "that across its endpoint(s). Smaller is better - closer to zero means your clicks "
        "were precise and the measurement is reliable. See the measurement methodology "
        "whitepaper (Help menu) for the full derivation."
    ),
}
"""Header-hover tooltip text for each column. Text for the four new
Phase 16 columns (and text reused for existing columns moved into this
new tooltip surface) received the project owner's scientific-accuracy
sign-off 2026-08-19 - see this module's docstring."""

_TYPE_COLUMN_INDEX = RESULT_COLUMNS.index("type")


def _make_row_items(row):
    """Build one `QTableWidgetItem` per field in a row, color-coding the
    whole row by its "type" value so Point/Segment/Total rows are
    distinguishable at a glance, not just the Type cell itself.

    Args:
        row (tuple[str]): A full `RESULT_COLUMNS`-shaped row (or a padded
            one - see `restore_log_text`).

    Returns:
        list[QTableWidgetItem]: One item per field, in column order.
    """
    color = TYPE_COLORS.get(row[_TYPE_COLUMN_INDEX]) if len(row) > _TYPE_COLUMN_INDEX else None
    items = []
    for value in row:
        item = QTableWidgetItem(str(value))
        if color is not None:
            item.setForeground(color)
        items.append(item)
    return items


def _apply_visual_column_order(table):
    """Reorder a table's on-screen columns to match `VISUAL_COLUMN_ORDER`.

    Args:
        table (QTableWidget): The table to reorder (moves visual
            sections only - never touches logical column indices).

    Returns:
        None
    """
    header = table.horizontalHeader()
    for target_visual, col in enumerate(VISUAL_COLUMN_ORDER):
        logical = RESULT_COLUMNS.index(col)
        current_visual = header.visualIndex(logical)
        if current_visual != target_visual:
            header.moveSection(current_visual, target_visual)


def _apply_view_mode(table, simple):
    """Show/hide a table's columns to match the current Simple/Advanced toggle.

    Args:
        table (QTableWidget): The table to update.
        simple (bool): True to hide every column not in
            `SIMPLE_VIEW_COLUMNS`; False to show every column.

    Returns:
        None
    """
    for col in RESULT_COLUMNS:
        logical = RESULT_COLUMNS.index(col)
        table.setColumnHidden(logical, simple and col not in SIMPLE_VIEW_COLUMNS)


def _visible_columns_in_order(table):
    """Get a table's currently-visible logical column indices, left to right.

    Args:
        table (QTableWidget): The table to inspect.

    Returns:
        list[int]: Logical column indices, in on-screen visual order,
        excluding any column currently hidden by the Simple/Advanced
        toggle.
    """
    header = table.horizontalHeader()
    return [
        header.logicalIndex(v)
        for v in range(table.columnCount())
        if not table.isColumnHidden(header.logicalIndex(v))
    ]


def _table_rows_as_lines(table):
    """Build tab-separated lines from a table's currently-visible columns.

    Used by Copy to Clipboard/Export to CSV, which respect whatever the
    Simple/Advanced toggle currently shows - unlike `get_log_text`, which
    always serializes every column for the project file regardless of
    what's currently visible on screen.

    Args:
        table (QTableWidget): The table to read.

    Returns:
        list[str]: One tab-separated header line, then one per row.
    """
    cols = _visible_columns_in_order(table)
    lines = ["\t".join(RESULT_HEADERS[RESULT_COLUMNS[c]] for c in cols)]
    for r in range(table.rowCount()):
        lines.append("\t".join(table.item(r, c).text() if table.item(r, c) else "" for c in cols))
    return lines


def write_table_rows_csv(path, header, rows):
    """Write a header row plus data rows to a CSV file.

    Dialog-free so it's directly testable, matching this project's usual
    split between a thin file-dialog wrapper and the actual logic (see
    `calibration_io.py`/`project_io.py`).

    Args:
        path (str): Destination file path.
        header (list[str]): Column header labels.
        rows (list[list[str]]): Data rows, same column count as `header`.

    Returns:
        None
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


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
        window hasn't been built yet. Not user-editable - it's fully
        regenerated on every point change, so nothing typed into it
        would ever persist."""

        self.log_table = None
        """The accumulating recorded-measurements `QTableWidget`, or None
        if the window hasn't been built yet. Only grows when the user
        presses Record (`record_current_measurement`) - never
        auto-populated by `update_window`. Cells stay user-editable, and
        rows can be deleted (Delete key or right-click) to fix/remove a
        bad recorded measurement."""

        self.error_label = None
        """The `QLabel` backing the error/status line, or None if the
        window hasn't been built yet."""

        self.record_button = None
        """The "Record" `QPushButton`, or None if the window hasn't been
        built yet - stored on `self` (not just a local variable in
        `ensure_window`) so ROADMAP.md Phase 15's Tutorial mode can
        highlight it directly."""

        self._view_mode_button = None
        """The "Show Advanced View"/"Show Simple View" toggle button, or
        None if the window hasn't been built yet."""

        self.results_copy_button = None
        """The Results table's "Copy to Clipboard" `QPushButton`, or None
        if the window hasn't been built yet - stored on `self` (not just
        a local variable) so Tutorial mode can highlight it directly."""

        self._view_mode = "simple"
        """Either "simple" or "advanced" - which columns are currently
        visible on both tables. Not persisted anywhere (resets to
        "simple" every time the window is rebuilt), matching how the
        rest of this window's layout state isn't saved either."""

        self._last_rows = []
        """The most recently displayed measurement's rows (same shape
        `update_window` received), kept so `record_current_measurement`
        can append exactly what's currently shown without needing the
        caller to recompute or resend anything."""

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

        Clears the stored widget references so the next `update_window`
        call knows to rebuild via `ensure_window` first. Recorded log
        content is lost when the window closes, same as the rest of its
        state - there's no separate persistence for it.

        Returns:
            None
        """
        self.win = None
        self.results_table = None
        self.log_table = None
        self.error_label = None
        self.record_button = None
        self._view_mode_button = None
        self.results_copy_button = None
        self._view_mode = "simple"
        self._next_measurement_id = 1

    def ensure_window(self):
        """Create the measurement results window if it doesn't already exist.

        Creates the measurement results window, including the unified
        results table, error message line, and the Record button + Log
        table. If the window already exists, exits without creating
        another one. Only builds the UI widgets; measurement values are
        filled in later by `update_window`.

        Returns:
            None
        """
        if self.win is not None:
            return

        win = ClosableDialog(self._on_close)
        win.setWindowTitle(self.app._app_window_title())
        win.resize(1100, 800)

        outer = QVBoxLayout(win)

        error_label = QLabel("")
        error_label.setStyleSheet("color: #ef5350;")
        outer.addWidget(error_label)

        # -------------------------------------------------------------------------
        # Unified results table.
        # -------------------------------------------------------------------------

        results_header = QHBoxLayout()
        results_label = QLabel("Results")
        results_label.setStyleSheet("font-weight: bold;")
        results_header.addWidget(results_label, stretch=1)

        self._view_mode_button = QPushButton("Show Advanced View")
        self._view_mode_button.clicked.connect(self._on_toggle_view_mode)
        results_header.addWidget(self._view_mode_button)
        outer.addLayout(results_header)

        results_table = QTableWidget(0, len(RESULT_COLUMNS))
        results_table.setHorizontalHeaderLabels([RESULT_HEADERS[col] for col in RESULT_COLUMNS])
        results_table.verticalHeader().setVisible(False)
        results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        _apply_visual_column_order(results_table)
        self._set_header_tooltips(results_table)
        outer.addWidget(results_table, stretch=1)
        results_row, self.results_copy_button = self._build_copy_export_row(results_table)
        outer.addLayout(results_row)

        # -------------------------------------------------------------------------
        # Log (accumulated recorded measurements).
        # -------------------------------------------------------------------------

        log_header = QHBoxLayout()
        log_label = QLabel("Log (recorded measurements — editable; select row(s) and press Delete to remove one)")
        log_label.setStyleSheet("font-weight: bold;")
        log_header.addWidget(log_label, stretch=1)

        # Explicit action, not auto-logged on every recalculation - see
        # this module's "Design notes" and ROADMAP.md Phase 7.
        self.record_button = QPushButton("Record")
        self.record_button.setIcon(qta.icon("fa5s.circle", color=ICON_COLOR))
        self.record_button.setStyleSheet(
            f"QPushButton {{ background-color: {RECORD_BUTTON_COLOR}; }} "
            f"QPushButton:hover {{ background-color: #9c3b3b; }}"
        )
        self.record_button.clicked.connect(self.record_current_measurement)
        log_header.addWidget(self.record_button)
        outer.addLayout(log_header)

        log_table = QTableWidget(0, len(RESULT_COLUMNS))
        log_table.setHorizontalHeaderLabels([RESULT_HEADERS[col] for col in RESULT_COLUMNS])
        log_table.verticalHeader().setVisible(False)
        log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        log_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        log_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        log_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        log_table.customContextMenuRequested.connect(lambda pos: self._show_log_context_menu(log_table, pos))
        _apply_visual_column_order(log_table)
        self._set_header_tooltips(log_table)
        outer.addWidget(log_table, stretch=1)
        log_row, _log_copy_button = self._build_copy_export_row(log_table)
        outer.addLayout(log_row)

        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), log_table)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_shortcut.activated.connect(self._delete_selected_log_rows)

        # Store the window and widgets for later update/record calls.
        self.win = win
        self.results_table = results_table
        self.log_table = log_table
        self.error_label = error_label

        _apply_view_mode(self.results_table, True)
        _apply_view_mode(self.log_table, True)

        # Size columns from their header text now, not just after the
        # first row lands - an empty table's columns otherwise default to
        # a fixed generic width, clipping/overlapping longer header text
        # (e.g. "Distance (mm)") until Log gets its first recorded row.
        self.results_table.resizeColumnsToContents()
        self.log_table.resizeColumnsToContents()

        win.show()

        # This window auto-opens the first time a measurement becomes
        # available - while the project owner's hands are still on the
        # mouse, over the video panes - unlike the other three sub-
        # windows, which only ever open from an explicit menu click.
        # Anchoring it to the screen's right edge (rather than dead
        # center, which would land right on top of the video/cursor)
        # keeps it out of the way of what's actually being worked on.
        # Positioned only after show() (which finalizes the window's real
        # layout-driven size) so this reads `win.width()`/`win.height()`
        # accurately rather than stale.
        screen = self.app.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            x = available.x() + max(0, available.width() - win.width() - 20)
            y = available.y() + max(0, (available.height() - win.height()) // 2)
            win.move(x, y)

    def _set_header_tooltips(self, table):
        """Attach `RESULT_TOOLTIPS`' hover text to every column header.

        Args:
            table (QTableWidget): The table whose headers to annotate.

        Returns:
            None
        """
        for i, col in enumerate(RESULT_COLUMNS):
            item = table.horizontalHeaderItem(i)
            if item is not None:
                item.setToolTip(RESULT_TOOLTIPS[col])

    def _build_copy_export_row(self, table):
        """Build a "Copy to Clipboard"/"Export to CSV" button row for one table.

        Args:
            table (QTableWidget): The table these buttons act on.

        Returns:
            tuple[QHBoxLayout, QPushButton]: The button row (not yet
            added to any layout) and the Copy to Clipboard button itself
            (the caller may want to store it, e.g. so Tutorial mode can
            highlight it).
        """
        row = QHBoxLayout()
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.setIcon(qta.icon("fa5s.copy", color=ICON_COLOR))
        copy_button.clicked.connect(lambda: self._copy_table_to_clipboard(table))
        row.addWidget(copy_button)

        export_button = QPushButton("Export to CSV")
        export_button.setIcon(qta.icon("fa5s.file-csv", color=ICON_COLOR))
        export_button.clicked.connect(lambda: self._export_table_csv(table))
        row.addWidget(export_button)
        row.addStretch(1)
        return row, copy_button

    def _copy_table_to_clipboard(self, table):
        """Copy a table's currently-visible columns to the clipboard.

        Args:
            table (QTableWidget): The table to copy from.

        Returns:
            None
        """
        QApplication.clipboard().setText("\n".join(_table_rows_as_lines(table)))
        self.app.tutorial_window.notify_action("copy_to_clipboard")

    def _export_table_csv(self, table):
        """Prompt for a save path and write a table's currently-visible
        columns to it as CSV.

        Args:
            table (QTableWidget): The table to export.

        Returns:
            None
        """
        path, _filter = QFileDialog.getSaveFileName(self.win, "Export to CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        cols = _visible_columns_in_order(table)
        header = [RESULT_HEADERS[RESULT_COLUMNS[c]] for c in cols]
        rows = [
            [table.item(r, c).text() if table.item(r, c) else "" for c in cols] for r in range(table.rowCount())
        ]
        write_table_rows_csv(path, header, rows)

    def _on_toggle_view_mode(self):
        """Flip between Simple and Advanced View on both tables.

        Returns:
            None
        """
        self._view_mode = "advanced" if self._view_mode == "simple" else "simple"
        simple = self._view_mode == "simple"
        _apply_view_mode(self.results_table, simple)
        _apply_view_mode(self.log_table, simple)
        self.results_table.resizeColumnsToContents()
        self.log_table.resizeColumnsToContents()
        self._view_mode_button.setText("Show Simple View" if not simple else "Show Advanced View")

    def _show_log_context_menu(self, log_table, pos):
        """Show a right-click "Delete Row(s)" menu on the Log table.

        Args:
            log_table (QTableWidget): The Log table.
            pos (QPoint): Local position the context menu was requested at.

        Returns:
            None
        """
        menu = QMenu(log_table)
        delete_action = menu.addAction("Delete Row(s)")
        delete_action.triggered.connect(self._delete_selected_log_rows)
        menu.exec(log_table.viewport().mapToGlobal(pos))

    def _delete_selected_log_rows(self):
        """Delete the Log's currently-selected row(s), after confirming.

        Per the project owner's explicit request: unlike the old
        plain-text Log (where deleting meant manually selecting exact
        text spans), a real table makes a whole-row delete one keypress
        - a confirmation prompt keeps that from being an accidental,
        irreversible click.

        Returns:
            None
        """
        rows = sorted({idx.row() for idx in self.log_table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        reply = QMessageBox.question(
            self.win,
            "Delete Row(s)",
            f"Delete {len(rows)} row(s) from the Log?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for r in rows:
            self.log_table.removeRow(r)

    def update_window(self, rows, error_msg):
        """Refresh the measurement window with the latest computed rows.

        Clears any previous results and inserts the latest rows into the
        unified results table. Does not touch the Log - that only grows
        via an explicit `record_current_measurement` call.

        Args:
            rows (list[tuple]): Already-formatted result rows, each
                matching `RESULT_COLUMNS`'s order - a mix of "Point",
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

        if error_msg:
            self.error_label.setText(error_msg)
        else:
            self.error_label.setText(f"Assumed click σ = {self.app.click_sigma_px:.1f} px")

        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, item in enumerate(_make_row_items(row)):
                self.results_table.setItem(r, c, item)
        self.results_table.resizeColumnsToContents()

    def record_current_measurement(self):
        """Append the currently displayed measurement to the Log.

        Inserts one row per row from the most recent `update_window`
        call, each stamped with the same measurement ID (see
        `_next_measurement_id`), so every row from this one Record click
        can be found (and deleted) together later. Does nothing if
        there's no current measurement to record (e.g. the window was
        just opened, or the last update had zero valid rows).

        Notifies `self.app._on_measurement_recorded()` afterward so the
        app can snapshot enough state to restore this exact measurement
        later from a project file, and
        `self.app.tutorial_window.notify_action("record_measurement")`
        for ROADMAP.md Phase 15's Tutorial mode completion detection.

        Returns:
            None
        """
        if not self._last_rows:
            return

        id_index = RESULT_COLUMNS.index("measurement_id")
        measurement_id = self._next_measurement_id

        for row in self._last_rows:
            stamped = list(row)
            stamped[id_index] = str(measurement_id)
            r = self.log_table.rowCount()
            self.log_table.insertRow(r)
            for c, item in enumerate(_make_row_items(stamped)):
                self.log_table.setItem(r, c, item)
        self.log_table.resizeColumnsToContents()

        self._next_measurement_id += 1

        self.app._on_measurement_recorded()
        self.app.tutorial_window.notify_action("record_measurement")

    def get_log_text(self):
        """Serialize the Log's full content to tab-separated text, for
        saving to a project file.

        Always includes every `RESULT_COLUMNS` field regardless of the
        current Simple/Advanced toggle - the project file needs the full
        data, not just whatever happens to be visible on screen (unlike
        `_table_rows_as_lines`, which the Copy/Export buttons use).

        Returns:
            str: The Log's full content as one header line plus one line
            per row, or "" if the window/log hasn't been built yet or is
            empty.
        """
        if self.log_table is None or self.log_table.rowCount() == 0:
            return ""
        lines = ["\t".join(RESULT_HEADERS[col] for col in RESULT_COLUMNS)]
        for r in range(self.log_table.rowCount()):
            lines.append(
                "\t".join(
                    self.log_table.item(r, c).text() if self.log_table.item(r, c) else ""
                    for c in range(len(RESULT_COLUMNS))
                )
            )
        return "\n".join(lines)

    def restore_log_text(self, text):
        """Replace the Log's content with previously-saved text.

        Builds the window if it doesn't exist yet. Each line is
        padded/truncated to `RESULT_COLUMNS`'s current width before
        inserting, so a project saved before Phase 16's five new columns
        existed restores cleanly - those fields just come back blank on
        old rows, never a crash or schema-mismatch error. Recomputes
        `_next_measurement_id` from the restored content by scanning
        every row's measurement_id column for the highest value seen.

        Args:
            text (str): Previously-saved Log text, as returned by
                `get_log_text`.

        Returns:
            None
        """
        self.ensure_window()
        self.log_table.setRowCount(0)

        if not text:
            self._next_measurement_id = 1
            return

        id_index = RESULT_COLUMNS.index("measurement_id")
        max_id = 0
        for line in text.splitlines()[1:]:  # header line skipped unconditionally, never validated
            if not line.strip():
                continue
            fields = line.split("\t")
            fields = (fields + [""] * len(RESULT_COLUMNS))[: len(RESULT_COLUMNS)]

            r = self.log_table.rowCount()
            self.log_table.insertRow(r)
            for c, item in enumerate(_make_row_items(fields)):
                self.log_table.setItem(r, c, item)

            try:
                max_id = max(max_id, int(fields[id_index]))
            except ValueError:
                pass

        self.log_table.resizeColumnsToContents()
        self._next_measurement_id = max_id + 1
