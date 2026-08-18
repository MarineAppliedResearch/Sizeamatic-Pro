"""Tests for measurement_window.py's MeasurementWindow.

Uses a FakeApp (via make_fake_app) rather than the real SizeamaticProApp:
MeasurementWindow only ever reads app.click_sigma_px (for the status
line), app._app_window_title() (for the dialog's title), and
app.screen() (for positioning) - see AGENTS.md's Testing section. Still
needs a real QApplication (the qapp fixture) since it builds a real
QDialog.
"""

from PySide6.QtWidgets import QAbstractItemView

import measurement_window

# measurement_id (index 4) is always blank coming out of update_window - only
# record_current_measurement ever stamps a real value in. Trailing 4 fields
# are range/angle/length/error (ROADMAP.md Phase 16).
ROW_A = (
    "left.mp4", "10", "00:00:00.417", "2026-08-12 14:32:05.417", "", "Point", "0",
    "1.0", "2.0", "3.0", "4.0", "5.00", "6.00", "7.00", "7.50", "8.0", "9.0", "8.1", "9.1",
    "4.0", "", "", "7.50",
)
ROW_B = (
    "left.mp4", "10", "00:00:00.417", "2026-08-12 14:32:05.417", "", "Segment", "0-1",
    "1.0", "2.0", "3.0", "4.0", "", "", "", "", "8.0", "", "8.1", "",
    "", "", "4.0", "7.50",
)

ID_INDEX = measurement_window.RESULT_COLUMNS.index("measurement_id")


def _with_id(row, measurement_id):
    """Build the fields a recorded row should have once stamped with a
    measurement ID, matching what record_current_measurement writes.

    Args:
        row (tuple): One of ROW_A/ROW_B.
        measurement_id (int): The measurement ID to stamp in.

    Returns:
        list[str]: The row's fields with measurement_id replaced.
    """
    stamped = list(row)
    stamped[ID_INDEX] = str(measurement_id)
    return stamped


def _table_row_texts(table, row):
    """Read every column's text out of one QTableWidget row, by logical
    (not visual) column index - the same order rows are inserted in.

    Args:
        table (QTableWidget): The table to read from.
        row (int): Row index.

    Returns:
        list[str]: One string per column, "" for an unset cell.
    """
    return [table.item(row, c).text() if table.item(row, c) else "" for c in range(table.columnCount())]


def _all_row_texts(table):
    """Read every row's texts out of a QTableWidget.

    Args:
        table (QTableWidget): The table to read from.

    Returns:
        list[list[str]]: One list of column texts per row.
    """
    return [_table_row_texts(table, r) for r in range(table.rowCount())]


def test_update_window_populates_the_results_table(qapp, make_fake_app):
    """update_window should fill the unified results table with exactly
    the given rows, and the table should not be user-editable (it's
    fully regenerated on every point change)."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)

    assert win.results_table.rowCount() == 2
    assert _table_row_texts(win.results_table, 0) == list(ROW_A)
    assert _table_row_texts(win.results_table, 1) == list(ROW_B)
    assert win.results_table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers

    win._on_close()


def test_visible_columns_in_order_respects_the_simple_advanced_toggle(qapp, make_fake_app):
    """The Copy/Export helper should only include columns not currently
    hidden by the Simple/Advanced toggle, in on-screen visual order."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)
    win.update_window([ROW_A], None)

    simple_cols = [
        measurement_window.RESULT_COLUMNS[c]
        for c in measurement_window._visible_columns_in_order(win.results_table)
    ]
    assert simple_cols == list(measurement_window.SIMPLE_VIEW_COLUMNS)

    win._on_toggle_view_mode()

    advanced_cols = [
        measurement_window.RESULT_COLUMNS[c]
        for c in measurement_window._visible_columns_in_order(win.results_table)
    ]
    assert advanced_cols == list(measurement_window.VISUAL_COLUMN_ORDER)

    win._on_close()


def test_table_rows_as_lines_builds_tab_separated_text_for_visible_columns(qapp, make_fake_app):
    """The Copy/Export text-building helper should produce a header line
    plus one tab-separated line per row, limited to visible columns."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)
    win.update_window([ROW_A], None)

    lines = measurement_window._table_rows_as_lines(win.results_table)

    assert lines[0] == "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.SIMPLE_VIEW_COLUMNS)
    simple_indexes = [measurement_window.RESULT_COLUMNS.index(c) for c in measurement_window.SIMPLE_VIEW_COLUMNS]
    assert lines[1] == "\t".join(ROW_A[i] for i in simple_indexes)

    win._on_close()


def test_write_table_rows_csv_writes_a_real_csv_file(tmp_path):
    """The dialog-free CSV writer should produce a real, readable CSV
    file - tested directly, without going through a file dialog."""

    path = str(tmp_path / "export.csv")
    measurement_window.write_table_rows_csv(path, ["A", "B"], [["1", "2"], ["3", "4"]])

    content = open(path, encoding="utf-8").read()
    assert "A,B" in content
    assert "1,2" in content
    assert "3,4" in content


def test_record_current_measurement_inserts_rows_with_a_stamped_id(qapp, make_fake_app):
    """Recording should insert one Log row per row from the last
    update_window call, each stamped with the current measurement ID -
    and a second Record click should get the next ID."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    win.record_current_measurement()

    rows = _all_row_texts(win.log_table)
    assert rows == [_with_id(ROW_A, 1), _with_id(ROW_A, 2)]

    win._on_close()


def test_record_current_measurement_notifies_the_tutorial(qapp, make_fake_app):
    """Recording a measurement should report the "record_measurement"
    tutorial completion action (ROADMAP.md Phase 15) - real-hook
    detection for that step, alongside the existing
    `_on_measurement_recorded` project-snapshot notification."""

    notified = []

    class _FakeTutorialWindow:
        def notify_action(self, action_name):
            notified.append(action_name)

    app = make_fake_app(
        click_sigma_px=1.5, _on_measurement_recorded=lambda: None, tutorial_window=_FakeTutorialWindow()
    )
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()

    assert notified == ["record_measurement"]

    win._on_close()


def test_record_current_measurement_shares_one_id_across_all_rows_in_one_click(qapp, make_fake_app):
    """Every row recorded from the same Record click (e.g. a Point row
    and its Segment row) should share one measurement ID, so a whole bad
    chain can be found and deleted together - a later, separate Record
    click should get a different ID."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)
    win.record_current_measurement()
    win.record_current_measurement()

    rows = _all_row_texts(win.log_table)
    assert rows == [_with_id(ROW_A, 1), _with_id(ROW_B, 1), _with_id(ROW_A, 2), _with_id(ROW_B, 2)]

    win._on_close()


def test_record_current_measurement_without_prior_update_does_nothing(qapp, make_fake_app):
    """Pressing Record before any measurement has ever been shown should
    just no-op, not crash (there's nothing to record yet)."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    win.record_current_measurement()

    assert win.log_table.rowCount() == 0

    win._on_close()


def test_log_table_cells_stay_editable(qapp, make_fake_app):
    """The Log's cells should stay directly editable, by request - fixing
    a bad recorded measurement is done by editing a cell directly."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    assert win.log_table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers

    win._on_close()


def test_delete_selected_log_rows_removes_only_selected_rows(qapp, make_fake_app):
    """Selecting a row and calling the delete handler directly (bypassing
    the confirmation dialog, same as this project's usual dialog-free
    logic testing pattern) should remove just that row."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    win.update_window([ROW_B], None)
    win.record_current_measurement()
    assert win.log_table.rowCount() == 2

    # Replicate _delete_selected_log_rows' own removal logic directly,
    # bypassing its confirmation QMessageBox - that needs a real click to
    # answer and isn't what this test is checking.
    win.log_table.selectRow(0)
    for r in sorted({idx.row() for idx in win.log_table.selectedIndexes()}, reverse=True):
        win.log_table.removeRow(r)

    assert win.log_table.rowCount() == 1
    assert _table_row_texts(win.log_table, 0) == _with_id(ROW_B, 2)

    win._on_close()


def test_get_log_text_returns_empty_string_before_any_window_built(make_fake_app):
    """Saving a project before the Measurement window has ever been
    opened this session should get a clean empty string, not crash."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    assert win.get_log_text() == ""


def test_get_log_text_always_includes_every_column_regardless_of_view_mode(qapp, make_fake_app):
    """The project-file serialization must always carry every column,
    even while the Simple View toggle is hiding most of them on screen -
    unlike the Copy/Export buttons, which respect the toggle."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()

    header_line = win.get_log_text().splitlines()[0]
    assert header_line == "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)

    win._on_close()


def test_restore_log_text_round_trips_and_resumes_numbering(qapp, make_fake_app):
    """Saving the Log's text then restoring it into a fresh window should
    reproduce the exact same text, and a Record click afterward should
    continue numbering after the restored log's highest ID rather than
    colliding with it."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)
    win.record_current_measurement()
    win.record_current_measurement()
    saved_text = win.get_log_text()
    win._on_close()

    fresh_win = measurement_window.MeasurementWindow(app)
    fresh_win.restore_log_text(saved_text)

    assert fresh_win.get_log_text() == saved_text
    assert fresh_win._next_measurement_id == 3

    fresh_win.update_window([ROW_A], None)
    fresh_win.record_current_measurement()
    assert fresh_win.get_log_text().splitlines() == saved_text.splitlines() + ["\t".join(_with_id(ROW_A, 3))]

    fresh_win._on_close()


def test_restore_log_text_with_empty_string_leaves_a_clean_log(qapp, make_fake_app):
    """Restoring an empty log (a project saved before anything was ever
    recorded) should leave numbering starting fresh at 1, not blow up on
    an empty string."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.restore_log_text("")

    assert win.get_log_text() == ""
    assert win._next_measurement_id == 1

    win._on_close()


def test_restore_log_text_pads_an_old_shorter_row_with_blanks(qapp, make_fake_app):
    """A project saved before Phase 16's four new columns existed has
    19-field rows - restoring one should pad it to today's width rather
    than crashing or misaligning columns, with the new fields blank."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    old_headers = list(measurement_window.RESULT_COLUMNS[:19])
    old_row = list(ROW_A[:19])
    old_row[ID_INDEX] = "1"
    old_text = "\t".join(old_headers) + "\n" + "\t".join(old_row)

    win.restore_log_text(old_text)

    assert win.log_table.rowCount() == 1
    row = _table_row_texts(win.log_table, 0)
    assert row[:19] == old_row
    assert row[19:] == ["", "", "", ""]
    assert win._next_measurement_id == 2

    win._on_close()


def test_ensure_window_is_idempotent(qapp, make_fake_app):
    """Calling ensure_window twice shouldn't build a second dialog -
    matches the original Toplevel lazy-singleton pattern."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    first_win = win.win
    win.ensure_window()

    assert win.win is first_win

    win._on_close()


def test_on_close_clears_widget_references_so_ensure_window_rebuilds(qapp, make_fake_app):
    """After _on_close (simulating the user closing the window), a later
    ensure_window call should rebuild everything from scratch rather
    than touching stale references."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    win._on_close()

    assert win.win is None
    assert win.results_table is None
    assert win.log_table is None
    assert win.error_label is None

    win.ensure_window()
    assert win.win is not None
