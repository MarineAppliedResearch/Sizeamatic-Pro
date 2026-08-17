"""Tests for measurement_window.py's MeasurementWindow.

Uses a FakeApp (via make_fake_app) rather than the real SizeamaticProApp:
MeasurementWindow only ever reads app.click_sigma_px (for the status
line), app._app_window_title() (for the dialog's title), and
app.screen() (for positioning) - see AGENTS.md's Testing section. Still
needs a real QApplication (the qapp fixture) since it builds a real
QDialog.
"""

import measurement_window

# measurement_id (index 4) is always blank coming out of update_window - only
# record_current_measurement ever stamps a real value in.
ROW_A = ("left.mp4", "10", "00:00:00.417", "2026-08-12 14:32:05.417", "", "Point", "0", "1.0", "2.0", "3.0", "4.0", "5.00", "6.00", "7.00", "7.50", "8.0", "9.0", "8.1", "9.1")
ROW_B = ("left.mp4", "10", "00:00:00.417", "2026-08-12 14:32:05.417", "", "Segment", "0-1", "1.0", "2.0", "3.0", "4.0", "", "", "", "", "8.0", "", "8.1", "")

ID_INDEX = measurement_window.RESULT_COLUMNS.index("measurement_id")


def _with_id(row, measurement_id):
    """Build the line a recorded row should produce once stamped with a
    measurement ID, matching what record_current_measurement writes.

    Args:
        row (tuple): One of ROW_A/ROW_B.
        measurement_id (int): The measurement ID to stamp in.

    Returns:
        str: The expected tab-separated log line.
    """
    stamped = list(row)
    stamped[ID_INDEX] = str(measurement_id)
    return "\t".join(stamped)


def _table_row_texts(table, row):
    """Read every column's text out of one QTableWidget row.

    Args:
        table (QTableWidget): The table to read from.
        row (int): Row index.

    Returns:
        list[str]: One string per column, "" for an unset cell.
    """
    return [table.item(row, c).text() if table.item(row, c) else "" for c in range(table.columnCount())]


def test_update_window_populates_results_table_and_copy_text(qapp, make_fake_app):
    """update_window should fill the unified results table and build a
    copy block with a header line plus one line per row."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)

    assert win.results_table.rowCount() == 2
    assert _table_row_texts(win.results_table, 0) == list(ROW_A)
    assert _table_row_texts(win.results_table, 1) == list(ROW_B)

    copy_content = win.copy_text.toPlainText()
    lines = copy_content.split("\n")
    assert lines[0] == "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)
    assert lines[1] == "\t".join(ROW_A)
    assert lines[2] == "\t".join(ROW_B)

    win._on_close()


def test_record_current_measurement_appends_header_once_and_stamps_a_new_id_each_time(qapp, make_fake_app):
    """The Log should get the header line exactly once, then one line per
    row per Record click, each stamped with the next measurement ID -
    not a repeated header, and not a blank/reused ID."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    win.record_current_measurement()

    log_content = win.log_text.toPlainText()
    lines = log_content.split("\n")

    header_line = "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)
    assert lines[0] == header_line
    assert lines.count(header_line) == 1
    assert lines[1] == _with_id(ROW_A, 1)
    assert lines[2] == _with_id(ROW_A, 2)

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

    log_content = win.log_text.toPlainText()
    lines = log_content.split("\n")

    assert lines[1] == _with_id(ROW_A, 1)
    assert lines[2] == _with_id(ROW_B, 1)
    assert lines[3] == _with_id(ROW_A, 2)
    assert lines[4] == _with_id(ROW_B, 2)

    win._on_close()


def test_record_current_measurement_without_prior_update_does_nothing(qapp, make_fake_app):
    """Pressing Record before any measurement has ever been shown should
    just no-op, not crash (there's nothing to record yet)."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    win.record_current_measurement()

    assert win.log_text.toPlainText() == ""

    win._on_close()


def test_log_text_is_never_read_only_so_it_can_be_manually_edited(qapp, make_fake_app):
    """The Log widget should stay directly editable, by request - fixing
    or removing a bad recorded measurement is done by editing it
    directly, not through a separate delete UI."""

    app = make_fake_app(click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    assert win.log_text.isReadOnly() is False

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    assert win.log_text.isReadOnly() is False

    win._on_close()


def test_copy_text_is_read_only(qapp, make_fake_app):
    """The copy block should be read-only - it's generated output, not
    something the user edits directly (unlike the Log)."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    assert win.copy_text.isReadOnly() is True

    win._on_close()


def test_get_log_text_returns_empty_string_before_any_window_built(make_fake_app):
    """Saving a project before the Measurement window has ever been
    opened this session should get a clean empty string, not crash."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    assert win.get_log_text() == ""


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
    assert fresh_win.get_log_text().splitlines() == saved_text.splitlines() + [_with_id(ROW_A, 3)]

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
    assert win.copy_text is None
    assert win.log_text is None
    assert win.error_label is None

    win.ensure_window()
    assert win.win is not None
