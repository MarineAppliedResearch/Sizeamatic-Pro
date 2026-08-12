"""Tests for measurement_window.py's MeasurementWindow.

Uses a FakeApp (via make_fake_app) rather than the real SizeamaticProApp:
MeasurementWindow only ever reads app.root (for the Toplevel) and
app.click_sigma_px (for the status line), so a minimal stand-in is
enough - see AGENTS.md's Testing section.
"""

import measurement_window

# measurement_id (index 3) is always blank coming out of update_window - only
# record_current_measurement ever stamps a real value in.
ROW_A = ("left.mp4", "10", "00:00:00.417", "", "Point", "0", "1.0", "2.0", "3.0", "4.0", "5.00", "6.00", "7.00", "8.0", "9.0")
ROW_B = ("left.mp4", "10", "00:00:00.417", "", "Segment", "0-1", "1.0", "2.0", "3.0", "4.0", "", "", "", "8.0", "")

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


def test_update_window_populates_results_tree_and_copy_text(hidden_tk_root, make_fake_app):
    """update_window should fill the unified results table and build a
    copy block with a header line plus one line per row."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)

    assert len(win.results_tree.get_children()) == 2

    copy_content = win.copy_text.get("1.0", "end-1c")
    lines = copy_content.split("\n")
    assert lines[0] == "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)
    assert lines[1] == "\t".join(ROW_A)
    assert lines[2] == "\t".join(ROW_B)

    win._on_close()


def test_record_current_measurement_appends_header_once_and_stamps_a_new_id_each_time(
    hidden_tk_root, make_fake_app
):
    """The Log should get the header line exactly once, then one line per
    row per Record click, each stamped with the next measurement ID -
    not a repeated header, and not a blank/reused ID."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    win.record_current_measurement()

    log_content = win.log_text.get("1.0", "end-1c")
    lines = log_content.split("\n")

    header_line = "\t".join(measurement_window.RESULT_HEADERS[c] for c in measurement_window.RESULT_COLUMNS)
    assert lines[0] == header_line
    assert lines.count(header_line) == 1
    assert lines[1] == _with_id(ROW_A, 1)
    assert lines[2] == _with_id(ROW_A, 2)

    win._on_close()


def test_record_current_measurement_shares_one_id_across_all_rows_in_one_click(
    hidden_tk_root, make_fake_app
):
    """Every row recorded from the same Record click (e.g. a Point row
    and its Segment row) should share one measurement ID, so a whole bad
    chain can be found and deleted together - a later, separate Record
    click should get a different ID."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.update_window([ROW_A, ROW_B], None)
    win.record_current_measurement()
    win.record_current_measurement()

    log_content = win.log_text.get("1.0", "end-1c")
    lines = log_content.split("\n")

    assert lines[1] == _with_id(ROW_A, 1)
    assert lines[2] == _with_id(ROW_B, 1)
    assert lines[3] == _with_id(ROW_A, 2)
    assert lines[4] == _with_id(ROW_B, 2)

    win._on_close()


def test_record_current_measurement_without_prior_update_does_nothing(
    hidden_tk_root, make_fake_app
):
    """Pressing Record before any measurement has ever been shown should
    just no-op, not crash (there's nothing to record yet)."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    win.record_current_measurement()

    assert win.log_text.get("1.0", "end-1c") == ""

    win._on_close()


def test_log_text_is_never_disabled_so_it_can_be_manually_edited(
    hidden_tk_root, make_fake_app
):
    """The Log widget should stay directly editable, by request - fixing
    or removing a bad recorded measurement is done by editing it
    directly, not through a separate delete UI."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
    win = measurement_window.MeasurementWindow(app)

    win.ensure_window()
    assert str(win.log_text.cget("state")) == "normal"

    win.update_window([ROW_A], None)
    win.record_current_measurement()
    assert str(win.log_text.cget("state")) == "normal"

    win._on_close()


def test_get_log_text_returns_empty_string_before_any_window_built(make_fake_app):
    """Saving a project before the Measurement window has ever been
    opened this session should get a clean empty string, not crash."""

    app = make_fake_app(click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    assert win.get_log_text() == ""


def test_restore_log_text_round_trips_and_resumes_numbering(
    hidden_tk_root, make_fake_app
):
    """Saving the Log's text then restoring it into a fresh window should
    reproduce the exact same text, and a Record click afterward should
    continue numbering after the restored log's highest ID rather than
    colliding with it."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5, _on_measurement_recorded=lambda: None)
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


def test_restore_log_text_with_empty_string_leaves_a_clean_log(
    hidden_tk_root, make_fake_app
):
    """Restoring an empty log (a project saved before anything was ever
    recorded) should leave numbering starting fresh at 1, not blow up on
    an empty string."""

    app = make_fake_app(root=hidden_tk_root, click_sigma_px=1.5)
    win = measurement_window.MeasurementWindow(app)

    win.restore_log_text("")

    assert win.get_log_text() == ""
    assert win._next_measurement_id == 1

    win._on_close()
