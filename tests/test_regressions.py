"""One test per bug fixed in FINDINGS.md, so none of them can silently
come back.
"""

import anaglyph_preview
import calibration_summary


def test_calibration_window_close_then_update_does_not_crash(hidden_tk_root, make_fake_app):
    """FINDINGS.md #1: closing the calibration summary window used to
    leave cal_win/cal_tree/cal_copy_text pointing at destroyed widgets
    (because the nested _on_close was missing its own `global`
    declaration), crashing the next update_calibration_window call.
    """

    # Start from a known-clean module state regardless of test order.
    calibration_summary.cal_win = None
    calibration_summary.cal_tree = None
    calibration_summary.cal_copy_text = None

    app = make_fake_app(root=hidden_tk_root, cal=None)

    calibration_summary.ensure_calibration_window(app)
    assert calibration_summary.cal_win is not None
    assert calibration_summary.cal_tree is not None

    # Simulate the user closing the window — the same callback the real
    # WM_DELETE_WINDOW protocol triggers.
    calibration_summary._on_calibration_window_close(calibration_summary.cal_win)

    assert calibration_summary.cal_win is None
    assert calibration_summary.cal_tree is None
    assert calibration_summary.cal_copy_text is None

    # This used to raise TclError because cal_tree still referenced a
    # destroyed Treeview. Should now just no-op (app.cal is also None).
    calibration_summary.update_calibration_window(app)


def test_stop_anaglyph_preview_without_prior_tick_does_not_raise(make_fake_app):
    """FINDINGS.md #2: anaglyph_after_id used to only come into existence
    once anaglyph_tick reached its own assignment to it, so calling
    stop_anaglyph_preview before any tick had ever run raised NameError.
    """

    anaglyph_preview.anaglyph_after_id = None
    anaglyph_preview.anaglyph_active = True

    app = make_fake_app()
    anaglyph_preview.stop_anaglyph_preview(app)  # should not raise NameError

    assert anaglyph_preview.anaglyph_active is False


def test_on_app_close_with_active_anaglyph_preview_does_not_raise(sizeamatic_app):
    """FINDINGS.md #3: on_app_close used to call stop_anaglyph_preview()
    with no app argument, raising TypeError whenever the anaglyph preview
    was open at the moment the app closed.
    """

    anaglyph_preview.anaglyph_active = True
    anaglyph_preview.anaglyph_after_id = None

    # on_app_close's real job here is calling stop_anaglyph_preview(self)
    # correctly — not tearing down the session's shared Tk root, which
    # Tcl/Tk doesn't reliably support doing repeatedly within one process
    # (see hidden_tk_root's docstring). Patch out just the final destroy.
    sizeamatic_app.root.destroy = lambda: None

    sizeamatic_app.on_app_close()  # should not raise TypeError
