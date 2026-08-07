"""One test per bug fixed in FINDINGS.md, so none of them can silently
come back.
"""

import anaglyph_preview
import calibration_summary


def test_calibration_window_close_then_update_does_not_crash(hidden_tk_root, make_fake_app):
    """FINDINGS.md #1: closing the calibration summary window used to
    leave cal_win/cal_tree/cal_copy_text pointing at destroyed widgets
    (because the nested _on_close closure was missing its own `global`
    declaration), crashing the next update_calibration_window call.

    That failure mode is now structurally impossible:
    `CalibrationSummaryWindow` is a real class (see calibration_summary.py's
    "Design notes"), so there's no module-level global and no `global`
    declaration to miss. This test just confirms the close-then-update
    behavior still works as intended.
    """

    app = make_fake_app(root=hidden_tk_root, cal=None)
    win = calibration_summary.CalibrationSummaryWindow(app)

    win.ensure_window()
    assert win.win is not None
    assert win.tree is not None

    # Simulate the user closing the window — the same callback the real
    # WM_DELETE_WINDOW protocol triggers.
    win._on_close()

    assert win.win is None
    assert win.tree is None
    assert win.copy_text is None

    # Should just no-op (app.cal is also None).
    win.update_window()


def test_stop_anaglyph_preview_without_prior_tick_does_not_raise(make_fake_app):
    """FINDINGS.md #2: `anaglyph_after_id` used to only come into existence
    once `anaglyph_tick` reached its own assignment to it (as an
    uninitialized module global), so calling `stop_anaglyph_preview`
    before any tick had ever run raised NameError.

    That failure mode is now structurally impossible: `AnaglyphPreview` is
    a real class (see anaglyph_preview.py's "Design notes") whose
    `__init__` always initializes `after_id`, so there's no uninitialized
    module global to hit. This test just confirms `stop` still works as
    intended when called before `tick` ever ran.
    """

    app = make_fake_app()
    preview = anaglyph_preview.AnaglyphPreview(app)
    preview.active = True

    preview.stop()  # should not raise NameError

    assert preview.active is False


def test_on_app_close_with_active_anaglyph_preview_does_not_raise(sizeamatic_app):
    """FINDINGS.md #3: `on_app_close` used to call
    `anaglyph_preview.stop_anaglyph_preview()` with no `app` argument,
    raising TypeError whenever the anaglyph preview was open at the
    moment the app closed.

    That failure mode is now structurally impossible: `on_app_close` calls
    `self.anaglyph_preview.stop()`, a bound method that always has access
    to the right app — there's no separate argument to forget to pass.
    """

    sizeamatic_app.anaglyph_preview.active = True

    # on_app_close's real job here is calling self.anaglyph_preview.stop()
    # correctly — not tearing down the session's shared Tk root, which
    # Tcl/Tk doesn't reliably support doing repeatedly within one process
    # (see hidden_tk_root's docstring). Patch out just the final destroy.
    sizeamatic_app.root.destroy = lambda: None

    sizeamatic_app.on_app_close()  # should not raise TypeError
