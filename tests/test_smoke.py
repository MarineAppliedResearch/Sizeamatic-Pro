"""Wraps smoke_test.py's launch/close check into the pytest suite, so
`pytest` alone covers everything without needing a separate manual step.
"""

import smoke_test


def test_app_launches_and_closes_cleanly(hidden_tk_root):
    """The app should construct, run briefly, and close via the real
    on_app_close path without raising."""

    # on_app_close's last line is self.root.destroy(), which both stops
    # mainloop() AND tears down the interpreter. We want the former
    # (mainloop must actually exit for this test to return) but not the
    # latter (see hidden_tk_root's docstring in conftest.py for why
    # multiple real Tk() destroy cycles in one process aren't reliable
    # here) — root.quit() stops mainloop() without destroying anything,
    # so redirect destroy() to it for the duration of this test.
    hidden_tk_root.destroy = hidden_tk_root.quit

    # Short duration keeps the suite fast; this is still a real Tk root
    # and a real SizeamaticProApp, closed via the real on_app_close path.
    smoke_test.run_smoke_test(duration_ms=200, root=hidden_tk_root)
