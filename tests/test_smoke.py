"""Wraps smoke_test.py's launch/close check into the pytest suite, so
`pytest` alone covers everything without needing a separate manual step.
"""

import smoke_test


def test_app_launches_and_closes_cleanly(qapp):
    """The app should construct, run briefly, and close via the real
    closeEvent path without raising."""

    # Short duration keeps the suite fast; this is still a real
    # SizeamaticProApp, closed via the real closeEvent path
    # (window.close(), the same thing a user clicking the window's
    # close button triggers).
    smoke_test.run_smoke_test(duration_ms=200, app=qapp)
