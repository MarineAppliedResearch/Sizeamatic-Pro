"""Smoke test for Sizeamatic Pro.

Launches the app, lets the Qt event loop run briefly, then triggers the
same close path a user closing the window would (`window.close()`,
which fires the real `closeEvent`), and exits. Intended for an agent
(or a developer) to quickly verify the app at least launches and shuts
down cleanly, without a full test suite.

Usage:
    python smoke_test.py [duration_ms]

Exit code 0 means the app launched and closed without raising. Any
exception during setup, the event loop, or close is allowed to propagate
with its traceback and a non-zero exit code — this is intentionally a
smoke test, not a suite that catches and reports failures gracefully.
"""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from main import SizeamaticProApp


def run_smoke_test(duration_ms: int = 1000, app: "QApplication | None" = None) -> None:
    """Launch, briefly run, and cleanly close the app.

    Args:
        duration_ms (int): How long to let the Qt event loop run before
            triggering close, in milliseconds.
        app (QApplication | None): An existing QApplication to reuse
            instead of creating a new one. Defaults to None, which
            creates a fresh one — the right choice for standalone
            script usage, where this is the only QApplication the
            process will ever create. The pytest wrapper
            (tests/test_smoke.py) passes in the test session's shared
            `qapp` fixture instead, since Qt does not support more than
            one QApplication per process.

    Returns:
        None
    """
    if app is None:
        app = QApplication.instance() or QApplication(sys.argv)

    window = SizeamaticProApp()
    window.show()

    def _finish():
        # Exercises the real close path first (closeEvent's anaglyph/
        # capture cleanup, video capture release), then quits explicitly
        # rather than relying on Qt's quit-on-last-window-closed - when
        # reusing a shared QApplication (the pytest wrapper's qapp
        # fixture), other real top-level windows built by earlier tests
        # in the same process can still count as "open" even if this is
        # the only one this test itself created, which would otherwise
        # leave app.exec() blocked forever.
        window.close()
        app.quit()

    QTimer.singleShot(duration_ms, _finish)

    app.exec()


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_smoke_test(duration)
    print("Smoke test passed: app launched and closed cleanly.")
