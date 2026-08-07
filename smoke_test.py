"""Smoke test for Sizeamatic Pro.

Launches the app, lets the Tk event loop run briefly, then triggers the
same close path a user closing the window would (`app.on_app_close`), and
exits. Intended for an agent (or a developer) to quickly verify the app at
least launches and shuts down cleanly, without a full test suite — that's
Phase 4 in `ROADMAP.md`.

Usage:
    python smoke_test.py [duration_ms]

Exit code 0 means the app launched and closed without raising. Any
exception during setup, the event loop, or close is allowed to propagate
with its traceback and a non-zero exit code — this is intentionally a
smoke test, not a suite that catches and reports failures gracefully.
"""

import sys

import tkinter as tk

from main import SizeamaticProApp


def run_smoke_test(duration_ms: int = 1000, root: "tk.Tk | None" = None) -> None:
    """Launch, briefly run, and cleanly close the app.

    Args:
        duration_ms (int): How long to let the Tk event loop run before
            triggering close, in milliseconds.
        root (tkinter.Tk | None): An existing Tk root to reuse instead of
            creating a new one. Defaults to None, which creates a fresh
            root — the right choice for standalone script usage, where
            this is the only Tk() the process will ever create. The
            pytest wrapper (tests/test_smoke.py) passes in the test
            session's shared root instead, since Tcl/Tk does not reliably
            support multiple create-then-destroy cycles within one
            process — see hidden_tk_root's docstring in
            tests/conftest.py.

    Returns:
        None
    """

    # Build the same root/app pair main() would, without the icon/taskbar
    # setup that's irrelevant to whether the app itself works.
    if root is None:
        root = tk.Tk()
    app = SizeamaticProApp(root)

    # Schedule the same close path a real window-close would trigger, so
    # this also exercises on_app_close (including anaglyph/capture cleanup).
    root.after(duration_ms, app.on_app_close)

    root.mainloop()


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_smoke_test(duration)
    print("Smoke test passed: app launched and closed cleanly.")
