"""Shared pytest fixtures for the Sizeamatic Pro test suite.

Most of the app's actual measurement/calibration logic lives in functions
that take an `app` parameter but only read a handful of specific
attributes off it (`app.cal`, `app.ptsL`/`app.ptsR`,
`app.view_rectified.get()`, etc.) — see `ARCHITECTURE.md`. That means most
of these functions can be unit-tested with a lightweight stand-in object
instead of constructing the real Tkinter GUI. `FakeApp` (via
`make_fake_app`) is that stand-in; reach for the real `sizeamatic_app`
fixture only when a test genuinely needs the GUI (e.g. a regression test
for a bug in window-close handling).

The synthetic calibration fixtures (`synthetic_cal`, `known_point_pixels`)
exist because there's no "known correct answer" to assert against with
real captured calibration data — a small hand-built rectified stereo rig
with a chosen baseline/focal length lets tests compute the exact expected
pixel coordinates for a given 3D point analytically, then assert
triangulation recovers that same point.
"""

import matplotlib

# generate_calibration_report.py imports matplotlib.pyplot, which on
# Windows can auto-select a Tk-based GUI backend. That backend competes
# with our own direct tkinter usage (hidden_tk_root/sizeamatic_app) for
# the process's single Tcl/Tk interpreter, causing intermittent
# "invalid command name tcl_findLibrary" errors when a test that creates
# its own tk.Tk() runs after generate_calibration_report has been
# imported. Force a non-GUI backend before anything else in the suite
# gets a chance to import pyplot.
matplotlib.use("Agg")

import tkinter as tk

import numpy as np
import pytest


class FakeVar:
    """Minimal stand-in for `tkinter.BooleanVar`/`tkinter.IntVar`.

    Supports only `.get()`/`.set()`, which is all the app-parameter
    functions under test actually call on these Tkinter variable types.
    """

    def __init__(self, value):
        """Store the initial value.

        Args:
            value: The initial value to hold.

        Returns:
            None
        """
        self._value = value

    def get(self):
        """Return the current value.

        Returns:
            The stored value.
        """
        return self._value

    def set(self, value):
        """Set a new value.

        Args:
            value: The new value to store.

        Returns:
            None
        """
        self._value = value


class FakeApp:
    """Minimal stand-in for `main.SizeamaticProApp`.

    Deliberately not a full `SizeamaticProApp` — only carries whichever
    attributes a given test assigns onto it via `make_fake_app`, so
    testing app-parameter functions doesn't require a real Tk root.
    """

    def _set_status_mid(self, text):
        """No-op stand-in for the real status bar update, so calls to it
        from the code under test don't raise `AttributeError`.

        Args:
            text (str): The status text that would have been displayed.

        Returns:
            None
        """


@pytest.fixture
def make_fake_app():
    """Fixture factory for building a `FakeApp` with specific attributes.

    Returns:
        Callable[..., FakeApp]: A function that takes keyword arguments
        and returns a `FakeApp` with each one set as an attribute.
    """

    def _make(**kwargs):
        """Build one FakeApp with the given attributes set on it.

        Args:
            **kwargs: Attribute name/value pairs to set on the FakeApp.

        Returns:
            FakeApp: The constructed stand-in app.
        """
        app = FakeApp()
        for key, value in kwargs.items():
            setattr(app, key, value)
        return app

    return _make


# Synthetic rectified stereo rig parameters. Not real captured calibration
# values — chosen arbitrarily but self-consistently, purely so tests have a
# known-correct expected triangulation result to assert against.
SYNTHETIC_FX = 800.0
SYNTHETIC_FY = 800.0
SYNTHETIC_CX = 320.0
SYNTHETIC_CY = 240.0
SYNTHETIC_BASELINE_MM = 100.0


def project_through(P, X, Y, Z):
    """Project a 3D point through a projection matrix (test helper only).

    Deliberately independent of `stereo_matching.project_point` — this is
    used to construct known expected pixel coordinates for a synthetic 3D
    point, so it must not share an implementation with the code under
    test.

    Args:
        P (numpy.ndarray): 3x4 projection matrix.
        X (float): 3D point X coordinate.
        Y (float): 3D point Y coordinate.
        Z (float): 3D point Z coordinate.

    Returns:
        tuple[float, float]: The projected (u, v) pixel coordinates.
    """
    p = P @ np.array([X, Y, Z, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


@pytest.fixture
def synthetic_cal():
    """A small, self-consistent rectified stereo calibration dict.

    Returns:
        dict: A calibration dict with just the "PL"/"PR" keys that
        `stereo_matching.py`'s triangulation functions need.
    """
    PL = np.array(
        [
            [SYNTHETIC_FX, 0.0, SYNTHETIC_CX, 0.0],
            [0.0, SYNTHETIC_FY, SYNTHETIC_CY, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    # The right camera sits at world position (+baseline, 0, 0) relative to
    # the left camera, with the same orientation (rectified), so its
    # translation column is K @ (-baseline, 0, 0).
    PR = np.array(
        [
            [SYNTHETIC_FX, 0.0, SYNTHETIC_CX, -SYNTHETIC_FX * SYNTHETIC_BASELINE_MM],
            [0.0, SYNTHETIC_FY, SYNTHETIC_CY, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    return {"PL": PL, "PR": PR}


@pytest.fixture
def known_point_pixels(synthetic_cal):
    """A known 3D point plus its exact left/right pixel projections.

    Args:
        synthetic_cal (dict): The `synthetic_cal` fixture.

    Returns:
        dict: Keys "X"/"Y"/"Z" (the 3D point, in millimeters) and
        "xL"/"yL"/"xR"/"yR" (its exact projected pixel coordinates).
    """
    X, Y, Z = 30.0, -20.0, 2000.0
    xL, yL = project_through(synthetic_cal["PL"], X, Y, Z)
    xR, yR = project_through(synthetic_cal["PR"], X, Y, Z)
    return {"X": X, "Y": Y, "Z": Z, "xL": xL, "yL": yL, "xR": xR, "yR": yR}


@pytest.fixture(scope="session")
def hidden_tk_root():
    """A withdrawn (invisible) Tk root window, shared for the whole test
    session and torn down once at the very end.

    Session-scoped deliberately: Tcl/Tk does not reliably support
    repeated full create-then-destroy cycles within a single process —
    empirically, giving each GUI-touching test its own fresh `tk.Tk()`
    made the suite intermittently fail with
    `TclError: invalid command name "tcl_findLibrary"` on the second or
    third such cycle. One shared root for the whole session avoids that.
    Tests that need to build/destroy something should create a
    `tk.Toplevel(root)` child instead of touching the root itself — see
    `test_regressions.py`'s calibration-window test for that pattern, and
    the on_app_close test for how to test root-destroying code without
    actually destroying this shared root.

    Returns:
        tkinter.Tk: The shared hidden root window.
    """
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def sizeamatic_app(hidden_tk_root):
    """A real `SizeamaticProApp` built on the shared hidden root, for
    tests that genuinely need the GUI rather than a `FakeApp` stand-in
    (e.g. regression tests for window-close bugs).

    Note:
        Built on the session-shared `hidden_tk_root`, not a fresh root of
        its own. If a test needs to exercise code that calls
        `self.root.destroy()` (like `on_app_close`), patch that method out
        first rather than letting it actually tear down the shared root —
        see `test_regressions.py`'s on_app_close test.

    Args:
        hidden_tk_root (tkinter.Tk): The `hidden_tk_root` fixture.

    Returns:
        main.SizeamaticProApp: The constructed app instance.
    """
    import main

    return main.SizeamaticProApp(hidden_tk_root)
