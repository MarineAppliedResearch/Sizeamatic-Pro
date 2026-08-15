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

import os

# Force Qt's offscreen platform plugin before anything in the suite gets a
# chance to construct a QApplication - the four ported sub-window modules
# (measurement_window.py, calibration_summary.py, etc.) build real QDialogs
# in their tests, and without this a normal `pytest` run would flash a real,
# visible window on screen for every one of them. setdefault so an explicit
# override (e.g. a developer deliberately watching a test run on-screen)
# still wins.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from PySide6.QtWidgets import QApplication, QWidget

import recent_projects


@pytest.fixture(autouse=True)
def _isolate_recent_projects_file(tmp_path, monkeypatch):
    """Redirect recent_projects.py's persistent file to a throwaway
    tmp_path location for every test, automatically.

    Without this, a test that exercises `on_save_project`/
    `on_open_project` without its own explicit monkeypatch for
    `recent_projects.get_recent_projects_path` silently reads and writes
    the real per-user `%APPDATA%\\SizeamaticPro\\recent_projects.json`
    on whatever machine runs the suite — which is exactly what happened
    before this fixture existed: running the tests clobbered the
    project owner's actual Recent Projects list with pytest tmp-path
    entries. Autouse means every test gets this protection whether or
    not it remembers to ask for it.
    """
    monkeypatch.setattr(
        "recent_projects.get_recent_projects_path",
        lambda: str(tmp_path / "recent_projects.json"),
    )


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

    def _app_window_title(self):
        """Fixed stand-in for the real project-name-aware window title,
        so `ensure_window` methods that set it during tests don't raise
        `AttributeError`.

        Returns:
            str: A fixed placeholder title.
        """
        return "Sizeamatic Pro"

    def screen(self):
        """No-op stand-in for `QWidget.screen()`, so the four ported
        sub-window modules' `ensure_window()` (which positions the new
        dialog on whichever screen the main app window is on) falls back
        to `QApplication.primaryScreen()` cleanly instead of raising
        `AttributeError` on a `FakeApp` that isn't a real `QWidget`.

        Returns:
            None
        """
        return None


@pytest.fixture(scope="session")
def qapp():
    """A shared `QApplication` instance for the whole test session.

    Qt requires at least one `QApplication` to exist before any `QWidget`
    can be constructed, and doesn't support creating more than one per
    process - session-scoped so every test that touches real Qt widgets
    (the four ported sub-window modules' dialogs) shares the same one.

    Returns:
        PySide6.QtWidgets.QApplication: The shared application instance.
    """
    return QApplication.instance() or QApplication([])


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


class FakeAppWidget(QWidget):
    """A real `QWidget`-based stand-in for `main.SizeamaticProApp`,
    for tests that need `self.app` to actually be a `QObject` -
    `perform_calibration.py`'s `ensure_window()` parents a `QShortcut`
    to `self.app` (the Space-bar capture shortcut, active while the
    main app window has focus), which requires a real `QObject`/
    `QWidget`, not the plain-object `FakeApp` above. Everything else
    about it matches `FakeApp` (same stub methods) - `screen()` doesn't
    need a stub here since a real `QWidget` already has a working one.
    """

    _set_status_mid = FakeApp._set_status_mid
    _app_window_title = FakeApp._app_window_title


@pytest.fixture
def make_fake_app_widget(qapp):
    """Fixture factory for building a `FakeAppWidget` with specific
    attributes - like `make_fake_app`, but for tests that call code
    requiring `self.app` to be a real `QObject` (see `FakeAppWidget`).
    Depends on `qapp` directly since constructing any `QWidget` requires
    a `QApplication` to already exist.

    Returns:
        Callable[..., FakeAppWidget]: A function that takes keyword
        arguments and returns a `FakeAppWidget` with each one set as an
        attribute.
    """

    def _make(**kwargs):
        """Build one FakeAppWidget with the given attributes set on it.

        Args:
            **kwargs: Attribute name/value pairs to set on the FakeAppWidget.

        Returns:
            FakeAppWidget: The constructed stand-in app.
        """
        app = FakeAppWidget()
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


@pytest.fixture
def known_chain_pixels(synthetic_cal):
    """Three known 3D points (a 2-segment connected chain) plus their
    exact left/right pixel projections, for testing multi-point/
    multi-segment measurement math (ROADMAP.md Phase 7's measurement
    output item) with an exact expected answer — same rationale as
    `known_point_pixels`, just extended to a chain instead of one point.

    Args:
        synthetic_cal (dict): The `synthetic_cal` fixture.

    Returns:
        dict: Keys "points" (list[dict], each like `known_point_pixels`'s
        return value) and "total_length_mm" (float), the exact Euclidean
        chain length computed directly from the 3D points (independent of
        any triangulation math, since it's just distance between the
        known inputs).
    """
    points_3d = [
        (30.0, -20.0, 2000.0),
        (60.0, -20.0, 2000.0),
        (60.0, 10.0, 2200.0),
    ]

    points = []
    for X, Y, Z in points_3d:
        xL, yL = project_through(synthetic_cal["PL"], X, Y, Z)
        xR, yR = project_through(synthetic_cal["PR"], X, Y, Z)
        points.append({"X": X, "Y": Y, "Z": Z, "xL": xL, "yL": yL, "xR": xR, "yR": yR})

    total_length_mm = 0.0
    for (X0, Y0, Z0), (X1, Y1, Z1) in zip(points_3d, points_3d[1:]):
        dX, dY, dZ = X1 - X0, Y1 - Y0, Z1 - Z0
        total_length_mm += (dX * dX + dY * dY + dZ * dZ) ** 0.5

    return {"points": points, "total_length_mm": total_length_mm}


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
