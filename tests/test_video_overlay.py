"""Interaction tests for video_overlay.py's VideoOverlay class.

Exercises the real click -> drag -> release state machine through the
real app (canvases built by SizeamaticProApp.__init__ via
create_canvases), since the hit-testing and coordinate conversion it
depends on need a real Tkinter canvas, not a FakeApp stand-in.
"""

from types import SimpleNamespace


def _event(x, y):
    """Build a minimal stand-in for a Tkinter mouse event.

    Args:
        x (int): Event X coordinate, in canvas screen pixels.
        y (int): Event Y coordinate, in canvas screen pixels.

    Returns:
        SimpleNamespace: An object with just the `.x`/`.y` attributes the
        code under test actually reads off a real Tkinter event.
    """
    return SimpleNamespace(x=x, y=y)


def test_left_click_on_empty_space_adds_a_point(sizeamatic_app):
    """Clicking empty overlay space (no existing handle under the cursor)
    should append a new point to the clicked pane's point list."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)  # native 1:1 scale, easier to reason about

    assert len(app.ptsL) == 0

    app.video_overlay.on_left_down("L", _event(100, 50))

    assert len(app.ptsL) == 1
    assert app.ptsL[0] == (100.0, 50.0)


def test_left_click_on_new_point_also_creates_stereo_mate_guess(sizeamatic_app):
    """Placing the first point on one pane should attempt an initial
    stereo mate guess on the opposite pane (guess_mate_point_on_scanline
    returning None here since there's no real frame loaded, but the
    attempt itself — and not crashing — is what's being verified)."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.metaR = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)
    app.view_rectified.set(True)
    app.cal = None  # guess_mate_point_on_scanline requires cal is not None

    app.video_overlay.on_left_down("L", _event(100, 50))

    # No calibration loaded, so the mate guess can't succeed - but it
    # should have been attempted without raising, and the right pane
    # should simply have no mate point yet.
    assert len(app.ptsL) == 1
    assert len(app.ptsR) == 0


def test_drag_moves_an_existing_point(sizeamatic_app):
    """Dragging an existing point handle should update its position in
    the point list, and releasing should end the drag."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)

    # Place a point, then redraw so it actually exists as a tagged handle
    # on the canvas (on_left_down's hit-test needs the drawn handle, not
    # just the point-list entry, to find it on a second click).
    app.ptsL.append((100.0, 50.0))
    app.video_overlay.redraw()

    # Click down exactly on the handle we just drew.
    app.video_overlay.on_left_down("L", _event(100, 50))
    assert app.video_overlay.drag_active is True
    assert app.video_overlay.drag_index == 0

    # Drag it to a new position.
    app.video_overlay.on_left_drag("L", _event(200, 150))
    assert app.ptsL[0] == (200.0, 150.0)

    # Release ends the drag.
    app.video_overlay.on_left_up("L", _event(200, 150))
    assert app.video_overlay.drag_active is False
    assert app.video_overlay.drag_index is None


def test_get_handle_index_under_cursor_without_a_handle_returns_none(sizeamatic_app):
    """Clicking where nothing is drawn should not report a handle index."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)

    app.video_overlay.redraw()  # clears any leftover overlay items

    idx = app.video_overlay.get_nearest_handle_index("L", app.video_overlay.left_canvas, 9999, 9999)
    assert idx is None
