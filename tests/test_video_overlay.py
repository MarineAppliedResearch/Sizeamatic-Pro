"""Interaction tests for video_overlay.py's VideoOverlay class.

Exercises the real click -> drag -> release state machine through the
real app (canvases built by SizeamaticProApp.__init__ via
create_canvases), since the hit-testing and coordinate conversion it
depends on need a real Tkinter canvas, not a FakeApp stand-in.
"""

from types import SimpleNamespace

import video_overlay


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


def test_draw_pane_draws_a_center_dot_pinpointing_the_exact_point(sizeamatic_app):
    """Each point should get a small solid center dot in addition to the
    surrounding handle ring, so the exact clicked/dragged pixel is visible
    rather than just the ring's general vicinity (ROADMAP.md Phase 8's
    point visibility item)."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)

    app.ptsL.append((100.0, 50.0))
    app.video_overlay.redraw()

    canvas = app.video_overlay.left_canvas
    r = video_overlay.CENTER_DOT_RADIUS_PX

    # The center dot is a small "overlay"-tagged oval that isn't also tagged
    # "handle" - distinct both from the larger handle-ring oval at the same
    # center, and from the untagged placeholder graphics the "no video
    # loaded" canvas state draws (which draw_pane's own "overlay"-tagged
    # clear/redraw cycle deliberately leaves alone).
    overlay_ovals = [
        item for item in canvas.find_withtag("overlay") if canvas.type(item) == "oval"
    ]
    dot_candidates = [item for item in overlay_ovals if "handle" not in canvas.gettags(item)]

    assert len(dot_candidates) == 1
    bbox = canvas.coords(dot_candidates[0])
    assert bbox == [100.0 - r, 50.0 - r, 100.0 + r, 50.0 + r]
    assert canvas.itemcget(dot_candidates[0], "fill") == "#ff0000"


def test_left_click_on_new_point_places_mate_at_same_image_pixel(sizeamatic_app):
    """Placing the first point on one pane should immediately place its
    mate on the opposite pane at the exact same image pixel coordinates
    (ROADMAP.md Phase 7's usability quiz: the previous scanline-matcher
    guess was found unhelpful in practice, and was replaced with this
    simpler default — the user drags it into place manually, or
    right-click-drags it to invoke the matcher explicitly instead).

    Deliberately doesn't depend on calibration being loaded at all — no
    stereo matching is attempted for this initial placement anymore.
    """

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.metaR = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)
    app.view_rectified.set(True)
    app.cal = None  # should have no effect on this initial placement now

    app.video_overlay.on_left_down("L", _event(100, 50))

    assert len(app.ptsL) == 1
    assert len(app.ptsR) == 1
    assert app.ptsR[0] == app.ptsL[0] == (100.0, 50.0)


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


def test_multiple_clicks_build_a_connected_chain_up_to_the_point_cap(sizeamatic_app):
    """Clicking empty space repeatedly should keep appending points (a
    connected chain, not just a single pair) up to `max_points_per_pane`,
    then silently ignore further clicks past that cap.

    Regression test for ROADMAP.md Phase 7's measurement output item:
    the point cap used to be hardcoded to 2 (main.py FINDINGS.md-adjacent
    history), which made a multi-segment chain impossible to place at
    all regardless of whether the underlying segment math supported one.
    """

    app = sizeamatic_app
    app.metaL = {"width": 2000, "height": 2000, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)

    assert app.max_points_per_pane == 20

    # Click at max_points_per_pane distinct, well-separated locations.
    for i in range(app.max_points_per_pane):
        app.video_overlay.on_left_down("L", _event(10 + i * 20, 10))

    assert len(app.ptsL) == app.max_points_per_pane

    # One more click at yet another empty location should be ignored - the
    # pane is already at its cap.
    app.video_overlay.on_left_down("L", _event(10 + app.max_points_per_pane * 20, 10))
    assert len(app.ptsL) == app.max_points_per_pane


def test_get_handle_index_under_cursor_without_a_handle_returns_none(sizeamatic_app):
    """Clicking where nothing is drawn should not report a handle index."""

    app = sizeamatic_app
    app.metaL = {"width": 640, "height": 480, "fps": 30.0, "frame_count": 10}
    app.fit_to_window.set(False)

    app.video_overlay.redraw()  # clears any leftover overlay items

    idx = app.video_overlay.get_nearest_handle_index("L", app.video_overlay.left_canvas, 9999, 9999)
    assert idx is None
