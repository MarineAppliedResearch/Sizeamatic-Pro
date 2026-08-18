"""Interaction tests for video_overlay.py's VideoPane class.

Exercises the real click -> drag -> release state machine directly
against a real `VideoPane` widget (needs the `qapp` fixture - any
`QWidget` requires a `QApplication` to exist), using a `FakeApp` stand-in
since `VideoPane` only ever reads/writes a handful of specific
attributes off its owning app (see `video_overlay.py`'s own module
docstring for the full list).

Unlike the original Tkinter version - which drew onto a real
`tk.Canvas` (a *retained-mode* API: shapes are persistent, queryable
"canvas items" you can inspect after the fact via `find_withtag`/
`itemcget`/`coords`) - `VideoPane` paints via `QPainter` in `paintEvent`,
an *immediate-mode* API: nothing persists once painting finishes, so
there's no item to query afterward. Where the original tests asserted
against a canvas item's exact color/position, these instead render the
pane to a real `QPixmap` (`pane.grab()`) and sample actual pixel colors
at the expected screen coordinates - the same regression-catching power
(did the right color end up in the right place?), just via Qt's actual
rendering pipeline instead of retained canvas items.
"""

from PySide6.QtCore import QPointF, Qt

import video_overlay


class _FakeMouseEvent:
    """Minimal stand-in for a PySide6 `QMouseEvent`.

    Exposes only `.position()`/`.button()` - the only methods
    `VideoPane`'s mouse handlers actually call on a real one, mirroring
    the original Tkinter suite's own minimal `SimpleNamespace(x=x, y=y)`
    event stand-in.
    """

    def __init__(self, x, y, button=Qt.MouseButton.LeftButton):
        """Store the event's position and button.

        Args:
            x (float): Local X coordinate, in pane screen pixels.
            y (float): Local Y coordinate, in pane screen pixels.
            button (Qt.MouseButton): Which button this event is for.

        Returns:
            None
        """
        self._pos = QPointF(x, y)
        self._button = button

    def position(self):
        """Return this event's local position.

        Returns:
            QPointF: The stored position.
        """
        return self._pos

    def button(self):
        """Return this event's button.

        Returns:
            Qt.MouseButton: The stored button.
        """
        return self._button


def _make_fake_app(**kwargs):
    """Build a minimal stand-in for `main.SizeamaticProApp`, with
    every attribute `VideoPane` might read defaulted to a harmless
    empty/inactive value, then overridden by `**kwargs`.

    Args:
        **kwargs: Attribute name/value pairs to override the defaults.

    Returns:
        object: A plain object with all the attributes `VideoPane`
        needs already set.
    """

    class _FakeVar:
        """Minimal `.get()` stand-in for `main.py`'s `Var` shim."""

        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    class _FakeTutorialWindow:
        """Minimal stand-in for `tutorial_window.TutorialController` -
        just records every `notify_action` call so a test can assert on
        it, without needing a real Tutorial run behind it."""

        def __init__(self):
            self.notified_actions = []

        def notify_action(self, action_name):
            self.notified_actions.append(action_name)

    class _FakeApp:
        pass

    app = _FakeApp()
    app.ptsL = []
    app.ptsR = []
    app.metaL = None
    app.metaR = None
    app.cal = None
    app.current_frameL = None
    app.current_frameR = None
    app.view_rectified = _FakeVar(True)
    app.zoom_min = 1.0
    app.zoom_max = 10.0
    app.zoom_step = 1.10
    app.handle_radius_px = 8
    app.max_points_per_pane = 20
    app.on_points_changed = lambda: None
    app.redisplay_current_frames = lambda: None
    app.render_current_frames = lambda: None
    app.tutorial_window = _FakeTutorialWindow()

    for key, value in kwargs.items():
        setattr(app, key, value)
    return app


def _make_pane(qapp, app, which="L", width=640, height=480):
    """Build a `VideoPane` sized to exactly match its video's pixel
    dimensions, so `image_to_screen`/`screen_to_image` reduce to a
    plain 1:1, zero-offset mapping - the same simplification the
    original suite got "for free" via `fit_to_window=False`, which no
    longer exists (this Qt port always fits/letterboxes; see
    `video_overlay.VideoPane._display_rect`'s docstring).

    Args:
        qapp (QApplication): The shared test-session `QApplication`.
        app: The fake app to own this pane.
        which (str): `"L"` or `"R"`.
        width (int): Pane width - should match the fake app's
            `metaL`/`metaR` width for the 1:1 mapping to hold.
        height (int): Pane height - should match height similarly.

    Returns:
        video_overlay.VideoPane: The constructed, sized pane.
    """
    pane = video_overlay.VideoPane(app, which)
    pane.resize(width, height)
    return pane


def test_left_click_on_empty_space_adds_a_point(qapp):
    """Clicking empty overlay space (no existing handle under the cursor)
    should append a new point to the clicked pane's point list."""

    app = _make_fake_app(metaL={"width": 640, "height": 480})
    pane = _make_pane(qapp, app)

    assert len(app.ptsL) == 0

    pane.mousePressEvent(_FakeMouseEvent(100, 50))

    assert len(app.ptsL) == 1
    assert app.ptsL[0] == (100.0, 50.0)


def test_draw_pane_draws_a_center_dot_pinpointing_the_exact_point(qapp):
    """Each point should get a small solid center dot in addition to the
    surrounding handle ring, so the exact clicked/dragged pixel is visible
    rather than just the ring's general vicinity (ROADMAP.md Phase 8's
    point visibility item)."""

    app = _make_fake_app(metaL={"width": 640, "height": 480}, ptsL=[(100.0, 50.0)])
    pane = _make_pane(qapp, app)
    pane.repaint()

    image = pane.grab().toImage()

    # The center dot is small (CENTER_DOT_RADIUS_PX) and always drawn
    # dead-center on the point - sampling that exact pixel should read
    # its fixed, rectification-independent color.
    assert image.pixelColor(100, 50).name() == video_overlay.CENTER_DOT_COLOR.name()


def test_draw_pane_uses_orange_overlay_color_when_not_rectified(qapp):
    """The ring/line/index-label color should switch from green to orange
    when view_rectified is off, so it's visually obvious a measurement
    isn't real-world-accurate yet (ROADMAP.md Phase 8's rectified/
    not-rectified indicator item). The small red center dot is
    deliberately unaffected - it marks the exact clicked pixel, a
    purpose unrelated to rectification state."""

    class _FakeVar:
        def get(self):
            return False

    app = _make_fake_app(
        metaL={"width": 640, "height": 480},
        ptsL=[(100.0, 50.0), (150.0, 90.0)],
        view_rectified=_FakeVar(),
    )
    pane = _make_pane(qapp, app)
    pane.repaint()

    image = pane.grab().toImage()

    # Sample a pixel on the ring itself (handle_radius_px above the
    # point's center, where the ring's outline is drawn) rather than the
    # point's exact center (which is always the fixed-color dot).
    ring_color = image.pixelColor(100, 50 - app.handle_radius_px).name()
    assert ring_color == video_overlay.NOT_RECTIFIED_OVERLAY_COLOR.name()

    # Sample a pixel along the connecting line between the two points
    # (its midpoint).
    line_color = image.pixelColor(125, 70).name()
    assert line_color == video_overlay.NOT_RECTIFIED_OVERLAY_COLOR.name()

    # The center dot stays red regardless of rectification state.
    assert image.pixelColor(100, 50).name() == video_overlay.CENTER_DOT_COLOR.name()


def test_left_click_on_new_point_does_not_place_a_mate_on_the_other_pane(qapp):
    """Placing a point on one pane should NOT auto-place a matching point
    on the opposite pane (ROADMAP.md Phase 16's usability feedback: users
    wanted to click both sides themselves - left, left, left, then right,
    right, right - rather than fixing an auto-guessed mate every time)."""

    app = _make_fake_app(metaL={"width": 640, "height": 480}, metaR={"width": 640, "height": 480})
    pane = _make_pane(qapp, app)

    pane.mousePressEvent(_FakeMouseEvent(100, 50))

    assert app.ptsL == [(100.0, 50.0)]
    assert app.ptsR == []


def test_drag_moves_an_existing_point(qapp):
    """Dragging an existing point handle should update its position in
    the point list, and releasing should end the drag."""

    app = _make_fake_app(metaL={"width": 640, "height": 480}, ptsL=[(100.0, 50.0)])
    pane = _make_pane(qapp, app)

    # Click down exactly on the point we just placed.
    pane.mousePressEvent(_FakeMouseEvent(100, 50))
    assert pane.drag_active is True
    assert pane.drag_index == 0

    # Drag it to a new position.
    pane.mouseMoveEvent(_FakeMouseEvent(200, 150))
    assert app.ptsL[0] == (200.0, 150.0)

    # Release ends the drag.
    pane.mouseReleaseEvent(_FakeMouseEvent(200, 150))
    assert pane.drag_active is False
    assert pane.drag_index is None


def test_multiple_clicks_build_a_connected_chain_up_to_the_point_cap(qapp):
    """Clicking empty space repeatedly should keep appending points (a
    connected chain, not just a single pair) up to `max_points_per_pane`,
    then silently ignore further clicks past that cap.

    Regression test for ROADMAP.md Phase 7's measurement output item:
    the point cap used to be hardcoded to 2, which made a multi-segment
    chain impossible to place at all regardless of whether the
    underlying segment math supported one.
    """

    app = _make_fake_app(metaL={"width": 2000, "height": 2000})
    pane = _make_pane(qapp, app, width=2000, height=2000)

    assert app.max_points_per_pane == 20

    # Click at max_points_per_pane distinct, well-separated locations.
    for i in range(app.max_points_per_pane):
        pane.mousePressEvent(_FakeMouseEvent(10 + i * 20, 10))

    assert len(app.ptsL) == app.max_points_per_pane

    # One more click at yet another empty location should be ignored - the
    # pane is already at its cap.
    pane.mousePressEvent(_FakeMouseEvent(10 + app.max_points_per_pane * 20, 10))
    assert len(app.ptsL) == app.max_points_per_pane


def test_nearest_handle_index_without_a_handle_returns_none(qapp):
    """Clicking where nothing is drawn should not report a handle index."""

    app = _make_fake_app(metaL={"width": 640, "height": 480})
    pane = _make_pane(qapp, app)

    idx = pane._nearest_handle_index(9999, 9999)
    assert idx is None


class _FakeWheelEvent:
    """Minimal stand-in for a PySide6 `QWheelEvent` - exposes only
    `.position()`/`.angleDelta()`, the only methods `wheelEvent` reads."""

    def __init__(self, x, y, delta_y=120):
        """Store the event's position and vertical scroll delta.

        Args:
            x (float): Local X coordinate, in pane screen pixels.
            y (float): Local Y coordinate, in pane screen pixels.
            delta_y (int): Positive scrolls "up" (zoom in), negative
                scrolls "down" (zoom out) - matches Qt's own convention.

        Returns:
            None
        """
        self._pos = QPointF(x, y)
        self._delta_y = delta_y

    def position(self):
        """Return this event's local position.

        Returns:
            QPointF: The stored position.
        """
        return self._pos

    def angleDelta(self):
        """Return an object whose `.y()` is this event's scroll delta.

        Returns:
            _FakeWheelEvent: `self` - `y()` below reads `_delta_y`
            directly, so this doubles as its own angleDelta result.
        """
        return self

    def y(self):
        """Return the stored vertical scroll delta.

        Returns:
            int: `delta_y` as given to `__init__`.
        """
        return self._delta_y


def test_wheel_zoom_notifies_the_tutorial_of_pan_or_zoom(qapp):
    """Zooming with the scroll wheel should report the "pan_or_zoom"
    tutorial completion action - real-hook detection for that step,
    since pan/zoom has no single discrete handler to hook instead."""

    app = _make_fake_app(metaL={"width": 640, "height": 480})
    pane = _make_pane(qapp, app)

    pane.wheelEvent(_FakeWheelEvent(320, 240, delta_y=120))

    assert "pan_or_zoom" in app.tutorial_window.notified_actions


def test_middle_drag_pan_notifies_the_tutorial_of_pan_or_zoom(qapp):
    """Panning (middle-drag) should report the same "pan_or_zoom"
    tutorial completion action as zooming does - either one teaches
    the step."""

    app = _make_fake_app(metaL={"width": 640, "height": 480})
    pane = _make_pane(qapp, app)
    pane.pan_active = True
    pane.pan_last_pos = (100, 100)

    pane.mouseMoveEvent(_FakeMouseEvent(120, 110))

    assert "pan_or_zoom" in app.tutorial_window.notified_actions
