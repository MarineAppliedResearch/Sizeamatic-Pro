"""Tests for tutorial_window.py's Qt widgets and TutorialController.

Uses the real `qapp`/`sizeamatic_app` fixtures (see tests/conftest.py)
since this module is all real Qt widgets/windows, not the app-parameter-
only functions `make_fake_app` stands in for elsewhere. Rendering
assertions follow test_video_overlay.py's pattern: sample real pixel
colors from a `.grab()`'d image rather than inspecting any retained
drawing state (Qt's `QPainter` is immediate-mode - nothing persists
after `paintEvent` returns).
"""

import os

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QPushButton, QToolBar, QToolButton

import tutorial_engine
import tutorial_fixtures
import tutorial_window


@pytest.fixture(autouse=True)
def _fast_tutorial_fixtures(monkeypatch, tmp_path):
    """Replace the real synthetic fixture generator (a real, but ~1s,
    300-frame video encode - see test_tutorial_fixtures.py for testing
    that directly) with a near-instant 2-frame stand-in for every test
    in this file. These tests exercise `TutorialController`/the engine's
    wiring, not `tutorial_fixtures.py`'s own video-generation code, so
    they don't need the real thing - just something real enough to
    actually load through `main.py`'s loaders without erroring."""

    def _fast_generate():
        calibration_folder = str(tmp_path / "calibration")
        left_path = str(tmp_path / "left.mp4")
        right_path = str(tmp_path / "right.mp4")

        tutorial_window.tutorial_fixtures.generate_tutorial_calibration(calibration_folder)

        size = (tutorial_window.tutorial_fixtures.TUTORIAL_VIDEO_WIDTH, tutorial_window.tutorial_fixtures.TUTORIAL_VIDEO_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for path in (left_path, right_path):
            writer = cv2.VideoWriter(path, fourcc, tutorial_window.tutorial_fixtures.TUTORIAL_VIDEO_FPS, size)
            writer.write(frame)
            writer.write(frame)
            writer.release()

        return {"left_video_path": left_path, "right_video_path": right_path, "calibration_folder": calibration_folder}

    monkeypatch.setattr(tutorial_window.tutorial_fixtures, "generate_tutorial_fixtures", _fast_generate)


def _make_steps(n):
    """Build `n` simple steps targeting the main window's own attributes
    that every `SizeamaticProApp` has regardless of loaded state
    (`btn_play_pause`, `btn_step_back`, ...) - a shared helper so tests
    don't repeat this boilerplate."""

    widget_names = ["btn_play_pause", "btn_step_back", "btn_step_forward", "btn_to_start", "btn_to_end"]
    return [
        tutorial_engine.TutorialStep(
            step_id=f"step{i}",
            title=f"Title {i}",
            description=f"Description {i}",
            details=f"Details {i}",
            target=("main", "widget", widget_names[i % len(widget_names)]),
            completion_action=f"step{i}",
        )
        for i in range(n)
    ]


def _make_mouse_event(widget, local_pos, buttons_down=False):
    """Build a real QMouseEvent for drag-simulation tests - DraggablePanel
    reads `.globalPosition()`/`.button()`/`.buttons()`, which a hand-rolled
    stand-in (like test_video_overlay.py's `_FakeMouseEvent`) would need to
    reimplement in full; a real QMouseEvent is simpler here since global
    coordinates need real screen-mapping math."""

    global_pos = widget.mapToGlobal(local_pos)
    buttons = Qt.MouseButton.LeftButton if buttons_down else Qt.MouseButton.NoButton
    return QMouseEvent(
        QEvent.Type.MouseMove if buttons_down else QEvent.Type.MouseButtonPress,
        QPointF(local_pos),
        QPointF(global_pos),
        Qt.MouseButton.LeftButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


# -------------------------------------------------------------------------
# TutorialStepBubble
# -------------------------------------------------------------------------


def test_bubble_show_step_populates_every_field(qapp):
    """show_step should set the step counter, title, description, and
    Details text, and leave the Details section expanded (the default -
    per the project owner's request, Details is shown by default and
    the user explicitly hides it, not the other way around)."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)
    bubble.show()  # a child's isVisible() depends on the whole ancestor chain being shown
    step = tutorial_engine.TutorialStep(
        step_id="x", title="My Title", description="My description.", details="My details.",
        target=("main", "widget", "btn_play_pause"),
    )

    bubble.show_step(step, step_number=2, total_steps=5)

    assert bubble.step_label.text() == "Step 2 of 5"
    assert bubble.title_label.text() == "My Title"
    assert bubble.desc_label.text() == "My description."
    assert bubble.details_label.text() == "My details."
    assert bubble.details_panel.isVisible() is True


def test_bubble_details_button_toggles_the_details_panel(qapp):
    """Details starts expanded by default - clicking "Hide details"
    should collapse it and relabel the button; clicking again should
    re-expand it."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)
    bubble.show()  # a child's isVisible() depends on the whole ancestor chain being shown

    assert bubble.details_panel.isVisible() is True
    assert "Hide details" in bubble.details_btn.text()

    bubble.details_btn.click()
    assert bubble.details_panel.isVisible() is False
    assert "Show details" in bubble.details_btn.text()

    bubble.details_btn.click()
    assert bubble.details_panel.isVisible() is True
    assert "Hide details" in bubble.details_btn.text()


def test_bubble_next_button_invokes_the_callback(qapp):
    """The Next button is the manual-advance escape hatch - clicking it
    should call whatever `on_next` the caller provided."""

    calls = []
    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: calls.append(1))

    # The Next button is the second (last) button added to the header
    # button row - locate it by its own text rather than assuming
    # layout position, so this test doesn't break on cosmetic reorders.
    next_button = next(b for b in bubble.findChildren(type(bubble.details_btn)) if b.text() == "Next →")
    next_button.click()

    assert calls == [1]


def test_set_next_highlighted_pulses_and_can_be_turned_back_off(qapp):
    """set_next_highlighted(True) should start animating the Next
    button's border (a real inline stylesheet, not just a flag);
    turning it back off should clear that override."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)

    assert bubble.next_btn.styleSheet() == ""

    bubble.set_next_highlighted(True)
    bubble._tick_next_pulse()
    assert bubble.next_btn.styleSheet() != ""

    bubble.set_next_highlighted(False)
    assert bubble.next_btn.styleSheet() == ""


def test_tick_next_pulse_is_a_no_op_when_not_highlighted(qapp):
    """The pulse timer ticks continuously regardless of whether
    highlighting is active - it must not touch the button's style
    unless set_next_highlighted(True) was actually called."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)

    bubble._tick_next_pulse()

    assert bubble.next_btn.styleSheet() == ""


def test_show_completion_swaps_to_the_completion_panel(qapp):
    """show_completion() should hide the regular step content and show
    the completion screen (with its celebration animation running)."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)
    bubble.show()

    bubble.show_completion()

    assert bubble.content_panel.isVisible() is False
    assert bubble.completion_panel.isVisible() is True
    assert bubble._celebration_animation.state() == bubble._celebration_animation.State.Running


def test_show_step_after_completion_swaps_back_to_regular_content(qapp):
    """Showing a regular step again (e.g. the user jumped back via the
    checklist after finishing) should undo the completion screen and
    stop its animation."""

    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None)
    bubble.show()
    bubble.show_completion()

    step = tutorial_engine.TutorialStep(
        step_id="x", title="X", description="X", details="X", target=("main", "widget", "x")
    )
    bubble.show_step(step, 1, 1)

    assert bubble.content_panel.isVisible() is True
    assert bubble.completion_panel.isVisible() is False
    assert bubble._celebration_animation.state() == bubble._celebration_animation.State.Stopped


def test_close_tutorial_button_invokes_on_close_tutorial(qapp):
    """The completion screen's Close Tutorial button should call
    whatever `on_close_tutorial` the caller provided - expected to be
    the controller's full stop() (bubble + checklist + overlay), unlike
    this panel's own independent "x"."""

    calls = []
    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None, on_close_tutorial=lambda: calls.append(1))
    bubble.show_completion()

    close_tutorial_btn = next(b for b in bubble.completion_panel.findChildren(QPushButton) if b.text() == "Close Tutorial")
    close_tutorial_btn.click()

    assert calls == [1]


def test_close_button_closes_the_panel_and_fires_on_closed(qapp):
    """Each panel's own "x" should close that panel and report it via
    on_closed - what the caller does with that report (TutorialController
    postpones the whole tutorial - see test_tutorial_window.py's
    TutorialController tests) is a separate concern from this widget's
    own close mechanism."""

    closed = []
    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None, on_closed=lambda: closed.append(1))
    bubble.show()

    close_button = next(b for b in bubble.findChildren(type(bubble.details_btn)) if b.text() == "✕")
    close_button.click()

    assert closed == [1]
    assert bubble.isVisible() is False


# -------------------------------------------------------------------------
# ChecklistPanel
# -------------------------------------------------------------------------


def test_checklist_scroll_area_caps_height_for_long_step_lists(qapp):
    """A step list longer than CHECKLIST_VISIBLE_ROWS should be shown in
    a fixed-height, scrollable area rather than growing the panel to fit
    every row - the v1 content runs to ~27 steps."""

    many_titles = [f"Step {i}" for i in range(30)]
    checklist = tutorial_window.ChecklistPanel(many_titles, on_step_clicked=lambda i: None)

    scroll_areas = checklist.findChildren(tutorial_window.QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.height() == tutorial_window.CHECKLIST_VISIBLE_ROWS * tutorial_window.CHECKLIST_ROW_HEIGHT_PX
    # All 30 buttons should still exist (just scrollable), not truncated.
    assert len(checklist._buttons) == 30


def test_checklist_scroll_area_shrinks_for_short_step_lists(qapp):
    """A short step list shouldn't reserve a full CHECKLIST_VISIBLE_ROWS
    worth of (mostly empty) scroll area."""

    checklist = tutorial_window.ChecklistPanel(["A", "B", "C"], on_step_clicked=lambda i: None)

    scroll_area = checklist.findChildren(tutorial_window.QScrollArea)[0]
    assert scroll_area.height() == 3 * tutorial_window.CHECKLIST_ROW_HEIGHT_PX


def test_checklist_marks_done_current_and_upcoming_steps_distinctly(qapp):
    """Every row's marker/objectName should reflect done vs. current vs.
    upcoming, independent of each other - a done step stays marked done
    even if it's also (somehow) the "current" index."""

    checklist = tutorial_window.ChecklistPanel(["A", "B", "C"], on_step_clicked=lambda i: None)

    checklist.set_current_step(current_index=1, done=[True, False, False])

    assert checklist._buttons[0].text() == "✓  A"
    assert checklist._buttons[0].objectName() == "doneStep"
    assert checklist._buttons[1].text() == "▶  B"
    assert checklist._buttons[1].objectName() == "currentStep"
    assert checklist._buttons[2].text() == "○  C"
    assert checklist._buttons[2].objectName() == ""


def test_checklist_row_click_reports_its_own_index(qapp):
    """Clicking a checklist row is the decided skip/jump escape hatch -
    it should report that row's index regardless of current progress."""

    clicked = []
    checklist = tutorial_window.ChecklistPanel(["A", "B", "C"], on_step_clicked=lambda i: clicked.append(i))

    checklist._buttons[2].click()

    assert clicked == [2]


# -------------------------------------------------------------------------
# HighlightOverlay
# -------------------------------------------------------------------------


def test_highlight_overlay_is_hidden_until_a_target_is_set(sizeamatic_app):
    """The overlay should start hidden - nothing is highlighted before
    the tutorial actually shows a step."""

    overlay = tutorial_window.HighlightOverlay(sizeamatic_app)

    assert overlay.isVisible() is False


def test_highlight_overlay_shows_after_set_target_rect_and_hides_after_clear(sizeamatic_app):
    """set_target_rect should make the overlay visible; clear_target
    should hide it again - the mechanism `TutorialController` uses to
    turn a step's highlight on/off."""

    sizeamatic_app.show()
    overlay = tutorial_window.HighlightOverlay(sizeamatic_app)

    overlay.set_target_rect(sizeamatic_app.btn_play_pause.rect())
    assert overlay.isVisible() is True

    overlay.clear_target()
    assert overlay.isVisible() is False


# -------------------------------------------------------------------------
# TutorialController
# -------------------------------------------------------------------------


def test_controller_start_builds_widgets_and_begins_at_step_zero(sizeamatic_app, monkeypatch):
    """start() should build the bubble/checklist on first call and begin
    a fresh Tutorial at step 0.

    Uses a fake step list whose completion_actions don't match any real
    handler ("step0", not "load_left_video") - the *real* content's
    first three steps auto-complete themselves as a side effect of
    start()'s own fixture-loading (see the dedicated test for that
    behavior below), which would otherwise make "begins at step 0" a
    false claim for this specific controller/step-list combination.
    """

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)

    controller.start()

    assert controller.tutorial is not None
    assert controller.tutorial.current_index == 0
    assert controller.bubble is not None
    assert controller.checklist is not None
    assert controller.bubble.isVisible() is True
    assert controller.checklist.isVisible() is True

    controller.stop()


def test_start_generates_fixtures_but_does_not_auto_load_them(sizeamatic_app, monkeypatch):
    """start() generates a fresh sample video/calibration set and
    remembers where it put them (`fixture_paths`), but - per the
    project owner's explicit correction - does NOT load them into the
    app itself. The user still practices the real File/Calibration menu
    actions to load the samples themselves, so a fresh tutorial run
    should leave the app's own video/calibration/points state completely
    untouched and the very first step still not done.
    """

    # Bypass the fast fixture stand-in for this one test - it needs the
    # real generator's real output to prove fixture_paths point at
    # actually-generated, loadable files.
    monkeypatch.setattr(
        tutorial_window.tutorial_fixtures, "generate_tutorial_fixtures", tutorial_fixtures.generate_tutorial_fixtures
    )

    sizeamatic_app.show()
    sizeamatic_app.left_video_path = None
    sizeamatic_app.cal = None

    sizeamatic_app.tutorial_window.start()

    assert sizeamatic_app.left_video_path is None
    assert sizeamatic_app.right_video_path is None
    assert sizeamatic_app.cal is None

    fixture_paths = sizeamatic_app.tutorial_window.fixture_paths
    assert os.path.isfile(fixture_paths["left_video_path"])
    assert os.path.isfile(fixture_paths["right_video_path"])
    assert os.path.isdir(fixture_paths["calibration_folder"])

    tutorial = sizeamatic_app.tutorial_window.tutorial
    assert tutorial.current_step().step_id == "load_left_video"
    assert tutorial.done[0] is False

    sizeamatic_app.tutorial_window.stop()


def test_load_dialogs_default_to_the_tutorial_fixture_folder_once_generated(sizeamatic_app, monkeypatch):
    """Once a tutorial has generated its sample files, the real File/
    Calibration load dialogs should default to opening wherever those
    files are - the user still has to pick them, but shouldn't have to
    hunt through an arbitrary temp directory to find them."""

    monkeypatch.setattr(
        tutorial_window.tutorial_fixtures, "generate_tutorial_fixtures", tutorial_fixtures.generate_tutorial_fixtures
    )
    sizeamatic_app.show()
    sizeamatic_app.tutorial_window.start()

    fixture_paths = sizeamatic_app.tutorial_window.fixture_paths
    assert sizeamatic_app._tutorial_fixture_dialog_dir("left_video") == os.path.dirname(fixture_paths["left_video_path"])
    assert sizeamatic_app._tutorial_fixture_dialog_dir("right_video") == os.path.dirname(fixture_paths["right_video_path"])
    assert sizeamatic_app._tutorial_fixture_dialog_dir("calibration") == fixture_paths["calibration_folder"]

    sizeamatic_app.tutorial_window.stop()


def test_load_dialog_default_dir_is_empty_before_any_tutorial_has_run(sizeamatic_app):
    """Before Start Tutorial has ever been clicked, the load dialogs
    should fall back to the OS's own default location, not error."""

    assert sizeamatic_app.tutorial_window.fixture_paths is None
    assert sizeamatic_app._tutorial_fixture_dialog_dir("left_video") == ""


def test_refresh_display_highlights_next_only_for_steps_with_no_completion_action(sizeamatic_app, monkeypatch):
    """A step with a real completion hook shouldn't pulse Next (the
    highlighted GUI control is the actual signal); a step with none
    should, so the user has some signal for how to continue."""

    steps = _make_steps(2)  # both have a completion_action set
    steps.append(
        tutorial_engine.TutorialStep(
            step_id="info_step", title="Info", description="Look at this.", details="Details.",
            target=("main", "widget", "btn_play_pause"),
        )
    )
    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", steps)
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    assert controller.bubble._next_highlighted is False  # step 0 has a completion_action

    controller._on_step_clicked(2)  # jump to the informational step
    assert controller.bubble._next_highlighted is True

    controller.stop()


def test_controller_notify_action_with_no_tutorial_running_is_a_safe_no_op(sizeamatic_app):
    """Handlers call notify_action() unconditionally, whether or not a
    tutorial is running - this must never raise."""

    controller = tutorial_window.TutorialController(sizeamatic_app)

    controller.notify_action("load_left_video")  # no tutorial started - should just do nothing


def test_controller_notify_action_completes_the_current_step_and_advances(sizeamatic_app, monkeypatch):
    """Firing the current step's real completion action should mark it
    done and move the tutorial forward - real-hook detection, the
    default decided in Step 0."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller.notify_action("step0")

    assert controller.tutorial.done[0] is True
    assert controller.tutorial.current_index == 1
    assert controller.bubble.title_label.text() == "Title 1"

    controller.stop()


def test_controller_notify_action_refreshes_the_display_even_when_already_credited(sizeamatic_app, monkeypatch):
    """Regression test for a real bug found during manual proof-testing:
    if step1's action happened to fire early while step0 was still
    current (e.g. a stray click matching a later step's action), step1
    gets credited (`done`) immediately - but the bubble was still
    showing step0, so nothing looked different. Once step0 completes
    for real and the view moves to step1, step1's action firing again
    (genuinely, this time while it's actually shown) must still visibly
    advance the bubble - not silently do nothing just because `done`
    was already True from the earlier, out-of-order call."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller.notify_action("step1")  # fires early, while step0 is current
    assert controller.tutorial.current_index == 0
    assert controller.bubble.title_label.text() == "Title 0"

    controller.notify_action("step0")  # real advance to step1
    assert controller.tutorial.current_index == 1
    assert controller.bubble.title_label.text() == "Title 1"

    controller.notify_action("step1")  # the real action, step1 now current
    assert controller.tutorial.current_index == 2
    assert controller.bubble.title_label.text() == "Title 2"

    controller.stop()


def test_controller_next_button_advances_the_bubble_to_the_next_step(sizeamatic_app, monkeypatch):
    """The bubble's Next button should drive the same advance() path a
    real completion hook would, for steps that need the manual fallback."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(2))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller._on_next_clicked()

    assert controller.tutorial.current_index == 1
    assert controller.bubble.title_label.text() == "Title 1"

    controller.stop()


def test_controller_checklist_click_jumps_directly_to_that_step(sizeamatic_app, monkeypatch):
    """Clicking a checklist row should jump straight there, regardless
    of linear progress - the decided skip/jump escape hatch."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(4))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller._on_step_clicked(3)

    assert controller.tutorial.current_index == 3
    assert controller.bubble.title_label.text() == "Title 3"

    controller.stop()


def test_resolve_target_rect_for_a_main_window_widget(sizeamatic_app):
    """A "main"/"widget" target should resolve against the app window
    itself."""

    sizeamatic_app.show()
    controller = tutorial_window.TutorialController(sizeamatic_app)

    host_window, rect = controller._resolve_target_rect(("main", "widget", "btn_play_pause"))

    assert host_window is sizeamatic_app
    assert rect is not None


def test_resolve_target_rect_for_a_main_window_menu(sizeamatic_app):
    """A "main"/"menu" target should resolve against the named
    top-level menu-bar entry, per Step 0's decision to only highlight
    the closed entry, never an open submenu item."""

    sizeamatic_app.show()
    controller = tutorial_window.TutorialController(sizeamatic_app)

    host_window, rect = controller._resolve_target_rect(("main", "menu", "File"))

    assert host_window is sizeamatic_app
    assert rect is not None


def test_resolve_target_rect_for_measurement_window_before_it_exists_returns_none(sizeamatic_app):
    """The measurement window is built lazily - a step targeting it
    before any measurement has ever been placed shouldn't force it open
    or crash, just report it can't be highlighted yet."""

    controller = tutorial_window.TutorialController(sizeamatic_app)
    assert sizeamatic_app.measurement_window.win is None  # precondition: never opened

    host_window, rect = controller._resolve_target_rect(("measurement_window", "widget", "results_table"))

    assert host_window is None
    assert rect is None


def test_resolve_target_rect_for_measurement_window_widgets_once_it_exists(sizeamatic_app):
    """Regression test for a real bug found during this phase's manual
    proof-test: `results_table`/`record_button`/etc. are attributes of
    the owning `MeasurementWindow` controller object
    (`sizeamatic_app.measurement_window`), NOT of the dialog widget
    itself (`sizeamatic_app.measurement_window.win`) - reading them off
    the wrong object silently returned None, so these steps never
    highlighted anything even once the window existed."""

    sizeamatic_app.show()
    sizeamatic_app.measurement_window.ensure_window()
    controller = tutorial_window.TutorialController(sizeamatic_app)

    for ref in ("results_table", "record_button", "log_table"):
        host_window, rect = controller._resolve_target_rect(("measurement_window", "widget", ref))
        assert host_window is sizeamatic_app.measurement_window.win, ref
        assert rect is not None, ref


def test_resolve_target_rect_falls_back_to_the_toolbar_extension_button_when_hidden(sizeamatic_app):
    """Regression test: a narrow window can collapse a trailing toolbar
    widget behind a ">>" overflow extension button - the widget still
    technically exists but `isVisible()` is False and its geometry is
    meaningless to highlight. `_resolve_target_rect` should fall back to
    the toolbar's own real, visible extension button instead of
    pointing at that stale rect.

    Fakes the extension button rather than narrowing the real window
    (issue #17 moved several widgets off the toolbar specifically to
    fix real overflow - the toolbar is now light enough that it no
    longer overflows at the app's own minimum window width, so this
    scenario can't be reproduced live anymore; the fallback logic
    itself still needs covering for whatever does end up overflowing
    on some future narrower/heavier toolbar)."""

    sizeamatic_app.show()
    controller = tutorial_window.TutorialController(sizeamatic_app)

    real_widget = sizeamatic_app.btn_clear_points
    toolbar = real_widget.parent()
    while not isinstance(toolbar, QToolBar):
        toolbar = toolbar.parent()

    # Qt's toolbar already has its own real (currently invisible)
    # extension button as a child - rename it out of the way first so
    # findChild("qt_toolbar_ext_button") can only match the fake, shown
    # one below, rather than ambiguously returning Qt's own hidden one.
    real_extension = toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
    if real_extension is not None:
        real_extension.setObjectName("")

    extension = QToolButton(toolbar)
    extension.setObjectName("qt_toolbar_ext_button")
    extension.show()
    real_widget.setVisible(False)

    host_window, rect = controller._resolve_target_rect(("main", "widget", "btn_clear_points"))

    assert host_window is sizeamatic_app
    assert rect is not None
    expected = extension.rect()
    expected.moveTopLeft(extension.mapTo(sizeamatic_app, expected.topLeft()))
    assert rect == expected


def test_resolve_target_rect_uses_the_hidden_widget_when_no_toolbar_extension_is_visible(sizeamatic_app):
    """If a widget is hidden for some other reason (not toolbar
    overflow, so the extension button isn't actually showing), there's
    no better fallback - resolving should just return that widget's own
    rect rather than raising."""

    sizeamatic_app.show()
    controller = tutorial_window.TutorialController(sizeamatic_app)

    sizeamatic_app.btn_clear_points.setVisible(False)

    host_window, rect = controller._resolve_target_rect(("main", "widget", "btn_clear_points"))

    assert host_window is sizeamatic_app
    assert rect is not None


def test_refresh_display_raises_and_activates_the_measurement_window_for_its_steps(sizeamatic_app, monkeypatch):
    """Regression test for a real bug found during manual proof-testing:
    a step targeting the measurement window gave no indication that
    window needed to come to the front if the main window currently had
    focus instead - the highlight could be glowing on a buried window."""

    sizeamatic_app.show()
    sizeamatic_app.measurement_window.ensure_window()

    steps = [
        tutorial_engine.TutorialStep(
            step_id="x", title="X", description="X", details="X",
            target=("measurement_window", "widget", "results_table"),
        )
    ]
    monkeypatch.setattr("tutorial_window.STEPS", steps)
    controller = tutorial_window.TutorialController(sizeamatic_app)

    raised = []
    activated = []
    win = sizeamatic_app.measurement_window.win
    win.raise_ = lambda: raised.append(1)
    win.activateWindow = lambda: activated.append(1)

    controller.start()

    assert raised == [1]
    assert activated == [1]

    controller.stop()


def test_controller_shows_completion_screen_once_every_step_is_done(sizeamatic_app, monkeypatch):
    """Advancing past the very last step should trigger the bubble's
    completion screen, not a (nonexistent) step past the end of the
    list."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(2))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller._on_next_clicked()  # step0 -> step1
    assert controller.bubble.completion_panel.isVisible() is False

    controller._on_next_clicked()  # step1 done -> every step now done
    assert controller.tutorial.is_finished() is True
    assert controller.bubble.completion_panel.isVisible() is True
    assert controller.bubble.content_panel.isVisible() is False

    controller.stop()


def test_controller_stop_closes_every_widget_and_clears_the_tutorial(sizeamatic_app):
    """stop() should close the bubble/checklist/overlay and drop the
    active Tutorial, matching how the app's own close handling tears
    down its other independent top-level windows."""

    sizeamatic_app.show()
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller.stop()

    assert controller.tutorial is None
    assert controller.bubble.isVisible() is False
    assert controller.checklist.isVisible() is False


def test_closing_the_bubble_postpones_the_whole_tutorial(sizeamatic_app, monkeypatch):
    """Closing the bubble via its own "x" should hide the checklist too
    and leave the Tutorial's progress intact - per the project owner's
    request, either panel's "x" postpones the whole tutorial rather than
    closing just that one panel."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()
    controller.notify_action("step0")  # advance past step 0, so there's real progress to preserve

    controller.bubble.close()

    assert controller.bubble.isVisible() is False
    assert controller.checklist.isVisible() is False
    assert controller.tutorial is not None
    assert controller.tutorial.current_index == 1  # progress preserved, not reset

    controller.stop()


def test_closing_the_checklist_postpones_the_whole_tutorial(sizeamatic_app, monkeypatch):
    """Same as above, closing the checklist instead - either panel's "x"
    has the same postpone-everything effect."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()

    controller.checklist.close()

    assert controller.bubble.isVisible() is False
    assert controller.checklist.isVisible() is False
    assert controller.tutorial is not None

    controller.stop()


def test_start_after_postponing_resumes_without_resetting_progress_or_fixtures(sizeamatic_app, monkeypatch):
    """Calling start() again (e.g. Help > Start Tutorial…) after a
    postpone should resume exactly where the user left off - same
    Tutorial instance, same progress, same generated fixture paths -
    not regenerate fixtures or reset to step 0."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(3))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()
    controller.notify_action("step0")
    tutorial_before = controller.tutorial
    fixture_paths_before = controller.fixture_paths

    controller.bubble.close()  # postpone
    controller.start()  # "Start Tutorial…" clicked again

    assert controller.tutorial is tutorial_before
    assert controller.fixture_paths is fixture_paths_before
    assert controller.tutorial.current_index == 1
    assert controller.bubble.isVisible() is True
    assert controller.checklist.isVisible() is True

    controller.stop()


def test_start_after_finishing_starts_completely_fresh(sizeamatic_app, monkeypatch):
    """A *finished* tutorial has nothing left to resume - start() should
    behave like a first-ever call (new fixtures, new Tutorial at step 0),
    not try to resume a completed run."""

    sizeamatic_app.show()
    monkeypatch.setattr("tutorial_window.STEPS", _make_steps(1))
    controller = tutorial_window.TutorialController(sizeamatic_app)
    controller.start()
    controller.notify_action("step0")  # completes the only step -> finished
    assert controller.tutorial.is_finished() is True
    finished_tutorial = controller.tutorial

    controller.start()

    assert controller.tutorial is not finished_tutorial
    assert controller.tutorial.current_index == 0
    assert controller.tutorial.done == [False]

    controller.stop()


# -------------------------------------------------------------------------
# DraggablePanel (exercised via TutorialStepBubble, a concrete subclass)
# -------------------------------------------------------------------------


def test_dragging_the_bubble_moves_it_and_reports_the_new_position(qapp):
    """Press-hold-move-release on the bubble's own background should
    move the whole panel and call on_user_moved once, on release."""

    moved = []
    bubble = tutorial_window.TutorialStepBubble(on_next=lambda: None, on_user_moved=lambda: moved.append(1))
    bubble.move(100, 100)
    bubble.show()

    start_pos = bubble.pos()

    press_event = _make_mouse_event(bubble, QPoint(10, 10))
    bubble.mousePressEvent(press_event)

    move_event = _make_mouse_event(bubble, QPoint(60, 60), buttons_down=True)
    bubble.mouseMoveEvent(move_event)

    release_event = _make_mouse_event(bubble, QPoint(60, 60))
    bubble.mouseReleaseEvent(release_event)

    assert bubble.pos() != start_pos
    assert moved == [1]
