"""PySide6 tutorial window(s) for Sizeamatic Pro (ROADMAP.md Phase 15).

Ports the highlight-overlay/current-step-bubble/checklist mechanism
validated live against a throwaway prototype during this phase's
iterative-design step (drag-to-move, minimize-tracking, independent
close buttons - all confirmed working against the real app shell before
any of this became real code) into real, tested app modules.

Contents:
    - `DraggablePanel` - shared base for the two floating panels below:
      click-and-drag repositioning, plus a small "x" - closing *either*
      panel postpones the whole tutorial (see `TutorialController.
      _postpone`) rather than dismissing that one panel independently,
      per the project owner's request.
    - `HighlightOverlay` - dims a host window except a pulsing-bordered
      cutout around one target widget.
    - `TutorialStepBubble` - the roaming current-step panel (title,
      description, expandable Details, Next button).
    - `ChecklistPanel` - the fixed-corner panel listing every step with
      a completion mark; clicking a step jumps straight to it (the
      decided skip/jump escape hatch).
    - `WindowTracker` - a `QObject` event filter that calls back on
      move/resize/minimize-state-change - a plain instance-patched
      `moveEvent` doesn't reliably fire in PySide6 for a window-manager-
      driven move.
    - `TutorialController` - owns one `tutorial_engine.Tutorial` plus
      all of the above, and is the single object `main.py`'s handlers
      (and `video_overlay.py`'s/`measurement_window.py`'s) call into via
      `notify_action` to report real completion-detection events.

Design notes:
    A step's target can live on the main window OR the separate
    `MeasurementWindow` dialog (see `tutorial_engine.TutorialStep`'s
    docstring) - `TutorialController` keeps one `HighlightOverlay` per
    host window, lazily creating the measurement-window one the first
    time a step actually needs it, since that dialog itself is built
    lazily (`measurement_window.py`'s `ensure_window`) and may not exist
    yet when the tutorial starts.

    The current-step bubble and checklist panel always track the *main*
    window's position, never the measurement window's - they're the
    tutorial's own "home base" UI, regardless of which window the
    current step's highlight happens to be pointing at.

Assumptions:
    - The owning app exposes everything `main.py`'s `SizeamaticProApp`
      already does: `menuBar()`, the toolbar attributes named in
      `tutorial_content_operational.py`'s steps (e.g. `btn_play_pause`,
      `pane_left`), and `measurement_window` (a `measurement_window.
      MeasurementWindow`, whose `.win` is the actual dialog once built).

Author:
    Isaac Travers

Created:
    2026-08-17
"""

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import tutorial_engine
import tutorial_fixtures
from tutorial_content_operational import STEPS

PULSE_TICK_MS = 30
"""How often the highlight overlay's pulsing glow border re-paints, in
milliseconds - fast enough to look smooth, cheap enough to not matter."""

PULSE_STEP = 0.05
"""How much the glow's pulse phase (0.0-1.0) advances per tick - a full
dim-to-bright-to-dim cycle takes `2 / PULSE_STEP` ticks."""

RECTIFIED_GLOW_COLOR = QColor("#2f6fed")
"""The highlight overlay's pulsing border color - matches this app's
existing accent blue (`main.py`'s `DARK_QSS` selection/focus color),
not a new color introduced just for the tutorial."""


def _visible_toolbar_fallback(widget):
    """Find a visible stand-in to highlight for a widget hidden behind a
    toolbar's ">>" overflow extension.

    Args:
        widget (QWidget): A hidden widget, possibly a toolbar child that
            got collapsed into the overflow because the window is too
            narrow to show every toolbar item.

    Returns:
        QWidget | None: The enclosing `QToolBar`'s extension button, if
        `widget` is inside a toolbar and that button is currently
        visible - otherwise None, in which case the caller just
        highlights the original (invisible) widget's rect as before.
    """
    ancestor = widget.parent()
    while ancestor is not None and not isinstance(ancestor, QToolBar):
        ancestor = ancestor.parent()
    if ancestor is None:
        return None
    # Qt's own internal object name for a QToolBar's overflow button -
    # there's no public API to fetch it directly. Confirmed against a
    # real narrow-window overflow (not guessed) - PySide6 names it
    # "qt_toolbar_ext_button", not the more guessable
    # "qt_toolbar_extension_button".
    extension = ancestor.findChild(QToolButton, "qt_toolbar_ext_button")
    if extension is not None and extension.isVisible():
        return extension
    return None

CHECKLIST_VISIBLE_ROWS = 10
"""How many step rows `ChecklistPanel` shows before scrolling - the v1
content list (ROADMAP.md Phase 15) runs to ~27 steps, too many to just
keep growing the panel's height for."""

CHECKLIST_ROW_HEIGHT_PX = 32
"""Approximate height of one checklist row (button + spacing) - used to
size the scroll area to `CHECKLIST_VISIBLE_ROWS` rows; doesn't need to
be pixel-exact, just close enough that ~10 rows are visible without
scrolling."""


class DraggablePanel(QFrame):
    """A frameless floating panel the user can click-and-drag anywhere.

    `TutorialStepBubble`/`ChecklistPanel` are frameless `Tool` windows
    with no native title bar to grab, so dragging is implemented by
    hand here: press-and-hold anywhere on the panel's own background
    (not on a child button, which consumes the click for itself) and
    drag. `on_user_moved` fires once, on release, so the caller can
    remember the drop position as an offset relative to the main
    window - a manually-repositioned panel then tracks *that* location
    as the main window moves, per the project owner's request, rather
    than snapping back to a fixed default anchor. `on_closed` fires
    when the user dismisses this panel via its own "x" - distinct from
    this panel merely being `.hide()`-d while the main window is
    minimized, so the caller can tell "the user closed this on purpose"
    apart from "temporarily hidden" and not resurrect a deliberately-
    closed panel the next time the main window moves.
    """

    def __init__(self, flags, on_user_moved=None, on_closed=None):
        """Construct as an independent, frameless, always-on-top window.

        Args:
            flags (Qt.WindowType): Qt window flags - both subclasses
                pass `Tool | FramelessWindowHint | WindowStaysOnTopHint`.
            on_user_moved (Callable[[], None] | None): Called once,
                after a drag ends, so the caller can remember the new
                position.
            on_closed (Callable[[], None] | None): Called when the
                user closes this panel via its own "x".

        Returns:
            None
        """
        super().__init__(None, flags)
        self._on_user_moved = on_user_moved
        self._on_closed = on_closed

        self._drag_offset = None
        """Screen-space offset between the cursor and this panel's
        top-left corner, captured on press - or None while no drag is
        active. Used to keep the panel's position locked to the cursor
        throughout the drag rather than jumping to re-center on it."""

    def closeEvent(self, event):
        """Notify `on_closed` before the close proceeds normally.

        Args:
            event (QCloseEvent): The close event.

        Returns:
            None
        """
        if self._on_closed is not None:
            self._on_closed()
        super().closeEvent(event)

    def mousePressEvent(self, event):
        """Start a drag if the left button was pressed on this panel.

        Args:
            event (QMouseEvent): The press event.

        Returns:
            None
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Continue an active drag, following the cursor.

        Args:
            event (QMouseEvent): The move event.

        Returns:
            None
        """
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End an active drag and report the new position.

        Args:
            event (QMouseEvent): The release event.

        Returns:
            None
        """
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            if self._on_user_moved is not None:
                self._on_user_moved()
        super().mouseReleaseEvent(event)

    def _make_close_button(self):
        """Build the small "x" this panel closes with.

        Wired to `self.close`, which triggers `closeEvent` ->
        `on_closed` - `TutorialController` connects that to postponing
        the *whole* tutorial (both panels), not just this one, per the
        project owner's request (a later correction from this phase's
        earlier prototyping, which had them close independently).

        Returns:
            QPushButton: A small, minimally-styled close button wired
            to `self.close`. Caller places it in its own header row.
        """
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8ea2c6; border: none;"
            " font-size: 11pt; font-weight: bold; padding: 0px; }"
            "QPushButton:hover { color: #ef5350; background: transparent; }"
        )
        close_btn.clicked.connect(self.close)
        return close_btn


class HighlightOverlay(QWidget):
    """Dims a host window except a pulsing-bordered cutout around one
    target widget.

    Parented directly to the host window itself (not its central
    widget), sized to cover the whole window, and raised above
    everything - a toolbar/menu bar/central widget are all siblings
    under the same host `QMainWindow`/`QDialog`, so z-order via
    `raise_()` puts this on top of all of them regardless of the host's
    own layout.

    Cannot highlight something *inside* an already-open `QMenu` - an
    open menu is its own always-on-top popup window that paints above
    this overlay regardless of z-order tricks, since it isn't a sibling
    child widget of the host at all. Confirmed live during this phase's
    prototyping; decided not to build a second, independently-top-level
    overlay just for that case - a menu-targeted step only ever
    highlights the closed top-level menu-bar entry, and its own
    description text names the exact item to click.
    """

    def __init__(self, host_window):
        """Store the host window and start the pulse animation timer.

        Args:
            host_window (QWidget): The top-level window this overlay
                dims/highlights against (the main app window, or the
                measurement window's dialog).

        Returns:
            None
        """
        super().__init__(host_window)
        self.host_window = host_window
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._target_rect = None
        """The current highlight cutout, in `host_window`-local
        coordinates, or None while nothing is being highlighted."""

        self._pulse = 0.0
        """Current pulse phase, 0.0-1.0, driving the glow border's
        width/opacity - see `_tick`."""

        self._pulse_dir = 1
        """+1 while the pulse is brightening, -1 while dimming - flips
        at each end of the 0.0-1.0 range."""

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(PULSE_TICK_MS)

        self.hide()

    def set_target_rect(self, rect):
        """Show (or move) the highlight cutout at a new rect.

        Args:
            rect (QRect): The target widget's rect, in `host_window`-
                local coordinates.

        Returns:
            None
        """
        self._target_rect = rect
        self.setGeometry(self.host_window.rect())
        self.raise_()
        self.show()
        self.update()

    def clear_target(self):
        """Hide the overlay entirely - nothing on this host is
        currently being highlighted.

        Returns:
            None
        """
        self._target_rect = None
        self.hide()

    def _tick(self):
        """Advance the pulse animation by one step and repaint.

        Returns:
            None
        """
        self._pulse += PULSE_STEP * self._pulse_dir
        if self._pulse >= 1.0:
            self._pulse, self._pulse_dir = 1.0, -1
        elif self._pulse <= 0.0:
            self._pulse, self._pulse_dir = 0.0, 1
        if self._target_rect is not None:
            self.update()

    def paintEvent(self, event):
        """Paint the dimmed background and the pulsing cutout border.

        Args:
            event (QPaintEvent): Unused - always repaints the whole
                overlay.

        Returns:
            None
        """
        if self._target_rect is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # A small margin around the target's own rect so the glow
        # doesn't hug the widget's exact edge pixel-for-pixel.
        margin = 6
        cutout = self._target_rect.adjusted(-margin, -margin, margin, margin)

        # Dim everything except the cutout - "whole minus hole" via
        # QPainterPath subtraction, then fill just that leftover shape.
        whole = QPainterPath()
        whole.addRect(QRectF(self.rect()))
        hole = QPainterPath()
        hole.addRoundedRect(QRectF(cutout), 8, 8)
        painter.fillPath(whole.subtracted(hole), QColor(0, 0, 0, 140))

        # Pulsing glow border around the cutout - alpha and width both
        # animate with the same pulse phase for a "breathing" look.
        glow = QColor(RECTIFIED_GLOW_COLOR)
        glow.setAlpha(int(120 + 135 * self._pulse))
        pen = QPen(glow)
        pen.setWidthF(2 + 3 * self._pulse)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(cutout), 8, 8)


class TutorialStepBubble(DraggablePanel):
    """Floating, non-modal panel showing the current step: a step
    counter, title, instruction text, an expandable "Details" section,
    and a Next button.

    Sets its own font sizes explicitly on every child - `main.py`'s
    `DARK_QSS`, applied app-wide via `QApplication.setStyleSheet`,
    forces bold 11pt on every `QWidget` by default (fine for the main
    app's toolbar/menu text, but oversized for a compact info panel).
    """

    def __init__(self, on_next, on_user_moved=None, on_closed=None, on_close_tutorial=None):
        """Build the bubble's layout and widgets.

        Args:
            on_next (Callable[[], None]): Called when the Next button
                is clicked - the manual-advance fallback/escape hatch,
                always available regardless of whether the current
                step also has a real completion hook.
            on_user_moved (Callable[[], None] | None): See
                `DraggablePanel`.
            on_closed (Callable[[], None] | None): See `DraggablePanel` -
                fires only for this panel's own "x", not the completion
                screen's "Close Tutorial" button (see `on_close_tutorial`).
            on_close_tutorial (Callable[[], None] | None): Called when
                the completion screen's "Close Tutorial" button is
                clicked - unlike this panel's own "x" (`on_closed`),
                this is expected to tear down the *whole* tutorial
                (bubble, checklist, and overlay together), since by this
                point there's nothing left to independently keep open.

        Returns:
            None
        """
        super().__init__(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            on_user_moved=on_user_moved,
            on_closed=on_closed,
        )

        self._next_highlighted = False
        """Whether Next should currently pulse - true for steps with no
        real completion hook (`completion_action is None`), so the user
        has a clear signal for "this is how you continue" instead of
        waiting for an action detection that will never fire. Set by
        `TutorialController._refresh_display` via `set_next_highlighted`."""

        self._next_pulse = 0.0
        """Current Next-button pulse phase, 0.0-1.0 - same animation
        shape as `HighlightOverlay`'s glow, just applied to a button's
        border instead of an overlay cutout."""

        self._next_pulse_dir = 1
        self._next_pulse_timer = QTimer(self)
        self._next_pulse_timer.timeout.connect(self._tick_next_pulse)
        self._next_pulse_timer.start(PULSE_TICK_MS)
        self.setStyleSheet(
            "QFrame { background-color: #121a2b; border: 1px solid #2f6fed; border-radius: 8px; }"
            "QLabel { color: #e8eefc; background: transparent; border: none; font-weight: bold; font-size: 9pt; }"
            "QPushButton { background-color: #1a2438; color: #e8eefc; border: 1px solid #263351;"
            " border-radius: 5px; padding: 6px 14px; font-weight: bold; font-size: 8pt; }"
            "QPushButton:hover { background-color: #22304d; }"
            "QFrame#detailsPanel { background-color: #0a0f1a; border: 1px solid #263351; border-radius: 6px; }"
        )
        self.setFixedWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        # WA_TransparentForMouseEvents on every purely-decorative label
        # (and the details sub-panel's own background) so a click
        # anywhere on the bubble's content - not just its bare margins -
        # reaches DraggablePanel's mousePressEvent and starts a drag.
        # Only the actual buttons stay interactive/non-transparent.
        # The header row (step counter + close "x") stays visible in
        # both normal-step and completion mode - only `content_panel`
        # vs `completion_panel` below toggles.
        header_row = QHBoxLayout()
        self.step_label = QLabel("")
        """Shows "Step N of M" (or a completion message) - set by
        `TutorialController._refresh_display`/`show_completion`."""
        self.step_label.setStyleSheet("color: #8ea2c6; font-size: 9pt;")
        self.step_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_row.addWidget(self.step_label, stretch=1)
        header_row.addWidget(self._make_close_button())
        layout.addLayout(header_row)

        self.content_panel = QWidget()
        """Holds everything specific to displaying one regular step -
        swapped out for `completion_panel` once the tutorial finishes."""
        content_layout = QVBoxLayout(self.content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        layout.addWidget(self.content_panel)

        self.title_label = QLabel("")
        """The current step's title."""
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        self.title_label.setWordWrap(True)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        content_layout.addWidget(self.title_label)

        self.desc_label = QLabel("")
        """The current step's main instruction text."""
        self.desc_label.setWordWrap(True)
        self.desc_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        content_layout.addWidget(self.desc_label)

        # A real "Details" section - a distinct bordered panel with its
        # own heading, not a bare label that appears/disappears - per
        # the project owner's feedback during this phase's prototyping.
        # Still collapsible (floating-panel space is limited), but
        # reads as a section of the tutorial's content, not an
        # incidental tooltip-style aside.
        self.details_panel = QFrame()
        """Bordered sub-panel holding the expandable Details text."""
        self.details_panel.setObjectName("detailsPanel")
        self.details_panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(12, 10, 12, 10)
        details_heading = QLabel("DETAILS")
        details_heading.setStyleSheet("color: #2f6fed; font-weight: bold; font-size: 8pt; letter-spacing: 1px;")
        details_heading.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        details_layout.addWidget(details_heading)
        self.details_label = QLabel("")
        """The current step's Details text."""
        self.details_label.setWordWrap(True)
        self.details_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        details_layout.addWidget(self.details_label)
        # Expanded by default - per the project owner's request, Details
        # should be the default view, with an explicit action to hide it
        # rather than the other way around.
        self.details_panel.setVisible(True)
        content_layout.addWidget(self.details_panel)

        btn_row = QHBoxLayout()
        self.details_btn = QPushButton("Hide details ▴")
        """Toggles `details_panel`'s visibility."""
        self.details_btn.clicked.connect(self._toggle_details)
        btn_row.addWidget(self.details_btn)
        btn_row.addStretch(1)
        self.next_btn = QPushButton("Next →")
        """The manual-advance button - pulses when the current step has
        no real completion hook (see `set_next_highlighted`)."""
        self.next_btn.clicked.connect(on_next)
        btn_row.addWidget(self.next_btn)
        content_layout.addLayout(btn_row)

        self.completion_panel = self._build_completion_panel(on_close_tutorial)
        """Shown instead of `content_panel` once every step is done -
        see `_build_completion_panel`/`show_completion`."""
        self.completion_panel.setVisible(False)
        layout.addWidget(self.completion_panel)

    def _build_completion_panel(self, on_close_tutorial):
        """Build the "Tutorial Complete" screen: a pulsing emoji, a
        congratulations message, and a Close Tutorial button.

        Args:
            on_close_tutorial (Callable[[], None] | None): See
                `__init__`'s docstring.

        Returns:
            QWidget: The completion panel, not yet added to any layout.
        """
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 12, 0, 0)
        panel_layout.setSpacing(10)

        celebration_label = QLabel("🎉")
        celebration_label.setStyleSheet("font-size: 30pt;")
        celebration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        celebration_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        panel_layout.addWidget(celebration_label)

        # A simple opacity pulse (Qt's standard way to animate a plain
        # QLabel, which has no directly-animatable "size"/"scale"
        # property of its own) - not full confetti physics, but enough
        # of a celebratory flourish to mark this as a distinct, happy
        # ending rather than just another step.
        self._celebration_opacity_effect = QGraphicsOpacityEffect(celebration_label)
        celebration_label.setGraphicsEffect(self._celebration_opacity_effect)
        self._celebration_animation = QPropertyAnimation(self._celebration_opacity_effect, b"opacity", self)
        """Loops the celebration emoji's opacity in a smooth pulse -
        started in `show_completion`, stopped in `show_step` (so it
        doesn't keep animating, invisibly, once a regular step shows
        again)."""
        self._celebration_animation.setDuration(900)
        self._celebration_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._celebration_animation.setKeyValueAt(0.0, 0.35)
        self._celebration_animation.setKeyValueAt(0.5, 1.0)
        self._celebration_animation.setKeyValueAt(1.0, 0.35)
        self._celebration_animation.setLoopCount(-1)

        title_label = QLabel("Tutorial Complete!")
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        panel_layout.addWidget(title_label)

        subtitle_label = QLabel("You've walked through the full Sizeamatic Pro operational workflow.")
        subtitle_label.setWordWrap(True)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        panel_layout.addWidget(subtitle_label)

        close_tutorial_btn = QPushButton("Close Tutorial")
        if on_close_tutorial is not None:
            close_tutorial_btn.clicked.connect(on_close_tutorial)
        panel_layout.addWidget(close_tutorial_btn)

        return panel

    def _toggle_details(self):
        """Show or hide the Details section, resizing to fit.

        Returns:
            None
        """
        visible = not self.details_panel.isVisible()
        self.details_panel.setVisible(visible)
        self.details_btn.setText("Hide details ▴" if visible else "Show details ▾")
        self.adjustSize()

    def set_next_highlighted(self, active):
        """Turn the Next button's pulsing highlight on or off.

        Args:
            active (bool): True to start pulsing (the current step has
                no real completion hook), False to return to the plain
                default button style.

        Returns:
            None
        """
        self._next_highlighted = active
        if not active:
            self.next_btn.setStyleSheet("")

    def _tick_next_pulse(self):
        """Advance the Next button's pulse animation by one step, if active.

        Returns:
            None
        """
        if not self._next_highlighted:
            return

        self._next_pulse += PULSE_STEP * self._next_pulse_dir
        if self._next_pulse >= 1.0:
            self._next_pulse, self._next_pulse_dir = 1.0, -1
        elif self._next_pulse <= 0.0:
            self._next_pulse, self._next_pulse_dir = 0.0, 1

        glow = QColor(RECTIFIED_GLOW_COLOR)
        glow.setAlpha(int(120 + 135 * self._next_pulse))
        self.next_btn.setStyleSheet(
            "background-color: #1a2438; color: #e8eefc; border-radius: 5px;"
            " font-weight: bold; font-size: 8pt; padding: 5px 13px;"
            f" border: 2px solid rgba({glow.red()}, {glow.green()}, {glow.blue()}, {glow.alpha()});"
        )

    def show_step(self, step, step_number, total_steps):
        """Update every field to show one step's content.

        Args:
            step (tutorial_engine.TutorialStep): The step to display.
            step_number (int): 1-based position for the "Step N of M"
                label.
            total_steps (int): Total step count for the same label.

        Returns:
            None
        """
        # Undo show_completion(), if that's currently showing - a step
        # display always means the tutorial isn't (or is no longer)
        # finished (e.g. the user jumped back via the checklist).
        self._celebration_animation.stop()
        self.completion_panel.setVisible(False)
        self.content_panel.setVisible(True)

        self.step_label.setText(f"Step {step_number} of {total_steps}")
        self.title_label.setText(step.title)
        self.desc_label.setText(step.description)
        self.details_label.setText(step.details)
        # Expanded by default on every step change (see __init__'s
        # docstring note) - the text itself always changes to match the
        # new step, so there's no stale-content risk in leaving it open.
        self.details_panel.setVisible(True)
        self.details_btn.setText("Hide details ▴")
        self.adjustSize()

    def show_completion(self):
        """Show the "Tutorial Complete" screen instead of a regular step.

        Called once `tutorial_engine.Tutorial.is_finished()` is True -
        hides the normal step content, starts the celebration pulse
        animation, and shows the Close Tutorial button.

        Returns:
            None
        """
        self.step_label.setText("All steps complete")
        self.content_panel.setVisible(False)
        self.completion_panel.setVisible(True)
        self._celebration_animation.start()
        self.adjustSize()


class ChecklistPanel(DraggablePanel):
    """Floating, non-modal panel listing every tutorial step with a
    completion mark - separate from `TutorialStepBubble` (the roaming
    current-step panel) per the project owner's request for "both": a
    fixed-position overview of the whole tutorial, plus the compact
    per-step panel.

    Anchors at a fixed corner of the main window by default so the two
    panels don't fight for the same screen space or overlap as the
    bubble roams - but like `TutorialStepBubble`, the user can drag it
    anywhere, and it then tracks that chosen spot instead.
    """

    def __init__(self, step_titles, on_step_clicked, on_user_moved=None, on_closed=None):
        """Build one row per step, each a clickable jump-to-step button.

        Args:
            step_titles (list[str]): Every step's title, in order.
            on_step_clicked (Callable[[int], None]): Called with a
                step's index when its row is clicked - the decided
                skip/jump escape hatch (no locking; any step is always
                reachable).
            on_user_moved (Callable[[], None] | None): See
                `DraggablePanel`.
            on_closed (Callable[[], None] | None): See `DraggablePanel`.

        Returns:
            None
        """
        super().__init__(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            on_user_moved=on_user_moved,
            on_closed=on_closed,
        )
        self.setStyleSheet(
            "QFrame { background-color: #121a2b; border: 1px solid #263351; border-radius: 8px; }"
            "QLabel { color: #e8eefc; background: transparent; border: none; font-weight: bold; font-size: 9pt; }"
            "QPushButton { background-color: transparent; color: #8ea2c6; border: none; text-align: left;"
            " padding: 5px 8px; font-weight: bold; font-size: 9pt; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a2438; }"
            "QPushButton#currentStep { color: #e8eefc; background-color: #1a2438; border: 1px solid #2f6fed; }"
            "QPushButton#doneStep { color: #2fbf71; }"
        )
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        header_row = QHBoxLayout()
        heading = QLabel("TUTORIAL PROGRESS")
        heading.setStyleSheet("color: #2f6fed; font-weight: bold; font-size: 8pt; letter-spacing: 1px;")
        heading.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_row.addWidget(heading, stretch=1)
        header_row.addWidget(self._make_close_button())
        layout.addLayout(header_row)

        # The v1 content list runs to ~27 steps - too many to just keep
        # growing the panel's height for, so the buttons live inside a
        # scroll area capped to CHECKLIST_VISIBLE_ROWS rows tall instead
        # of directly in this panel's own layout.
        buttons_container = QWidget()
        buttons_container.setStyleSheet("background: transparent;")
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(4)

        self._buttons = []
        """One QPushButton per step, in order - see `set_current_step`
        for how each one's text/style reflects done/current/upcoming."""
        for i, title in enumerate(step_titles):
            btn = QPushButton(f"○  {title}")
            btn.clicked.connect(lambda _checked=False, idx=i: on_step_clicked(idx))
            buttons_layout.addWidget(btn)
            self._buttons.append(btn)
        buttons_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(buttons_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_area.setFixedHeight(min(len(step_titles), CHECKLIST_VISIBLE_ROWS) * CHECKLIST_ROW_HEIGHT_PX)
        layout.addWidget(scroll_area)

    def set_current_step(self, current_index, done):
        """Refresh every row's marker to reflect current progress.

        Args:
            current_index (int): The step currently being shown.
            done (list[bool]): `tutorial_engine.Tutorial.done` - which
                steps are complete, parallel to `self._buttons`.

        Returns:
            None
        """
        for i, btn in enumerate(self._buttons):
            # Recover the plain title text (strip whichever marker +
            # two spaces currently prefixes it) so re-marking a row
            # doesn't accumulate marker characters across calls.
            title = btn.text().split("  ", 1)[1]
            if done[i]:
                btn.setObjectName("doneStep")
                btn.setText(f"✓  {title}")
            elif i == current_index:
                btn.setObjectName("currentStep")
                btn.setText(f"▶  {title}")
            else:
                btn.setObjectName("")
                btn.setText(f"○  {title}")
            # setObjectName alone doesn't re-evaluate this button's own
            # QSS selector match - force a style refresh so the
            # #currentStep/#doneStep rules actually take effect.
            btn.style().unpolish(btn)
            btn.style().polish(btn)


class WindowTracker(QObject):
    """Calls back whenever the watched window moves, resizes, or
    changes minimized/restored state.

    Instance-patching a Qt virtual method (`window.moveEvent = ...`)
    doesn't reliably fire for a window-manager-driven move on Windows -
    confirmed during this phase's prototyping. An event filter is a
    proper virtual method on this `QObject` subclass, so Qt actually
    calls it regardless of how the watched widget's own class
    implements (or doesn't implement) those handlers.
    `WindowStateChange` covers minimize/restore - the floating panels
    are independent top-level windows, so minimizing the main window
    does not minimize them for free; the project owner asked for them
    to disappear/reappear along with it instead of being left floating
    with nothing behind them.
    """

    def __init__(self, callback):
        """Store the callback to invoke on every watched event.

        Args:
            callback (Callable[[], None]): Called with no arguments on
                every move/resize/state-change event.

        Returns:
            None
        """
        super().__init__()
        self._callback = callback

    def eventFilter(self, watched, event):
        """Invoke the callback for move/resize/state-change events.

        Args:
            watched (QObject): The watched window (unused - this
                filter is only ever installed on one window at a time).
            event (QEvent): The event being filtered.

        Returns:
            bool: Always False - never consumes the event, only
            observes it.
        """
        if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowStateChange):
            self._callback()
        return False


class TutorialController:
    """Owns one `tutorial_engine.Tutorial` run plus every widget that
    displays it - the single object `main.py`'s (and
    `video_overlay.py`'s/`measurement_window.py`'s) handlers call into.

    One instance lives on the main application (`app.tutorial_window`),
    created lazily on first use (matching this app's existing
    `ensure_window()` pattern for its other sub-windows) and reused for
    the app's lifetime - but `start()` always begins a completely fresh
    run (a new `Tutorial` from `tutorial_content_operational.STEPS`,
    reset to step 0), per Step 0's "no persistence" decision.
    """

    def __init__(self, app):
        """Store the owning app and initialize widget/state references.

        Args:
            app: The main application object (`main.SizeamaticProApp`).

        Returns:
            None
        """
        self.app = app

        self.tutorial = None
        """The active `tutorial_engine.Tutorial`, or None if no
        tutorial run is currently in progress (before the first
        `start()` call, or after the app closes)."""

        self.bubble = None
        """The current-step `TutorialStepBubble`, or None until `start()`
        first builds it."""

        self.checklist = None
        """The `ChecklistPanel`, or None until `start()` first builds it."""

        self.main_overlay = None
        """`HighlightOverlay` hosted on the main app window, or None
        until first needed."""

        self.measurement_overlay = None
        """`HighlightOverlay` hosted on the measurement window's dialog,
        or None until a step actually targets it (that dialog itself is
        built lazily - see this module's docstring)."""

        self._window_tracker = None
        """`WindowTracker` installed on the main app window, or None
        until `start()` first builds it."""

        self.fixture_paths = None
        """The most recent `tutorial_fixtures.generate_tutorial_fixtures()`
        result (a dict with `"left_video_path"`/`"right_video_path"`/
        `"calibration_folder"` keys), or None before the first `start()`
        call. `main.py`'s load-dialog handlers read this to default their
        file dialog's starting directory to wherever the generated
        sample files actually are, rather than the user needing to hunt
        for a temp directory - the tutorial doesn't load these files
        itself; the user still does, through the real menu actions."""

        self._bubble_offset = {"dx": None, "dy": None}
        """The bubble's position relative to the main window's top-left,
        in pixels - None means "use the default anchor formula" (see
        `_position_bubble`); set once the user drags the bubble, and
        persists across step changes and window moves until dragged
        again or the tutorial is restarted."""

        self._checklist_offset = {"dx": None, "dy": None}
        """Same as `_bubble_offset`, for the checklist panel."""

        self._postponed = False
        """Whether the user postponed the tutorial via either panel's
        own "x" this run - distinct from the panels being `.hide()`-d
        while the main window is minimized, so restoring from minimized
        doesn't resurrect a postponed tutorial on its own; only the next
        `start()` call does. Closing *either* panel postpones the whole
        tutorial (hides both) - per the project owner's request, the
        "x" isn't independent-per-panel dismissal here, it's "pause this
        for later"."""

    # -------------------------------------------------------------------------
    # Starting/stopping a run
    # -------------------------------------------------------------------------

    def start(self):
        """Begin a tutorial run - resuming a postponed one if there is
        one, or starting completely fresh otherwise.

        Either panel's own "x" postpones the tutorial rather than ending
        it (see `_postpone`) - clicking Help > Start Tutorial… again
        after that resumes exactly where it was left off, per the
        project owner's explicit request, rather than discarding
        progress. Only starts fresh (new fixtures, new `Tutorial`, step
        0) if there's no unfinished run to resume - either this is the
        very first call, or the previous run was already completed (the
        completion screen's Close Tutorial button called `stop()`).

        Returns:
            None
        """
        if self.tutorial is not None and not self.tutorial.is_finished():
            self._resume()
            return
        self._start_fresh()

    def _resume(self):
        """Re-show a postponed (or already-open) tutorial run as-is -
        no new fixtures, no progress reset.

        Returns:
            None
        """
        self._postponed = False
        self.bubble.show()
        self.checklist.show()
        self._refresh_display()

    def _start_fresh(self):
        """Begin a completely fresh tutorial run.

        Generates a brand new synthetic tutorial video+calibration set
        (`tutorial_fixtures.generate_tutorial_fixtures`) - but, per the
        project owner's explicit correction, does NOT load it
        automatically. The user still goes through the real File > Load
        Left/Right Video…/Calibration > Load Calibration… menu actions
        themselves and picks the generated sample files, same as they
        would with real footage - `self.fixture_paths` just remembers
        where those files ended up, so `main.py`'s load dialogs can
        default to opening there instead of an arbitrary temp directory.
        Builds a new `Tutorial` from `tutorial_content_operational.STEPS`
        (discarding any previous run's progress - there is no
        persistence across a *completed*/stopped run, per Step 0's
        decision), builds the bubble/checklist/window tracker on first
        call, resets any manually-dragged positions and the postponed
        flag back to their defaults, and shows step 0.

        Returns:
            None
        """
        # A fresh run also resets manual placement/postponed state - a
        # panel dragged or a tutorial postponed during a *previous*,
        # already-completed run shouldn't silently carry over into this
        # one.
        self._bubble_offset = {"dx": None, "dy": None}
        self._checklist_offset = {"dx": None, "dy": None}
        self._postponed = False

        if self.bubble is None:
            self.bubble = TutorialStepBubble(
                on_next=self._on_next_clicked,
                on_user_moved=self._remember_bubble_offset,
                on_closed=self._on_panel_closed,
                on_close_tutorial=self.stop,
            )
        if self.checklist is None:
            self.checklist = ChecklistPanel(
                [step.title for step in STEPS],
                on_step_clicked=self._on_step_clicked,
                on_user_moved=self._remember_checklist_offset,
                on_closed=self._on_panel_closed,
            )
        if self.main_overlay is None:
            self.main_overlay = HighlightOverlay(self.app)
        if self._window_tracker is None:
            self._window_tracker = WindowTracker(self._on_main_window_moved_or_resized)
            self.app.installEventFilter(self._window_tracker)

        self.bubble.show()
        self.checklist.show()

        self.tutorial = tutorial_engine.Tutorial(list(STEPS))
        self.fixture_paths = tutorial_fixtures.generate_tutorial_fixtures()

        self._refresh_display()

    def stop(self):
        """Tear down every tutorial widget - called from the completion
        screen's Close Tutorial button, and from the app's own close
        handling, since these are independent top-level windows Qt's
        parent/child cleanup won't close on its own.

        Unlike postponing (either panel's own "x"), this really does end
        the run - the next `start()` call begins completely fresh.

        Returns:
            None
        """
        for widget in (self.bubble, self.checklist, self.main_overlay, self.measurement_overlay):
            if widget is not None:
                widget.close()
        self.tutorial = None
        self._postponed = False

    # -------------------------------------------------------------------------
    # Real completion detection - the entry point every wired-up handler calls
    # -------------------------------------------------------------------------

    def notify_action(self, action_name):
        """Report that a real app action just happened.

        Safe to call unconditionally from every wired-up handler,
        whether or not a tutorial is currently running - a no-op
        whenever `self.tutorial` is None.

        Args:
            action_name (str): The action-name key that just fired
                (e.g. `"load_left_video"`) - must match some step's
                `completion_action` in `tutorial_content_operational.py`
                to have any visible effect.

        Returns:
            None
        """
        if self.tutorial is None:
            return

        index_before = self.tutorial.current_index
        newly_done = self.tutorial.mark_action_done(action_name)
        # Refresh on a current-step advance too, not just on newly_done -
        # `mark_action_done` can move `current_index` forward even when
        # `newly_done` comes back empty (the current step's `done` flag
        # was already set early by an out-of-order action while a
        # different step was current; see its docstring). Gating the
        # refresh on `newly_done` alone would silently advance the
        # engine's internal state without ever updating what's on
        # screen - exactly the "did the right thing but nothing visibly
        # happened" bug this call exists to prevent.
        if newly_done or self.tutorial.current_index != index_before:
            self._refresh_display()

    # -------------------------------------------------------------------------
    # Navigation callbacks
    # -------------------------------------------------------------------------

    def _on_next_clicked(self):
        """Handle the bubble's Next button - the manual-advance
        fallback/escape hatch, always available.

        Returns:
            None
        """
        if self.tutorial is None:
            return
        self.tutorial.advance()
        self._refresh_display()

    def _on_step_clicked(self, index):
        """Handle a checklist row click - jump straight to that step.

        Args:
            index (int): The clicked step's position.

        Returns:
            None
        """
        if self.tutorial is None:
            return
        self.tutorial.jump_to(index)
        self._refresh_display()

    def _on_panel_closed(self):
        """Handle either panel's own "x" - postpones the whole tutorial.

        Wired as the `on_closed` callback for both the bubble and the
        checklist (see `_start_fresh`) - per the project owner's
        request, closing *either* one is "pause this for later", not
        independent-per-panel dismissal.

        Returns:
            None
        """
        self._postpone()

    def _postpone(self):
        """Hide both panels without ending the tutorial run.

        Progress (`self.tutorial`) and the generated fixtures
        (`self.fixture_paths`) stay exactly as they are - the next
        `start()` call resumes here instead of starting over.

        Returns:
            None
        """
        self._postponed = True
        if self.bubble is not None:
            self.bubble.hide()
        if self.checklist is not None:
            self.checklist.hide()
        for overlay in (self.main_overlay, self.measurement_overlay):
            if overlay is not None:
                overlay.clear_target()

    def _remember_bubble_offset(self):
        """Capture the bubble's current position as its new default
        offset from the main window, after the user finishes dragging it.

        Returns:
            None
        """
        self._bubble_offset["dx"] = self.bubble.x() - self.app.x()
        self._bubble_offset["dy"] = self.bubble.y() - self.app.y()

    def _remember_checklist_offset(self):
        """Same as `_remember_bubble_offset`, for the checklist panel.

        Returns:
            None
        """
        self._checklist_offset["dx"] = self.checklist.x() - self.app.x()
        self._checklist_offset["dy"] = self.checklist.y() - self.app.y()

    def _on_main_window_moved_or_resized(self):
        """Keep every tutorial widget in sync with the main window.

        The panels are independent top-level windows, so minimizing the
        main window doesn't minimize them for free - hide them along
        with it instead of leaving them floating with nothing behind
        them, and bring them back (repositioned/re-highlighted for the
        current window layout) on restore - unless the tutorial is
        currently postponed (either panel's own "x"), in which case only
        an explicit `start()` call should bring them back.

        Returns:
            None
        """
        if self.tutorial is None or self._postponed:
            return

        if self.app.isMinimized():
            self.bubble.hide()
            self.checklist.hide()
            return

        if not self.bubble.isVisible():
            self.bubble.show()
        if not self.checklist.isVisible():
            self.checklist.show()

        self._refresh_display()

    # -------------------------------------------------------------------------
    # Display refresh
    # -------------------------------------------------------------------------

    def _resolve_target_rect(self, target):
        """Turn a step's `(host, kind, ref)` target into a concrete
        rect in its host window's local coordinates.

        Args:
            target (tuple[str, str, str]): See
                `tutorial_engine.TutorialStep`'s docstring.

        Returns:
            tuple[QWidget, QRect] | tuple[None, None]: The resolved
            host window and rect, or `(None, None)` if the target
            can't be resolved right now (e.g. a measurement-window
            step before that dialog has ever been built).
        """
        host_name, kind, ref = target

        if host_name == "main":
            # The app itself both IS the top-level window to highlight
            # against AND holds every widget attribute directly.
            host_window = self.app
            attr_source = self.app
        else:
            # The measurement window is built lazily - it may not exist
            # yet if the user hasn't placed a measurement this run.
            # Don't force it open just to satisfy a jump/checklist click;
            # the relevant step simply can't be highlighted yet.
            host_window = self.app.measurement_window.win
            if host_window is None:
                return None, None
            # Unlike the main app, the measurement window's actual
            # QDialog (`host_window`, aka `.win`) does NOT hold
            # `results_table`/`record_button`/etc. as its own
            # attributes - those live on the owning `MeasurementWindow`
            # controller object instead (`measurement_window.py`'s
            # `ensure_window`, e.g. `self.record_button = ...`). Reading
            # the widget from the wrong object was a real bug found
            # during this phase's manual proof-test: steps targeting the
            # measurement window never highlighted anything.
            attr_source = self.app.measurement_window

        if kind == "menu":
            action = next((a for a in host_window.menuBar().actions() if a.text() == ref), None)
            if action is None:
                return None, None
            rect = host_window.menuBar().actionGeometry(action)
            rect.moveTopLeft(host_window.menuBar().mapTo(host_window, rect.topLeft()))
            return host_window, rect

        widget = getattr(attr_source, ref, None)
        if widget is None:
            return None, None
        if not widget.isVisible():
            # A narrow window can collapse trailing toolbar widgets (e.g.
            # "Set Time Sync", built last in `_build_real_time_sync_group`)
            # behind a ">>" extension button - the widget still exists as
            # a hidden child at that point, so highlighting it directly
            # would compute a meaningless rect. Fall back to the
            # toolbar's own extension button so there's still something
            # real to point at; the user opens it, the widget becomes
            # visible again inside the popup, and the real click still
            # fires the same completion hook either way.
            widget = _visible_toolbar_fallback(widget) or widget
        rect = widget.rect()
        rect.moveTopLeft(widget.mapTo(host_window, rect.topLeft()))
        return host_window, rect

    def _overlay_for_host(self, host_window):
        """Get (lazily building) the `HighlightOverlay` for one host.

        Args:
            host_window (QWidget): The main app window, or the
                measurement window's dialog.

        Returns:
            HighlightOverlay: The overlay already attached to that
            host, building a new one on first use.
        """
        if host_window is self.app:
            if self.main_overlay is None:
                self.main_overlay = HighlightOverlay(self.app)
            return self.main_overlay

        if self.measurement_overlay is None or self.measurement_overlay.host_window is not host_window:
            # Rebuild if the measurement window itself was closed and
            # reopened since - `ensure_window` builds a brand new dialog
            # each time, so a stale overlay's host_window would be a
            # since-destroyed widget.
            self.measurement_overlay = HighlightOverlay(host_window)
        return self.measurement_overlay

    def _position_bubble(self):
        """Move the bubble to its default anchor, or the user's
        last-dragged offset if one was set.

        Returns:
            None
        """
        if self._bubble_offset["dx"] is None:
            self.bubble.move(self.app.x() + self.app.width() - self.bubble.width() - 40, self.app.y() + 80)
        else:
            self.bubble.move(self.app.x() + self._bubble_offset["dx"], self.app.y() + self._bubble_offset["dy"])

    def _position_checklist(self):
        """Move the checklist to its default anchor, or the user's
        last-dragged offset if one was set.

        Returns:
            None
        """
        if self._checklist_offset["dx"] is None:
            self.checklist.move(self.app.x() + 20, self.app.y() + 80)
        else:
            self.checklist.move(
                self.app.x() + self._checklist_offset["dx"], self.app.y() + self._checklist_offset["dy"]
            )

    def _refresh_display(self):
        """Redraw everything to match the tutorial's current state:
        the highlight overlay's target, the bubble's content/position,
        and the checklist's progress markers/position.

        Once every step is done, shows the bubble's completion screen
        instead of a regular step - this can happen regardless of which
        step is "current" (e.g. the user completed every other step out
        of order via the checklist's jump escape hatch, and the very
        last one to finish wasn't the last step in the list).

        Returns:
            None
        """
        if self.tutorial.is_finished():
            self._show_completion()
            return

        step = self.tutorial.current_step()
        host_window, rect = self._resolve_target_rect(step.target)

        # Bring whichever window this step's target actually lives on to
        # the front - a highlight glowing on a window buried behind
        # another one doesn't help. Real bug found during this phase's
        # manual proof-test: steps targeting the measurement window gave
        # no indication it needed to be raised if the main window
        # currently had focus instead. Harmless to call even when
        # `host_window` is already frontmost - `raise_`/`activateWindow`
        # are no-ops in that case, not a visible flicker.
        if host_window is not None:
            host_window.raise_()
            host_window.activateWindow()

        # Resolve the one overlay actually in use this call (None if the
        # target couldn't be resolved at all) *before* the clearing loop
        # below, rather than re-deriving it per overlay - avoids lazily
        # building an overlay just to compare against it.
        active_overlay = self._overlay_for_host(host_window) if host_window is not None else None

        # Clear whichever overlay isn't the one currently in use, so a
        # step targeting the main window doesn't leave a stale
        # highlight glowing on the measurement window (or vice versa).
        for overlay in (self.main_overlay, self.measurement_overlay):
            if overlay is not None and overlay is not active_overlay:
                overlay.clear_target()

        if active_overlay is not None and rect is not None:
            active_overlay.set_target_rect(rect)

        self.bubble.show_step(step, self.tutorial.current_index + 1, len(self.tutorial.steps))
        # A step with no real completion hook has nothing else to detect -
        # pulsing Next is the only signal for "this is how you continue".
        self.bubble.set_next_highlighted(step.completion_action is None)
        self._position_bubble()

        self.checklist.set_current_step(self.tutorial.current_index, self.tutorial.done)
        self._position_checklist()

    def _show_completion(self):
        """Show the "Tutorial Complete" screen - every step is done.

        Clears any active highlight (nothing left to point at), shows
        the bubble's completion screen, and leaves the checklist as-is -
        it already reads as fully checked off via its normal per-step
        rendering once every entry in `tutorial.done` is True, so it
        needs no special-casing here.

        Returns:
            None
        """
        for overlay in (self.main_overlay, self.measurement_overlay):
            if overlay is not None:
                overlay.clear_target()

        self.bubble.show_completion()
        self._position_bubble()

        self.checklist.set_current_step(self.tutorial.current_index, self.tutorial.done)
        self._position_checklist()
