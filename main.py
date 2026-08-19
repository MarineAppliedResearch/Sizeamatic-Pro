"""Sizeamatic Pro main application module (PySide6).

Sizeamatic Pro is a desktop tool for measuring real world distances from
stereo video. This module defines `SizeamaticProApp`, the Qt GUI
container that owns window/menu construction, video playback and
timeline state, calibration loading, and the top-level event wiring
that ties together the supporting feature modules (`stereo_matching`,
`calibration_io`, `video_overlay`).

This is a PySide6 port of the original Tkinter app (ROADMAP.md Phase
11) - the project owner decided Tkinter/ttk couldn't reach the
polish/consistency bar this app needs (its native menu bar can't be
dark-themed at all, and a hand-built ttk replacement menu system was
tried and rejected as too much reimplemented-from-scratch risk for
something used constantly). PySide6's QSS stylesheets style
everything, including menus, without fighting the OS. The underlying
measurement/calibration math (`stereo_matching.py`, `calibration_io.py`,
`project_io.py`) is completely framework-independent and needed zero
changes for this port.

Every screen from the original Tkinter app is now ported: video
loading/playback/sync, pan/zoom, and point placement/dragging/
refinement (see `video_overlay.py`); the measurement pipeline, results
window (`measurement_window.py`), and real-time sync; calibration
summary (`calibration_summary.py`); frame-pair capture and running a
new calibration (`perform_calibration.py`); printable calibration
target generation (`generate_calibration_target.py`); the anaglyph 3D
preview (`anaglyph_preview.py`); and project save/load/recent projects.
Small shared Qt-only helpers (PIL-to-`QPixmap` conversion, a
close-callback `QDialog` base) live in `qt_helpers.py` so the four
sub-window modules can use them without importing back from this file.

See `ARCHITECTURE.md` for how responsibilities are currently split
across files, and `README.md` for the user-facing description of the
app.
"""

import ctypes
import datetime
import os
import signal
import sys
import time
import tomllib

import cv2
import qtawesome as qta
from PIL import Image

from PySide6.QtCore import QSize, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QCursor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplashScreen,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import anaglyph_preview
import calibration_io
import calibration_summary
import generate_calibration_target
import measurement_window
import perform_calibration
import prepare_splash_image
import project_io
import recent_projects
import stereo_matching
import tutorial_window
import video_overlay
from qt_helpers import enable_dark_title_bar, pil_image_to_qpixmap


def resource_path(relative_path):
    """Get the correct path to a bundled resource file.

    When running normally, this returns a path relative to the source
    folder. When running from PyInstaller, this returns a path inside
    the bundled app. Unchanged from the original Tkinter app - no
    framework dependency to begin with.

    Args:
        relative_path (str): The file path relative to the project root.

    Returns:
        str: The absolute path to the requested resource.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def get_app_version():
    """Read this app's version string from `pyproject.toml`.

    Unchanged from the original - reads via `resource_path` rather than
    a path relative to `__file__`, since a packaged build's `__file__`
    resolves inside PyInstaller's temporary extraction folder.

    Returns:
        str: The version string (e.g. "0.1.0"), or "unknown" if
        `pyproject.toml` can't be found or parsed for any reason.
    """
    try:
        with open(resource_path("pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
        return str(data["project"]["version"])
    except Exception:
        return "unknown"


STARTUP_SPLASH_MIN_SECONDS = 2.0
"""Minimum time the startup splash (`_show_startup_splash`) stays on
screen, even if building the app finishes faster than that - so it's
actually readable rather than a barely-visible flash on a fast machine
or a source run. Unchanged from the original."""

STARTUP_SPLASH_MAX_WIDTH_PX = 720
"""Cap the splash's on-screen width, scaling the source art down
proportionally if it's larger. Unchanged from the original."""


def _show_startup_splash():
    """Show a splash screen while the rest of the app builds.

    Uses Qt's own `QSplashScreen` rather than a hand-rolled borderless
    window (the original Tkinter app had no built-in splash widget, so
    it had to build one from a plain `Toplevel`) - reuses
    `prepare_splash_image.py`'s existing flatten/version-text pipeline
    unchanged, since that module never depended on Tkinter to begin
    with.

    Best-effort: returns None (skipping the splash entirely) if the
    splash image can't be prepared for any reason, rather than blocking
    startup over a missing/corrupt art asset.

    Returns:
        QSplashScreen | None: The already-shown splash screen, with a
        `.progress_bar` attribute (a `QProgressBar` child widget near
        the bottom, next to the version text) the caller should drive
        from 0 to 100 over the splash's on-screen duration - or None.
    """
    try:
        img = prepare_splash_image.flatten_splash_image(resource_path("assets/splash-pro.png"))
        img = prepare_splash_image.add_version_text(img, f"v{get_app_version()}")

        if img.width > STARTUP_SPLASH_MAX_WIDTH_PX:
            scale = STARTUP_SPLASH_MAX_WIDTH_PX / img.width
            img = img.resize((STARTUP_SPLASH_MAX_WIDTH_PX, round(img.height * scale)), Image.LANCZOS)

        pixmap = pil_image_to_qpixmap(img)
    except Exception:
        return None

    splash = QSplashScreen(pixmap)

    # Center on whichever screen the cursor is actually on, not Qt's
    # "primary" screen - on a multi-monitor setup those aren't always
    # the same one, and a new top-level window generally ends up on
    # whichever screen the user's actually working on/looking at, not
    # necessarily the one Qt considers primary.
    screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        x = available.x() + (available.width() - pixmap.width()) // 2
        y = available.y() + (available.height() - pixmap.height()) // 2
        splash.move(x, y)

    # A thin progress bar near the bottom, on its own row *above* the
    # version text's own bottom-margin row (not sharing it) - driven by
    # the caller (main()) from 0 to 100 across the splash's minimum
    # on-screen duration, so there's a visible sign of progress rather
    # than a static image.
    bar_height = 4
    margin = max(12, pixmap.width() // 40)
    # add_version_text sizes its font as roughly height/22 (before this
    # image was scaled down to fit STARTUP_SPLASH_MAX_WIDTH_PX, but that
    # scale-down preserves aspect ratio, so the ratio of text height to
    # image height carries over) - used here purely to clear that text
    # row, not to match it exactly.
    version_text_row_estimate = max(12, pixmap.height() // 22)
    bar_y = pixmap.height() - margin - version_text_row_estimate - 8 - bar_height
    progress_bar = QProgressBar(splash)
    progress_bar.setRange(0, 100)
    progress_bar.setValue(0)
    progress_bar.setTextVisible(False)
    progress_bar.setGeometry(margin, bar_y, pixmap.width() - margin * 2, bar_height)
    progress_bar.setStyleSheet(
        "QProgressBar { background-color: rgba(255, 255, 255, 40); border: none; border-radius: 2px; }"
        "QProgressBar::chunk { background-color: #2f6fed; border-radius: 2px; }"
    )
    progress_bar.show()
    splash.progress_bar = progress_bar

    splash.show()
    return splash


ICON_COLOR = "#e8eefc"
"""Toolbar icon color - matches DARK_QSS's primary text color, so
qtawesome-rendered Font Awesome icons (`qta.icon(name, color=ICON_COLOR)`)
read consistently with the surrounding button/label text."""

ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
ZOOM_STEP = 1.10
"""Per-wheel-notch zoom multiplier and bounds - see
`video_overlay.VideoPane.wheelEvent`."""

HANDLE_RADIUS_PX = 8
"""On-screen radius, in pixels, of each drawn point handle ring."""

MAX_POINTS_PER_PANE = 20
"""Hard cap on how many measurement points one pane can hold."""

DEFAULT_VIDEO_ASPECT_RATIO = 1280 / 800
"""Width/height ratio of the stereo rigs' actual footage - used to size
the main window's default height so the video panes end up close to
this ratio instead of leaving letterbox bars, whatever screen the app
opens on. Individual videos can still be a different ratio - the panes
just letterbox/pillarbox as usual in that case."""

WINDOW_SCREEN_FRACTION = 0.9
"""Fraction of the target screen's available width/height the main
window sizes itself to on startup - see `_size_window_to_screen`."""

SPEED_TABLE = {
    "0.25x": 0.25,
    "0.5x": 0.5,
    "1x": 1.0,
    "2x": 2.0,
    "4x": 4.0,
}
"""Playback speed label -> speed multiplier. "1x" means the video's own
real fps, not a fixed assumption - `_compute_playback_timing` turns a
multiplier and the loaded video's actual fps into a concrete
`(frame_step, tick_delay_ms)` pair. Previously this table hardcoded
`(step, delay_ms)` pairs derived from an assumed ~25fps, so footage at
any other fps played at the wrong rate even at "1x" - see FINDINGS.md
#10 and ROADMAP.md Phase 7's deferred playback-speed item."""

MAX_SEQUENTIAL_SKIP_FRAMES = 10
"""Largest forward frame-index gap `_read_frame_at` will close by
decoding-and-discarding rather than seeking. Comfortably covers every
`SPEED_TABLE` step (currently up to 4, for "4x") with headroom for a
faster speed being added later, while staying well under the point
where decode-and-discard's linear per-frame cost would outgrow a
seek's flat cost."""

DEFAULT_PLAYBACK_FPS = 25.0
"""Fallback fps for `_compute_playback_timing` if a loaded video
somehow reports a nonsensical fps (e.g. 0, from corrupt metadata) -
`_open_video_capture` doesn't currently validate fps the way it
validates width/height/frame_count are positive."""


def _compute_playback_timing(fps, speed_label):
    """Turn a video's real fps and a `SPEED_TABLE` speed label into a
    concrete `(frame_step, tick_delay_ms)` pair for `_playback_tick`.

    "1x" ticks once every `1000/fps` ms (the video's actual native
    rate), advancing one frame per tick. Slower-than-1x speeds keep
    that one-frame step but stretch the tick interval; faster-than-1x
    speeds keep the native tick interval but skip more frames per
    tick - same tradeoff the old hardcoded `SPEED_TABLE` used, just
    parameterized by the real fps instead of assuming ~25fps.

    Args:
        fps (float): The loaded video's actual frames per second.
            Falls back to `DEFAULT_PLAYBACK_FPS` if not positive.
        speed_label (str): One of `SPEED_TABLE`'s keys (e.g. "1x").
            Falls back to a 1.0 multiplier if not a recognized label.

    Returns:
        tuple[int, int]: `(frame_step, tick_delay_ms)`, both at least 1.
    """
    if not fps or fps <= 0:
        fps = DEFAULT_PLAYBACK_FPS

    multiplier = SPEED_TABLE.get(speed_label, 1.0)
    native_delay_ms = 1000.0 / fps

    if multiplier >= 1.0:
        step = max(1, round(multiplier))
        delay_ms = round(native_delay_ms)
    else:
        step = 1
        delay_ms = round(native_delay_ms / multiplier)

    return step, max(1, delay_ms)

DARK_QSS = """
QMainWindow, QWidget { background-color: #0a0f1a; color: #e8eefc; font-family: "Segoe UI"; font-weight: bold; font-size: 11pt; }
QMenuBar { background-color: #121a2b; color: #e8eefc; padding: 4px 6px; }
QMenuBar::item { padding: 6px 14px; border-radius: 4px; }
QMenuBar::item:selected { background-color: #2f6fed; }
QMenu { background-color: #121a2b; color: #e8eefc; border: 1px solid #263351; padding: 6px; }
QMenu::item { padding: 8px 28px 8px 16px; border-radius: 4px; }
QMenu::item:selected { background-color: #2f6fed; }
QMenu::separator { height: 1px; background: #263351; margin: 6px 10px; }
QToolBar { background-color: #121a2b; border: none; spacing: 10px; padding: 8px; }
QToolButton#qt_toolbar_ext_button { background-color: #1a2438; border: 1px solid #263351; border-radius: 5px; }
QToolButton#qt_toolbar_ext_button:hover { background-color: #22304d; }
QPushButton { background-color: #1a2438; color: #e8eefc; border: 1px solid #263351; border-radius: 5px; padding: 8px 16px; }
QPushButton:hover { background-color: #22304d; }
QPushButton:checked { background-color: #2f6fed; }
QComboBox, QCheckBox, QSpinBox, QLineEdit { color: #e8eefc; background-color: transparent; }
QComboBox, QSpinBox, QLineEdit { border: 1px solid #263351; border-radius: 5px; }
QComboBox, QSpinBox { padding: 6px 10px; }
QLineEdit { padding: 4px 6px; }
QCheckBox { padding: 4px 8px; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSlider::groove:horizontal { background: #1a2438; height: 4px; border-radius: 2px; margin: 0 4px; }
QSlider::handle:horizontal { background: #2f6fed; width: 14px; margin: -6px 0; border-radius: 7px; }
QStatusBar { color: #8ea2c6; padding: 4px 10px; }
QLabel { background-color: transparent; padding: 2px 4px; }
QLabel#rectifiedIndicator[state="rectified"] { color: #2fbf71; }
QLabel#rectifiedIndicator[state="not_rectified"] { color: #ef5350; }
QHeaderView::section { background-color: #121a2b; padding: 6px 10px; border: none; border-right: 1px solid #263351; }
"""


class Var:
    """Minimal `.get()`/`.set()` box, used only where a ported pure-
    logic module (`stereo_matching.py`) expects Tk-Variable-style
    access (`app.view_rectified.get()`). Everywhere else in this Qt
    app, plain attributes are used directly - this shim exists purely
    for that one cross-module contract, not as a general pattern.
    """

    def __init__(self, value):
        """Store the initial value.

        Args:
            value: The initial value `.get()` should return.

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
        """Store a new value.

        Args:
            value: The new value to store.

        Returns:
            None
        """
        self._value = value


class SizeamaticProApp(QMainWindow):
    """Owns the main window, video state, and top-level event wiring.

    One instance is the whole app - built and shown by `main()`.
    """

    def __init__(self):
        """Initialize all app state and build the window.

        Returns:
            None
        """
        super().__init__()
        self.setWindowTitle("Sizeamatic Pro")
        # Fallback default size, used if `main()` can't detect a screen to
        # size against (see `_size_window_to_screen`) - height still tuned
        # to DEFAULT_VIDEO_ASPECT_RATIO so the panes don't letterbox by
        # default even in that fallback case.
        self.resize(1400, round(1400 / 2 / DEFAULT_VIDEO_ASPECT_RATIO) + 170)

        # assets/icon.ico is regenerated from assets/default-icon.png on
        # every build (create_app_icon.py); stays best-effort so a missing/
        # invalid icon file doesn't crash startup.
        icon_path = resource_path("assets/icon.ico")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # ---- Video capture/metadata state ----
        self.capL = None
        self.capR = None
        self.metaL = None
        self.metaR = None
        self.left_video_path = None
        self.right_video_path = None

        # ---- Frame index/range state ----
        self.left_frame_index = 0
        self.right_frame_index = 0
        self.left_frame_max = 0
        self.right_frame_max = 0

        # ---- Cached currently-displayed frames (post-rectification, if
        # rectified view is on) - read by the scanline matcher. ----
        self.current_frameL = None
        self.current_frameR = None

        # ---- View/sync state ----
        self.lock_lr = True
        self.lock_offset_frames = 0
        """`right_index - left_index` while locked."""

        self.view_rectified = Var(False)
        """Wrapped in `Var` - `stereo_matching.py` expects
        `app.view_rectified.get()`."""

        self.cal = None
        """Loaded calibration dict (`calibration_io.load_calibration_bundle`'s
        result), or None."""

        self.calibration_folder = None
        """The folder `self.cal` was loaded from, or None - separate
        from `self.cal` itself since project files save/restore the
        folder path, not the loaded arrays."""

        # ---- Measurement point state ----
        self.ptsL = []
        self.ptsR = []

        self.click_sigma_px = 3.0
        """Assumed user click-placement uncertainty, in image pixels. An
        explicit modeling assumption (not a measured value) fed into
        `stereo_matching.py`'s perturbation-based uncertainty estimates
        (`estimate_point_sigma_mm`, `estimate_segment_sigma_len_mm`) to
        translate pixel-level click imprecision into millimeter-level
        depth/length uncertainty estimates."""

        # ---- Project save/load state ----
        self.current_project_name = None
        self.last_recorded_snapshot = None

        self.real_time_anchor_frame = None
        """The left-timeline frame index the user was on when they last
        set the real-world time anchor (`on_real_time_entered`), or None
        if no anchor has been set."""

        self.real_time_anchor_iso = None
        """The persisted (project-file-friendly) ISO-8601 form of the
        real-time anchor, or None. `self.real_time_anchor_dt` below is
        the live `datetime.datetime` parsed from this - kept as a
        separate attribute rather than parsing on every read, and kept
        as an ISO string here (rather than storing the `datetime`
        directly) since that's the format `project_io.py` round-trips
        through JSON."""

        self.real_time_anchor_dt = None
        """The `datetime.datetime` parsed from `self.real_time_anchor_iso`,
        read off whatever real-world clock is burned into the video at
        `self.real_time_anchor_frame`. Together with the left video's
        fps, this is what `_format_actual_time` uses to project the
        real-world time at any other frame."""

        # ---- Shared pan/zoom/interaction constants (read by VideoPane) ----
        self.zoom_min = ZOOM_MIN
        self.zoom_max = ZOOM_MAX
        self.zoom_step = ZOOM_STEP
        self.handle_radius_px = HANDLE_RADIUS_PX
        self.max_points_per_pane = MAX_POINTS_PER_PANE

        # ---- Playback state ----
        self.is_playing = False
        self.playback_timer = QTimer(self)
        self.playback_timer.setSingleShot(True)
        # Qt's default QTimer uses a "coarse" timer type - intentionally
        # imprecise (Qt reserves the right to fire up to ~5% or so late)
        # so the OS can batch wakeups for power efficiency. That slack
        # was the dominant cost in real playback throughput: profiling
        # showed decode+render alone takes under 1ms/frame, yet the real
        # self-rescheduling timer loop averaged ~51ms between ticks
        # against a 40ms request. PreciseTimer asks the OS for its most
        # accurate timer facility instead.
        self.playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.playback_timer.timeout.connect(self._playback_tick)

        self._suppress_slider_callbacks = False

        self.measurement_window = measurement_window.MeasurementWindow(self)
        """Owns the measurement results dialog and its widgets. See
        `measurement_window.MeasurementWindow`."""

        self.cal_summary_window = calibration_summary.CalibrationSummaryWindow(self)
        """Owns the calibration summary dialog and its widgets. See
        `calibration_summary.CalibrationSummaryWindow`."""

        self.anaglyph_preview = anaglyph_preview.AnaglyphPreview(self)
        """Owns the anaglyph preview's OpenCV window and playback state.
        See `anaglyph_preview.AnaglyphPreview`."""
        self.anaglyph_preview.window_name = "Sizeamatic Pro - Anaglyph 3D"

        self.perform_calibration_window = perform_calibration.PerformCalibrationWindow(self)
        """Owns the Perform Calibration dialog and its widgets - capturing
        calibration frame pairs from the currently loaded left/right
        video and running the calibration computation on them. See
        `perform_calibration.PerformCalibrationWindow`."""

        self.generate_calibration_target_window = generate_calibration_target.GenerateCalibrationTargetWindow(self)
        """Owns the Generate Calibration Target dialog and its widgets -
        printing a checkerboard/ChArUco calibration board. See
        `generate_calibration_target.GenerateCalibrationTargetWindow`."""

        self.tutorial_window = tutorial_window.TutorialController(self)
        """Owns the in-app Tutorial mode's overlay/current-step bubble/
        checklist and the active `tutorial_engine.Tutorial` run, if any
        (ROADMAP.md Phase 15). See `tutorial_window.TutorialController`.
        Every handler below that completes a tutorial step calls
        `self.tutorial_window.notify_action(...)` unconditionally - safe
        to call even when no tutorial is running."""

        self._build_menu()
        self._build_toolbar()
        self._build_central_widget()
        self._build_statusbar()
        self._build_lock_arrow_shortcuts()

        self._update_slider_ranges()
        self._refresh_window_title()

    def closeEvent(self, event):
        """Stop playback/preview loops and release video captures before closing.

        Direct port of the original's `on_app_close`. Also explicitly
        closes the four sub-window dialogs if still open - they're
        deliberately built with no Qt parent (see `qt_helpers.ClosableDialog`),
        so Qt's default quit-on-last-window-closed can't be relied on to
        clean them up as a side effect of the main window closing.

        Args:
            event (QCloseEvent): The close event.

        Returns:
            None
        """
        self.is_playing = False
        self.playback_timer.stop()

        if self.anaglyph_preview.active:
            self.anaglyph_preview.stop()

        if self.capL:
            self.capL.release()
        if self.capR:
            self.capR.release()

        for sub_window in (
            self.measurement_window,
            self.cal_summary_window,
            self.perform_calibration_window,
            self.generate_calibration_target_window,
        ):
            if sub_window.win is not None:
                sub_window.win.close()

        # The tutorial's bubble/checklist/overlay(s) are also independent
        # top-level windows, same reasoning as the loop above - `stop()`
        # is a safe no-op if no tutorial was ever started.
        self.tutorial_window.stop()

        super().closeEvent(event)

    def resizeEvent(self, event):
        """Give the toolbar's overflow button an icon once it exists.

        When the window gets too narrow for the toolbar's full content,
        Qt creates a "..." overflow button on demand (internally named
        "qt_toolbar_ext_button") that opens a menu of whatever no longer
        fits - QSS styles its background/border/hover fine (see
        DARK_QSS), but Qt gives it its own plain black "..." icon
        *before* this ever runs, so a guard like "only set an icon if
        it doesn't have one yet" never actually fires - that icon is
        never null to begin with. Set unconditionally instead so this
        app's `qtawesome` icon set always wins. `findChild` is cheap
        and safe to call on every resize - the button doesn't exist at
        all until the window is actually narrow enough to need one.

        Args:
            event (QResizeEvent): The resize event.

        Returns:
            None
        """
        super().resizeEvent(event)

        ext_button = self.findChild(QToolButton, "qt_toolbar_ext_button")
        if ext_button is not None:
            ext_button.setIcon(qta.icon("fa5s.chevron-down", color=ICON_COLOR))
            ext_button.setText("")
            ext_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

    def _build_lock_arrow_shortcuts(self):
        """Make locked arrow-key stepping work regardless of what has focus.

        `video_overlay.VideoPane.keyPressEvent` already steps both
        timelines together on Left/Right when Lock L and R is checked and
        the pane itself has keyboard focus - but a plain `keyPressEvent`
        override on this window turned out NOT to be a usable fallback
        for "focus is somewhere else entirely": toolbar buttons keep
        keyboard focus by default (confirmed live - a fresh window
        already has one focused) and Qt's own `QToolBar` gives its
        buttons built-in Left/Right focus-navigation, which silently
        swallows the key before it would ever bubble up here. A
        `QShortcut` with `WindowShortcut` context sidesteps that - Qt's
        shortcut map is consulted ahead of normal widget key delivery,
        so it fires even though a toolbar button has focus (confirmed
        live). That same priority is exactly why it must stay disabled
        whenever Lock is off: enabled, it would intercept Left/Right
        before a focused `VideoPane` ever saw them, breaking the
        pane-scoped "nudge one side" behavior that's the whole point of
        the unlocked case. `on_toggle_lock` flips `setEnabled` on both
        to match, so this only ever fires while Lock is actually checked.

        Returns:
            None
        """
        self.shortcut_step_forward_locked = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_step_forward_locked.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_step_forward_locked.activated.connect(self.on_step_forward)

        self.shortcut_step_back_locked = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_step_back_locked.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_step_back_locked.activated.connect(self.on_step_back)

        self._update_lock_arrow_shortcuts()

    def _update_lock_arrow_shortcuts(self):
        """Enable the locked-arrow-key shortcuts only while Lock L and R is on.

        See `_build_lock_arrow_shortcuts` for why these must stay
        disabled while unlocked - a `WindowShortcut` takes priority over
        a focused `VideoPane`'s own `keyPressEvent`, so leaving them
        enabled unlocked would break per-pane nudging.

        Returns:
            None
        """
        self.shortcut_step_forward_locked.setEnabled(self.lock_lr)
        self.shortcut_step_back_locked.setEnabled(self.lock_lr)

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------

    def _build_menu(self):
        """Build the File/View/Calibration/Help menus.

        Returns:
            None
        """
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Load Left Video…", self.on_load_left_video)
        file_menu.addAction("Load Right Video…", self.on_load_right_video)
        file_menu.addSeparator()
        file_menu.addAction("Save Project…", self.on_save_project)
        file_menu.addAction("Open Project…", self.on_open_project)

        # Rebuilt fresh every time it's about to be shown (aboutToShow is
        # Qt's equivalent of tk.Menu's postcommand), not once at startup -
        # so a project moved/renamed/deleted since the last time this menu
        # opened just quietly drops out of the list instead of showing an
        # entry that would only error if clicked.
        self.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self.recent_projects_menu.aboutToShow.connect(self._refresh_recent_projects_menu)

        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        view_menu = menubar.addMenu("View")
        self.action_show_rectified = QAction("Show Rectified", self, checkable=True)
        self.action_show_rectified.toggled.connect(self.on_toggle_view_rectified)
        view_menu.addAction(self.action_show_rectified)

        view_menu.addAction("Anaglyph 3D Preview…", self.on_toggle_anaglyph_preview)
        view_menu.addAction("Reset Pan/Zoom", self.on_reset_pan_zoom)
        view_menu.addSeparator()
        view_menu.addAction("Calibration Summary…", self.on_show_calibration_summary)

        calibration_menu = menubar.addMenu("Calibration")
        calibration_menu.addAction("Load Calibration…", self.on_load_calibration_folder)
        calibration_menu.addSeparator()
        calibration_menu.addAction("Perform Calibration…", self.on_perform_calibration)
        calibration_menu.addAction("Generate Calibration Target…", self.on_generate_calibration_target)

        # New in ROADMAP.md Phase 15 - reachable any time, not just on
        # first launch, per that phase's Step 0 decision.
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Start Tutorial…", self.on_start_tutorial)
        help_menu.addAction("Open Measurement Method Whitepaper…", self.on_open_whitepaper)

    # -------------------------------------------------------------------------
    # Toolbar
    # -------------------------------------------------------------------------

    def _build_toolbar(self):
        """Build the transport/speed/lock/offset/rectified-indicator toolbar.

        Returns:
            None
        """
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        icon_size = QSize(18, 18)

        self.btn_to_start = QPushButton()
        self.btn_to_start.setIcon(qta.icon("fa5s.fast-backward", color=ICON_COLOR))
        self.btn_step_back = QPushButton()
        self.btn_step_back.setIcon(qta.icon("fa5s.step-backward", color=ICON_COLOR))
        self.btn_play_pause = QPushButton()
        self.btn_step_forward = QPushButton()
        self.btn_step_forward.setIcon(qta.icon("fa5s.step-forward", color=ICON_COLOR))
        self.btn_to_end = QPushButton()
        self.btn_to_end.setIcon(qta.icon("fa5s.fast-forward", color=ICON_COLOR))
        for btn in (self.btn_to_start, self.btn_step_back, self.btn_play_pause, self.btn_step_forward, self.btn_to_end):
            btn.setFixedWidth(36)
            btn.setIconSize(icon_size)
            toolbar.addWidget(btn)

        self._update_play_pause_icon()

        self.btn_to_start.clicked.connect(self.on_to_start)
        self.btn_step_back.clicked.connect(self.on_step_back)
        self.btn_play_pause.clicked.connect(self.on_play_pause)
        self.btn_step_forward.clicked.connect(self.on_step_forward)
        self.btn_to_end.clicked.connect(self.on_to_end)

        toolbar.addWidget(QLabel("  Speed  "))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(list(SPEED_TABLE.keys()))
        self.speed_combo.setCurrentText("1x")
        toolbar.addWidget(self.speed_combo)

        self.btn_clear_points = QPushButton("  Clear Points")
        self.btn_clear_points.setIcon(qta.icon("fa5s.trash", color=ICON_COLOR))
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        toolbar.addWidget(self.btn_clear_points)

        # Reset Pan/Zoom and Record live here too now (issue #17) - in
        # addition to, not replacing, the existing View menu item and the
        # Measurement window's own Record button respectively; both call
        # the identical handler either way, so behavior never diverges.
        self.btn_reset_pan_zoom = QPushButton("  Reset Pan/Zoom")
        self.btn_reset_pan_zoom.setIcon(qta.icon("fa5s.compress", color=ICON_COLOR))
        self.btn_reset_pan_zoom.clicked.connect(self.on_reset_pan_zoom)
        toolbar.addWidget(self.btn_reset_pan_zoom)

        self.btn_record = QPushButton("  Record")
        self.btn_record.setIcon(qta.icon("fa5s.circle", color=ICON_COLOR))
        self.btn_record.setStyleSheet(
            f"QPushButton {{ background-color: {measurement_window.RECORD_BUTTON_COLOR}; }} "
            f"QPushButton:hover {{ background-color: #9c3b3b; }}"
        )
        self.btn_record.clicked.connect(lambda: self.measurement_window.record_current_measurement())
        toolbar.addWidget(self.btn_record)

        self.rectified_indicator = QLabel("  NOT RECTIFIED  ")
        self.rectified_indicator.setObjectName("rectifiedIndicator")
        self.rectified_indicator.setProperty("state", "not_rectified")
        toolbar.addWidget(self.rectified_indicator)

    def _build_real_time_sync_group(self, layout):
        """Build the real-world time anchor entry group.

        Six separate plain text boxes (year/month/day/hour/minute/
        second), not one free-text field to parse and not spinners -
        the project owner specifically didn't want either. Starts
        empty, not pre-filled with "now": the boxes exist to be typed
        into, matching whatever's burned into the video, not edited
        from a default that has nothing to do with the footage.
        Nothing here is validated/applied until "Set Time Sync" is
        pressed (`on_real_time_entered`).

        Args:
            layout (QHBoxLayout): The row layout to add this group to
                (the sync-controls row between the scrub bars and the
                Frame/Video Time/Actual Time readout, per issue #17 -
                previously the toolbar itself, before it got moved off
                to make room).

        Returns:
            None
        """
        box_specs = [
            ("real_time_year_edit", 4, "YYYY", "enter_time_year"),
            ("real_time_month_edit", 2, "MM", "enter_time_month"),
            ("real_time_day_edit", 2, "DD", "enter_time_day"),
            ("real_time_hour_edit", 2, "HH", "enter_time_hour"),
            ("real_time_minute_edit", 2, "MM", "enter_time_minute"),
            ("real_time_second_edit", 2, "SS", "enter_time_second"),
        ]
        # Separator text drawn between consecutive boxes (index i sits
        # between box i and box i+1) - one shorter than the number of boxes.
        separators = ["-", "-", "  ", ":", ":"]

        self.real_time_entries = []
        """The six real-time-anchor `QLineEdit`s (year/month/day/hour/
        minute/second, in that order) - kept so each box's auto-advance
        handler can focus the *next* one, and so tests can drive them
        uniformly without naming each one."""

        # A single tight container for the boxes + separators, rather than
        # adding each one straight to the row layout - its own `spacing`
        # applies between every item added to it, which otherwise stacks
        # with the "-"/":" separator labels themselves and spreads
        # "YYYY-MM-DD" out into visibly gapped characters instead of one
        # tight date/time group.
        box_group = QWidget()
        box_group.setStyleSheet("background-color: transparent;")
        box_layout = QHBoxLayout(box_group)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(2)

        for i, (attr_name, max_len, placeholder, _action_name) in enumerate(box_specs):
            entry = QLineEdit()
            entry.setPlaceholderText(placeholder)
            # Wide enough for the placeholder text plus this QLineEdit's
            # QSS padding/border (~14px of non-text chrome) with a little
            # slack - too tight and Qt silently elides the placeholder to
            # "..." for wider letter combinations (e.g. "MM"/"DD"/"HH")
            # while narrower ones (e.g. "SS") happen to still fit.
            entry.setFixedWidth(70 if max_len == 4 else 48)
            entry.setMaxLength(max_len)
            setattr(self, attr_name, entry)
            self.real_time_entries.append(entry)
            box_layout.addWidget(entry)

            if i < len(separators):
                box_layout.addWidget(QLabel(separators[i]))

        layout.addWidget(box_group)

        # Auto-advance to the next box once this one looks full - purely a
        # focus convenience, not validation (nothing is checked/applied here).
        # Also reports the matching tutorial completion action once a box
        # is full, box-by-box, for ROADMAP.md Phase 15's per-field steps.
        for i, entry in enumerate(self.real_time_entries):
            max_len = box_specs[i][1]
            action_name = box_specs[i][3]
            next_entry = self.real_time_entries[i + 1] if i + 1 < len(self.real_time_entries) else None
            entry.textChanged.connect(
                lambda _text, e=entry, n=next_entry, m=max_len, a=action_name: self._advance_real_time_focus(
                    e, n, m, a
                )
            )

        self.btn_set_time_sync = QPushButton("Set Time Sync")
        self.btn_set_time_sync.clicked.connect(self.on_real_time_entered)
        layout.addWidget(self.btn_set_time_sync)

        # Synced indicator: a checkmark next to the button, shown by
        # on_real_time_entered once an anchor is actually set; empty
        # until then.
        self.time_sync_indicator = QLabel("")
        self.time_sync_indicator.setStyleSheet("color: #2fbf71;")
        layout.addWidget(self.time_sync_indicator)

    def _advance_real_time_focus(self, entry, next_entry, max_len, action_name):
        """Move focus to the next real-time-anchor box once this one looks
        full, and report the matching tutorial completion action.

        Purely a typing convenience as far as the app's own state goes -
        does not validate or apply anything; that only happens when
        "Set Time Sync" is pressed (`on_real_time_entered`). The tutorial
        notification is real completion detection, though (ROADMAP.md
        Phase 15) - each of the six boxes is its own step.

        Args:
            entry (QLineEdit): The box that was just typed into.
            next_entry (QLineEdit | None): The box to focus next, or
                None if this is the last one (seconds).
            max_len (int): How many characters this box is expected to
                hold (e.g. 4 for year, 2 for the rest) before advancing.
            action_name (str): The tutorial completion action this box
                reports once full (e.g. `"enter_time_year"`).

        Returns:
            None
        """
        if len(entry.text()) < max_len:
            return

        self.tutorial_window.notify_action(action_name)

        if next_entry is not None:
            next_entry.setFocus()
            next_entry.selectAll()

    # -------------------------------------------------------------------------
    # Central widget: dual video panes + sliders
    # -------------------------------------------------------------------------

    def _build_central_widget(self):
        """Build the dual video panes and their sliders/frame labels.

        Returns:
            None
        """
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        self.pane_left, self.left_slider, self.left_frame_label = self._build_pane_column(splitter, "L")
        self.pane_right, self.right_slider, self.right_frame_label = self._build_pane_column(splitter, "R")

        self._build_sync_controls_row(root_layout)
        self._build_time_readout_row(root_layout)

    def _build_sync_controls_row(self, root_layout):
        """Build the Lock/Offset/real-time-sync controls row.

        Moved here from the top toolbar (issue #17) - that toolbar was
        crowded enough to silently overflow behind an unlabeled ">>"
        arrow on completely ordinary window widths, hiding this entire
        group. Sits between the video panes and the Frame/Video Time/
        Actual Time readout, spanning the full window width like that
        readout already did, since these are shared/cross-pane controls
        rather than per-pane ones.

        Args:
            root_layout (QVBoxLayout): The central widget's root layout.

        Returns:
            None
        """
        row = QHBoxLayout()

        self.lock_checkbox = QCheckBox("  Lock L and R")
        self.lock_checkbox.setChecked(True)
        self.lock_checkbox.toggled.connect(self.on_toggle_lock)
        row.addWidget(self.lock_checkbox)

        row.addWidget(QLabel("  Offset:"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-100000, 100000)
        self.offset_spin.valueChanged.connect(self.on_offset_changed)
        row.addWidget(self.offset_spin)

        self._build_real_time_sync_group(row)
        row.addStretch(1)

        root_layout.addLayout(row)

    def _build_time_readout_row(self, root_layout):
        """Build the styled Frame/Video Time/Actual Time readout row.

        Three separate small caption+value "chips" instead of one long
        plain-text line (issue #17) - shared across both panes (one
        anchor, not duplicated per pane), by the scrub bars rather than
        up in the toolbar.

        Args:
            root_layout (QVBoxLayout): The central widget's root layout.

        Returns:
            None
        """
        row = QHBoxLayout()
        row.addStretch(1)
        self.frame_readout_value = self._build_readout_chip(row, "FRAME")
        self.video_time_readout_value = self._build_readout_chip(row, "VIDEO TIME")
        self.actual_time_readout_value = self._build_readout_chip(row, "ACTUAL TIME")
        row.addStretch(1)
        root_layout.addLayout(row)

    def _build_readout_chip(self, row, caption):
        """Build one small caption-over-value "chip" widget.

        Args:
            row (QHBoxLayout): The readout row to add this chip to.
            caption (str): The chip's caption (e.g. "FRAME").

        Returns:
            QLabel: The chip's value label - callers set its text later
            as the readout updates.
        """
        chip = QFrame()
        chip.setStyleSheet(
            "QFrame { background-color: #121a2b; border: 1px solid #263351; border-radius: 6px; }"
        )
        chip_layout = QVBoxLayout(chip)
        chip_layout.setContentsMargins(14, 4, 14, 6)
        chip_layout.setSpacing(0)

        caption_label = QLabel(caption)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_label.setStyleSheet("color: #8ea2c6; font-size: 8pt; font-weight: normal; background: transparent;")
        chip_layout.addWidget(caption_label)

        value_label = QLabel("")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("background: transparent;")
        chip_layout.addWidget(value_label)

        row.addWidget(chip)
        return value_label

    def _build_pane_column(self, parent, which):
        """Build one side's video pane + slider + frame label column.

        Args:
            parent (QWidget): The splitter to add this column to.
            which (str): `"L"` or `"R"`.

        Returns:
            tuple[video_overlay.VideoPane, QSlider, QLabel]: The pane,
            its slider, and its frame-index label.
        """
        column = QWidget()
        parent.addWidget(column)
        layout = QVBoxLayout(column)

        pane = video_overlay.VideoPane(self, which)
        layout.addWidget(pane, stretch=1)

        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        row.addWidget(slider, stretch=1)

        frame_label = QLabel("Frame: 0/0")
        row.addWidget(frame_label)
        layout.addLayout(row)

        if which == "L":
            slider.valueChanged.connect(self.on_left_slider_changed)
        else:
            slider.valueChanged.connect(self.on_right_slider_changed)

        return pane, slider, frame_label

    def _build_statusbar(self):
        """Build the 3-section status bar (left/mid/right labels).

        Direct port of the original's `_build_statusbar` - one QStatusBar
        holding three QLabels instead of Tkinter's 3-column grid frame.
        Left shows file/cal/view state, mid shows transient action
        messages, right shows the live measurement summary.

        Returns:
            None
        """
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        self.status_left = QLabel("")
        self.status_right = QLabel("")

        self.status_mid = QLabel("")
        self.status_mid.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # A plain QLabel's minimumSizeHint equals its full unwrapped text
        # width (it won't shrink/elide on its own) - status_left/
        # status_right's content is bounded by design (`_short_path`'s
        # truncation, small fixed-format numbers) so their natural width
        # is fine left alone, but status_mid carries arbitrary transient
        # messages (e.g. the anaglyph preview's ~65-character keyboard-
        # controls tip) that would otherwise force this whole window
        # wider to fit, growing every time a longer message came along.
        # `Ignored` tells the layout to disregard its size hint for
        # sizing purposes, so a long message clips instead of ever
        # resizing the window - matching the original Tkinter status
        # bar's fixed-character-width label, which had the same "just
        # clip it" behavior. Relies on the stretch factor below to still
        # give it real width to work with, since `Ignored` alone would
        # otherwise let it collapse to zero.
        self.status_mid.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        status_bar.addWidget(self.status_left)
        status_bar.addWidget(self.status_mid, 1)
        status_bar.addWidget(self.status_right)

        self._set_status_mid("Ready")

    def _set_status_mid(self, text):
        """Set the status bar's center (message/warning) text.

        Args:
            text (str): The text to display.

        Returns:
            None
        """
        self.status_mid.setText(text)

    def _set_status_right(self, text):
        """Set the status bar's right (measurement results) text.

        Args:
            text (str): The text to display.

        Returns:
            None
        """
        self.status_right.setText(text)

    def _both_videos_loaded(self):
        """Check whether both left and right video captures and metadata exist.

        Returns:
            bool: True only when both `capL`/`capR` and `metaL`/`metaR`
            are set.
        """
        return self.capL is not None and self.capR is not None and self.metaL is not None and self.metaR is not None

    def _refresh_status_left(self):
        """Refresh the status bar's left section with view/lock state.

        Keeps the visible text short and glanceable (issue #17 - the
        status bar previously spelled out the full L/R video and
        calibration paths, which most users don't need day to day); the
        full file paths are still one hover away, via this same label's
        tooltip.

        Returns:
            None
        """
        l = self.left_video_path if self.left_video_path else "(none)"
        r = self.right_video_path if self.right_video_path else "(none)"
        c = self.calibration_folder if self.calibration_folder else "(none)"
        view = "Rectified" if self.view_rectified.get() else "Raw"
        lock = "Locked" if self.lock_lr else "Unlocked"

        # Only show an offset when lock is enabled and both videos are loaded.
        # This keeps the status line clean when you are still loading files.
        offset_txt = ""
        if self.lock_lr and self._both_videos_loaded():
            offset_txt = f" | Offset: {self.lock_offset_frames:+d}f"

        self.status_left.setText(f"View: {view} | {lock}{offset_txt}")
        self.status_left.setToolTip(
            f"L: {self._short_path(l)} | R: {self._short_path(r)} | Cal: {self._short_path(c)}"
        )

    # -------------------------------------------------------------------------
    # Video loading
    # -------------------------------------------------------------------------

    def _open_video_capture(self, path):
        """Open a video file and read its metadata.

        Direct port of the original's helper of the same name - no
        Tkinter dependency there to begin with.

        Args:
            path (str): Path to the video file to open.

        Returns:
            tuple[cv2.VideoCapture, dict] | tuple[None, None]: The
            opened capture and its metadata dict (`"fps"`/`"width"`/
            `"height"`/`"frame_count"`), or `(None, None)` if the file
            couldn't be opened or reports a nonsensical size/frame
            count.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            return None, None

        return cap, {"fps": fps, "width": width, "height": height, "frame_count": frame_count}

    def _tutorial_fixture_dialog_dir(self, kind):
        """Get the directory a load dialog should default to, if the
        tutorial's generated sample files are relevant right now.

        The tutorial deliberately doesn't load its generated files
        automatically - the user still picks them via these same real
        dialogs - but defaulting to wherever they were generated saves
        hunting through an arbitrary temp directory for them.

        Args:
            kind (str): `"left_video"`, `"right_video"`, or
                `"calibration"`.

        Returns:
            str: The directory to open the dialog in, or `""` (the
            dialog's own default) if no tutorial has generated fixtures
            yet this session.
        """
        fixture_paths = self.tutorial_window.fixture_paths
        if fixture_paths is None:
            return ""
        if kind == "left_video":
            return os.path.dirname(fixture_paths["left_video_path"])
        if kind == "right_video":
            return os.path.dirname(fixture_paths["right_video_path"])
        return fixture_paths["calibration_folder"]

    def on_load_left_video(self):
        """Prompt for and load the left video.

        Just handles the file dialog; the actual loading logic lives
        in `_load_left_video_from_path` so `_open_project_from_path`
        can reuse it with a path read from a project file instead.

        Returns:
            None
        """
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Left Video", self._tutorial_fixture_dialog_dir("left_video"), "MP4 Video (*.mp4);;All Files (*)"
        )
        if not path:
            return
        self._load_left_video_from_path(path)

    def _load_left_video_from_path(self, path):
        """Load the left video from an already-known path, updating UI state.

        Dialog-free so it can be driven by either `on_load_left_video`
        (file picker) or `_open_project_from_path` (a path stored in a
        project file).

        Args:
            path (str): Path to the left video file to open.

        Returns:
            bool: True if the video opened successfully, False
            otherwise (with an error dialog already shown).
        """
        if self.capL:
            self.capL.release()
            self.capL = None
            self.metaL = None

        cap, meta = self._open_video_capture(path)
        if cap is None:
            QMessageBox.critical(self, "Load Left Video", "Failed to open the selected video file.")
            return False

        self.left_video_path = path
        self.capL = cap
        self.metaL = meta
        self.left_frame_index = 0
        self.pane_left.reset_view()

        self._update_slider_ranges()
        self.render_current_frames()
        self._set_status_mid(
            f"Loaded left video ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )
        self.tutorial_window.notify_action("load_left_video")
        return True

    def on_load_right_video(self):
        """Prompt for and load the right video.

        Just handles the file dialog; the actual loading logic lives
        in `_load_right_video_from_path` so `_open_project_from_path`
        can reuse it with a path read from a project file instead.

        Returns:
            None
        """
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Right Video", self._tutorial_fixture_dialog_dir("right_video"), "MP4 Video (*.mp4);;All Files (*)"
        )
        if not path:
            return
        self._load_right_video_from_path(path)

    def _load_right_video_from_path(self, path):
        """Load the right video from an already-known path, updating UI state.

        Mirrors `_load_left_video_from_path` for the right pane.

        Args:
            path (str): Path to the right video file to open.

        Returns:
            bool: True if the video opened successfully, False
            otherwise (with an error dialog already shown).
        """
        if self.capR:
            self.capR.release()
            self.capR = None
            self.metaR = None

        cap, meta = self._open_video_capture(path)
        if cap is None:
            QMessageBox.critical(self, "Load Right Video", "Failed to open the selected video file.")
            return False

        self.right_video_path = path
        self.capR = cap
        self.metaR = meta
        self.right_frame_index = 0
        self.pane_right.reset_view()

        self._update_slider_ranges()
        self.render_current_frames()
        self._set_status_mid(
            f"Loaded right video ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )
        self.tutorial_window.notify_action("load_right_video")
        return True

    def on_load_calibration_folder(self):
        """Prompt for and load a calibration folder.

        Just handles the file dialog; the actual loading logic lives in
        `_load_calibration_from_folder` so `_open_project_from_path` can
        reuse it with a folder path read from a project file instead.

        Note:
            Deliberately uses an *open file* dialog rather than a folder
            picker, even though what's actually wanted is a folder.
            Windows' native folder-picker dialog only shows folder
            names, never the files inside them - so a user comparing
            several candidate folders has no way to see which one
            actually contains the expected NPZ files before picking.
            Asking for "any file inside the calibration folder" instead,
            filtered to `calibration_*.npz`, means the picker's own file
            list does the job the dialog title alone couldn't: the four
            expected files are right there to look at. The containing
            folder is then just the dirname of whichever one gets
            picked.

        Returns:
            None
        """
        sample_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select any file inside the calibration folder — expects "
            "calibration_intrinsics.npz, calibration_extrinsics.npz, "
            "calibration_rectification.npz, calibration_maps.npz",
            self._tutorial_fixture_dialog_dir("calibration"),
            "Calibration NPZ (calibration_*.npz);;All Files (*)",
        )
        if not sample_path:
            return

        folder = os.path.dirname(sample_path)
        self._load_calibration_from_folder(folder)

    def on_show_calibration_summary(self):
        """Open (or focus) the calibration summary window.

        Requires calibration to already be loaded.

        Returns:
            None
        """
        if self.cal is None:
            self._set_status_mid("Load calibration first")
            return

        self.cal_summary_window.ensure_window()
        self.cal_summary_window.update_window()

    def on_perform_calibration(self):
        """Open (or focus) the Perform Calibration window.

        Unlike `on_show_calibration_summary`, this doesn't require an
        existing calibration to already be loaded — capturing frame
        pairs is how a *new* calibration gets built in the first place.

        Returns:
            None
        """
        self.perform_calibration_window.ensure_window()

    def on_generate_calibration_target(self):
        """Open (or focus) the Generate Calibration Target window.

        Doesn't require any video or calibration to be loaded - printing
        a target is a prep step done before capturing anything.

        Returns:
            None
        """
        self.generate_calibration_target_window.ensure_window()

    def on_start_tutorial(self):
        """Start (or restart) the in-app Tutorial mode.

        `TutorialController.start` generates a fresh synthetic sample
        video/calibration set but does not load it - the user still
        loads it themselves through the real File/Calibration menu
        actions (see `on_load_left_video`/`on_load_right_video`/
        `on_load_calibration_folder`, which default their file dialog to
        wherever the tutorial's generated files are once a tutorial is
        active), so nothing about clicking this replaces the app's
        current state by itself. Always begins a completely fresh
        tutorial run - `TutorialController.start` discards any previous
        run's progress rather than resuming it, per ROADMAP.md Phase
        15's Step 0 "no persistence" decision, so clicking this again
        mid-tutorial is a legitimate way to restart from step 0 rather
        than an error.

        Returns:
            None
        """
        self.tutorial_window.start()

    def on_open_whitepaper(self):
        """Open the stereo length measurement method whitepaper PDF.

        Added alongside the Phase 16 Measurement window column tooltips
        (`measurement_window.py`'s `RESULT_TOOLTIPS`) as the actual
        reference those tooltips point to - a `QToolTip` can't contain a
        clickable link (tooltips don't accept mouse events at all), so a
        real Help menu entry that opens the PDF directly is this app's
        only way to make that reference actually usable rather than just
        naming a file path the user has to go find themselves. Uses the
        OS's own PDF viewer (`QDesktopServices.openUrl`) rather than
        rendering it in-app - same reasoning as not building a custom
        video player instead of `cv2.VideoCapture` + Qt widgets.

        Returns:
            None
        """
        path = resource_path("docs/Sizeamatic_Pro_Stereo_Length_Measurement_Method.pdf")
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open Whitepaper", f"Whitepaper file not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _load_calibration_from_folder(self, folder):
        """Load a calibration bundle from an already-known folder path.

        Dialog-free so it can be driven by either
        `on_load_calibration_folder` (folder picker) or
        `_open_project_from_path` (a path stored in a project file).

        Args:
            folder (str): Path to the calibration folder to load.

        Returns:
            bool: True if the calibration loaded successfully, False
            otherwise (with an error dialog already shown).
        """
        cal, error = calibration_io.load_calibration_bundle(folder, self.metaL, self.metaR)
        if error is not None:
            QMessageBox.critical(self, "Load Calibration", error)
            return False

        self.cal = cal
        self.calibration_folder = folder
        self._set_status_mid(f"Loaded calibration from {folder}")
        self._refresh_status_left()
        self.tutorial_window.notify_action("load_calibration")
        return True

    # -------------------------------------------------------------------------
    # Frame decode/render
    # -------------------------------------------------------------------------

    def _read_frame_at(self, cap, index):
        """Decode the frame at a specific index, seeking only if needed.

        A `cap.set(CAP_PROP_POS_FRAMES)` seek is expensive (profiled at
        ~54-105ms/frame, vs. ~3ms/frame for a plain sequential
        `cap.read()`) - the original optimization here avoided it for
        the exact-same-index case (nothing to do) and the pure
        one-frame-forward case (already the default outcome of the
        previous `cap.read()`). That left one real gap: 2x/4x playback
        deliberately requests indices 2/4 frames ahead each tick,
        which - being neither "same index" nor "exactly one frame
        ahead" - fell through to a seek on *every single tick*,
        making faster-than-1x playback slower in wall-clock terms than
        1x despite needing fewer ticks to cover the same duration (this
        is what a "2x plays even slower than 1x" report traced back
        to). Small forward gaps now decode-and-discard the skipped
        frames instead (confirmed via profiling to stay ~10-12x cheaper
        than seeking for the gaps current playback speeds actually
        produce); only a genuinely large or backward jump (e.g. a
        slider drag) still seeks.

        Args:
            cap (cv2.VideoCapture): The capture to read from.
            index (int): The zero-based frame index to read.

        Returns:
            numpy.ndarray | None: The decoded BGR frame, or None if the
            seek/decode failed.
        """
        index = int(index)
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gap = index - current_pos

        if gap != 0:
            if 0 < gap <= MAX_SEQUENTIAL_SKIP_FRAMES:
                for _ in range(gap):
                    cap.read()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame_bgr = cap.read()
        if not ok:
            return None
        return frame_bgr

    def render_current_frames(self):
        """Decode, (if rectified) remap, cache, and display both panes' current frames.

        Returns:
            None
        """
        if self.capL:
            frame_l = self._read_frame_at(self.capL, self.left_frame_index)
            if self.view_rectified.get() and self.cal is not None:
                frame_l = cv2.remap(frame_l, self.cal["mapLx"], self.cal["mapLy"], interpolation=cv2.INTER_LINEAR)
            self.current_frameL = frame_l
            self.pane_left.set_frame(frame_l)

        if self.capR:
            frame_r = self._read_frame_at(self.capR, self.right_frame_index)
            if self.view_rectified.get() and self.cal is not None:
                frame_r = cv2.remap(frame_r, self.cal["mapRx"], self.cal["mapRy"], interpolation=cv2.INTER_LINEAR)
            self.current_frameR = frame_r
            self.pane_right.set_frame(frame_r)

        self._update_frame_labels()

    def redisplay_current_frames(self):
        """Repaint both panes from their already-decoded cached frames.

        Used for interactions that only change the on-screen transform
        (panning) rather than which frame is showing - avoids a costly
        re-seek+re-decode on every mouse-move event. The `VideoPane`s
        already have the current `QImage` cached; this just asks Qt to
        repaint them with their (already-updated) pan/zoom state.

        Returns:
            None
        """
        self.pane_left.update()
        self.pane_right.update()

    def on_points_changed(self):
        """Redraw both panes and recompute measurements after a point
        was placed/moved.

        Returns:
            None
        """
        self.pane_left.update()
        self.pane_right.update()
        self._update_measurement_status_stub()

        # Tutorial completion detection, checked by count so it fires the
        # same way for a click or a drag. Each side is independent (no
        # more auto-mirroring) - all four checks run every call, harmless
        # since mark_action_done is idempotent.
        if len(self.ptsL) >= 1:
            self.tutorial_window.notify_action("place_left_point_1")
        if len(self.ptsR) >= 1:
            self.tutorial_window.notify_action("place_right_point_1")
        if len(self.ptsL) >= 2:
            self.tutorial_window.notify_action("place_left_point_2")
        if len(self.ptsR) >= 2:
            self.tutorial_window.notify_action("place_right_point_2")

    def _current_measurement_context(self):
        """Build the video/frame/timestamp identifying info for the
        current measurement.

        This is what lets a copied-and-pasted measurement row still
        mean something once it's sitting in a spreadsheet with no other
        context. Always reads the *left* timeline/video, since
        measurements are computed in the rectified left camera
        coordinate frame (see `README.md`'s "Measurement notes").

        Returns:
            dict: Keys "video_name" (str), "frame_index" (int),
            "timestamp" (str, elapsed video time since frame 0), and
            "actual_time" (str, the calculated real-world time if a
            real-time anchor is set, or "" if not).
        """
        if self.left_video_path:
            video_name = os.path.basename(self.left_video_path)
        else:
            video_name = "(no video)"

        frame_index = int(self.left_frame_index)
        fps = self.metaL["fps"] if self.metaL else None

        actual_time = self._format_actual_time(frame_index)
        if actual_time == "(not set)":
            # A spreadsheet column should be empty when there's nothing to
            # show, not carry a placeholder string as if it were real data.
            actual_time = ""

        return {
            "video_name": video_name,
            "frame_index": frame_index,
            "timestamp": self._format_timestamp(frame_index, fps),
            "actual_time": actual_time,
        }

    def _update_measurement_status_stub(self):
        """Recompute measurements from current points and refresh the UI.

        Triangulates all currently paired left/right points, builds one
        unified list of result rows (a "Point" row per point, a
        "Segment" row per consecutive pair, and - for 2+ points - one
        "Total" row summing the connected chain's segment lengths),
        updates the status bar's right section with a short summary (or
        the reason measurement isn't available), and refreshes the
        measurement results window.

        Note:
            If a point in the middle of the list fails to triangulate,
            the loop below stops there (via `break`) but the function
            still continues on to report a summary count for whatever
            points triangulated successfully beforehand - the partial-
            failure `err_msg` is passed on to the measurement popup
            window (which does display it), but is not shown in this
            window's own status bar, which instead gets overwritten with
            the "Measured N pts" summary.

        Returns:
            None
        """
        l_count = len(self.ptsL)
        r_count = len(self.ptsR)

        # Quick gating messages stay in the status bar.
        if l_count == 0 and r_count == 0:
            self._set_status_right("")
            self._refresh_measurement_window([], "No points placed")
            return

        if l_count != r_count:
            message = f"Point pair incomplete: L={l_count} R={r_count}"
            self._set_status_right(message)
            self._refresh_measurement_window([], message)
            return

        if not self.view_rectified.get():
            self._set_status_right("Enable rectified view to measure")
            self._refresh_measurement_window([], "Enable rectified view to measure")
            return

        if self.cal is None:
            self._set_status_right("Load calibration to measure")
            self._refresh_measurement_window([], "Load calibration to measure")
            return

        n = min(l_count, r_count)

        # Compute 3D points as many as we can.
        pts3d = []
        err_msg = ""

        for i in range(n):
            P, err = stereo_matching.triangulate_point_pair(self, i)
            if err is not None:
                err_msg = f"Point {i} failed: {err}"
                break
            pts3d.append(P)

        # If we got nothing, show the error and return.
        if len(pts3d) == 0:
            message = err_msg if err_msg else "No valid points"
            self._set_status_right(message)
            self._refresh_measurement_window([], message)
            return

        # Video/frame/timestamp context, shared by every row this call produces.
        ctx = self._current_measurement_context()
        video_col = ctx["video_name"]
        frame_col = str(ctx["frame_index"])
        time_col = ctx["timestamp"]
        actual_time_col = ctx["actual_time"]

        rows = []
        sigma_px = float(self.click_sigma_px)
        # Each point's raw ray_residual_mm (or None), parallel to pts3d -
        # the segment loop below averages a pair of these for "error".
        point_ray_residuals = []

        # Build one "Point" row per clicked point pair.
        for i, (X, Y, Z) in enumerate(pts3d):
            R = (X * X + Y * Y + Z * Z) ** 0.5

            # Assumption-free quality metric (pixel-space reprojection
            # consistency, using the Y-averaged triangulated point).
            erms = stereo_matching.reprojection_rms_px(self, i)
            erms_str = f"{erms:.2f}" if erms is not None else ""

            # Assumption-free quality metric (object-space distance between
            # the two original, un-averaged left/right viewing rays) - a
            # different quantity from erms above, not a duplicate; see
            # stereo_matching.ray_residual_mm's docstring.
            ray_residual = stereo_matching.stereo_ray_residual_mm(self, i)
            point_ray_residuals.append(ray_residual)
            ray_residual_str = f"{ray_residual:.2f}" if ray_residual is not None else ""

            # Assumption-based uncertainty in mm (sample-standard-deviation
            # estimate, shown side by side with the Jacobian estimate below -
            # ROADMAP.md Phase 13, a project owner decision not to replace
            # this pair outright).
            sig = stereo_matching.estimate_point_sigma_mm(self, i, sigma_px)
            if sig is None:
                sZ_str = ""
                sR_str = ""
            else:
                sZ, sR = sig
                sZ_str = f"{sZ:.1f}"
                sR_str = f"{sR:.1f}"

            # Assumption-based uncertainty in mm (Jacobian/covariance-
            # propagation estimate - see stereo_matching's docstring for how
            # this differs from the sample-standard-deviation estimate above).
            sig_jac = stereo_matching.estimate_point_sigma_mm_jacobian(self, i, sigma_px)
            if sig_jac is None:
                sZ_jac_str = ""
                sR_jac_str = ""
            else:
                sZ_jac, sR_jac = sig_jac
                sZ_jac_str = f"{sZ_jac:.1f}"
                sR_jac_str = f"{sR_jac:.1f}"

            # Read the clicked left and right pixels for this point.
            xL, yL = self.ptsL[i]
            xR, yR = self.ptsR[i]

            # Compute disparity, which drives stereo depth.
            disp = xL - xR

            # Compute rectified Y mismatch between left and right clicks.
            dy = yR - yL

            # Trailing 4: range=R, angle=(not calculated yet), length=(N/A
            # for a Point), error=ray_residual.
            rows.append((
                video_col, frame_col, time_col, actual_time_col, "",
                "Point", str(i),
                f"{X:.1f}", f"{Y:.1f}", f"{Z:.1f}", f"{R:.1f}",
                f"{disp:.2f}", f"{dy:.2f}", erms_str, ray_residual_str, sZ_str, sR_str,
                sZ_jac_str, sR_jac_str,
                f"{R:.1f}", "", "", ray_residual_str,
            ))

        # Build one "Segment" row per consecutive point pair (the chain is a
        # single connected polyline: 0-1, 1-2, 2-3, ...), plus a running
        # total length and quadrature-summed sigma across the whole chain.
        total_len_mm = 0.0
        total_var_mm2 = 0.0
        have_total_sigma = True
        total_var_jac_mm2 = 0.0
        have_total_sigma_jac = True
        total_error_var_mm2 = 0.0
        have_total_error = True

        if len(pts3d) >= 2:
            for i in range(1, len(pts3d)):
                X0, Y0, Z0 = pts3d[i - 1]
                X1, Y1, Z1 = pts3d[i]
                dX = X1 - X0
                dY = Y1 - Y0
                dZ = Z1 - Z0
                L = (dX * dX + dY * dY + dZ * dZ) ** 0.5
                total_len_mm += L

                # Segment sigma length estimate (sample-standard-deviation).
                seg_est = stereo_matching.estimate_segment_sigma_len_mm(self, i - 1, i, sigma_px)
                if seg_est is None:
                    sL_str = ""
                    # Can't propagate a total sigma if any segment along the
                    # chain is missing one.
                    have_total_sigma = False
                else:
                    _L0, sL = seg_est
                    sL_str = f"{sL:.1f}"
                    total_var_mm2 += sL * sL

                # Segment sigma length estimate (Jacobian/covariance
                # propagation - ROADMAP.md Phase 13, shown side by side with
                # the sample-standard-deviation estimate above).
                seg_est_jac = stereo_matching.estimate_segment_sigma_len_mm_jacobian(self, i - 1, i, sigma_px)
                if seg_est_jac is None:
                    sL_jac_str = ""
                    have_total_sigma_jac = False
                else:
                    _L0_jac, sL_jac = seg_est_jac
                    sL_jac_str = f"{sL_jac:.1f}"
                    total_var_jac_mm2 += sL_jac * sL_jac

                # This segment's "error" is the average of its two
                # endpoints' ray_residual - there's no separate per-
                # segment ray-residual diagnostic to draw from instead.
                r0 = point_ray_residuals[i - 1]
                r1 = point_ray_residuals[i]
                if r0 is not None and r1 is not None:
                    seg_error = (r0 + r1) / 2.0
                    error_str = f"{seg_error:.2f}"
                    total_error_var_mm2 += seg_error * seg_error
                else:
                    error_str = ""
                    have_total_error = False

                # Trailing 4: range=(N/A for a Segment - it has two ends,
                # not one distance from the camera), angle=(not
                # calculated yet), length=L, error=averaged ray_residual.
                rows.append((
                    video_col, frame_col, time_col, actual_time_col, "",
                    "Segment", f"{i-1}-{i}",
                    f"{dX:.1f}", f"{dY:.1f}", f"{dZ:.1f}", f"{L:.1f}",
                    "", "", "", "", sL_str, "", sL_jac_str, "",
                    "", "", f"{L:.1f}", error_str,
                ))

            # Total: sum of the connected chain's segment lengths. Segment
            # sigmas are each estimated independently, so a sum of
            # independent errors adds in quadrature:
            # sigma_total = sqrt(sum(sigma_i^2)). Done separately for each
            # of the two sigma estimators.
            total_sigma_str = f"{total_var_mm2 ** 0.5:.1f}" if have_total_sigma else ""
            total_sigma_jac_str = f"{total_var_jac_mm2 ** 0.5:.1f}" if have_total_sigma_jac else ""
            # Total's error is the same quadrature-sum treatment as its
            # sigma, applied to each segment's averaged-ray-residual error.
            total_error_str = f"{total_error_var_mm2 ** 0.5:.2f}" if have_total_error else ""
            rows.append((
                video_col, frame_col, time_col, actual_time_col, "",
                "Total", "",
                "", "", "", f"{total_len_mm:.1f}",
                "", "", "", "", total_sigma_str, "", total_sigma_jac_str, "",
                "", "", f"{total_len_mm:.1f}", total_error_str,
            ))

            self._set_status_right(
                f"Measured {len(pts3d)} pts, {len(pts3d) - 1} segs, total {total_len_mm:.1f}mm"
            )
        else:
            self._set_status_right("Measured 1 point")

        # Update popup window (creates it on first valid measurement).
        self.measurement_window.update_window(rows, err_msg)

    def _refresh_measurement_window(self, rows, message):
        """Refresh an already-open measurement window with "nothing to
        show" state, without forcing it open.

        Used by every early-return branch in `_update_measurement_status_stub`
        (points cleared, mismatched counts, view/calibration not ready, no
        valid points) - the window should stop showing stale results from
        a previous measurement the moment that's no longer true, but a
        user who has never placed a measurement yet shouldn't see it pop
        open just because they clicked Clear Points.

        Args:
            rows (list[tuple]): Result rows to show - normally empty here.
            message (str): Error/status line to show in the window.

        Returns:
            None
        """
        if self.measurement_window.win is not None:
            self.measurement_window.update_window(rows, message)

    def _on_measurement_recorded(self):
        """Snapshot enough state to restore this exact measurement later.

        Called by `self.measurement_window.record_current_measurement`
        right after it successfully appends to the Log. Records which
        frame each timeline was on and the exact clicked points at this
        moment, so a saved project file can jump back to "the very last
        place that was recorded" and show those same points again on
        reopen (`_open_project_from_path`).

        Returns:
            None
        """
        self.last_recorded_snapshot = {
            "left_frame_index": int(self.left_frame_index),
            "right_frame_index": int(self.right_frame_index),
            "ptsL": [list(p) for p in self.ptsL],
            "ptsR": [list(p) for p in self.ptsR],
        }

    def on_clear_points(self):
        """Clear all measurement points in both panes.

        Direct port of the original's `on_clear_points` - also cancels
        any active drag/pan/refine state on both panes, since clearing
        the point lists out from under an in-progress drag would leave
        a stale index pointing at nothing.

        Returns:
            None
        """
        self.ptsL.clear()
        self.ptsR.clear()

        for pane in (self.pane_left, self.pane_right):
            pane.drag_active = False
            pane.drag_index = None
            pane.refine_drag_active = False
            pane.refine_drag_index = None
            pane.pan_active = False
            pane.pan_last_pos = None

        self.on_points_changed()
        self._set_status_mid("Cleared all points")
        self.tutorial_window.notify_action("clear_points")

    def _update_frame_labels(self):
        """Refresh the "Frame: i/max" labels, the shared Frame/Video
        Time/Actual Time readout, the status bar's left section, and
        (if an anchor is set) the six real-time entry boxes themselves.

        Returns:
            None
        """
        lmax = max(0, int(self.left_frame_max))
        rmax = max(0, int(self.right_frame_max))
        li = int(self.left_frame_index)
        ri = int(self.right_frame_index)

        self.left_frame_label.setText(f"Frame: {li}/{lmax}")
        self.right_frame_label.setText(f"Frame: {ri}/{rmax}")

        # The shared readout is referenced to the left/master timeline, same
        # as the real-world time anchor itself.
        video_time = self._format_timestamp(li, self.metaL["fps"] if self.metaL else None)
        actual_time = self._format_actual_time(li)
        self.frame_readout_value.setText(f"{li}/{lmax}")
        self.video_time_readout_value.setText(video_time)
        self.actual_time_readout_value.setText(actual_time)

        self._refresh_status_left()

        # Keep the entry boxes live-tracking the current frame's actual time,
        # once an anchor exists (a no-op before that, so typing a fresh
        # anchor isn't clobbered by this running on every frame change).
        self._refresh_real_time_entries(li)

    # -------------------------------------------------------------------------
    # Real-time sync
    # -------------------------------------------------------------------------

    def _format_timestamp(self, frame_index, fps):
        """Format a frame index as an HH:MM:SS:FF timecode.

        The trailing "FF" is the frame number *within* that second
        (0-based, wrapping at the video's own fps) - not a fraction of a
        second - so scrubbing to a specific frame shows exactly which
        frame that is, the same way professional video timecode does.

        Args:
            frame_index (int): Zero-based frame index.
            fps (float | None): The video's frames-per-second, or None/0
                if unknown.

        Returns:
            str: The formatted timecode, or "?" if `fps` isn't a usable
            positive number (e.g. no video loaded yet).
        """
        if not fps or fps <= 0:
            return "?"

        fps_int = max(1, round(float(fps)))
        frame_index = int(frame_index)

        whole_seconds, frame_in_second = divmod(frame_index, fps_int)
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame_in_second:02d}"

    def _format_actual_time(self, frame_index):
        """Calculate and format the real-world time at a given frame.

        Uses the real-world time anchor (`self.real_time_anchor_frame`/
        `self.real_time_anchor_dt`, set by `on_real_time_entered`) plus
        the left video's fps to project forward/backward from that one
        known point, in whole frames rather than fractional seconds -
        see `_format_timestamp`'s docstring for why.

        Args:
            frame_index (int): The left-timeline frame index to
                calculate the real-world time for.

        Returns:
            str: The calculated real-world time as
            "YYYY-MM-DD HH:MM:SS:FF" (FF = frame number within that
            second), or "(not set)" if no anchor has been set yet or the
            left video's fps isn't known.
        """
        if self.real_time_anchor_frame is None or self.real_time_anchor_dt is None:
            return "(not set)"

        fps = self.metaL["fps"] if self.metaL else None
        if not fps or fps <= 0:
            return "(not set)"

        fps_int = max(1, round(float(fps)))
        frame_delta = int(frame_index) - int(self.real_time_anchor_frame)

        # divmod floors toward negative infinity for a positive divisor, so a
        # negative frame_delta still lands on a frame_in_second in [0, fps_int)
        # rather than a negative frame count - e.g. one frame before the
        # anchor is "the second before, frame fps_int - 1", not "-1 frames".
        whole_seconds_delta, frame_in_second = divmod(frame_delta, fps_int)

        actual_dt = self.real_time_anchor_dt + datetime.timedelta(seconds=whole_seconds_delta)
        return actual_dt.strftime("%Y-%m-%d %H:%M:%S") + f":{frame_in_second:02d}"

    def on_real_time_entered(self):
        """Handle the user pressing "Set Time Sync".

        Reads the six year/month/day/hour/minute/second boxes and, if
        they form a valid date/time, anchors it to the left timeline's
        current frame index - from then on, `_format_actual_time` can
        calculate the real-world time at any other frame. Requires the
        left video to already be loaded (its fps is needed for that
        calculation). Nothing is validated or applied by typing alone -
        only this explicit action does that, deliberately, so a
        half-typed date never triggers a premature error dialog.

        Returns:
            None
        """
        try:
            year = int(self.real_time_year_edit.text().strip())
            month = int(self.real_time_month_edit.text().strip())
            day = int(self.real_time_day_edit.text().strip())
            hour = int(self.real_time_hour_edit.text().strip())
            minute = int(self.real_time_minute_edit.text().strip())
            second = int(self.real_time_second_edit.text().strip())
            parsed = datetime.datetime(year, month, day, hour, minute, second)
        except ValueError:
            # A box is empty/non-numeric, or the values parsed as ints fine
            # but don't form a real date (e.g. day 31 in a 30-day month).
            # Either way, there's nothing safe to anchor yet.
            QMessageBox.critical(
                self,
                "Real Time",
                "That's not a valid date/time — check that every box is "
                "filled in and the day of month is valid.",
            )
            return

        if not self.metaL:
            self._set_status_mid("Load the left video before setting a real-time anchor")
            return

        self.real_time_anchor_frame = int(self.left_frame_index)
        self.real_time_anchor_dt = parsed
        self.real_time_anchor_iso = parsed.isoformat()

        self._set_status_mid(f"Real time anchored at frame {self.real_time_anchor_frame}")
        self.time_sync_indicator.setText("✓ Synced")
        self._update_frame_labels()
        self.tutorial_window.notify_action("set_real_time_sync")

    def _refresh_real_time_entries(self, frame_index):
        """Update the six real-time anchor boxes to the calculated actual
        time at a given frame.

        A no-op if no anchor is set yet - so the boxes stay exactly as
        the user is typing them until "Set Time Sync" actually
        establishes an anchor; once one exists, this keeps the boxes
        live-tracking the calculated real-world time as the frame
        changes (scrubbing, playback, stepping), not frozen at the
        original anchor value.

        Args:
            frame_index (int): The left-timeline frame index to display
                the calculated actual time for.

        Returns:
            None
        """
        if self.real_time_anchor_frame is None or self.real_time_anchor_dt is None:
            return

        fps = self.metaL["fps"] if self.metaL else None
        if not fps or fps <= 0:
            return

        elapsed_seconds = (float(frame_index) - float(self.real_time_anchor_frame)) / float(fps)
        current_dt = self.real_time_anchor_dt + datetime.timedelta(seconds=elapsed_seconds)

        self.real_time_year_edit.setText(f"{current_dt.year:04d}")
        self.real_time_month_edit.setText(f"{current_dt.month:02d}")
        self.real_time_day_edit.setText(f"{current_dt.day:02d}")
        self.real_time_hour_edit.setText(f"{current_dt.hour:02d}")
        self.real_time_minute_edit.setText(f"{current_dt.minute:02d}")
        self.real_time_second_edit.setText(f"{current_dt.second:02d}")

    # -------------------------------------------------------------------------
    # Slider ranges / lock-sync
    # -------------------------------------------------------------------------

    def _clamp(self, value, lo, hi):
        """Clamp a value to `[lo, hi]`.

        Args:
            value (int): Value to clamp.
            lo (int): Minimum.
            hi (int): Maximum.

        Returns:
            int: The clamped value.
        """
        return max(lo, min(hi, value))

    def _update_slider_ranges(self):
        """Update slider max ranges and clamp indices based on lock mode.

        Direct port of the original's method of the same name.

        Returns:
            None
        """
        self.left_frame_max = (self.metaL["frame_count"] - 1) if self.metaL else 0
        self.right_frame_max = (self.metaR["frame_count"] - 1) if self.metaR else 0

        self._suppress_slider_callbacks = True
        try:
            if self.lock_lr and self.metaL and self.metaR:
                master_max = min(self.left_frame_max, self.right_frame_max)

                li = self._clamp(int(self.left_frame_index), 0, master_max)

                self.left_frame_index = li
                self.right_frame_index = li

                self.left_slider.setRange(0, master_max)
                self.right_slider.setRange(0, master_max)
                self.left_slider.setValue(li)
                self.right_slider.setValue(li)
            else:
                self.left_slider.setRange(0, int(self.left_frame_max))
                self.right_slider.setRange(0, int(self.right_frame_max))

                li = self._clamp(int(self.left_frame_index), 0, int(self.left_frame_max))
                ri = self._clamp(int(self.right_frame_index), 0, int(self.right_frame_max))

                self.left_frame_index = li
                self.right_frame_index = ri

                self.left_slider.setValue(li)
                self.right_slider.setValue(ri)
        finally:
            self._suppress_slider_callbacks = False

        self._update_frame_labels()

    def _jump_frames_locked_with_offset(self, master_side, target_index):
        """Move both timelines together, preserving the locked offset.

        Direct port of the original's method of the same name - if one
        side hits an end stop, shifts the *other* side to preserve the
        offset rather than changing the offset itself ("Option A"
        clamping).

        Args:
            master_side (str): `"L"` or `"R"` - which side's index
                `target_index` refers to.
            target_index (int): The requested frame index for
                `master_side`.

        Returns:
            None
        """
        lmax = self.left_frame_max
        rmax = self.right_frame_max

        if master_side == "L":
            li = target_index
            ri = li + int(self.lock_offset_frames)
        else:
            ri = target_index
            li = ri - int(self.lock_offset_frames)

        if li < 0:
            li = 0
            ri = li + int(self.lock_offset_frames)
        elif li > lmax:
            li = lmax
            ri = li + int(self.lock_offset_frames)

        if ri < 0:
            ri = 0
            li = ri - int(self.lock_offset_frames)
        elif ri > rmax:
            ri = rmax
            li = ri - int(self.lock_offset_frames)

        li = self._clamp(li, 0, lmax)
        ri = self._clamp(ri, 0, rmax)

        self.left_frame_index = li
        self.right_frame_index = ri

        self._suppress_slider_callbacks = True
        try:
            self.left_slider.setValue(li)
            self.right_slider.setValue(ri)
        finally:
            self._suppress_slider_callbacks = False

        self.render_current_frames()

    def on_left_slider_changed(self, value):
        """Handle the left slider moving (by user or programmatically).

        Args:
            value (int): The slider's new value.

        Returns:
            None
        """
        if self._suppress_slider_callbacks:
            return

        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", value)
        else:
            self.left_frame_index = value
            self.render_current_frames()

    def on_right_slider_changed(self, value):
        """Handle the right slider moving (by user or programmatically).

        Args:
            value (int): The slider's new value.

        Returns:
            None
        """
        if self._suppress_slider_callbacks:
            return

        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("R", value)
        else:
            self.right_frame_index = value
            self.render_current_frames()

    def on_toggle_lock(self, checked):
        """Handle the Lock L and R checkbox toggling.

        Enabling lock captures the current alignment as the new offset
        without moving either timeline - a direct port of the
        original's `on_toggle_lock`.

        Args:
            checked (bool): The checkbox's new state.

        Returns:
            None
        """
        self.lock_lr = checked
        if checked and self.capL and self.capR:
            self.lock_offset_frames = self.right_frame_index - self.left_frame_index
            self.offset_spin.blockSignals(True)
            self.offset_spin.setValue(self.lock_offset_frames)
            self.offset_spin.blockSignals(False)
        self._update_slider_ranges()
        self._refresh_status_left()
        self._update_lock_arrow_shortcuts()

    def on_offset_changed(self, value):
        """Handle the Offset spin box changing.

        Args:
            value (int): The spin box's new value.

        Returns:
            None
        """
        self.lock_offset_frames = value
        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", self.left_frame_index)

        self.tutorial_window.notify_action("set_resync_offset")

    # -------------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------------

    def on_play_pause(self):
        """Toggle play/pause.

        Returns:
            None
        """
        if not self.capL and not self.capR:
            return

        self.is_playing = not self.is_playing
        self._update_play_pause_icon()
        if self.is_playing:
            self._playback_tick()
        else:
            self.playback_timer.stop()

    def _update_play_pause_icon(self):
        """Set the play/pause button's icon to match `self.is_playing`.

        Returns:
            None
        """
        icon_name = "fa5s.pause" if self.is_playing else "fa5s.play"
        self.btn_play_pause.setIcon(qta.icon(icon_name, color=ICON_COLOR))

    def _playback_tick(self):
        """Advance playback by one speed-dependent step, then reschedule.

        See `_compute_playback_timing`'s docstring for the exact
        step/delay semantics - "1x" plays at the loaded video's actual
        native fps rather than a fixed assumption.

        Returns:
            None
        """
        if not self.is_playing:
            return

        fps = None
        if self.metaL:
            fps = self.metaL["fps"]
        elif self.metaR:
            fps = self.metaR["fps"]
        step, delay_ms = _compute_playback_timing(fps, self.speed_combo.currentText())

        if self.lock_lr and self.capL and self.capR:
            master_max = min(self.left_frame_max, self.right_frame_max)
            nxt = self.left_frame_index + step
            if nxt > master_max:
                self.is_playing = False
                self._update_play_pause_icon()
                return
            self._jump_frames_locked_with_offset("L", nxt)
        else:
            if self.capL:
                nxt = self._clamp(self.left_frame_index + step, 0, self.left_frame_max)
                self.left_frame_index = nxt
            if self.capR:
                nxt = self._clamp(self.right_frame_index + step, 0, self.right_frame_max)
                self.right_frame_index = nxt
            self.render_current_frames()

        if self.is_playing:
            self.playback_timer.start(delay_ms)

    def on_step_forward(self):
        """Step forward by one frame (or one locked pair).

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", self.left_frame_index + 1)
        else:
            if self.capL:
                self.left_frame_index = self._clamp(self.left_frame_index + 1, 0, self.left_frame_max)
            if self.capR:
                self.right_frame_index = self._clamp(self.right_frame_index + 1, 0, self.right_frame_max)
            self.render_current_frames()
            self._sync_slider_positions()

    def on_step_back(self):
        """Step back by one frame (or one locked pair).

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", self.left_frame_index - 1)
        else:
            if self.capL:
                self.left_frame_index = self._clamp(self.left_frame_index - 1, 0, self.left_frame_max)
            if self.capR:
                self.right_frame_index = self._clamp(self.right_frame_index - 1, 0, self.right_frame_max)
            self.render_current_frames()
            self._sync_slider_positions()

    def on_step_forward_single_pane(self, which):
        """Step one pane's timeline forward by one frame.

        Respects Lock L and R exactly like the sliders do (`on_left_slider_changed`/
        `on_right_slider_changed`): when locked, both timelines move
        together preserving the offset - Lock's whole purpose is that
        the two timelines never move independently while it's on, and a
        keyboard shortcut is no exception. Only steps the *given* pane
        alone when unlocked, which is when nudging one side to find/
        confirm the right offset actually makes sense.

        Args:
            which (str): `"L"` or `"R"` - which pane's timeline this
                step request came from.

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            current = self.left_frame_index if which == "L" else self.right_frame_index
            self._jump_frames_locked_with_offset(which, current + 1)
            return

        if which == "L" and self.capL:
            self.left_frame_index = self._clamp(self.left_frame_index + 1, 0, self.left_frame_max)
        elif which == "R" and self.capR:
            self.right_frame_index = self._clamp(self.right_frame_index + 1, 0, self.right_frame_max)
        self.render_current_frames()
        self._sync_slider_positions()

    def on_step_back_single_pane(self, which):
        """Step one pane's timeline back by one frame.

        See `on_step_forward_single_pane` - same Lock L and R handling.

        Args:
            which (str): `"L"` or `"R"` - which pane's timeline this
                step request came from.

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            current = self.left_frame_index if which == "L" else self.right_frame_index
            self._jump_frames_locked_with_offset(which, current - 1)
            return

        if which == "L" and self.capL:
            self.left_frame_index = self._clamp(self.left_frame_index - 1, 0, self.left_frame_max)
        elif which == "R" and self.capR:
            self.right_frame_index = self._clamp(self.right_frame_index - 1, 0, self.right_frame_max)
        self.render_current_frames()
        self._sync_slider_positions()

    def on_to_start(self):
        """Jump to frame 0 (or a locked pair at frame 0).

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", 0)
        else:
            if self.capL:
                self.left_frame_index = 0
            if self.capR:
                self.right_frame_index = 0
            self.render_current_frames()
            self._sync_slider_positions()

    def on_to_end(self):
        """Jump to the last frame (or a locked pair at the last common frame).

        Returns:
            None
        """
        if self.lock_lr and self.capL and self.capR:
            self._jump_frames_locked_with_offset("L", min(self.left_frame_max, self.right_frame_max))
        else:
            if self.capL:
                self.left_frame_index = self.left_frame_max
            if self.capR:
                self.right_frame_index = self.right_frame_max
            self.render_current_frames()
            self._sync_slider_positions()

    def _sync_slider_positions(self):
        """Move both sliders to match the current frame indices without recursing.

        Returns:
            None
        """
        self._suppress_slider_callbacks = True
        try:
            self.left_slider.setValue(self.left_frame_index)
            self.right_slider.setValue(self.right_frame_index)
        finally:
            self._suppress_slider_callbacks = False
        self._update_frame_labels()

    # -------------------------------------------------------------------------
    # View toggles
    # -------------------------------------------------------------------------

    def on_toggle_view_rectified(self, checked):
        """Handle the Show Rectified action toggling, with validation.

        Direct port of the original's `on_toggle_view_rectified` -
        requires calibration to be loaded, and (if videos are loaded)
        their resolution to match the calibrated resolution; forces the
        toggle back off with a status message otherwise.

        Args:
            checked (bool): The action's new (requested) state.

        Returns:
            None
        """
        if checked:
            if self.cal is None:
                self._force_rectified_off("Rectified view requires calibration")
                return
            if self.metaL and (self.metaL["width"] != self.cal["w"] or self.metaL["height"] != self.cal["h"]):
                self._force_rectified_off("Rectified view disabled: LEFT video resolution mismatch")
                return
            if self.metaR and (self.metaR["width"] != self.cal["w"] or self.metaR["height"] != self.cal["h"]):
                self._force_rectified_off("Rectified view disabled: RIGHT video resolution mismatch")
                return

        self.view_rectified.set(checked)
        self.rectified_indicator.setText("  RECTIFIED  " if checked else "  NOT RECTIFIED  ")
        self.rectified_indicator.setProperty("state", "rectified" if checked else "not_rectified")
        self.rectified_indicator.style().polish(self.rectified_indicator)
        self.render_current_frames()

        # Only the "switched it on" direction completes the tutorial's
        # "Switch to Rectified View" step - reaching here with checked
        # False (or forced off above) isn't that action.
        if checked:
            self.tutorial_window.notify_action("toggle_rectified")

    def _force_rectified_off(self, message):
        """Force the Show Rectified action back off and show a status message.

        Args:
            message (str): The status bar message to show.

        Returns:
            None
        """
        self.action_show_rectified.blockSignals(True)
        self.action_show_rectified.setChecked(False)
        self.action_show_rectified.blockSignals(False)
        self.view_rectified.set(False)
        self._set_status_mid(message)

    def on_toggle_anaglyph_preview(self):
        """Start or stop the anaglyph preview window.

        Requires both videos to be loaded. Toggles based on the current
        `self.anaglyph_preview.active` state.

        Returns:
            None
        """
        if not self._both_videos_loaded():
            self._set_status_mid("Load both videos to use anaglyph preview")
            return

        if self.anaglyph_preview.active:
            self.anaglyph_preview.stop()
            return

        self.anaglyph_preview.start()

    def on_reset_pan_zoom(self):
        """Reset both panes' zoom/pan back to the plain fit view.

        A plain one-shot action, not a toggle/mode - there is no
        separate "native size" display mode to switch between (see
        `video_overlay.py`'s `_display_rect`); this just resets zoom
        and pan, nothing else.

        Returns:
            None
        """
        self.pane_left.reset_view()
        self.pane_right.reset_view()

    # -------------------------------------------------------------------------
    # Project save/load
    # -------------------------------------------------------------------------

    def _app_window_title(self):
        """Build the title text this window should show.

        Returns:
            str: "Sizeamatic Pro vX.Y.Z" (or just "Sizeamatic Pro" if
            the version couldn't be read), plus " - <project name>"
            once a project has been saved/opened this session.
        """
        version = get_app_version()
        base = f"Sizeamatic Pro v{version}" if version != "unknown" else "Sizeamatic Pro"
        if self.current_project_name:
            return f"{base} - {self.current_project_name}"
        return base

    def _refresh_window_title(self):
        """Reapply this window's title, plus any open sub-window's, e.g.
        after a project is saved/opened.

        Returns:
            None
        """
        title = self._app_window_title()
        self.setWindowTitle(title)
        if self.measurement_window.win is not None:
            self.measurement_window.win.setWindowTitle(title)
        if self.cal_summary_window.win is not None:
            self.cal_summary_window.win.setWindowTitle(title)

    def on_save_project(self):
        """Prompt for a save location and write the current project state.

        Saves the left/right video paths, calibration folder, current
        resync offset, and rectified-view toggle state, plus whatever
        this port already has ported for the measurement log/last-
        recorded-snapshot/real-time-anchor/perform-calibration-capture-
        folder fields (see this class's `__init__` for why those are
        still plain passthrough attributes rather than backed by real
        UI yet).

        Returns:
            None
        """
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Project", "", "Sizeamatic Project (*.json);;All Files (*)"
        )
        if not path:
            return

        error = project_io.save_project(
            path,
            self.left_video_path,
            self.right_video_path,
            self.calibration_folder,
            self.lock_offset_frames,
            self.view_rectified.get(),
            get_app_version(),
            self.measurement_window.get_log_text(),
            self.last_recorded_snapshot,
            self.real_time_anchor_frame,
            self.real_time_anchor_iso,
            self.perform_calibration_window.capture_folder,
        )

        if error is not None:
            QMessageBox.critical(self, "Save Project", error)
            return

        recent_projects.add_recent_project(path)

        self.current_project_name = os.path.splitext(os.path.basename(path))[0]
        self._refresh_window_title()
        self._set_status_mid("Project saved")
        self.tutorial_window.notify_action("save_project")

    def on_open_project(self):
        """Prompt for a project file and reload the saved video/calibration state.

        Just handles the file dialog - the actual load/restore logic
        is shared with the File > Recent Projects submenu via
        `_open_project_from_path`.

        Returns:
            None
        """
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Sizeamatic Project (*.json);;All Files (*)"
        )
        if not path:
            return
        if self._open_project_from_path(path):
            self.tutorial_window.notify_action("open_project")

    def _open_project_from_path(self, path):
        """Load and apply a project manifest from an already-known path.

        Reads the project manifest via `project_io.load_project`, then
        reuses the same dialog-free loading helpers the file-picker
        menu items use, so a saved path that's since become invalid
        (moved/deleted file) surfaces the exact same error dialogs a
        manual reload would, one per stage, rather than failing the
        whole project load silently. Shared by `on_open_project` (after
        its file dialog) and the File > Recent Projects submenu
        (`on_open_recent_project`) - each of those, not this shared
        helper, reports its own distinct tutorial completion action on
        success, since "Open Project…" and "Recent Projects" are two
        separate tutorial steps despite sharing this same load logic.

        Args:
            path (str): Path to the project file to open.

        Returns:
            bool: True if the project loaded successfully, False
            otherwise (with an error dialog already shown).
        """
        project, error = project_io.load_project(path)
        if error is not None:
            QMessageBox.critical(self, "Open Project", error)
            return False

        if project["left_video_path"]:
            self._load_left_video_from_path(project["left_video_path"])

        if project["right_video_path"]:
            self._load_right_video_from_path(project["right_video_path"])

        if project["calibration_folder"]:
            self._load_calibration_from_folder(project["calibration_folder"])

        # Restore the resync offset after both videos are loaded, so it
        # doesn't get overwritten by anything the video loads above do.
        self.lock_offset_frames = int(project["lock_offset_frames"])
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(self.lock_offset_frames)
        self.offset_spin.blockSignals(False)

        # Restore the rectified-view toggle, after calibration is loaded -
        # go through the real toggle handler (not just the flag) so its
        # existing resolution-mismatch validation still applies, in case
        # the saved calibration folder no longer matches these videos.
        self.action_show_rectified.blockSignals(True)
        self.action_show_rectified.setChecked(bool(project["view_rectified"]))
        self.action_show_rectified.blockSignals(False)
        self.on_toggle_view_rectified(bool(project["view_rectified"]))

        self.last_recorded_snapshot = project["last_recorded_snapshot"]
        self.real_time_anchor_frame = project["real_time_anchor_frame"]
        self.real_time_anchor_iso = project["real_time_anchor_iso"]
        self.real_time_anchor_dt = (
            datetime.datetime.fromisoformat(self.real_time_anchor_iso) if self.real_time_anchor_iso else None
        )
        if self.real_time_anchor_dt is not None:
            self.time_sync_indicator.setText("✓ Synced")

        # The videos were already loaded (and _update_frame_labels already
        # ran as a side effect) above, before the anchor was restored just
        # now - so the readout row and the six entry boxes rendered with
        # no anchor in effect yet and never got refreshed afterward,
        # leaving them showing "(not set)"/blank until something else
        # (e.g. pressing Play) happened to trigger a redraw. Refresh now
        # so a restored anchor is visible immediately.
        self._update_frame_labels()

        self.measurement_window.restore_log_text(project["measurement_log_text"])

        self.perform_calibration_window.capture_folder = project["perform_calibration_capture_folder"]
        if self.perform_calibration_window.win is not None:
            self.perform_calibration_window.folder_label.setText(
                self.perform_calibration_window.capture_folder or "(no capture folder chosen yet)"
            )
            self.perform_calibration_window._refresh_pairs_listbox()

        # Jump back to "the very last place that was recorded" and show
        # that same measurement on screen again - last, since it depends
        # on the video/calibration/offset state above already being in
        # place.
        snapshot = project["last_recorded_snapshot"]
        if snapshot:
            self.left_frame_index = int(snapshot["left_frame_index"])
            self.right_frame_index = int(snapshot["right_frame_index"])
            self._sync_slider_positions()

            self.ptsL = [tuple(p) for p in snapshot["ptsL"]]
            self.ptsR = [tuple(p) for p in snapshot["ptsR"]]

            self.render_current_frames()
            self.on_points_changed()

        recent_projects.add_recent_project(path)

        self.current_project_name = os.path.splitext(os.path.basename(path))[0]
        self._refresh_window_title()
        self._set_status_mid("Project opened")
        return True

    def on_open_recent_project(self, path):
        """Open a project path chosen from the File > Recent Projects submenu.

        Args:
            path (str): The project file path to open, as listed by
                `_refresh_recent_projects_menu`.

        Returns:
            None
        """
        if self._open_project_from_path(path):
            self.tutorial_window.notify_action("open_recent_project")

    def _refresh_recent_projects_menu(self):
        """Rebuild the File > Recent Projects submenu just before it's shown.

        Returns:
            None
        """
        self.recent_projects_menu.clear()

        recent = recent_projects.load_recent_projects()
        if not recent:
            action = self.recent_projects_menu.addAction("(none yet)")
            action.setEnabled(False)
            return

        for path in recent:
            self.recent_projects_menu.addAction(
                self._short_path(path, max_len=60), lambda p=path: self.on_open_recent_project(p)
            )

    def _short_path(self, path, max_len=45):
        """Truncate a file path for compact display.

        Direct port of the original's helper of the same name.

        Args:
            path (str | None): The path to shorten, or None.
            max_len (int): Maximum displayed length before truncating
                with a leading ellipsis.

        Returns:
            str: `"(none)"` if `path` is None, the path unchanged if it
            fits within `max_len`, otherwise an ellipsis-prefixed
            suffix of the path.
        """
        if path is None:
            return "(none)"
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1) :]


def _size_window_to_screen(window, screen):
    """Resize `window` to fill most of `screen`, keeping the video panes
    close to `DEFAULT_VIDEO_ASPECT_RATIO` rather than stretching them.

    Sizes to `WINDOW_SCREEN_FRACTION` of the screen's available width,
    then derives a matching height from the window's actual non-pane
    chrome (menu bar, toolbar, slider row, status bar) - that overhead
    is fixed regardless of window height, since the panes are the only
    `stretch=1` widgets in their layout, so measuring it once at an
    arbitrary height gives an exact answer.

    Args:
        window (SizeamaticProApp): The main window, already built (so
            its panes/layout exist), not yet shown.
        screen (QScreen | None): The screen to size against, or `None`
            to leave the window's current (fallback) size alone.

    Returns:
        None
    """
    if screen is None:
        return

    available = screen.availableGeometry()
    target_width = int(available.width() * WINDOW_SCREEN_FRACTION)

    window.resize(target_width, available.height())
    QApplication.processEvents()
    pane_width = window.pane_left.width()
    chrome_height = window.height() - window.pane_left.height()

    target_height = min(
        available.height(),
        chrome_height + round(pane_width / DEFAULT_VIDEO_ASPECT_RATIO),
    )
    window.resize(target_width, target_height)


def main():
    """Entry point: build the QApplication and main window, run the event loop.

    Sets the Windows taskbar application identity (so the app groups
    under its own taskbar icon rather than a generic Python one) -
    unlike Tkinter, Qt already applies the window icon to the taskbar
    itself, but the AppUserModelID is still needed for correct taskbar
    grouping/pinning behavior on Windows. Shows the startup splash
    before building the main window, and keeps it up for at least
    `STARTUP_SPLASH_MIN_SECONDS` even if building finished faster.

    Returns:
        None
    """
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SizeamaticPro.SizeamaticPro.App")

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    # Qt's C++ event loop (app.exec(), below) never yields back to the
    # Python interpreter on its own, so a Ctrl+C at the console (SIGINT)
    # has no chance to actually get processed - Python only checks for a
    # pending signal between bytecode instructions, and exec() blocks
    # entirely outside that. Restoring the default handler (so SIGINT
    # really does terminate rather than whatever Qt/PySide may have set)
    # plus a trivial repeating QTimer (so the interpreter regains control
    # briefly every 200ms) together make Ctrl+C work again.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    keep_alive_timer = QTimer()
    keep_alive_timer.start(200)
    keep_alive_timer.timeout.connect(lambda: None)

    icon_path = resource_path("assets/icon.ico")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash = _show_startup_splash()
    if splash is not None:
        # Force the splash to actually paint before starting the clock -
        # show() alone just schedules the paint, it doesn't happen until
        # the event loop processes it. Without this, the window-building
        # work below runs first, and the progress bar's very first paint
        # (once the loop below reaches processEvents) already shows
        # whatever fraction of STARTUP_SPLASH_MIN_SECONDS that took, i.e.
        # a bar that visibly starts mid-way rather than at 0.
        app.processEvents()
    splash_shown_at = time.monotonic()

    window = SizeamaticProApp()
    enable_dark_title_bar(window)

    # Park far off any real monitor, then show - still needs a real
    # show() so the layout actually goes live (a hidden top-level window
    # doesn't recompute its child widgets' sizes on resize(), which
    # `_size_window_to_screen` below depends on), but parking it off-
    # screen first means the user never sees it pop up/flash on top of
    # the splash before the splash's animation finishes - relying on the
    # splash merely "staying on top" isn't reliable enough on its own
    # (window activation on show can still bring the main window forward
    # mid-animation).
    window.move(-32000, -32000)
    window.show()

    # Open on whichever screen the splash just showed on (the one the
    # cursor's actually on) rather than wherever Qt/the OS would place a
    # brand new top-level window by default - on a multi-monitor setup
    # those aren't guaranteed to be the same screen, and the splash and
    # the real window ending up on different monitors reads as broken.
    screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
    _size_window_to_screen(window, screen)

    final_x, final_y = None, None
    if screen is not None:
        available = screen.availableGeometry()
        final_x = available.x() + (available.width() - window.width()) // 2
        final_y = available.y() + (available.height() - window.height()) // 2

    # Smoothly drive the splash's progress bar from 0 to 100 across
    # STARTUP_SPLASH_MIN_SECONDS (measured from when the splash first
    # appeared, not from here - building `window` above already used up
    # part of that budget), rather than jumping straight to 100 the
    # instant the app happens to finish building. The main window stays
    # parked off-screen (see above) for this entire loop.
    while True:
        elapsed = time.monotonic() - splash_shown_at
        fraction = min(1.0, elapsed / STARTUP_SPLASH_MIN_SECONDS)
        if splash is not None and hasattr(splash, "progress_bar"):
            splash.progress_bar.setValue(int(fraction * 100))
        if fraction >= 1.0:
            break
        app.processEvents()
        time.sleep(0.01)

    # Only now move the window onto the screen for real, right as the
    # splash is about to hand off to it.
    if final_x is not None:
        window.move(final_x, final_y)

    if splash is not None:
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
