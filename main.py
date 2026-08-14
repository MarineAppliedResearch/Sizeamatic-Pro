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

This first-pass port covers the highest-risk, most-used piece: video
loading/playback/sync, pan/zoom, and point placement/dragging/
refinement (see `video_overlay.py`). Project save/load, recent
projects, the measurement results/calibration summary/perform
calibration/generate target sub-windows, and real-time sync are not
yet ported - see ROADMAP.md Phase 11's tracking for what's left.

See `ARCHITECTURE.md` for how responsibilities are currently split
across files, and `README.md` for the user-facing description of the
app.
"""

import ctypes
import os
import sys
import time
import tomllib

import cv2
from PIL import Image

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QCursor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplashScreen,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import calibration_io
import prepare_splash_image
import project_io
import recent_projects
import stereo_matching
import video_overlay


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


STARTUP_SPLASH_MIN_SECONDS = 4.0
"""Minimum time the startup splash (`_show_startup_splash`) stays on
screen, even if building the app finishes faster than that - so it's
actually readable rather than a barely-visible flash on a fast machine
or a source run. Unchanged from the original."""

STARTUP_SPLASH_MAX_WIDTH_PX = 720
"""Cap the splash's on-screen width, scaling the source art down
proportionally if it's larger. Unchanged from the original."""


def _pil_image_to_qpixmap(img):
    """Convert a PIL RGB image to a `QPixmap`.

    Builds a `QImage` from the raw pixel bytes directly (same approach
    `video_overlay.py` uses for video frames) rather than depending on
    `PIL.ImageQt`'s own Qt-binding auto-detection.

    Args:
        img (PIL.Image.Image): An RGB image.

    Returns:
        QPixmap: The converted pixmap.
    """
    data = img.tobytes("raw", "RGB")
    qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


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

        pixmap = _pil_image_to_qpixmap(img)
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


ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
ZOOM_STEP = 1.10
"""Per-wheel-notch zoom multiplier and bounds - see
`video_overlay.VideoPane.wheelEvent`."""

HANDLE_RADIUS_PX = 8
"""On-screen radius, in pixels, of each drawn point handle ring."""

MAX_POINTS_PER_PANE = 20
"""Hard cap on how many measurement points one pane can hold."""

SPEED_TABLE = {
    "0.25x": (1, 160),
    "0.5x": (1, 80),
    "1x": (1, 40),
    "2x": (2, 40),
    "4x": (4, 40),
}
"""Playback speed -> `(frame_step, tick_delay_ms)`. Slower-than-1x
speeds lengthen the tick interval (frame_step stays 1); faster-than-1x
speeds skip frames per tick at the same 40ms interval - matches the
original Tkinter app's `_playback_tick` exactly."""

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
QPushButton { background-color: #1a2438; color: #e8eefc; border: 1px solid #263351; border-radius: 5px; padding: 8px 16px; }
QPushButton:hover { background-color: #22304d; }
QPushButton:checked { background-color: #2f6fed; }
QComboBox, QCheckBox, QSpinBox { color: #e8eefc; }
QComboBox, QSpinBox { background-color: #1a2438; border: 1px solid #263351; border-radius: 5px; padding: 6px 10px; }
QCheckBox { padding: 4px 8px; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSlider::groove:horizontal { background: #1a2438; height: 4px; border-radius: 2px; margin: 0 4px; }
QSlider::handle:horizontal { background: #2f6fed; width: 14px; margin: -6px 0; border-radius: 7px; }
QStatusBar { color: #8ea2c6; padding: 4px 10px; }
QLabel { padding: 2px 4px; }
QLabel#rectifiedIndicator[state="rectified"] { color: #2fbf71; }
QLabel#rectifiedIndicator[state="not_rectified"] { color: #ef5350; }
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
        # Height tuned so the two panes' aspect ratio roughly matches the
        # stereo rigs' actual footage (1280x800, a 1.6:1 ratio) by default,
        # instead of leaving tall letterbox bars above/below the video.
        self.resize(1400, 590)

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

        # ---- Project save/load state ----
        # Plain passthrough attributes for the sub-windows/features this
        # port hasn't reached yet (measurement log, real-time anchor,
        # perform-calibration capture folder) - not yet settable from
        # this app's UI, but preserved round-trip through save/open so a
        # project file saved by (or containing data from) a more-complete
        # version of this app doesn't lose that data if reopened and
        # resaved here in the meantime.
        self.current_project_name = None
        self.measurement_log_text = ""
        self.last_recorded_snapshot = None
        self.real_time_anchor_frame = None
        self.real_time_anchor_iso = None
        self.perform_calibration_capture_folder = None

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
        self.playback_timer.timeout.connect(self._playback_tick)

        self._suppress_slider_callbacks = False

        self._build_menu()
        self._build_toolbar()
        self._build_central_widget()
        self._build_statusbar()

        self._update_slider_ranges()
        self._refresh_window_title()

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------

    def _build_menu(self):
        """Build the File/View/Calibration menus.

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

        view_menu.addAction("Reset Pan/Zoom", self.on_reset_pan_zoom)

        calibration_menu = menubar.addMenu("Calibration")
        calibration_menu.addAction("Load Calibration…", self.on_load_calibration_folder)

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

        self.btn_to_start = QPushButton("⏮")
        self.btn_step_back = QPushButton("◀")
        self.btn_play_pause = QPushButton("⏯")
        self.btn_step_forward = QPushButton("▶")
        self.btn_to_end = QPushButton("⏭")
        for btn in (self.btn_to_start, self.btn_step_back, self.btn_play_pause, self.btn_step_forward, self.btn_to_end):
            btn.setFixedWidth(36)
            toolbar.addWidget(btn)

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

        self.lock_checkbox = QCheckBox("  Lock L and R")
        self.lock_checkbox.setChecked(True)
        self.lock_checkbox.toggled.connect(self.on_toggle_lock)
        toolbar.addWidget(self.lock_checkbox)

        toolbar.addWidget(QLabel("  Offset:"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-100000, 100000)
        self.offset_spin.valueChanged.connect(self.on_offset_changed)
        toolbar.addWidget(self.offset_spin)

        self.btn_clear_points = QPushButton("  Clear Points  ")
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        toolbar.addWidget(self.btn_clear_points)

        self.rectified_indicator = QLabel("  NOT RECTIFIED  ")
        self.rectified_indicator.setObjectName("rectifiedIndicator")
        self.rectified_indicator.setProperty("state", "not_rectified")
        toolbar.addWidget(self.rectified_indicator)

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
        """Build the status bar.

        Returns:
            None
        """
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

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

    def on_load_left_video(self):
        """Prompt for and load the left video.

        Just handles the file dialog; the actual loading logic lives
        in `_load_left_video_from_path` so `_open_project_from_path`
        can reuse it with a path read from a project file instead.

        Returns:
            None
        """
        path, _filter = QFileDialog.getOpenFileName(self, "Load Left Video", "", "MP4 Video (*.mp4);;All Files (*)")
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
        self.statusBar().showMessage(
            f"Loaded left video ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )
        return True

    def on_load_right_video(self):
        """Prompt for and load the right video.

        Just handles the file dialog; the actual loading logic lives
        in `_load_right_video_from_path` so `_open_project_from_path`
        can reuse it with a path read from a project file instead.

        Returns:
            None
        """
        path, _filter = QFileDialog.getOpenFileName(self, "Load Right Video", "", "MP4 Video (*.mp4);;All Files (*)")
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
        self.statusBar().showMessage(
            f"Loaded right video ({meta['width']}×{meta['height']}, fps={meta['fps']:.3f}, frames={meta['frame_count']})"
        )
        return True

    def on_load_calibration_folder(self):
        """Prompt for and load a calibration folder.

        Just handles the folder dialog; the actual loading logic lives
        in `_load_calibration_from_folder` so `_open_project_from_path`
        can reuse it with a path read from a project file instead.

        Returns:
            None
        """
        folder = QFileDialog.getExistingDirectory(self, "Load Calibration Folder")
        if not folder:
            return
        self._load_calibration_from_folder(folder)

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
        self.statusBar().showMessage(f"Loaded calibration from {folder}")
        return True

    # -------------------------------------------------------------------------
    # Frame decode/render
    # -------------------------------------------------------------------------

    def _read_frame_at(self, cap, index):
        """Decode the frame at a specific index, seeking only if needed.

        Direct port of the original's helper of the same name - see
        that docstring (in git history) for the seek-cost rationale;
        unchanged here since it never depended on Tkinter.

        Args:
            cap (cv2.VideoCapture): The capture to read from.
            index (int): The zero-based frame index to read.

        Returns:
            numpy.ndarray | None: The decoded BGR frame, or None if the
            seek/decode failed.
        """
        index = int(index)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
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
        """Redraw both panes after a measurement point was placed/moved.

        Returns:
            None
        """
        self.pane_left.update()
        self.pane_right.update()

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
        self.statusBar().showMessage("Cleared all points")

    def _update_frame_labels(self):
        """Refresh the "Frame: i/max" labels under each slider.

        Returns:
            None
        """
        self.left_frame_label.setText(f"Frame: {self.left_frame_index}/{self.left_frame_max}")
        self.right_frame_label.setText(f"Frame: {self.right_frame_index}/{self.right_frame_max}")

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
        if self.is_playing:
            self._playback_tick()
        else:
            self.playback_timer.stop()

    def _playback_tick(self):
        """Advance playback by one speed-dependent step, then reschedule.

        Direct port of the original's `_playback_tick` - see
        `SPEED_TABLE`'s docstring for the exact step/delay semantics.

        Returns:
            None
        """
        if not self.is_playing:
            return

        step, delay_ms = SPEED_TABLE.get(self.speed_combo.currentText(), (1, 40))

        if self.lock_lr and self.capL and self.capR:
            master_max = min(self.left_frame_max, self.right_frame_max)
            nxt = self.left_frame_index + step
            if nxt > master_max:
                self.is_playing = False
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
        self.statusBar().showMessage(message)

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
        """Reapply this window's title, e.g. after a project is saved/opened.

        Returns:
            None
        """
        self.setWindowTitle(self._app_window_title())

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
            self.measurement_log_text,
            self.last_recorded_snapshot,
            self.real_time_anchor_frame,
            self.real_time_anchor_iso,
            self.perform_calibration_capture_folder,
        )

        if error is not None:
            QMessageBox.critical(self, "Save Project", error)
            return

        recent_projects.add_recent_project(path)

        self.current_project_name = os.path.splitext(os.path.basename(path))[0]
        self._refresh_window_title()
        self.statusBar().showMessage("Project saved")

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
        self._open_project_from_path(path)

    def _open_project_from_path(self, path):
        """Load and apply a project manifest from an already-known path.

        Reads the project manifest via `project_io.load_project`, then
        reuses the same dialog-free loading helpers the file-picker
        menu items use, so a saved path that's since become invalid
        (moved/deleted file) surfaces the exact same error dialogs a
        manual reload would, one per stage, rather than failing the
        whole project load silently. Shared by `on_open_project` (after
        its file dialog) and the File > Recent Projects submenu
        (`on_open_recent_project`).

        Args:
            path (str): Path to the project file to open.

        Returns:
            None
        """
        project, error = project_io.load_project(path)
        if error is not None:
            QMessageBox.critical(self, "Open Project", error)
            return

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

        # Preserve (but don't yet act on) the fields this port hasn't
        # reached - see __init__'s note on these passthrough attributes.
        self.measurement_log_text = project["measurement_log_text"]
        self.last_recorded_snapshot = project["last_recorded_snapshot"]
        self.real_time_anchor_frame = project["real_time_anchor_frame"]
        self.real_time_anchor_iso = project["real_time_anchor_iso"]
        self.perform_calibration_capture_folder = project["perform_calibration_capture_folder"]

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
        self.statusBar().showMessage("Project opened")

    def on_open_recent_project(self, path):
        """Open a project path chosen from the File > Recent Projects submenu.

        Args:
            path (str): The project file path to open, as listed by
                `_refresh_recent_projects_menu`.

        Returns:
            None
        """
        self._open_project_from_path(path)

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

    # Open on whichever screen the splash just showed on (the one the
    # cursor's actually on) rather than wherever Qt/the OS would place a
    # brand new top-level window by default - on a multi-monitor setup
    # those aren't guaranteed to be the same screen, and the splash and
    # the real window ending up on different monitors reads as broken.
    screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        x = available.x() + (available.width() - window.width()) // 2
        y = available.y() + (available.height() - window.height()) // 2
        window.move(x, y)

    # Smoothly drive the splash's progress bar from 0 to 100 across
    # STARTUP_SPLASH_MIN_SECONDS (measured from when the splash first
    # appeared, not from here - building `window` above already used up
    # part of that budget), rather than jumping straight to 100 the
    # instant the app happens to finish building.
    while True:
        elapsed = time.monotonic() - splash_shown_at
        fraction = min(1.0, elapsed / STARTUP_SPLASH_MIN_SECONDS)
        if splash is not None and hasattr(splash, "progress_bar"):
            splash.progress_bar.setValue(int(fraction * 100))
        if fraction >= 1.0:
            break
        app.processEvents()
        time.sleep(0.01)

    if splash is not None:
        splash.finish(window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
