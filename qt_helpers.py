"""Small PySide6 helpers shared across Sizeamatic Pro's dialog windows.

Kept separate from `main.py` so `measurement_window.py`,
`calibration_summary.py`, `generate_calibration_target.py`, and
`perform_calibration.py` can all import from here without an import
cycle - `main.py` has to `import` each of those modules to wire up its
own menu actions, so they can't import back from `main.py`.

Contents:
    - `pil_image_to_qpixmap` - PIL Image -> QPixmap conversion, used by
      any window that previews a generated image (calibration boards,
      captured frame pairs).
    - `ClosableDialog` - a `QDialog` that calls back into its owner when
      the user closes it, mirroring Tkinter's `Toplevel` +
      `WM_DELETE_WINDOW` protocol pattern each of this app's lazily-built
      sub-windows was already using. Deliberately given no Qt parent
      widget - see its docstring for why.
    - `enable_dark_title_bar` - asks Windows' DWM to draw a window's
      native title bar in dark mode, matching the app's own dark theme.
    - `move_to_same_screen_as` - centers a widget on whichever monitor
      another (reference) widget currently occupies.

Author:
    Isaac Travers

Created:
    2026-08-14
"""

import ctypes
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog


def pil_image_to_qpixmap(img):
    """Convert a PIL RGB image to a `QPixmap`.

    Builds a `QImage` from the raw pixel bytes directly (same approach
    `video_overlay.py` uses for video frames) rather than depending on
    `PIL.ImageQt`'s own Qt-binding auto-detection.

    Args:
        img (PIL.Image.Image): An RGB image.

    Returns:
        QPixmap: The converted pixmap.
    """
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def enable_dark_title_bar(widget):
    """Ask Windows' DWM to draw `widget`'s native title bar in dark mode.

    Qt's QSS only styles Qt-drawn widgets - the native OS title bar/
    frame is drawn by the window manager and doesn't follow the app's
    palette at all, so without this every window's title bar reverts to
    the OS's default (light, on a typical Windows install) even though
    everything below it is dark. Best-effort and a no-op off Windows or
    on a Windows version that doesn't support this attribute.

    Args:
        widget (QWidget): The top-level window to dark-mode. Calling
            `.winId()` forces its native window handle to exist even if
            it hasn't been shown yet.

    Returns:
        None
    """
    if sys.platform != "win32":
        return

    try:
        hwnd = int(widget.winId())
        dwmwa_use_immersive_dark_mode = 20
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, dwmwa_use_immersive_dark_mode, ctypes.byref(value), ctypes.sizeof(value)
        )
    except Exception:
        pass


def move_to_same_screen_as(widget, reference_widget):
    """Center `widget` on whichever monitor `reference_widget` occupies.

    Without this, a freshly-built top-level window (a sub-window dialog,
    the anaglyph preview) opens wherever Qt/the OS defaults to placing a
    new window, which isn't guaranteed to be the same monitor the rest
    of the app is running on - the same multi-monitor issue already
    fixed for the startup splash screen in `main.py`.

    Args:
        widget (QWidget): The window to position - must already be
            sized (e.g. via `.resize()`) for the centering math to be
            correct.
        reference_widget (QWidget | None): The widget whose current
            screen to match, typically the main app window.

    Returns:
        None
    """
    screen = reference_widget.screen() if reference_widget is not None else None
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is None:
        return

    available = screen.availableGeometry()
    x = available.x() + max(0, (available.width() - widget.width()) // 2)
    y = available.y() + max(0, (available.height() - widget.height()) // 2)
    widget.move(x, y)


class ClosableDialog(QDialog):
    """A non-modal, independent top-level `QDialog` that notifies its
    owner on close.

    Each of this app's lazily-built sub-windows (measurement results,
    calibration summary, generate target, perform calibration) is owned
    by a plain Python class instance (`app.measurement_window`, etc.)
    that stores its own widget references and needs to clear them when
    the window closes, so a later `ensure_window()` call knows to
    rebuild rather than touch destroyed widgets - the same role
    Tkinter's `Toplevel` + `protocol("WM_DELETE_WINDOW", ...)` played.

    Deliberately constructed with NO Qt parent widget, even though the
    main app window is readily available - Qt/Windows keeps an owned
    top-level window (one constructed *with* a parent) permanently
    stacked above that parent, so clicking back on the main window
    would never bring it in front of this one. Tkinter's `Toplevel`
    doesn't enforce that, so parenting this the "obvious" way would have
    been a real behavioral regression (reported directly: "when you
    click on the other window, the data entering table stays on top").
    Omitting the parent makes this a fully independent window that
    stacks normally - `main.py`'s `closeEvent` explicitly closes any
    that are still open when the app quits, since Qt's default
    quit-on-last-window-closed can't rely on parent/child cleanup here.
    """

    def __init__(self, on_close):
        """Store the close callback and set up as an independent window.

        Args:
            on_close (Callable[[], None]): Called (with no arguments)
                right before the dialog actually closes.

        Returns:
            None
        """
        super().__init__(None, Qt.WindowType.Window)
        self._on_close = on_close
        enable_dark_title_bar(self)

    def closeEvent(self, event):
        """Notify the owner, then let the close proceed normally.

        Args:
            event (QCloseEvent): The close event.

        Returns:
            None
        """
        self._on_close()
        super().closeEvent(event)
