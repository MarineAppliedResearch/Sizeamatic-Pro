"""Video display + overlay interaction pane for Sizeamatic Pro (PySide6).

Ported from the original Tkinter `video_overlay.py` + the coordinate-
transform/pan/zoom methods that lived on `main.py`'s `SizeamaticProApp`.
Unlike the Tkinter version - which stacked a `tk.Canvas` for the video
frame under a second, separate `tk.Canvas` for the point/line overlay -
this is a single `QWidget` per pane that paints both the frame and the
overlay together in one `paintEvent`, since Qt's own repaint model
makes that layering unnecessary (see this module's "Design notes").

Contents:
    - `VideoPane` - one left/right video display + overlay widget. Two
      instances exist, owned by `main.py`'s app object.

Design notes:
    Coordinate system: measurement points are stored in *image pixel*
    coordinates (`app.ptsL`/`app.ptsR`); screen coordinates are this
    widget's own local pixel coordinates. Each pane keeps its own pan/
    zoom state (`self.view = {"zoom", "off_x", "off_y"}`) - `zoom` is a
    unitless multiplier on top of "fit to window" scale, `off_x`/
    `off_y` are pan offsets in screen pixels applied after scaling. All
    of `image_to_screen`/`screen_to_image`/the display-rect/scale math
    is a direct port of `main.py`'s original `_image_to_screen`/
    `_screen_to_image`/`_get_display_rect`/`_get_fit_scale`/
    `_get_total_scale` - same formulas, now as instance methods on the
    pane itself instead of app methods taking a `which`/`canvas` pair,
    since each pane now owns its own widget and state directly.

    Frame painting reuses that exact same crop-in-image-space math the
    original used (`_display_bgr_on_canvas`) so panning/zooming clips
    and letterboxes identically - but hands the actual scaling to
    `QPainter.drawImage`'s own source/target rects instead of manually
    `cv2.resize`-ing a cropped array first. This is a legitimate
    simplification, not a behavior change: same crop bounds, same
    edge-clamping, just letting Qt's painter do the scale+blit step
    Tkinter/Pillow had to do by hand.

    Resize handling is simpler here than the original for a genuine
    reason, not a dropped feature: the original had to explicitly
    debounce-then-re-decode-and-redraw on every `<Configure>` event,
    because Tkinter's canvas needed a fresh manually-drawn image.
    `paintEvent` already recomputes the display rect/scale from the
    widget's *current* size on every call, reading only the already-
    decoded, cached `QImage` - so Qt's own automatic repaint-on-resize
    reproduces the same "resize live-updates the display" behavior for
    free, with no manual re-render or debounce needed.

    Panning/point-dragging/refining call back into the owning app
    (`self.app.on_points_changed`/`redisplay_current_frames`/
    `render_current_frames`) rather than redrawing directly - the app
    object is what actually knows about both panes and the currently
    decoded frames, matching the original's app-owns-state split.

Assumptions:
    - The owning app exposes: `ptsL`/`ptsR` (plain lists of `(x, y)`
      tuples, image-pixel coordinates); `metaL`/`metaR` (dict with
      `"width"`/`"height"`, or None if that side isn't loaded);
      `view_rectified` (an object with `.get()`
      returning bool, for `stereo_matching.py` compatibility - see
      `main.py`); `cal` (calibration dict or None);
      `current_frameL`/`current_frameR` (the exact decoded/rectified
      BGR frame currently on screen, for the scanline matcher);
      `zoom_min`/`zoom_max`/`zoom_step`/`handle_radius_px`/
      `max_points_per_pane` (numeric constants); `on_points_changed()`/
      `redisplay_current_frames()`/`render_current_frames()` (redraw/
      recompute hooks).

Author:
    Isaac Travers

Date:
    2026-08-14
"""

import cv2

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

import stereo_matching

BG_DEEP = QColor("#0a0f1a")
BORDER = QColor("#263351")
TEXT_PRIMARY = QColor("#e8eefc")

CENTER_DOT_RADIUS_PX = 2
"""Screen-pixel radius of the small solid dot drawn at each point
handle's exact center - see `_paint_overlay`. Fixed regardless of
zoom, matching the ring's own radius behavior."""

RECTIFIED_OVERLAY_COLOR = QColor("#00ff66")
NOT_RECTIFIED_OVERLAY_COLOR = QColor("#ffa500")
CENTER_DOT_COLOR = QColor("#ff0000")
"""The small center dot deliberately stays this color in both
rectified/not-rectified modes - it exists to pinpoint the exact
clicked pixel, a purpose unrelated to rectification state."""


class VideoPane(QWidget):
    """One left/right video display + point-overlay interaction pane.

    Two instances exist (`which="L"` and `which="R"`), owned by the
    main app object.
    """

    def __init__(self, app, which, parent=None):
        """Store the owning app/side and initialize pan/zoom/drag state.

        Args:
            app: The main application object - see this module's
                docstring for the exact attributes assumed to exist.
            which (str): Which pane this is, `"L"` or `"R"`.
            parent (QWidget | None): Optional Qt parent widget.

        Returns:
            None
        """
        super().__init__(parent)
        self.app = app
        self.which = which

        self.setMouseTracking(True)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.current_qimage = None
        """The currently displayed frame, pre-converted to a `QImage`
        once per decode (not per repaint) - see `set_frame`."""

        self.view = {"zoom": 1.0, "off_x": 0.0, "off_y": 0.0}
        """This pane's own pan/zoom state - see this module's
        docstring for what each key means."""

        self.pan_active = False
        self.pan_last_pos = None
        """Screen-space `(x, y)` the last pan mouse-move event was at,
        or None while no pan is active - used to compute the delta for
        the next move."""

        self.drag_active = False
        self.drag_index = None
        """Index (into this pane's own `pts` list) of the point
        currently being dragged by a left-button drag, or None."""

        self.refine_drag_active = False
        self.refine_drag_index = None
        """Index of the point currently being refined by a right-
        button drag, or None. Distinct from `drag_index` - refinement
        only starts on a point that already has a matched mate in the
        opposite pane (see `mousePressEvent`)."""

    def reset_view(self):
        """Reset this pane's zoom/pan back to the plain, un-zoomed fit view.

        Called when a fresh video is loaded into this pane, and by the
        main window's plain "Reset Pan/Zoom" action - per the project
        owner's request, loading a new video should show it "the same
        way it was loaded" rather than carrying over whatever zoom/pan
        happened to be active, and resetting zoom/pan on demand is a
        one-shot action, not a mode toggle. This is a deliberate
        difference from the original Tkinter app, which never reset
        zoom/pan automatically.

        Returns:
            None
        """
        self.view["zoom"] = 1.0
        self.view["off_x"] = 0.0
        self.view["off_y"] = 0.0
        self.update()

    # -------------------------------------------------------------------------
    # Frame caching
    # -------------------------------------------------------------------------

    def set_frame(self, frame_bgr):
        """Cache a newly decoded frame for painting.

        Converts BGR to RGB and wraps it as a `QImage` once here,
        rather than doing that conversion inside `paintEvent` (which
        can fire many times per decoded frame, e.g. during a pan
        drag) - mirrors the original's separation of "expensive decode
        + convert" from "cheap redisplay with a new transform".

        Args:
            frame_bgr (numpy.ndarray | None): The decoded (and, if
                rectified view is on, already-remapped) BGR frame, or
                None if this pane has nothing to show right now.

        Returns:
            None
        """
        if frame_bgr is None:
            self.current_qimage = None
            self.update()
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)

        # .copy() so the QImage owns its own buffer - the numpy array
        # backing `rgb` gets overwritten by the next decoded frame,
        # and Qt does not copy image data by default.
        self.current_qimage = qimage.copy()
        self.update()

    # -------------------------------------------------------------------------
    # Coordinate transforms (ported from main.py's SizeamaticProApp)
    # -------------------------------------------------------------------------

    def _image_size(self):
        """Get this pane's loaded video's frame size, if known.

        Returns:
            tuple[int, int] | None: `(width, height)` in pixels, or
            None if this side has no video loaded yet.
        """
        meta = self.app.metaL if self.which == "L" else self.app.metaR
        if not meta:
            return None
        return meta["width"], meta["height"]

    def _display_rect(self):
        """Compute the on-widget rectangle where video should be drawn.

        Always fits the video to the pane, preserving aspect ratio -
        fits by width first, falling back to fitting by height if that
        would overflow the widget, then centers the result
        (letterboxing). There is deliberately no "native size" mode to
        toggle - zoom/pan (see `reset_view`) already cover wanting to
        see the video larger or at a specific crop.

        Returns:
            tuple[int, int, int, int]: `(dx, dy, dw, dh)` - the display
            rect's top-left corner and size, in this widget's own
            local pixel coordinates.
        """
        cw = max(1, self.width())
        ch = max(1, self.height())

        size = self._image_size()
        if size is None:
            return 0, 0, cw, ch
        img_w, img_h = size

        dw = cw
        dh = int(round(dw * (float(img_h) / float(img_w))))
        if dh > ch:
            dh = ch
            dw = int(round(dh * (float(img_w) / float(img_h))))

        dx = (cw - dw) // 2
        dy = (ch - dh) // 2
        return dx, dy, dw, dh

    def _fit_scale(self):
        """Compute the "fit to pane" base scale factor (before zoom).

        Returns:
            float: `1.0` if the image size isn't known yet, else the
            ratio of the display rect's width to the actual image
            width.
        """
        size = self._image_size()
        if size is None:
            return 1.0

        _dx, _dy, dw, _dh = self._display_rect()
        img_w, _img_h = size
        if img_w <= 0:
            return 1.0

        return float(dw) / float(img_w)

    def _total_scale(self):
        """Compute the full image-to-screen scale factor (fit * zoom).

        Returns:
            float: `_fit_scale() * self.view["zoom"]`.
        """
        return self._fit_scale() * float(self.view["zoom"])

    def image_to_screen(self, ix, iy):
        """Convert an image-pixel coordinate to this widget's local coordinates.

        Args:
            ix (float): Image X coordinate.
            iy (float): Image Y coordinate.

        Returns:
            tuple[float, float]: The corresponding `(x, y)` in this
            widget's own local pixel coordinates.
        """
        dx, dy, _dw, _dh = self._display_rect()
        scale = self._total_scale()
        sx = float(dx) + float(ix) * scale + float(self.view["off_x"])
        sy = float(dy) + float(iy) * scale + float(self.view["off_y"])
        return sx, sy

    def screen_to_image(self, sx, sy):
        """Convert this widget's local coordinates to an image-pixel coordinate.

        Exact inverse of `image_to_screen`.

        Args:
            sx (float): Local X coordinate.
            sy (float): Local Y coordinate.

        Returns:
            tuple[float, float]: The corresponding `(ix, iy)` in image
            pixel coordinates.
        """
        dx, dy, _dw, _dh = self._display_rect()
        scale = self._total_scale()
        if scale <= 0.0:
            scale = 1.0
        ix = (float(sx) - float(dx) - float(self.view["off_x"])) / scale
        iy = (float(sy) - float(dy) - float(self.view["off_y"])) / scale
        return ix, iy

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------

    def paintEvent(self, event):
        """Paint the current frame (if any) and the point/line overlay.

        Args:
            event (QPaintEvent): Unused - always repaints the whole
                widget.

        Returns:
            None
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG_DEEP)

        if self.current_qimage is not None:
            self._paint_frame(painter)
        else:
            self._paint_placeholder(painter)

        self._paint_overlay(painter)
        self._paint_border(painter)
        painter.end()

    def _paint_border(self, painter):
        """Draw a visible container border around the pane's edge.

        Drawn last (on top of the frame/overlay) so it always reads as
        a defined canvas edge rather than the pane blending into the
        window background - inset by half the pen width so the stroke
        is crisp rather than clipped at the widget bounds.

        Args:
            painter (QPainter): The active painter.

        Returns:
            None
        """
        painter.setPen(QPen(BORDER, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 4, 4)

    def _paint_placeholder(self, painter):
        """Draw a plain "no video loaded" placeholder label.

        Args:
            painter (QPainter): The active painter.

        Returns:
            None
        """
        painter.setPen(QPen(TEXT_PRIMARY))
        label = "LEFT VIEW" if self.which == "L" else "RIGHT VIEW"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, label)

    def _paint_frame(self, painter):
        """Draw the cached frame, cropped/scaled to the current pan/zoom view.

        Direct port of the original `_display_bgr_on_canvas`'s ROI
        math - computes the same image-space crop rect (clamped to
        image bounds, so panning off the edges shows blank space
        rather than erroring), then lets `QPainter.drawImage`'s
        source/target rects do the actual scale+blit.

        Args:
            painter (QPainter): The active painter.

        Returns:
            None
        """
        dx, dy, dw, dh = self._display_rect()
        size = self._image_size()
        if size is None:
            return
        img_w, img_h = size

        scale = self._total_scale()
        if scale <= 0.0:
            scale = 1.0

        off_x = float(self.view["off_x"])
        off_y = float(self.view["off_y"])

        ix0 = (0.0 - off_x) / scale
        iy0 = (0.0 - off_y) / scale
        ix1 = (float(dw) - off_x) / scale
        iy1 = (float(dh) - off_y) / scale

        x0 = min(ix0, ix1)
        x1 = max(ix0, ix1)
        y0 = min(iy0, iy1)
        y1 = max(iy0, iy1)

        x0 = max(0.0, min(float(img_w), x0))
        x1 = max(0.0, min(float(img_w), x1))
        y0 = max(0.0, min(float(img_h), y0))
        y1 = max(0.0, min(float(img_h), y1))

        rx0, ry0 = int(x0), int(y0)
        rx1, ry1 = int(x1 + 0.9999), int(y1 + 0.9999)

        if rx1 <= rx0 or ry1 <= ry0:
            return

        screen_x0 = float(dx) + float(rx0) * scale + off_x
        screen_y0 = float(dy) + float(ry0) * scale + off_y
        out_w = max(1.0, (rx1 - rx0) * scale)
        out_h = max(1.0, (ry1 - ry0) * scale)

        source_rect = QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
        target_rect = QRectF(screen_x0, screen_y0, out_w, out_h)
        painter.drawImage(target_rect, self.current_qimage, source_rect)

    def _paint_overlay(self, painter):
        """Draw connecting lines, point handles, center dots, and index labels.

        Direct port of the original `draw_pane` - one continuous
        polyline through consecutive points, then per point a ring, a
        small fixed-color center dot, and an index label.

        Args:
            painter (QPainter): The active painter.

        Returns:
            None
        """
        pts = self.app.ptsL if self.which == "L" else self.app.ptsR
        color = RECTIFIED_OVERLAY_COLOR if self.app.view_rectified.get() else NOT_RECTIFIED_OVERLAY_COLOR
        radius = float(self.app.handle_radius_px)

        pen = QPen(color)
        pen.setWidth(2)

        if len(pts) >= 2:
            painter.setPen(pen)
            for i in range(len(pts) - 1):
                x0, y0 = self.image_to_screen(*pts[i])
                x1, y1 = self.image_to_screen(*pts[i + 1])
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        for index, (ix, iy) in enumerate(pts):
            sx, sy = self.image_to_screen(ix, iy)

            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(sx, sy), radius, radius)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(CENTER_DOT_COLOR)
            painter.drawEllipse(QPointF(sx, sy), CENTER_DOT_RADIUS_PX, CENTER_DOT_RADIUS_PX)

            painter.setPen(pen)
            painter.drawText(QPointF(sx + radius + 6, sy - radius - 6), str(index))

    # -------------------------------------------------------------------------
    # Hit testing
    # -------------------------------------------------------------------------

    def _nearest_handle_index(self, sx, sy):
        """Find the closest point handle within hit range of a screen point.

        Direct port of the original `get_nearest_handle_index` - hit
        radius is `2x` the visual handle radius, generous on purpose
        since the visual ring is meant for easy clicking.

        Args:
            sx (float): Local X coordinate to test.
            sy (float): Local Y coordinate to test.

        Returns:
            int | None: The closest in-range point's index, or None if
            no point is within range.
        """
        pts = self.app.ptsL if self.which == "L" else self.app.ptsR
        hit_r = float(self.app.handle_radius_px) * 2.0
        hit_r2 = hit_r * hit_r

        best_index = None
        best_dist2 = None
        for index, (ix, iy) in enumerate(pts):
            px, py = self.image_to_screen(ix, iy)
            dx = px - sx
            dy = py - sy
            dist2 = dx * dx + dy * dy
            if dist2 <= hit_r2 and (best_dist2 is None or dist2 < best_dist2):
                best_dist2 = dist2
                best_index = index

        return best_index

    # -------------------------------------------------------------------------
    # Mouse interaction
    # -------------------------------------------------------------------------

    def wheelEvent(self, event):
        """Zoom in/out, anchored so the point under the cursor stays put.

        Direct port of the original `on_mouse_wheel` - re-derives
        `off_x`/`off_y` from scratch each tick (not an incremental
        drift correction) so the same image point the cursor was over
        before the zoom change maps back to the same screen position
        after it.

        Args:
            event (QWheelEvent): The wheel event.

        Returns:
            None
        """
        if self._image_size() is None:
            return

        pos = event.position()
        ix, iy = self.screen_to_image(pos.x(), pos.y())

        if event.angleDelta().y() > 0:
            new_zoom = float(self.view["zoom"]) * float(self.app.zoom_step)
        else:
            new_zoom = float(self.view["zoom"]) / float(self.app.zoom_step)
        new_zoom = max(float(self.app.zoom_min), min(float(self.app.zoom_max), new_zoom))
        self.view["zoom"] = new_zoom

        dx, dy, _dw, _dh = self._display_rect()
        scale_new = self._total_scale()
        self.view["off_x"] = (float(pos.x()) - float(dx)) - float(ix) * scale_new
        self.view["off_y"] = (float(pos.y()) - float(dy)) - float(iy) * scale_new

        self.app.render_current_frames()
        self.app.tutorial_window.notify_action("pan_or_zoom")

    def keyPressEvent(self, event):
        """Step this pane's own timeline by one frame on Left/Right arrow.

        Scoped to whichever pane actually has keyboard focus (this
        widget uses `Qt.FocusPolicy.ClickFocus`, set in `__init__`) -
        typing in the sync-time entry boxes or the Offset spinner steals
        focus away from both panes, so this never fires while typing
        there. Respects Lock L and R exactly like the regular transport
        controls do - see `main.py`'s `on_step_forward_single_pane`/
        `on_step_back_single_pane` for why this only moves the *other*
        pane along too when Lock is actually on. When Lock is on but
        neither pane has focus at all, `main.py`'s
        `_build_lock_arrow_shortcuts` (a `QShortcut` pair, enabled only
        while locked) is what still steps both timelines together -
        this method only needs to cover the per-pane, possibly-unlocked
        case, and is never reached at all while those shortcuts are
        enabled (a `WindowShortcut` takes priority over a focused
        widget's own `keyPressEvent`).

        Args:
            event (QKeyEvent): The key press event.

        Returns:
            None
        """
        if event.key() == Qt.Key.Key_Right:
            self.app.on_step_forward_single_pane(self.which)
        elif event.key() == Qt.Key.Key_Left:
            self.app.on_step_back_single_pane(self.which)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Start point placement, point drag, panning, or point refinement.

        Left button: place a new point (in empty space) or start
        dragging an existing one (on a handle) - unchanged behavior.
        Middle button: start panning. Right button: start refining an
        existing point that already has a matched mate in the opposite
        pane - but if the press *isn't* on such a point (empty space,
        or a handle with no mate yet), start panning instead. This
        makes panning available without a middle mouse button, per the
        project owner's request, while leaving left-click's own
        placement/drag behavior untouched.

        Args:
            event (QMouseEvent): The press event.

        Returns:
            None
        """
        pos = event.position()
        pts = self.app.ptsL if self.which == "L" else self.app.ptsR

        if event.button() == Qt.MouseButton.LeftButton:
            index = self._nearest_handle_index(pos.x(), pos.y())
            if index is not None:
                self.drag_active = True
                self.drag_index = index
                return

            if len(pts) >= int(self.app.max_points_per_pane):
                return

            ix, iy = self.screen_to_image(pos.x(), pos.y())
            pts.append((ix, iy))

            self.app.on_points_changed()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self.pan_active = True
            self.pan_last_pos = (pos.x(), pos.y())

        elif event.button() == Qt.MouseButton.RightButton:
            other_pts = self.app.ptsR if self.which == "L" else self.app.ptsL
            index = self._nearest_handle_index(pos.x(), pos.y())
            if index is not None and index < len(pts) and index < len(other_pts):
                self.refine_drag_active = True
                self.refine_drag_index = index
            else:
                self.pan_active = True
                self.pan_last_pos = (pos.x(), pos.y())

    def mouseMoveEvent(self, event):
        """Continue an active pan, point drag, or point refinement.

        Args:
            event (QMouseEvent): The move event.

        Returns:
            None
        """
        pos = event.position()

        if self.pan_active:
            last_x, last_y = self.pan_last_pos
            self.view["off_x"] += float(pos.x() - last_x)
            self.view["off_y"] += float(pos.y() - last_y)
            self.pan_last_pos = (pos.x(), pos.y())
            # Cheap redisplay only - no re-decode, matches the original's
            # `_redisplay_current_frames` optimization for pan drags.
            self.app.redisplay_current_frames()
            self.app.tutorial_window.notify_action("pan_or_zoom")
            return

        if self.drag_active and self.drag_index is not None:
            pts = self.app.ptsL if self.which == "L" else self.app.ptsR
            if self.drag_index < len(pts):
                ix, iy = self.screen_to_image(pos.x(), pos.y())
                pts[self.drag_index] = (ix, iy)
                self.app.on_points_changed()
            return

        if self.refine_drag_active and self.refine_drag_index is not None:
            pts = self.app.ptsL if self.which == "L" else self.app.ptsR
            if self.refine_drag_index < len(pts):
                ix, iy = self.screen_to_image(pos.x(), pos.y())
                pts[self.refine_drag_index] = (ix, iy)
                self.app.on_points_changed()

    def mouseReleaseEvent(self, event):
        """End an active pan, point drag, or trigger scanline refinement.

        Tutorial completion detection for point placement happens in
        `main.py`'s `on_points_changed` (called from every branch below
        that changes `pts`/`other_pts`), not here - it's checked by
        count rather than tied to a specific gesture, so it fires the
        same way whether a point got there by a fresh click or a drag.

        The right-button release triggers scanline refinement only if
        the press actually started a refine drag (on a point with a
        matched mate); if the right-button press instead started
        panning (empty space, or a point with no mate), this just ends
        the pan like a middle-button release would.

        Reading the just-dragged point's current position and the
        opposite pane's mate, then running the scanline matcher from
        the *opposite* (unmoved) pane back toward this one, hinting the
        search near where the point was just dropped, is a direct port
        of the original `on_right_up`.

        Args:
            event (QMouseEvent): The release event.

        Returns:
            None
        """
        if event.button() == Qt.MouseButton.LeftButton and self.drag_active:
            self.drag_active = False
            self.drag_index = None
            self.app.on_points_changed()

        elif event.button() == Qt.MouseButton.MiddleButton and self.pan_active:
            self.pan_active = False
            self.pan_last_pos = None

        elif event.button() == Qt.MouseButton.RightButton and self.pan_active:
            self.pan_active = False
            self.pan_last_pos = None

        elif event.button() == Qt.MouseButton.RightButton and self.refine_drag_active:
            index = self.refine_drag_index
            self.refine_drag_active = False
            self.refine_drag_index = None

            pts = self.app.ptsL if self.which == "L" else self.app.ptsR
            other_pts = self.app.ptsR if self.which == "L" else self.app.ptsL
            other_which = "R" if self.which == "L" else "L"

            if index is not None and index < len(pts) and index < len(other_pts):
                x_cur, y_cur = pts[index]
                x_other, y_other = other_pts[index]
                refined = stereo_matching.guess_mate_point_on_scanline(
                    self.app,
                    other_which,
                    x_other,
                    y_other,
                    x_hint=x_cur,
                    search_half_width=8,
                )
                if refined is not None:
                    pts[index] = refined

            self.app.on_points_changed()
