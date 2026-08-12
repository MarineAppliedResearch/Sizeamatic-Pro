"""Video overlay system for Sizeamatic Pro.

This module creates and manages the overlay canvases that sit above the
left and right video panes. It handles drawing measurement points, point
labels, connecting line segments, and mouse interaction for placing,
dragging, and refining stereo measurement points.

Contents:
    - `VideoOverlay` — owns the overlay canvases and overlay interaction
      state.
    - `get_handle_index_under_cursor` — standalone canvas-tag hit test, no
      state needed, independently testable.

Design notes:
    `VideoOverlay` is a plain class instance owned by the main application
    (`app.video_overlay`) — the same conversion already done for
    `calibration_summary.CalibrationSummaryWindow`,
    `measurement_window.MeasurementWindow`, and
    `anaglyph_preview.AnaglyphPreview`. This was the last module still
    using module-level globals for its state (`drag_active`,
    `left_overlay_canvas`, etc.).

    Note the construction-order constraint this creates: `create_canvases`
    is called from inside `main.py`'s `_build_viewers`, so
    `SizeamaticProApp.__init__` must construct `self.video_overlay` before
    calling `_build_viewers` — earlier than the other three classes, which
    only need to exist before their own `ensure_window`/`start` is first
    called by user action.

    The main application owns the video viewer layout and the actual point
    data. This class owns the overlay canvas widgets and overlay
    interaction state.

    Image points are stored in image pixel coordinates. Overlay drawing
    converts those image coordinates to screen coordinates so handles and
    line segments remain aligned with the displayed video frame.

    `draw_pane` draws a connecting line between every consecutive pair of
    points in the pane (0-1, 1-2, 2-3, ...), i.e. one continuous polyline
    through all placed points. This intentionally matches how `main.py`'s
    `_update_measurement_status_stub` computes `seg_rows` (also
    consecutive pairs), so a polyline with more than 2 points is measured
    as a chain of segments, not independent pairs.

Assumptions:
    - The main application provides left and right viewport frames before
      `create_canvases` is called.
    - The main application stores measurement point lists as `app.ptsL`
      and `app.ptsR`.
    - The main application provides coordinate conversion helpers for
      mapping between image coordinates and overlay canvas coordinates.
    - The overlay is a visual and interaction layer only. It does not own
      video frame rendering or stereo measurement math.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# tkinter provides the overlay canvases used for point drawing and mouse input.
import tkinter as tk

# stereo_matching provides scanline based mate point guessing for local point
# refinement.
import stereo_matching


CENTER_DOT_RADIUS_PX = 2

# Ring/line/label color while in rectified view (measurements are meaningful)
# versus raw view (they aren't - real-world sizes only come out right against
# rectified pixels). The small red center dot deliberately stays red in both
# modes - it exists to pinpoint the exact clicked pixel, a purpose unrelated
# to rectification state.
RECTIFIED_OVERLAY_COLOR = "#00ff66"
NOT_RECTIFIED_OVERLAY_COLOR = "#ffa500"
"""Screen-pixel radius of the small solid dot drawn at each point handle's
exact center (see `VideoOverlay.draw_pane`). Deliberately much smaller than
`app.handle_radius_px`'s ring — the ring is sized for easy clicking, this is
sized to pinpoint exactly where the point actually landed. Fixed in screen
pixels regardless of zoom, matching the ring's own radius (ROADMAP.md
Phase 8's point visibility item)."""


class VideoOverlay:
    """Owns the left/right overlay canvases and overlay interaction state.

    One instance lives on the main application (`app.video_overlay`),
    constructed early in `SizeamaticProApp.__init__` — before
    `_build_viewers` runs, since that method calls `create_canvases`.
    """

    def __init__(self, app):
        """Store the owning app and initialize overlay state to defaults.

        Args:
            app: The main application object, used for viewport frames,
                point lists, viewer settings, coordinate conversion
                helpers, and measurement refresh behavior.

        Returns:
            None
        """
        self.app = app

        self.left_canvas = None
        """The transparent overlay canvas stacked on top of the left
        video pane, used for drawing measurement points/handles/lines and
        for capturing mouse input. Set by `create_canvases`; other
        methods treat `None` as "not built yet, nothing safe to draw on
        or interact with"."""

        self.right_canvas = None
        """The transparent overlay canvas stacked on top of the right
        video pane. Mirrors `self.left_canvas` for the right side."""

        self.drag_active = False
        """Whether a left-button point handle drag is currently active
        (the "move an existing point" gesture, as opposed to placing a
        brand new one). Set in `on_left_down`, cleared in `on_left_up`."""

        self.drag_which = None
        """Which pane, "L" or "R", owns the point currently being
        dragged. Used so `on_left_drag`/`on_left_up` events from the
        *other* pane don't get misapplied to a drag that started
        elsewhere."""

        self.drag_index = None
        """Index of the point currently being dragged, within that
        pane's point list. `None` when no drag is active."""

        self.refine_drag_active = False
        """Whether an explicit right-button refinement drag is active.
        Distinct from `drag_active`: refinement only starts on an
        existing point that already has a matched point on the opposite
        pane (see `on_right_down`), and on release runs the scanline
        matcher to snap to a nearby feature rather than just placing the
        point wherever the mouse was."""

        self.refine_drag_which = None
        """Which pane, "L" or "R", owns the point currently being
        refined."""

        self.refine_drag_index = None
        """Index of the point currently being refined, within that
        pane's point list. `None` when no refine drag is active."""

    def create_canvases(self):
        """Create the left/right overlay canvases and bind their mouse events.

        Creates the overlay canvases used for point drawing and point
        interaction over the video panes. The main app owns the viewer
        layout, while this class owns the overlay canvas widgets and
        their mouse input bindings.

        Returns:
            None
        """

        app = self.app

        # Create the left overlay canvas on top of the left video viewport.
        self.left_canvas = tk.Canvas(
            app.left_viewport,
            bg="black",
            highlightthickness=0,
            bd=0,
        )

        # Make the left overlay canvas cover the left video viewport exactly.
        self.left_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

        # Create the right overlay canvas on top of the right video viewport.
        self.right_canvas = tk.Canvas(
            app.right_viewport,
            bg="black",
            highlightthickness=0,
            bd=0,
        )

        # Make the right overlay canvas cover the right video viewport exactly.
        self.right_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

        # When the left overlay canvas size changes, redraw the current frames through
        # the app resize handler.
        self.left_canvas.bind("<Configure>", app.on_canvas_resized)

        # When the right overlay canvas size changes, redraw the current frames through
        # the app resize handler.
        self.right_canvas.bind("<Configure>", app.on_canvas_resized)

        # Left overlay canvas receives manual left button input.
        self.left_canvas.bind("<Button-1>", lambda e: self.on_left_down("L", e))
        self.left_canvas.bind("<B1-Motion>", lambda e: self.on_left_drag("L", e))
        self.left_canvas.bind("<ButtonRelease-1>", lambda e: self.on_left_up("L", e))

        # Left overlay canvas also receives explicit right button refine input.
        self.left_canvas.bind("<Button-3>", lambda e: self.on_right_down("L", e))
        self.left_canvas.bind("<B3-Motion>", lambda e: self.on_right_drag("L", e))
        self.left_canvas.bind("<ButtonRelease-3>", lambda e: self.on_right_up("L", e))

        # Right overlay canvas receives manual left button input.
        self.right_canvas.bind("<Button-1>", lambda e: self.on_left_down("R", e))
        self.right_canvas.bind("<B1-Motion>", lambda e: self.on_left_drag("R", e))
        self.right_canvas.bind("<ButtonRelease-1>", lambda e: self.on_left_up("R", e))

        # Right overlay canvas also receives explicit right button refine input.
        self.right_canvas.bind("<Button-3>", lambda e: self.on_right_down("R", e))
        self.right_canvas.bind("<B3-Motion>", lambda e: self.on_right_drag("R", e))
        self.right_canvas.bind("<ButtonRelease-3>", lambda e: self.on_right_up("R", e))

        # Mouse wheel zoom for each pane. Windows uses <MouseWheel> with event.delta.
        self.left_canvas.bind("<MouseWheel>", lambda e: app.on_mouse_wheel("L", e))
        self.right_canvas.bind("<MouseWheel>", lambda e: app.on_mouse_wheel("R", e))

        # Middle-mouse-button drag pans each pane. A separate button from point
        # placement (left) and explicit point refinement (right) so panning never
        # collides with either — see main.py's `on_pan_down` docstring.
        self.left_canvas.bind("<Button-2>", lambda e: app.on_pan_down("L", e))
        self.left_canvas.bind("<B2-Motion>", lambda e: app.on_pan_drag("L", e))
        self.left_canvas.bind("<ButtonRelease-2>", lambda e: app.on_pan_up("L", e))

        self.right_canvas.bind("<Button-2>", lambda e: app.on_pan_down("R", e))
        self.right_canvas.bind("<B2-Motion>", lambda e: app.on_pan_drag("R", e))
        self.right_canvas.bind("<ButtonRelease-2>", lambda e: app.on_pan_up("R", e))

    def get_pane_scale(self, which, canvas):
        """Compute the image-to-screen scale factor for one video pane.

        Computes the scale used to draw image coordinate overlays on top
        of the video pane. When fit to window is disabled, image pixels
        map directly to screen pixels. When fit to window is enabled, the
        image is scaled by canvas width only, matching the current video
        display behavior.

        Args:
            which (str): Which pane to compute the scale for, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas being measured.

        Returns:
            float: The image-to-screen scale factor for the selected pane.
        """

        app = self.app

        # If fit to window is disabled, use native image pixel mapping.
        if not app.fit_to_window.get():
            return 1.0

        # Select the source image width for the requested pane.
        if which == "L":

            # If left video metadata is not available, fall back to native scale.
            if not app.metaL:
                return 1.0

            # Read the left source image width.
            src_w = float(app.metaL["width"])

        else:

            # If right video metadata is not available, fall back to native scale.
            if not app.metaR:
                return 1.0

            # Read the right source image width.
            src_w = float(app.metaR["width"])

        # Read the current canvas width. Clamp to at least 1 to avoid division by zero.
        canvas_w = float(max(1, canvas.winfo_width()))

        # Match the current fit to window behavior by scaling from width only.
        return canvas_w / src_w

    def redraw(self):
        """Redraw the measurement point overlays for both video panes.

        Clears and redraws the measurement point overlays for the left
        and right video panes. The overlay canvases are owned by this
        instance, while the point lists still come from the main
        application state.

        Returns:
            None
        """

        # If either overlay canvas has not been registered yet, there is nothing safe
        # to redraw.
        if self.left_canvas is None or self.right_canvas is None:
            return

        # Draw the left pane overlay using the current left image point list.
        self.draw_pane("L", self.left_canvas, self.app.ptsL)

        # Draw the right pane overlay using the current right image point list.
        self.draw_pane("R", self.right_canvas, self.app.ptsR)

    def get_canvas(self, which):
        """Look up the overlay canvas for a given video pane.

        Keeping this lookup in one place avoids repeating left/right
        canvas selection logic throughout the overlay mouse handlers.

        Args:
            which (str): Which pane's canvas to return, "L" or "R".

        Returns:
            tkinter.Canvas | None: The matching overlay canvas, or None if
            the pane identifier is invalid or the canvas has not been
            created.
        """

        # Return the left overlay canvas for the left pane.
        if which == "L":
            return self.left_canvas

        # Return the right overlay canvas for the right pane.
        if which == "R":
            return self.right_canvas

        # Unknown pane identifier.
        return None

    def on_left_down(self, which, event):
        """Handle a left mouse button press on an overlay canvas.

        If the click hits an existing point handle, enters drag mode for
        that point. If the click lands on empty overlay space, adds a new
        image space point to the clicked pane and optionally creates an
        initial stereo mate guess on the opposite pane.

        Args:
            which (str): Which pane was clicked, "L" or "R".
            event (tkinter.Event): The Tkinter mouse event, in overlay
                canvas coordinates.

        Returns:
            None
        """

        app = self.app

        # Choose the overlay canvas for this pane.
        canvas = self.get_canvas(which)

        # If the requested overlay canvas does not exist, there is nothing safe to do.
        if canvas is None:
            return

        # First try the exact canvas item hit test under the cursor.
        idx = get_handle_index_under_cursor(canvas)

        # If the exact item hit test fails, fall back to a nearest handle search in
        # screen space so points are easier to grab.
        if idx is None:
            idx = self.get_nearest_handle_index(which, canvas, event.x, event.y)

        # If a point handle was found, begin dragging that point.
        if idx is not None:
            self.drag_active = True
            self.drag_which = which
            self.drag_index = idx
            return

        # Otherwise, treat this click as a request to place a new point.
        pts = self.get_points_list(which)

        # If the point list could not be found, do not continue.
        if pts is None:
            return

        # If this pane is already at the point cap, ignore empty space clicks.
        if len(pts) >= int(app.max_points_per_pane):
            return

        # Convert the click from overlay canvas screen coordinates into image pixel
        # coordinates.
        ix, iy = app._screen_to_image(which, canvas, event.x, event.y)

        # Record the new point on the pane the user clicked.
        pts.append((ix, iy))

        # Remember the new point index so the opposite pane can receive the same
        # logical point pair index.
        new_idx = len(pts) - 1

        # Select the opposite pane's point list.
        other_which = "R" if which == "L" else "L"
        other_pts = self.get_points_list(other_which)

        # If the opposite point list is unavailable, update the overlay with the point
        # we did add and then exit.
        if other_pts is None:
            self.on_points_changed()
            return

        # Only auto create a mate if the opposite pane does not already have a point
        # at this pair index.
        if new_idx >= len(other_pts):

            # Place the initial mate at the exact same image pixel coordinates as
            # the point just clicked, rather than an automated scanline-matcher
            # guess (which the project owner found unhelpful in practice — see
            # ROADMAP.md Phase 7's usability quiz). Points are stored in image
            # pixel coordinates, and each pane's own draw pipeline
            # (_image_to_screen) already applies that pane's current zoom/pan
            # independently, so reusing (ix, iy) as-is lands correctly on-screen
            # in the opposite pane regardless of the two panes' current
            # zoom/pan state — no extra transform needed. The user drags it into
            # place manually, or right-click-drags it to trigger the scanline
            # matcher explicitly (on_right_up) if they want that assist.
            other_pts.append((ix, iy))

        # Redraw overlays and update measurement status after the point change.
        self.on_points_changed()

    def on_left_drag(self, which, event):
        """Handle mouse movement while dragging a point handle (left button).

        If a point handle drag is active for the requested pane, the
        cursor position is converted from screen coordinates to image
        coordinates and written back into the matching point list.

        Args:
            which (str): Which pane the drag event is for, "L" or "R".
            event (tkinter.Event): The Tkinter mouse event, in overlay
                canvas coordinates.

        Returns:
            None
        """

        # Only drag if a point handle drag is currently active.
        if not self.drag_active:
            return

        # Ignore drag events from the opposite pane.
        if self.drag_which != which:
            return

        # Choose the overlay canvas for this pane.
        canvas = self.get_canvas(which)

        # If the requested overlay canvas does not exist, there is nothing safe to do.
        if canvas is None:
            return

        # Get the point list for the pane currently being dragged.
        pts = self.get_points_list(which)

        # If the point list could not be found, do not continue.
        if pts is None:
            return

        # Validate that a drag index has been assigned.
        if self.drag_index is None:
            return

        # Validate that the drag index still points to an existing point.
        if self.drag_index < 0 or self.drag_index >= len(pts):
            return

        # Convert the current mouse position from overlay canvas coordinates into
        # image pixel coordinates.
        ix, iy = self.app._screen_to_image(which, canvas, event.x, event.y)

        # Update the dragged point in the pane's point list.
        pts[self.drag_index] = (ix, iy)

        # Redraw overlays and update measurement status continuously while dragging.
        self.on_points_changed()

    def on_left_up(self, which, _event):
        """Handle release of the left mouse button after a point handle drag.

        The drag only ends if the active drag belongs to the pane that
        received the release event.

        Args:
            which (str): Which pane received the release event, "L" or
                "R".
            _event (tkinter.Event): The Tkinter mouse event (unused).

        Returns:
            None
        """

        # Only finish a drag if this pane owns the active drag.
        if not self.drag_active or self.drag_which != which:
            return

        # Clear drag state now that the drag is finished.
        self.drag_active = False
        self.drag_which = None
        self.drag_index = None

        # Redraw overlays and recompute measurements using the user placed point.
        self.on_points_changed()

    def on_right_down(self, which, event):
        """Handle a right mouse button press on an overlay canvas.

        Right button input is used for explicit refinement, so it only
        starts dragging when the user clicks an existing point handle
        that already has a corresponding mate point on the opposite pane.

        Args:
            which (str): Which pane was clicked, "L" or "R".
            event (tkinter.Event): The Tkinter mouse event, in overlay
                canvas coordinates.

        Returns:
            None
        """

        # Choose the overlay canvas for this pane.
        canvas = self.get_canvas(which)

        # If the requested overlay canvas does not exist, there is nothing safe to do.
        if canvas is None:
            return

        # First try the exact canvas item hit test under the cursor.
        idx = get_handle_index_under_cursor(canvas)

        # If the exact item hit test fails, fall back to a nearest handle search in
        # screen space so points are easier to grab.
        if idx is None:
            idx = self.get_nearest_handle_index(which, canvas, event.x, event.y)

        # Right button refine mode only starts when an existing handle was selected.
        if idx is None:
            return

        # Get the point list for the clicked pane.
        pts = self.get_points_list(which)

        # Get the point list for the opposite pane.
        other_which = "R" if which == "L" else "L"
        other_pts = self.get_points_list(other_which)

        # If either point list is unavailable, do not enter refine mode.
        if pts is None or other_pts is None:
            return

        # Only refine points that exist in both panes at the same paired index.
        if idx >= len(pts) or idx >= len(other_pts):
            return

        # Begin explicit refine drag mode for this paired point.
        self.refine_drag_active = True
        self.refine_drag_which = which
        self.refine_drag_index = idx

    def on_right_drag(self, which, event):
        """Handle mouse movement while explicitly refining a point (right button).

        Right button dragging is explicit refinement mode: the selected
        point is moved manually in image coordinates while overlays and
        measurement output update continuously.

        Args:
            which (str): Which pane the refine drag is for, "L" or "R".
            event (tkinter.Event): The Tkinter mouse event, in overlay
                canvas coordinates.

        Returns:
            None
        """

        # Only drag if a refine drag is currently active.
        if not self.refine_drag_active:
            return

        # Ignore drag events from the opposite pane.
        if self.refine_drag_which != which:
            return

        # Choose the overlay canvas for this pane.
        canvas = self.get_canvas(which)

        # If the requested overlay canvas does not exist, there is nothing safe to do.
        if canvas is None:
            return

        # Get the point list for the pane currently being refined.
        pts = self.get_points_list(which)

        # If the point list could not be found, do not continue.
        if pts is None:
            return

        # Validate that a refine drag index has been assigned.
        if self.refine_drag_index is None:
            return

        # Validate that the refine drag index still points to an existing point.
        if self.refine_drag_index < 0 or self.refine_drag_index >= len(pts):
            return

        # Convert the current mouse position from overlay canvas coordinates into
        # image pixel coordinates.
        ix, iy = self.app._screen_to_image(which, canvas, event.x, event.y)

        # Update the refined point in the pane's point list.
        pts[self.refine_drag_index] = (ix, iy)

        # Redraw overlays and update measurement preview continuously while dragging.
        self.on_points_changed()

    def on_right_up(self, which, _event):
        """Handle release of the right mouse button after an explicit refine drag.

        The point is first manually positioned during the drag. On
        release, the stereo matcher is run in a narrow local search
        window near the user placed X position so the point can be
        snapped to a nearby matching feature without jumping far away
        from the user's intended placement.

        Args:
            which (str): Which pane received the release event, "L" or
                "R".
            _event (tkinter.Event): The Tkinter mouse event (unused).

        Returns:
            None
        """

        # Only finish a refine drag if this pane owns the active refine drag.
        if not self.refine_drag_active or self.refine_drag_which != which:
            return

        # Keep the point index before clearing refine drag state.
        idx = self.refine_drag_index

        # Refine only if there is still a valid point index.
        if idx is not None:

            # Get the point list for the pane being refined.
            pts = self.get_points_list(which)

            # Get the point list for the opposite pane.
            other_which = "R" if which == "L" else "L"
            other_pts = self.get_points_list(other_which)

            # Only refine if both point lists exist.
            if pts is not None and other_pts is not None:

                # Only refine if this paired point still exists in both panes.
                if idx < len(pts) and idx < len(other_pts):

                    # Read the current user placed point being refined.
                    x_cur, y_cur = pts[idx]

                    # Read the already paired mate point on the opposite pane.
                    x_other, y_other = other_pts[idx]

                    # Run the scanline matcher from the opposite pane back toward this
                    # pane. The x_hint keeps the search local to the user's placement.
                    refined = stereo_matching.guess_mate_point_on_scanline(
                        self.app,
                        other_which,
                        x_other,
                        y_other,
                        x_hint=x_cur,
                        search_half_width=8,
                    )

                    # If refinement succeeded, replace the user placed point with the
                    # locally refined result.
                    if refined is not None:
                        pts[idx] = refined

        # Clear refine drag state now that the gesture is complete.
        self.refine_drag_active = False
        self.refine_drag_which = None
        self.refine_drag_index = None

        # Redraw overlays and recompute measurements using the refined point.
        self.on_points_changed()

    def on_points_changed(self):
        """Refresh overlays and measurement status after a point change.

        Handles the common follow up work after overlay points are added,
        moved, refined, or cleared. Keeping this as the single point
        change hook makes it less likely that one mouse path updates the
        overlay but forgets to refresh measurement feedback.

        Returns:
            None
        """

        # Redraw the current point overlays for both video panes.
        self.redraw()

        # Update measurement status text and any measurement preview behavior.
        self.app._update_measurement_status_stub()

    def get_points_list(self, which):
        """Look up the image-coordinate point list for a given video pane.

        Provides one shared left/right point list lookup for overlay
        drawing and mouse interaction code. The point data still lives on
        the main app object.

        Args:
            which (str): Which pane's point list to return, "L" or "R".

        Returns:
            list[tuple[float, float]] | None: The point list for the
            requested pane, or None if the pane identifier is invalid.
        """

        # Return the left image point list for the left pane.
        if which == "L":
            return self.app.ptsL

        # Return the right image point list for the right pane.
        if which == "R":
            return self.app.ptsR

        # Unknown pane identifier.
        return None

    def get_nearest_handle_index(self, which, canvas, sx, sy):
        """Find the nearest point handle within a generous hit radius.

        Provides a forgiving fallback hit test when the exact Tkinter
        canvas item hit test misses. Each point is converted from image
        coordinates to screen coordinates, then compared against the
        mouse click using a generous hit radius.

        Args:
            which (str): Which pane to search, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for the pane being
                searched.
            sx (float): Click X position, in overlay canvas screen
                coordinates.
            sy (float): Click Y position, in overlay canvas screen
                coordinates.

        Returns:
            int | None: The nearest point index if the click is close
            enough to a handle, or None if no handle is within the hit
            radius.
        """

        app = self.app

        # Read the point list for this pane.
        pts = self.get_points_list(which)

        # If there are no points for this pane, there is no handle to find.
        if not pts:
            return None

        # Use a generous hit radius so handles are easier to grab than their exact
        # drawn oval outline.
        hit_r = float(app.handle_radius_px) * 2.0
        hit_r2 = hit_r * hit_r

        # Track the closest handle found inside the hit radius.
        best_idx = None
        best_d2 = None

        # Compare the click against each handle center in screen coordinates.
        for i, (ix, iy) in enumerate(pts):

            # Convert this point from image pixel coordinates to overlay screen
            # coordinates.
            hx, hy = app._image_to_screen(which, canvas, ix, iy)

            # Compute squared screen space distance from the click to the handle.
            dx = float(sx) - float(hx)
            dy = float(sy) - float(hy)
            d2 = dx * dx + dy * dy

            # Keep the closest handle that falls inside the hit radius.
            if d2 <= hit_r2:
                if best_d2 is None or d2 < best_d2:
                    best_idx = i
                    best_d2 = d2

        # Return the closest nearby handle index, or None if none was close enough.
        return best_idx

    def draw_pane(self, which, canvas, pts):
        """Draw the measurement overlay (points, handles, labels, lines) for one pane.

        Points are stored in image pixel coordinates, then converted into
        screen coordinates so handles and connecting segments line up
        with the displayed video frame. Handles are tagged for later
        mouse hit testing and dragging.

        Args:
            which (str): Which pane is being drawn, "L" or "R".
            canvas (tkinter.Canvas): The overlay canvas for the pane being
                drawn.
            pts (list[tuple[float, float]]): The pane's image-coordinate
                point list.

        Returns:
            None
        """

        app = self.app

        # Rectified measurements are real-world-accurate; raw ones aren't, so
        # give the ring/line/label a visibly different color in raw view
        # (ROADMAP.md Phase 8's rectified/not-rectified indicator item).
        overlay_color = RECTIFIED_OVERLAY_COLOR if app.view_rectified.get() else NOT_RECTIFIED_OVERLAY_COLOR

        # Clear only overlay tagged items so the canvas can be redrawn from current
        # point data without affecting unrelated canvas content.
        canvas.delete("overlay")

        # Draw line segments first so point handles and labels appear on top.
        # For N points, draw segment 0 to 1, 1 to 2, and so on.
        if len(pts) >= 2:
            for i in range(1, len(pts)):

                # Read the previous and current point in image pixel coordinates.
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]

                # Convert both segment endpoints into overlay screen coordinates.
                sx0, sy0 = app._image_to_screen(which, canvas, x0, y0)
                sx1, sy1 = app._image_to_screen(which, canvas, x1, y1)

                # Draw the segment connecting these two measurement points.
                canvas.create_line(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    width=2,
                    fill=overlay_color,
                    tags=("overlay",),
                )

        # Read the point handle radius in screen pixels.
        r = int(app.handle_radius_px)

        # Draw each point handle and its index label.
        for i, (x, y) in enumerate(pts):

            # Convert the point from image pixel coordinates to overlay screen
            # coordinates.
            sx, sy = app._image_to_screen(which, canvas, x, y)

            # Draw the draggable point handle. The "handle" tag marks this as a point
            # handle, and the "idx:<n>" tag stores which point index it represents.
            canvas.create_oval(
                sx - r,
                sy - r,
                sx + r,
                sy + r,
                outline=overlay_color,
                width=2,
                fill="",
                tags=("overlay", "handle", f"idx:{i}"),
            )

            # Draw a small solid dot exactly at the point's center. The ring alone
            # doesn't pinpoint the exact clicked/dragged pixel — this does, and
            # (like the ring's own radius) stays a fixed screen-pixel size
            # regardless of zoom, deliberately not tagged "handle" so it stays
            # purely visual and doesn't change hit-testing.
            canvas.create_oval(
                sx - CENTER_DOT_RADIUS_PX,
                sy - CENTER_DOT_RADIUS_PX,
                sx + CENTER_DOT_RADIUS_PX,
                sy + CENTER_DOT_RADIUS_PX,
                outline="",
                fill="#ff0000",
                tags=("overlay",),
            )

            # Draw the point index label near the handle.
            canvas.create_text(
                sx + r + 6,
                sy - r - 6,
                text=str(i),
                fill=overlay_color,
                font=("Segoe UI", 11, "bold"),
                tags=("overlay",),
            )


def get_handle_index_under_cursor(canvas):
    """Detect the point handle directly under the cursor, if any.

    Uses Tkinter canvas item tags to detect whether the mouse is over a
    drawn point handle. Handles are expected to have a "handle" tag and an
    "idx:<n>" tag that stores the point index.

    Args:
        canvas (tkinter.Canvas): The overlay canvas receiving the mouse
            event.

    Returns:
        int | None: The integer point index for the handle currently under
        the cursor, or None if the current canvas item is not a point
        handle.
    """

    # "current" is the Tkinter canvas item under the mouse pointer at event time.
    items = canvas.find_withtag("current")

    # If there is no current canvas item, the cursor is not over a handle.
    if not items:
        return None

    # Use the first item under the cursor.
    item_id = items[0]

    # Read the item's tags so we can identify handle items and point indexes.
    tags = canvas.gettags(item_id)

    # Only treat this canvas item as a draggable point if it has the handle tag.
    if "handle" not in tags:
        return None

    # Search for an index tag formatted like "idx:0", "idx:1", etc.
    for t in tags:

        # Ignore unrelated tags.
        if not t.startswith("idx:"):
            continue

        # Parse the point index from the tag.
        try:
            return int(t.split(":", 1)[1])

        # If the tag was malformed, treat this as no valid handle index.
        except ValueError:
            return None

    # No index tag was found.
    return None
