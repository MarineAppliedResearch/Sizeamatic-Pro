# -----------------------------------------------------------------------------
# video_overlay.py
#
# Author: Isaac Travers
# Created: 2026-05-18
# Project: Sizeamatic Pro
#
# Purpose:
#   Provides the video overlay system for Sizeamatic Pro.
#
#   This module creates and manages the overlay canvases that sit above the left
#   and right video panes. It handles drawing measurement points, point labels,
#   connecting line segments, and mouse interaction for placing, dragging, and
#   refining stereo measurement points.
#
# Contents:
#   - Left and right overlay canvas creation.
#   - Overlay canvas mouse event binding.
#   - Overlay redraw logic for points, handles, labels, and line segments.
#   - Point handle hit testing.
#   - Point list lookup helpers.
#   - Left button point placement and dragging.
#   - Right button point refinement.
#   - Shared point change handling.
#
# Design Notes:
#   The main application owns the video viewer layout and the actual point data.
#   This module owns the overlay canvas widgets and overlay interaction state.
#
#   Functions in this module receive the main application object when they need
#   access to point lists, viewer settings, coordinate conversion helpers,
#   measurement refresh behavior, or status updates.
#
#   Image points are stored in image pixel coordinates. Overlay drawing converts
#   those image coordinates to screen coordinates so handles and line segments
#   remain aligned with the displayed video frame.
#
# Assumptions:
#   - The main application provides left and right viewport frames before overlay
#     canvases are created.
#   - The main application stores measurement point lists as app.ptsL and
#     app.ptsR.
#   - The main application provides coordinate conversion helpers for mapping
#     between image coordinates and overlay canvas coordinates.
#   - The overlay is a visual and interaction layer only. It does not own video
#     frame rendering or stereo measurement math.
#
# Dependencies:
#   - tkinter provides the overlay Canvas widgets.
#   - stereo_matching provides scanline based mate point guessing for automatic
#     correspondence suggestions and local point refinement.
# -----------------------------------------------------------------------------


# tkinter provides the overlay canvases used for point drawing and mouse input.
import tkinter as tk

# stereo_matching provides scanline based mate point guessing for local point
# refinement.
import stereo_matching

# Module level overlay canvas state.

# Stores the transparent overlay canvas for the left video pane.
left_overlay_canvas = None

# Stores the transparent overlay canvas for the right video pane.
right_overlay_canvas = None

# Tracks whether a point handle drag is currently active.
drag_active = False

# Stores which pane owns the point currently being dragged: "L" or "R".
drag_which = None

# Stores the point index currently being dragged.
drag_index = None

# Tracks whether an explicit right button refinement drag is active.
refine_drag_active = False

# Stores which pane owns the point currently being refined: "L" or "R".
refine_drag_which = None

# Stores the point index currently being refined.
refine_drag_index = None


# -----------------------------------------------------------------------------
# set_overlay_canvases
#
# Inputs: left_canvas is the overlay canvas for the left video pane, and
# right_canvas is the overlay canvas for the right video pane.
# Outputs: stores both overlay canvas references in this module; returns nothing.
#
# Registers the overlay canvases used for drawing and editing measurement points.
# The canvases are created by the main viewer layout code, but the references are
# stored here so overlay drawing and mouse handling stay grouped in this module.
# -----------------------------------------------------------------------------
def set_overlay_canvases(left_canvas, right_canvas):

    # Use module level canvas references owned by the video overlay module.
    global left_overlay_canvas
    global right_overlay_canvas

    # Store the left and right overlay canvases for later redraw and event logic.
    left_overlay_canvas = left_canvas
    right_overlay_canvas = right_canvas

# -----------------------------------------------------------------------------
# create_overlay_canvases
#
# Inputs: app provides the left and right viewport frames and overlay event
# handlers needed by the overlay canvases.
# Outputs: creates the left and right overlay canvases, stores them in this
# module, binds overlay events, and returns nothing.
#
# Creates the overlay canvases used for point drawing and point interaction over
# the video panes. The main app owns the viewer layout, while this module owns the
# overlay canvas widgets and their mouse input bindings.
# -----------------------------------------------------------------------------
def create_overlay_canvases(app):

    # Use module level canvas references owned by the video overlay module.
    global left_overlay_canvas
    global right_overlay_canvas

    # Create the left overlay canvas on top of the left video viewport.
    left_overlay_canvas = tk.Canvas(
        app.left_viewport,
        bg="black",
        highlightthickness=0,
        bd=0,
    )

    # Make the left overlay canvas cover the left video viewport exactly.
    left_overlay_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

    # Create the right overlay canvas on top of the right video viewport.
    right_overlay_canvas = tk.Canvas(
        app.right_viewport,
        bg="black",
        highlightthickness=0,
        bd=0,
    )

    # Make the right overlay canvas cover the right video viewport exactly.
    right_overlay_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

    # When the left overlay canvas size changes, redraw the current frames through
    # the app resize handler.
    left_overlay_canvas.bind("<Configure>", app.on_canvas_resized)

    # When the right overlay canvas size changes, redraw the current frames through
    # the app resize handler.
    right_overlay_canvas.bind("<Configure>", app.on_canvas_resized)

    # Left overlay canvas receives manual left button input.
    left_overlay_canvas.bind("<Button-1>", lambda e: on_overlay_left_down(app, "L", e))
    left_overlay_canvas.bind("<B1-Motion>", lambda e: on_overlay_left_drag(app, "L", e))
    left_overlay_canvas.bind("<ButtonRelease-1>", lambda e: on_overlay_left_up(app, "L", e))

    # Left overlay canvas also receives explicit right button refine input.
    left_overlay_canvas.bind("<Button-3>", lambda e: on_overlay_right_down(app, "L", e))
    left_overlay_canvas.bind("<B3-Motion>", lambda e: on_overlay_right_drag(app, "L", e))
    left_overlay_canvas.bind("<ButtonRelease-3>", lambda e: on_overlay_right_up(app, "L", e))

    # Right overlay canvas receives manual left button input.
    right_overlay_canvas.bind("<Button-1>", lambda e: on_overlay_left_down(app, "R", e))
    right_overlay_canvas.bind("<B1-Motion>", lambda e: on_overlay_left_drag(app, "R", e))
    right_overlay_canvas.bind("<ButtonRelease-1>", lambda e: on_overlay_left_up(app, "R", e))

    # Right overlay canvas also receives explicit right button refine input.
    right_overlay_canvas.bind("<Button-3>", lambda e: on_overlay_right_down(app, "R", e))
    right_overlay_canvas.bind("<B3-Motion>", lambda e: on_overlay_right_drag(app, "R", e))
    right_overlay_canvas.bind("<ButtonRelease-3>", lambda e: on_overlay_right_up(app, "R", e))

    # Mouse wheel zoom for each pane. Windows uses <MouseWheel> with event.delta.
    left_overlay_canvas.bind("<MouseWheel>", lambda e: app.on_mouse_wheel("L", e))
    right_overlay_canvas.bind("<MouseWheel>", lambda e: app.on_mouse_wheel("R", e))

# -----------------------------------------------------------------------------
# get_pane_scale
#
# Inputs: app provides the fit to window setting and video metadata, which selects
# the left or right pane, and canvas is the Tkinter canvas being measured.
# Outputs: returns the current image to screen scale factor for the selected pane.
#
# Computes the scale used to draw image coordinate overlays on top of the video
# pane. When fit to window is disabled, image pixels map directly to screen
# pixels. When fit to window is enabled, the image is scaled by canvas width only,
# matching the current video display behavior.
# -----------------------------------------------------------------------------
def get_pane_scale(app, which, canvas):

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

# -----------------------------------------------------------------------------
# redraw_overlays
#
# Inputs: app provides the current left and right measurement point lists.
# Outputs: redraws overlay graphics for both video panes; returns nothing.
#
# Clears and redraws the measurement point overlays for the left and right video
# panes. The overlay canvas references are owned by this module, while the point
# lists still come from the main application state.
# -----------------------------------------------------------------------------
def redraw_overlays(app):

    # If either overlay canvas has not been registered yet, there is nothing safe
    # to redraw.
    if left_overlay_canvas is None or right_overlay_canvas is None:
        return

    # Draw the left pane overlay using the current left image point list.
    draw_overlay_for_pane(app, "L", left_overlay_canvas, app.ptsL)

    # Draw the right pane overlay using the current right image point list.
    draw_overlay_for_pane(app, "R", right_overlay_canvas, app.ptsR)


# -----------------------------------------------------------------------------
# get_overlay_canvas
#
# Inputs: which identifies the video pane, either "L" for left or "R" for right.
# Outputs: returns the matching overlay canvas, or None if the pane is invalid or
# the canvas has not been created.
#
# Returns the module owned overlay canvas for the requested video pane. Keeping
# this lookup in one place avoids repeating left/right canvas selection logic
# throughout the overlay mouse handlers.
# -----------------------------------------------------------------------------
def get_overlay_canvas(which):

    # Return the left overlay canvas for the left pane.
    if which == "L":
        return left_overlay_canvas

    # Return the right overlay canvas for the right pane.
    if which == "R":
        return right_overlay_canvas

    # Unknown pane identifier.
    return None


# -----------------------------------------------------------------------------
# on_overlay_left_down
#
# Inputs: app provides point lists and measurement update behavior, which
# identifies the clicked pane, and event provides the mouse position in overlay
# canvas coordinates.
# Outputs: either starts dragging an existing point handle or creates a new point;
# returns nothing.
#
# Handles a left mouse button press on one of the overlay canvases. If the click
# hits an existing point handle, the function enters drag mode for that point. If
# the click lands on empty overlay space, the function adds a new image space
# point to the clicked pane and optionally creates an initial stereo mate guess on
# the opposite pane.
# -----------------------------------------------------------------------------
def on_overlay_left_down(app, which, event):

    # Use module level drag state owned by the overlay system.
    global drag_active
    global drag_which
    global drag_index

    # Choose the module owned overlay canvas for this pane.
    canvas = get_overlay_canvas(which)

    # If the requested overlay canvas does not exist, there is nothing safe to do.
    if canvas is None:
        return

    # First try the exact canvas item hit test under the cursor.
    idx = get_handle_index_under_cursor(canvas)

    # If the exact item hit test fails, fall back to a nearest handle search in
    # screen space so points are easier to grab.
    if idx is None:
        idx = get_nearest_handle_index(app, which, canvas, event.x, event.y)

    # If a point handle was found, begin dragging that point.
    if idx is not None:
        drag_active = True
        drag_which = which
        drag_index = idx
        return

    # Otherwise, treat this click as a request to place a new point.
    pts = get_points_list(app, which)

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
    other_pts = get_points_list(app, other_which)

    # If the opposite point list is unavailable, update the overlay with the point
    # we did add and then exit.
    if other_pts is None:
        on_points_changed(app)
        return

    # Only auto create a mate if the opposite pane does not already have a point
    # at this pair index.
    if new_idx >= len(other_pts):

        # Ask the stereo scanline matcher for an initial guessed mate point in the
        # opposite image.
        mate = stereo_matching.guess_mate_point_on_scanline(app, which, ix, iy)

        # If matching succeeded, append the guessed mate at the same pair index.
        if mate is not None:
            other_pts.append(mate)

    # Redraw overlays and update measurement status after the point change.
    on_points_changed(app)

# -----------------------------------------------------------------------------
# on_overlay_left_drag
#
# Inputs: app provides the point lists and measurement update behavior, which
# identifies the pane being dragged, and event provides the current mouse
# position in overlay canvas coordinates.
# Outputs: updates the dragged point position if a valid drag is active; returns
# nothing.
#
# Handles mouse movement while the left button is held on an overlay canvas. If a
# point handle drag is active for the requested pane, the cursor position is
# converted from screen coordinates to image coordinates and written back into the
# matching point list.
# -----------------------------------------------------------------------------
def on_overlay_left_drag(app, which, event):

    # Use module level drag state owned by the overlay system.
    global drag_active
    global drag_which
    global drag_index

    # Only drag if a point handle drag is currently active.
    if not drag_active:
        return

    # Ignore drag events from the opposite pane.
    if drag_which != which:
        return

    # Choose the module owned overlay canvas for this pane.
    canvas = get_overlay_canvas(which)

    # If the requested overlay canvas does not exist, there is nothing safe to do.
    if canvas is None:
        return

    # Get the point list for the pane currently being dragged.
    pts = get_points_list(app, which)

    # If the point list could not be found, do not continue.
    if pts is None:
        return

    # Validate that a drag index has been assigned.
    if drag_index is None:
        return

    # Validate that the drag index still points to an existing point.
    if drag_index < 0 or drag_index >= len(pts):
        return

    # Convert the current mouse position from overlay canvas coordinates into
    # image pixel coordinates.
    ix, iy = app._screen_to_image(which, canvas, event.x, event.y)

    # Update the dragged point in the pane's point list.
    pts[drag_index] = (ix, iy)

    # Redraw overlays and update measurement status continuously while dragging.
    on_points_changed(app)


# -----------------------------------------------------------------------------
# on_overlay_left_up
#
# Inputs: app provides measurement update behavior, which identifies the pane
# receiving the mouse release, and _event is the unused Tkinter mouse event.
# Outputs: clears active drag state for the pane and refreshes overlays and
# measurements; returns nothing.
#
# Handles release of the left mouse button after dragging a point handle. The
# drag only ends if the active drag belongs to the pane that received the release
# event.
# -----------------------------------------------------------------------------
def on_overlay_left_up(app, which, _event):

    # Use module level drag state owned by the overlay system.
    global drag_active
    global drag_which
    global drag_index

    # Only finish a drag if this pane owns the active drag.
    if not drag_active or drag_which != which:
        return

    # Clear drag state now that the drag is finished.
    drag_active = False
    drag_which = None
    drag_index = None

    # Redraw overlays and recompute measurements using the user placed point.
    on_points_changed(app)


# -----------------------------------------------------------------------------
# on_overlay_right_down
#
# Inputs: app provides point lists, which identifies the clicked pane, and event
# provides the mouse position in overlay canvas coordinates.
# Outputs: starts explicit refine drag mode for an existing paired point, or
# returns without changing state if no valid paired point was clicked.
#
# Handles a right mouse button press on one of the overlay canvases. Right button
# input is used for explicit refinement, so it only starts dragging when the user
# clicks an existing point handle that already has a corresponding mate point on
# the opposite pane.
# -----------------------------------------------------------------------------
def on_overlay_right_down(app, which, event):

    # Use module level refine drag state owned by the overlay system.
    global refine_drag_active
    global refine_drag_which
    global refine_drag_index

    # Choose the module owned overlay canvas for this pane.
    canvas = get_overlay_canvas(which)

    # If the requested overlay canvas does not exist, there is nothing safe to do.
    if canvas is None:
        return

    # First try the exact canvas item hit test under the cursor.
    idx = get_handle_index_under_cursor(canvas)

    # If the exact item hit test fails, fall back to a nearest handle search in
    # screen space so points are easier to grab.
    if idx is None:
        idx = get_nearest_handle_index(app, which, canvas, event.x, event.y)

    # Right button refine mode only starts when an existing handle was selected.
    if idx is None:
        return

    # Get the point list for the clicked pane.
    pts = get_points_list(app, which)

    # Get the point list for the opposite pane.
    other_which = "R" if which == "L" else "L"
    other_pts = get_points_list(app, other_which)

    # If either point list is unavailable, do not enter refine mode.
    if pts is None or other_pts is None:
        return

    # Only refine points that exist in both panes at the same paired index.
    if idx >= len(pts) or idx >= len(other_pts):
        return

    # Begin explicit refine drag mode for this paired point.
    refine_drag_active = True
    refine_drag_which = which
    refine_drag_index = idx


# -----------------------------------------------------------------------------
# on_overlay_right_drag
#
# Inputs: app provides point lists and measurement update behavior, which
# identifies the pane being refined, and event provides the current mouse
# position in overlay canvas coordinates.
# Outputs: updates the refined point position if a valid refine drag is active;
# returns nothing.
#
# Handles mouse movement while the right button is held on an overlay canvas.
# Right button dragging is explicit refinement mode: the selected point is moved
# manually in image coordinates while overlays and measurement output update
# continuously.
# -----------------------------------------------------------------------------
def on_overlay_right_drag(app, which, event):

    # Use module level refine drag state owned by the overlay system.
    global refine_drag_active
    global refine_drag_which
    global refine_drag_index

    # Only drag if a refine drag is currently active.
    if not refine_drag_active:
        return

    # Ignore drag events from the opposite pane.
    if refine_drag_which != which:
        return

    # Choose the module owned overlay canvas for this pane.
    canvas = get_overlay_canvas(which)

    # If the requested overlay canvas does not exist, there is nothing safe to do.
    if canvas is None:
        return

    # Get the point list for the pane currently being refined.
    pts = get_points_list(app, which)

    # If the point list could not be found, do not continue.
    if pts is None:
        return

    # Validate that a refine drag index has been assigned.
    if refine_drag_index is None:
        return

    # Validate that the refine drag index still points to an existing point.
    if refine_drag_index < 0 or refine_drag_index >= len(pts):
        return

    # Convert the current mouse position from overlay canvas coordinates into
    # image pixel coordinates.
    ix, iy = app._screen_to_image(which, canvas, event.x, event.y)

    # Update the refined point in the pane's point list.
    pts[refine_drag_index] = (ix, iy)

    # Redraw overlays and update measurement preview continuously while dragging.
    on_points_changed(app)


# -----------------------------------------------------------------------------
# on_overlay_right_up
#
# Inputs: app provides point lists and stereo matching state, which identifies the
# pane receiving the mouse release, and _event is the unused Tkinter mouse event.
# Outputs: optionally refines the released point using local scanline matching,
# clears refine drag state, redraws overlays, and updates measurements.
#
# Handles release of the right mouse button after explicit refine dragging. The
# point is first manually positioned during the drag. On release, the stereo
# matcher is run in a narrow local search window near the user placed X position
# so the point can be snapped to a nearby matching feature without jumping far
# away from the user's intended placement.
# -----------------------------------------------------------------------------
def on_overlay_right_up(app, which, _event):

    # Use module level refine drag state owned by the overlay system.
    global refine_drag_active
    global refine_drag_which
    global refine_drag_index

    # Only finish a refine drag if this pane owns the active refine drag.
    if not refine_drag_active or refine_drag_which != which:
        return

    # Keep the point index before clearing refine drag state.
    idx = refine_drag_index

    # Refine only if there is still a valid point index.
    if idx is not None:

        # Get the point list for the pane being refined.
        pts = get_points_list(app, which)

        # Get the point list for the opposite pane.
        other_which = "R" if which == "L" else "L"
        other_pts = get_points_list(app, other_which)

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
                    app,
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
    refine_drag_active = False
    refine_drag_which = None
    refine_drag_index = None

    # Redraw overlays and recompute measurements using the refined point.
    on_points_changed(app)


# -----------------------------------------------------------------------------
# on_points_changed
#
# Inputs: app provides measurement status update behavior and current point state.
# Outputs: redraws point overlays, refreshes measurement status, and returns
# nothing.
#
# Handles the common follow up work after overlay points are added, moved,
# refined, or cleared. Keeping this as the single point change hook makes it less
# likely that one mouse path updates the overlay but forgets to refresh
# measurement feedback.
# -----------------------------------------------------------------------------
def on_points_changed(app):

    # Redraw the current point overlays for both video panes.
    redraw_overlays(app)

    # Update measurement status text and any measurement preview behavior.
    app._update_measurement_status_stub()


# -----------------------------------------------------------------------------
# get_points_list
#
# Inputs: app provides the left and right overlay point lists, and which identifies
# the requested pane as "L" or "R".
# Outputs: returns the point list for the requested pane, or None if the pane
# identifier is invalid.
#
# Provides one shared left/right point list lookup for overlay drawing and mouse
# interaction code. The point data still lives on the main app object, while the
# overlay module uses this helper to avoid repeating pane selection logic.
# -----------------------------------------------------------------------------
def get_points_list(app, which):

    # Return the left image point list for the left pane.
    if which == "L":
        return app.ptsL

    # Return the right image point list for the right pane.
    if which == "R":
        return app.ptsR

    # Unknown pane identifier.
    return None



# -----------------------------------------------------------------------------
# get_handle_index_under_cursor
#
# Inputs: canvas is the overlay canvas receiving the mouse event.
# Outputs: returns the integer point index for the handle currently under the
# cursor, or None if the current canvas item is not a point handle.
#
# Uses Tkinter canvas item tags to detect whether the mouse is over a drawn point
# handle. Handles are expected to have a "handle" tag and an "idx:<n>" tag that
# stores the point index.
# -----------------------------------------------------------------------------
def get_handle_index_under_cursor(canvas):

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


# -----------------------------------------------------------------------------
# get_nearest_handle_index
#
# Inputs: app provides point lists and handle radius settings, which identifies
# the pane to search, canvas is the overlay canvas, and sx/sy are the screen space
# click coordinates.
# Outputs: returns the nearest point index if the click is close enough to a
# handle, or None if no handle is within the hit radius.
#
# Provides a forgiving fallback hit test when the exact Tkinter canvas item hit
# test misses. Each point is converted from image coordinates to screen
# coordinates, then compared against the mouse click using a generous hit radius.
# -----------------------------------------------------------------------------
def get_nearest_handle_index(app, which, canvas, sx, sy):

    # Read the point list for this pane.
    pts = get_points_list(app, which)

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


# -----------------------------------------------------------------------------
# draw_overlay_for_pane
#
# Inputs: app provides overlay display settings, which identifies the pane being
# drawn, canvas is the overlay canvas for that pane, and pts is the pane's image
# coordinate point list.
# Outputs: clears and redraws overlay lines, point handles, and point labels for
# the selected pane; returns nothing.
#
# Draws the visible measurement overlay for one video pane. Points are stored in
# image pixel coordinates, then converted into screen coordinates so handles and
# connecting segments line up with the displayed video frame. Handles are tagged
# for later mouse hit testing and dragging.
# -----------------------------------------------------------------------------
def draw_overlay_for_pane(app, which, canvas, pts):

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
                fill="#00ff66",
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
            outline="#00ff66",
            width=2,
            fill="",
            tags=("overlay", "handle", f"idx:{i}"),
        )

        # Draw the point index label near the handle.
        canvas.create_text(
            sx + r + 6,
            sy - r - 6,
            text=str(i),
            fill="#00ff66",
            font=("Segoe UI", 11, "bold"),
            tags=("overlay",),
        )