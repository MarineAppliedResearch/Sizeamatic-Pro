"""Red/cyan anaglyph preview feature for Sizeamatic Pro.

This module manages the OpenCV preview window, preview playback state,
frame stepping controls, and red/cyan anaglyph image generation used to
visually inspect the current stereo video pair.

Contents:
    - Anaglyph preview active and playback state.
    - OpenCV preview window creation and cleanup.
    - Preview tick/update loop.
    - Keyboard controls for play, pause, stepping, and closing.
    - Optional rectified frame display when calibration is loaded.
    - Red/cyan anaglyph frame generation.

Design notes:
    The View menu command remains in the main application class because it
    is part of the main GUI menu wiring. This module owns the preview state
    and implementation details for the anaglyph preview feature itself.

    Functions in this module receive the main application object (`app`)
    when they need access to video captures, frame indexes, calibration
    maps, frame reading helpers, Tkinter scheduling, or status bar updates.

    The preview loop mixes OpenCV's own window/event handling
    (`cv2.imshow`/`cv2.waitKey`) with Tkinter's `after()` scheduling. This
    is a known-workable pattern but is a bit fragile: it depends on
    `cv2.waitKey()` being called on every tick to keep the OpenCV window
    responsive, and on `app.root.after()` continuing to fire on schedule.

Assumptions:
    - Both left and right videos are loaded before the preview is started.
    - The main application provides readable left and right video captures.
    - If rectified view is enabled, the loaded calibration dictionary
      contains `mapLx`, `mapLy`, `mapRx`, and `mapRy` remap arrays.
    - The preview is a visual inspection aid and does not change
      measurement points, measurement results, calibration values, or
      video state.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# OpenCV is used for stereo triangulation, template matching, projection, and
# other image-space measurement operations.
import cv2

# NumPy is used to build OpenCV-compatible point arrays and perform vector math.
import numpy as np

anaglyph_active = False
"""Whether the anaglyph preview loop is currently active. Set True by
`start_anaglyph_preview`, False by `stop_anaglyph_preview`; `anaglyph_tick`
checks this first thing on every tick to decide whether to keep
rescheduling itself."""

anaglyph_playing = False
"""Whether the preview auto-advances frames. The preview always opens
paused (so the user can inspect the first frame before anything moves) —
pressing Space toggles this in `anaglyph_tick`'s keyboard handling."""

anaglyph_index = 0
"""Frame index currently shown in the preview window. Deliberately
separate from the main app's left/right timeline indices — the preview
has its own scrubbing position, only seeded from the left timeline's
current index at the moment the preview opens."""

anaglyph_window_name = "Anaglyph 3D Preview"
"""OpenCV window title for the preview. `main.py` overrides this to
"Sizeamatic Pro - Anaglyph 3D" when constructing `SizeamaticProApp`, so
what the user actually sees matches the app's branding rather than this
module's generic default."""

anaglyph_after_id = None
"""Tkinter `after()` job ID for the scheduled preview tick, so it can be
cancelled when the preview stops. Initialized to `None` up front — rather
than only coming into existence the first time `anaglyph_tick` reaches its
own assignment to this variable — specifically so `stop_anaglyph_preview`
can safely check it even if the very first tick fails before getting that
far. See `FINDINGS.md` #2 for the bug this was guarding against."""


def start_anaglyph_preview(app):
    """Open the anaglyph preview window and start the preview loop.

    Starts the anaglyph preview at the current left video frame so the
    preview opens near the user's current timeline position. The preview
    starts paused by default, allowing the user to inspect the first
    anaglyph frame before playing.

    Args:
        app: The main application object, used to read the current left
            frame index (`app.left_frame_index`) and to display status
            text (`app._set_status_mid`).

    Returns:
        None
    """

    # Use module level state so anaglyph preview state lives with the feature
    # implementation instead of on the main application object.
    global anaglyph_active
    global anaglyph_playing
    global anaglyph_index

    # Start at the current left timeline index for convenience.
    anaglyph_index = int(app.left_frame_index.get())

    # Mark the preview active, but start paused so the user controls playback.
    anaglyph_active = True
    anaglyph_playing = False

    # Create a resizable OpenCV window for the preview image.
    cv2.namedWindow(anaglyph_window_name, cv2.WINDOW_NORMAL)

    # Show the keyboard controls in the main application status bar.
    app._set_status_mid("Anaglyph preview opened (Space: play/pause, A/D: step, Q: quit)")

    # Start the preview update loop.
    anaglyph_tick(app)


def stop_anaglyph_preview(app):
    """Stop the anaglyph preview and close its OpenCV window.

    Stops the red/cyan anaglyph preview feature. This function is safe to
    call from the menu toggle, from the preview tick when the OpenCV window
    is closed, or from keyboard handling when the user presses Q or ESC.

    Args:
        app: The main application object, used to cancel the scheduled
            Tkinter tick (`app.root.after_cancel`) and to display status
            text (`app._set_status_mid`).

    Returns:
        None
    """

    # Use module level state so the preview state stays inside this feature file.
    global anaglyph_active
    global anaglyph_playing
    global anaglyph_after_id

    # Cancel any scheduled Tkinter after() tick so the preview loop does not keep
    # running after the preview has been stopped.
    if anaglyph_after_id is not None:
        app.root.after_cancel(anaglyph_after_id)
        anaglyph_after_id = None

    # Reset the preview state flags.
    anaglyph_active = False
    anaglyph_playing = False

    # Try to close the OpenCV preview window. Ignore failures because OpenCV may
    # already consider the window closed depending on how the user exited it.
    try:
        cv2.destroyWindow(anaglyph_window_name)
    except Exception:
        pass

    # Update the main application status bar.
    app._set_status_mid("Anaglyph preview closed")


def anaglyph_tick(app):
    """Run one update pass of the anaglyph preview loop.

    Runs one update pass of the anaglyph preview loop. The function reads
    the left and right frames at the current anaglyph index, optionally
    remaps them into rectified view, builds a red/cyan anaglyph image,
    displays it in the OpenCV preview window, handles keyboard controls,
    and schedules the next tick.

    Args:
        app: The main application object, used for video captures
            (`app.capL`/`app.capR`), frame metadata, calibration state
            (`app.cal`), the frame reading helper (`app._read_frame_at`),
            and Tk root scheduling (`app.root.after`).

    Returns:
        None
    """

    # Use module level state so the preview state stays inside this feature file.
    global anaglyph_active
    global anaglyph_playing
    global anaglyph_index
    global anaglyph_after_id

    # If the preview was stopped, exit without scheduling another tick.
    if not anaglyph_active:
        return

    # Check whether the OpenCV preview window is still visible. If the user closed
    # it directly, stop the preview cleanly.
    try:
        vis = cv2.getWindowProperty(anaglyph_window_name, cv2.WND_PROP_VISIBLE)
        if vis < 1:
            stop_anaglyph_preview(app)
            return

    # If OpenCV raises while checking the window, assume the window is gone and
    # stop the preview cleanly.
    except Exception:
        stop_anaglyph_preview(app)
        return

    # Clamp the preview index to the shorter of the two video streams so frame
    # reads stay inside both videos.
    max_i = int(min(app.left_frame_max, app.right_frame_max))

    # Prevent negative frame indexes.
    if anaglyph_index < 0:
        anaglyph_index = 0

    # Prevent seeking past the end of the shorter stream.
    if anaglyph_index > max_i:
        anaglyph_index = max_i

    # Read the left and right frames at the current anaglyph preview index.
    frameL = app._read_frame_at(app.capL, anaglyph_index)
    frameR = app._read_frame_at(app.capR, anaglyph_index)

    # If either frame fails to read, show a black placeholder frame and keep the
    # preview loop alive.
    if frameL is None or frameR is None:
        blank = np.zeros(
            (int(app.metaL["height"]), int(app.metaL["width"]), 3),
            dtype=np.uint8,
        )
        cv2.imshow(anaglyph_window_name, blank)

    else:
        # Prefer rectified frames when rectified view is enabled and calibration
        # data is available.
        if app.view_rectified.get() and app.cal is not None:
            frameL = cv2.remap(
                frameL,
                app.cal["mapLx"],
                app.cal["mapLy"],
                interpolation=cv2.INTER_LINEAR,
            )
            frameR = cv2.remap(
                frameR,
                app.cal["mapRx"],
                app.cal["mapRy"],
                interpolation=cv2.INTER_LINEAR,
            )

        # Build the red/cyan anaglyph frame from the current left/right frames.
        ana = make_anaglyph_red_cyan(frameL, frameR)

        # Display the anaglyph frame in the OpenCV preview window.
        cv2.imshow(anaglyph_window_name, ana)

    # Read one key press from the OpenCV window. Space toggles play/pause, A/D
    # step backward/forward, and Q or ESC closes the preview.
    key = cv2.waitKey(1) & 0xFF

    # Q or ESC stops and closes the preview.
    if key == ord("q") or key == 27:
        stop_anaglyph_preview(app)
        return

    # Space toggles playback.
    if key == 32:
        anaglyph_playing = not anaglyph_playing

    # A steps one frame backward and pauses playback.
    if key == ord("a"):
        anaglyph_playing = False
        anaglyph_index -= 1

    # D steps one frame forward and pauses playback.
    if key == ord("d"):
        anaglyph_playing = False
        anaglyph_index += 1

    # Advance one frame if playback is active.
    if anaglyph_playing:
        anaglyph_index += 1

        # Stop playback at the end of the shorter stream.
        if anaglyph_index > max_i:
            anaglyph_index = max_i
            anaglyph_playing = False

    # Schedule the next preview tick using the Tkinter event loop. A 40 ms delay
    # targets roughly 25 frames per second.
    anaglyph_after_id = app.root.after(40, lambda: anaglyph_tick(app))


def make_anaglyph_red_cyan(frameL_bgr, frameR_bgr):
    """Build a red/cyan anaglyph image from a left/right frame pair.

    Builds a simple grayscale red/cyan anaglyph for stereo preview. The
    left frame supplies the red channel, and the right frame supplies the
    green and blue channels. This preview is meant for visual inspection
    only and does not affect measurement results.

    Args:
        frameL_bgr (numpy.ndarray): Left video frame in OpenCV BGR image
            format.
        frameR_bgr (numpy.ndarray): Right video frame in OpenCV BGR image
            format.

    Returns:
        numpy.ndarray: A red/cyan anaglyph image in BGR format, suitable
        for `cv2.imshow`.
    """

    # Convert the left frame to grayscale so it can be placed into the red channel.
    gL = cv2.cvtColor(frameL_bgr, cv2.COLOR_BGR2GRAY)

    # Convert the right frame to grayscale so it can be placed into the cyan
    # channels.
    gR = cv2.cvtColor(frameR_bgr, cv2.COLOR_BGR2GRAY)

    # Allocate an output image with the same shape and data type as the left frame.
    out = np.zeros_like(frameL_bgr)

    # OpenCV stores color images in BGR order. Put the left grayscale image into
    # the red channel.
    out[:, :, 2] = gL

    # Put the right grayscale image into the green and blue channels to make cyan.
    out[:, :, 1] = gR
    out[:, :, 0] = gR

    # Return a BGR image that can be displayed directly with cv2.imshow.
    return out
