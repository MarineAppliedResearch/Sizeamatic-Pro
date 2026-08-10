"""Red/cyan anaglyph preview feature for Sizeamatic Pro.

This module manages the OpenCV preview window, preview playback state,
frame stepping controls, and red/cyan anaglyph image generation used to
visually inspect the current stereo video pair.

Contents:
    - `AnaglyphPreview` — owns the preview window and playback state.
    - `make_anaglyph_red_cyan` — standalone red/cyan image generation, no
      state needed, independently testable.

Design notes:
    The View menu command remains in the main application class because it
    is part of the main GUI menu wiring. `AnaglyphPreview` owns the preview
    state and implementation details for the feature itself, as a class
    instance on the app (`app.anaglyph_preview`) — the same conversion
    already done for `calibration_summary.CalibrationSummaryWindow` and
    `measurement_window.MeasurementWindow` (removes the module-level-global
    fragility that caused `FINDINGS.md` #1 and #2).

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


class AnaglyphPreview:
    """Owns the anaglyph preview's OpenCV window and playback state.

    One instance lives on the main application (`app.anaglyph_preview`).
    """

    def __init__(self, app):
        """Store the owning app and initialize preview state.

        Args:
            app: The main application object, used for video captures,
                frame metadata, calibration state, the frame reading
                helper, and Tk root scheduling/status updates.

        Returns:
            None
        """
        self.app = app

        self.active = False
        """Whether the preview loop is currently active. Set True by
        `start`, False by `stop`; `tick` checks this first thing on every
        call to decide whether to keep rescheduling itself."""

        self.playing = False
        """Whether the preview auto-advances frames. The preview always
        opens paused (so the user can inspect the first frame before
        anything moves) — pressing Space toggles this in `tick`'s
        keyboard handling."""

        self.index = 0
        """Frame index currently shown in the preview window. Deliberately
        separate from the main app's left/right timeline indices — the
        preview has its own scrubbing position, only seeded from the left
        timeline's current index at the moment the preview opens."""

        self.window_name = "Anaglyph 3D Preview"
        """OpenCV window title for the preview. `main.py` overrides this
        to "Sizeamatic Pro - Anaglyph 3D" after constructing this
        instance, so what the user actually sees matches the app's
        branding rather than this generic default."""

        self.after_id = None
        """Tkinter `after()` job ID for the scheduled preview tick, so it
        can be cancelled when the preview stops. `stop` checks this
        defensively even though it's always set by the time a real
        window is open — see `FINDINGS.md` #2 for the bug this guarded
        against when this state lived as an uninitialized module
        global."""

    def start(self):
        """Open the anaglyph preview window and start the preview loop.

        Starts the anaglyph preview at the current left video frame so
        the preview opens near the user's current timeline position. The
        preview starts paused by default, allowing the user to inspect
        the first anaglyph frame before playing.

        Returns:
            None
        """

        # Start at the current left timeline index for convenience.
        self.index = int(self.app.left_frame_index.get())

        # Mark the preview active, but start paused so the user controls playback.
        self.active = True
        self.playing = False

        # Create a resizable OpenCV window for the preview image.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Show the keyboard controls in the main application status bar.
        self.app._set_status_mid("Anaglyph preview opened (Space: play/pause, A/D: step, Q: quit)")

        # Start the preview update loop.
        self.tick()

    def stop(self):
        """Stop the anaglyph preview and close its OpenCV window.

        Safe to call from the menu toggle, from the preview tick when the
        OpenCV window is closed, or from keyboard handling when the user
        presses Q or ESC.

        Returns:
            None
        """

        # Cancel any scheduled Tkinter after() tick so the preview loop does not keep
        # running after the preview has been stopped.
        if self.after_id is not None:
            self.app.root.after_cancel(self.after_id)
            self.after_id = None

        # Reset the preview state flags.
        self.active = False
        self.playing = False

        # Try to close the OpenCV preview window. Ignore failures because OpenCV may
        # already consider the window closed depending on how the user exited it.
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

        # Update the main application status bar.
        self.app._set_status_mid("Anaglyph preview closed")

    def tick(self):
        """Run one update pass of the anaglyph preview loop.

        Reads the left and right frames at the current preview index,
        optionally remaps them into rectified view, builds a red/cyan
        anaglyph image, displays it in the OpenCV preview window, handles
        keyboard controls, and schedules the next tick.

        Returns:
            None
        """

        # If the preview was stopped, exit without scheduling another tick.
        if not self.active:
            return

        # Check whether the OpenCV preview window is still visible. If the user closed
        # it directly, stop the preview cleanly.
        try:
            vis = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if vis < 1:
                self.stop()
                return

        # If OpenCV raises while checking the window, assume the window is gone and
        # stop the preview cleanly.
        except Exception:
            self.stop()
            return

        # Clamp the preview index to the shorter of the two video streams so frame
        # reads stay inside both videos.
        max_i = int(min(self.app.left_frame_max, self.app.right_frame_max))

        # Prevent negative frame indexes.
        if self.index < 0:
            self.index = 0

        # Prevent seeking past the end of the shorter stream.
        if self.index > max_i:
            self.index = max_i

        # Read the left and right frames at the current anaglyph preview index.
        frameL = self.app._read_frame_at(self.app.capL, self.index)
        frameR = self.app._read_frame_at(self.app.capR, self.index)

        # If either frame fails to read, show a black placeholder frame and keep the
        # preview loop alive.
        if frameL is None or frameR is None:
            blank = np.zeros(
                (int(self.app.metaL["height"]), int(self.app.metaL["width"]), 3),
                dtype=np.uint8,
            )
            cv2.imshow(self.window_name, blank)

        else:
            # Prefer rectified frames when rectified view is enabled and calibration
            # data is available.
            if self.app.view_rectified.get() and self.app.cal is not None:
                frameL = cv2.remap(
                    frameL,
                    self.app.cal["mapLx"],
                    self.app.cal["mapLy"],
                    interpolation=cv2.INTER_LINEAR,
                )
                frameR = cv2.remap(
                    frameR,
                    self.app.cal["mapRx"],
                    self.app.cal["mapRy"],
                    interpolation=cv2.INTER_LINEAR,
                )

            # Build the red/cyan anaglyph frame from the current left/right frames.
            ana = make_anaglyph_red_cyan(frameL, frameR)

            # Display the anaglyph frame in the OpenCV preview window.
            cv2.imshow(self.window_name, ana)

        # Read one key press from the OpenCV window. Space toggles play/pause, A/D
        # step backward/forward, and Q or ESC closes the preview.
        key = cv2.waitKey(1) & 0xFF

        # Q or ESC stops and closes the preview.
        if key == ord("q") or key == 27:
            self.stop()
            return

        # Space toggles playback.
        if key == 32:
            self.playing = not self.playing

        # A steps one frame backward and pauses playback.
        if key == ord("a"):
            self.playing = False
            self.index -= 1

        # D steps one frame forward and pauses playback.
        if key == ord("d"):
            self.playing = False
            self.index += 1

        # Advance one frame if playback is active.
        if self.playing:
            self.index += 1

            # Stop playback at the end of the shorter stream.
            if self.index > max_i:
                self.index = max_i
                self.playing = False

        # Schedule the next preview tick using the Tkinter event loop. A 40 ms delay
        # targets roughly 25 frames per second.
        self.after_id = self.app.root.after(40, self.tick)


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
