"""Tests for main.py's SizeamaticProApp, for behavior that's easier to
verify against the real GUI than through a FakeApp stand-in.
"""

import numpy as np

import video_overlay


def test_display_bgr_on_canvas_renders_without_error(sizeamatic_app):
    """Regression test for the Phase 5 Pillow render-path fix.

    Displaying a frame should succeed and store a real PhotoImage. This
    doesn't assert anything about the old PNG/base64 path directly (it's
    gone), but exercises the same call site that used to be the
    "unacceptably slow" bottleneck — see ROADMAP.md Phase 5.
    """

    sizeamatic_app.metaL = {"width": 64, "height": 48, "fps": 30.0, "frame_count": 10}
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

    sizeamatic_app._display_bgr_on_canvas(video_overlay.left_overlay_canvas, frame, "L")

    assert sizeamatic_app.tkimg_left is not None
