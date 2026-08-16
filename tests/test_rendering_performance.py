"""End-to-end rendering performance test.

Loads the real app, a real stereo video pair, and a real calibration
fixture, enables rectified view, and measures the actual achieved frames
per second through the full real pipeline (video decode + cv2.remap +
Pillow render) — not an isolated micro-benchmark. This is the test that
directly answers "is rectified rendering still slow" — see ROADMAP.md
Phase 5 for the Pillow render-path fix this exercises end-to-end.
"""

import os
import time

import pytest

import calibration_io

APRIL_CALIBRATION_DIR = "misc/AprilCalibration1"
LEFT_VIDEO = "examples/left_20260309_171631.mp4"
RIGHT_VIDEO = "examples/right_20260309_171631.mp4"

FRAMES_TO_RENDER = 60


@pytest.mark.skipif(
    not (os.path.isfile(LEFT_VIDEO) and os.path.isfile(RIGHT_VIDEO)),
    reason="Real example videos are gitignored/local-only, not present here.",
)
def test_rectified_rendering_framerate(qapp):
    """Rectified playback should sustain well above the old (pre-fix)
    performance ceiling.

    Renders a batch of real frame pairs through the actual app machinery
    (`render_current_frames`, the same method the real UI calls on every
    slider move and playback tick) and reports the achieved fps.
    """
    import main

    app = main.SizeamaticProApp()

    # Load left/right videos the same way on_load_left_video/
    # on_load_right_video do, without the file-picker dialog.
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    app.capR, app.metaR = app._open_video_capture(RIGHT_VIDEO)
    assert app.capL is not None, "Failed to open left example video"
    assert app.capR is not None, "Failed to open right example video"

    app._update_slider_ranges()

    # Load the real calibration fixture and enable rectified view.
    cal, err = calibration_io.load_calibration_bundle(
        APRIL_CALIBRATION_DIR, app.metaL, app.metaR
    )
    assert err is None, f"Calibration failed to load: {err}"
    app.cal = cal
    app.view_rectified.set(True)

    # Render a batch of frames (both panes, rectified) and time it.
    max_index = min(app.left_frame_max, app.right_frame_max, FRAMES_TO_RENDER - 1)

    t0 = time.perf_counter()
    for i in range(max_index + 1):
        app.left_frame_index = i
        app.right_frame_index = i
        app.render_current_frames()
    elapsed = time.perf_counter() - t0

    frames_rendered = max_index + 1
    fps = frames_rendered / elapsed

    print(
        f"\nRendered {frames_rendered} rectified frame pairs in "
        f"{elapsed:.2f}s ({fps:.1f} fps)"
    )

    # A generous floor, not a tight performance budget - this is a
    # regression guard against something becoming badly broken (e.g. the
    # old PNG/base64 render path coming back), not a strict performance
    # gate, since actual achievable fps depends on the machine running it.
    assert fps > 5.0, (
        f"Rectified rendering achieved only {fps:.1f} fps, which is "
        "suspiciously close to the old, pre-Pillow-fix performance."
    )
