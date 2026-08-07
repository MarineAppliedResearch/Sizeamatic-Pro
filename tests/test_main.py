"""Tests for main.py's SizeamaticProApp, for behavior that's easier to
verify against the real GUI than through a FakeApp stand-in.
"""

import os

import cv2
import numpy as np
import pytest

import video_overlay

LEFT_VIDEO = "examples/left_20260309_171631.mp4"


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


@pytest.mark.skipif(
    not os.path.isfile(LEFT_VIDEO),
    reason="Real example video is gitignored/local-only, not present here.",
)
def test_read_frame_at_correct_for_sequential_and_random_access(sizeamatic_app):
    """Regression test for the seek-skip optimization in _read_frame_at
    (ROADMAP.md Phase 5).

    _read_frame_at skips the explicit seek when the capture's own
    reported position already matches the requested index, purely for
    speed. This test verifies it still returns the actually-correct frame
    under three access patterns: pure random access (forces a seek every
    time), pure sequential access (exercises the seek-skip path
    specifically), and a mix that jumps backward then resumes sequential
    — the exact shape of access pattern that would expose a naive "assume
    position is last-read-plus-one" optimization as wrong, which is
    exactly why this checks the capture's real position instead.
    """

    app = sizeamatic_app
    app.capL, app.metaL = app._open_video_capture(LEFT_VIDEO)
    assert app.capL is not None

    # A second, independent capture used only to establish ground-truth
    # frame content, so reading via it never disturbs app.capL's position.
    ground_truth_cap = cv2.VideoCapture(LEFT_VIDEO)

    def ground_truth_frame(index):
        ground_truth_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = ground_truth_cap.read()
        assert ok
        return frame

    access_patterns = [
        [20, 5, 21, 6, 100, 0],           # pure random access
        [30, 31, 32, 33, 34],             # pure sequential access
        [10, 11, 40, 41, 42],             # backward jump, then resume sequential
    ]

    for indices in access_patterns:
        for index in indices:
            actual = app._read_frame_at(app.capL, index)
            expected = ground_truth_frame(index)
            assert np.array_equal(actual, expected), f"Mismatch at frame {index}"

    ground_truth_cap.release()
