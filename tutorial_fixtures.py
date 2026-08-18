"""Synthetic tutorial video + calibration generator (ROADMAP.md Phase 15).

Step 0's decision (b): the tutorial's "project" is synthetic/generated
at tutorial-start time, not a bundled real clip and not a fully mocked
experience with no real footage at all. This module builds a small,
self-consistent stereo rig from scratch (no real camera ever involved)
and renders a short left/right video pair directly from that rig's own
projection matrices, so a point clicked in both panes triangulates to a
sensible answer - and burns a real, readable clock into every frame so
the tutorial's real-time-sync steps have something concrete to read.

Pure logic, no Qt dependency - matches `stereo_matching.py`/
`calibration_io.py`/`project_io.py`'s "pure functions, no GUI code"
pattern. `tutorial_window.TutorialController.start` is the only caller -
it generates these fixtures but deliberately does NOT load them
automatically; the user still goes through the real File > Load Left/
Right Video…/Calibration > Load Calibration… actions themselves and
picks the generated files, same as they would with real footage, so the
tutorial's early "load" steps are genuine practice, not a shortcut past
them. `TutorialController.fixture_paths` remembers where these files
ended up purely so `main.py`'s load dialogs can default to opening
there.

Contents:
    - `generate_tutorial_fixtures` - the single entry point: builds a
      fresh temp directory holding both videos and the calibration
      folder, and returns their paths.
    - `generate_tutorial_calibration` - writes the four calibration
      NPZ files `calibration_io.py` already knows how to load.
    - `generate_tutorial_videos` - renders the left/right MP4 pair.

Design notes:
    The synthetic video is rendered directly from the rig's own
    rectified projection matrices (`PL`/`PR`), so there's no actual lens
    distortion or misalignment for the rectification maps to correct.
    A pure identity grid would be the literal truth of that, but it
    would also make toggling "Show Rectified" a visual no-op - unlike
    every real calibration, which crops in slightly around its valid
    ROI after undistorting. `TUTORIAL_RECTIFICATION_ZOOM` bakes in that
    same mild inward crop artificially, purely so the tutorial's
    "Switch to Rectified View" step looks and feels like flipping it on
    with a real calibration - a small, deliberate cosmetic embellishment
    on top of the otherwise-accurate synthetic rig, not a claim that
    this rig has real distortion to correct.

    The rendered scene is two small markers connected by a line, at a
    fixed real-world separation, drifting in a slow circle over time
    (so scrubbing/playback visibly shows *something* changing, not a
    frozen frame) - simple enough to render every frame cheaply, and a
    clear, unambiguous pair of points for the tutorial's point-placement
    steps to click on.

Author:
    Isaac Travers

Created:
    2026-08-17
"""

import datetime
import math
import os
import tempfile

import cv2
import numpy as np

TUTORIAL_VIDEO_WIDTH = 640
TUTORIAL_VIDEO_HEIGHT = 480
TUTORIAL_VIDEO_FPS = 25.0
TUTORIAL_VIDEO_DURATION_SECONDS = 12
"""~5 minutes' worth of tutorial clicking around fits comfortably within
a 12-second loop's worth of frames (300 at 25fps) - long enough to
scrub/pan/zoom meaningfully, short enough to generate and load quickly."""

TUTORIAL_FX = 800.0
TUTORIAL_FY = 800.0
TUTORIAL_CX = float(TUTORIAL_VIDEO_WIDTH) / 2.0
TUTORIAL_CY = float(TUTORIAL_VIDEO_HEIGHT) / 2.0
TUTORIAL_BASELINE_MM = 100.0
"""Synthetic rig parameters - not real captured calibration values,
chosen arbitrarily but self-consistently (mirroring the same approach
`tests/conftest.py`'s `synthetic_cal` fixture already uses for math
tests) so a clicked point triangulates to a real, sensible answer."""

TUTORIAL_OBJECT_DEPTH_MM = 2000.0
TUTORIAL_OBJECT_SEPARATION_MM = 150.0
"""The two rendered markers' real-world depth and separation - a
tutorial user placing a point on each and recording a Segment should
see a length close to this value, not an arbitrary/meaningless number."""

TUTORIAL_CLOCK_START = datetime.datetime(2026, 6, 1, 9, 0, 0)
"""The burned-in clock's value at frame 0 of the *left* video specifically
(see `TUTORIAL_SYNC_OFFSET_FRAMES`) - arbitrary but fixed, so the
real-time-sync tutorial steps always ask for (and can be typed back)
the same date/time regardless of which frame the user happens to anchor
from partway through the video."""

TUTORIAL_RECTIFICATION_ZOOM = 0.92
"""How far the rectification maps crop in toward center, purely for the
visible "zooms in a little and cuts off the edges" look a real
calibration's rectified view has - see this module's "Design notes"."""

TUTORIAL_SYNC_OFFSET_FRAMES = 15
"""The right video is deliberately generated out of sync with the left
by this many frames - real stereo camera pairs aren't guaranteed to
start recording at the exact same instant, and the tutorial's Lock/
Resync step has nothing real to teach if the two videos already line up
frame-for-frame by construction. Both videos burn in a frame counter
(see `_render_frame`) reflecting each one's own *true* underlying scene
moment, so the user can align them by eye - scrub the right video until
its displayed counter matches the left's, then set that difference as
the Offset."""


def _build_projection_matrices():
    """Build the synthetic rig's rectified left/right projection matrices.

    Same construction as `tests/conftest.py`'s `synthetic_cal` fixture -
    a shared left/right intrinsic matrix, with the right camera offset
    by the baseline along X and otherwise identically oriented (already
    rectified, by construction).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: `(PL, PR)`, each a 3x4
        `float64` array.
    """
    PL = np.array(
        [
            [TUTORIAL_FX, 0.0, TUTORIAL_CX, 0.0],
            [0.0, TUTORIAL_FY, TUTORIAL_CY, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    PR = np.array(
        [
            [TUTORIAL_FX, 0.0, TUTORIAL_CX, -TUTORIAL_FX * TUTORIAL_BASELINE_MM],
            [0.0, TUTORIAL_FY, TUTORIAL_CY, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return PL, PR


def _project(P, X, Y, Z):
    """Project one 3D point through a projection matrix.

    Args:
        P (numpy.ndarray): 3x4 projection matrix.
        X (float): 3D point X coordinate, millimeters.
        Y (float): 3D point Y coordinate, millimeters.
        Z (float): 3D point Z coordinate, millimeters.

    Returns:
        tuple[float, float]: The projected `(u, v)` pixel coordinates.
    """
    p = P @ np.array([X, Y, Z, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def _to_raw_pixel(u, v):
    """Map a point's true (rectified-space) pixel position to where it
    must be drawn in the RAW video so that applying the calibration's
    rectification maps reconstructs that exact position.

    `PL`/`PR` describe the rectified pixel space - the one this app's
    triangulation math actually assumes (see `README.md`'s "Measurement
    notes"), and the one the tutorial's own steps teach the user to
    measure in. Drawing markers straight at their `_project(...)`
    position would only be correct in the raw (un-rectified) video if
    the rectification maps were the identity - since they now crop in
    by `TUTORIAL_RECTIFICATION_ZOOM` instead (see this module's "Design
    notes"), the raw video has to draw each marker at the *inverse* of
    that crop so a click on the rectified marker still triangulates to
    the exact real position it was rendered from.

    Args:
        u (float): The point's true pixel X position, in rectified
            space (straight out of `_project`).
        v (float): The point's true pixel Y position, in rectified
            space.

    Returns:
        tuple[float, float]: The `(u, v)` position to actually draw in
        the raw video.
    """
    raw_u = TUTORIAL_CX + (u - TUTORIAL_CX) * TUTORIAL_RECTIFICATION_ZOOM
    raw_v = TUTORIAL_CY + (v - TUTORIAL_CY) * TUTORIAL_RECTIFICATION_ZOOM
    return raw_u, raw_v


def generate_tutorial_calibration(folder):
    """Write a self-consistent synthetic calibration bundle to `folder`.

    Writes exactly the four files `calibration_io.load_calibration_bundle`
    expects, matching every key it reads. The rectification maps crop in
    toward center by `TUTORIAL_RECTIFICATION_ZOOM` - see this module's
    "Design notes" for why.

    Args:
        folder (str): Path to write the four calibration NPZ files into
            - created if it doesn't already exist.

    Returns:
        None
    """
    os.makedirs(folder, exist_ok=True)

    w, h = TUTORIAL_VIDEO_WIDTH, TUTORIAL_VIDEO_HEIGHT
    PL, PR = _build_projection_matrices()

    mtx = np.array(
        [[TUTORIAL_FX, 0.0, TUTORIAL_CX], [0.0, TUTORIAL_FY, TUTORIAL_CY], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros(5, dtype=np.float64)
    np.savez(
        os.path.join(folder, "calibration_intrinsics.npz"),
        mtxL=mtx,
        distL=dist,
        mtxR=mtx,
        distR=dist,
        image_width=w,
        image_height=h,
    )

    # No rotation between the two cameras (already rectified by
    # construction) - just a pure baseline translation along X.
    R = np.eye(3, dtype=np.float64)
    T = np.array([[-TUTORIAL_BASELINE_MM], [0.0], [0.0]], dtype=np.float64)

    # Standard E/F formulas from R/T/intrinsics - included for
    # completeness (calibration_summary.py can display them) but not
    # read by this app's own measurement math, which uses PL/PR/Q
    # directly (see stereo_matching.py).
    tx = np.array(
        [[0.0, -T[2, 0], T[1, 0]], [T[2, 0], 0.0, -T[0, 0]], [-T[1, 0], T[0, 0], 0.0]], dtype=np.float64
    )
    E = tx @ R
    mtx_inv = np.linalg.inv(mtx)
    F = mtx_inv.T @ E @ mtx_inv
    np.savez(os.path.join(folder, "calibration_extrinsics.npz"), R=R, T=T, E=E, F=F, stereo_rms=0.3)

    RL = np.eye(3, dtype=np.float64)
    RR = np.eye(3, dtype=np.float64)
    # Standard disparity-to-depth Q matrix for a fronto-parallel rectified
    # pair with equal fx/fy - not read by this app's own triangulation
    # path (PL/PR-based, not disparity-based), but a real calibration
    # folder always has one, so this fixture does too.
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -TUTORIAL_CX],
            [0.0, 1.0, 0.0, -TUTORIAL_CY],
            [0.0, 0.0, 0.0, TUTORIAL_FX],
            [0.0, 0.0, -1.0 / TUTORIAL_BASELINE_MM, 0.0],
        ],
        dtype=np.float64,
    )
    roi = np.array([0, 0, w, h], dtype=np.int32)
    np.savez(os.path.join(folder, "calibration_rectification.npz"), PL=PL, PR=PR, Q=Q, RL=RL, RR=RR, roiL=roi, roiR=roi)

    # A mild inward-crop grid, not a pure identity one - see this
    # module's "Design notes" for why. Each output pixel samples a
    # source pixel pulled `TUTORIAL_RECTIFICATION_ZOOM` of the way in
    # toward the image center, so remapping visibly crops in and
    # magnifies, exactly like a real calibration's rectified view does.
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = (TUTORIAL_CX + (grid_x - TUTORIAL_CX) * TUTORIAL_RECTIFICATION_ZOOM).astype(np.float32)
    grid_y = (TUTORIAL_CY + (grid_y - TUTORIAL_CY) * TUTORIAL_RECTIFICATION_ZOOM).astype(np.float32)
    np.savez(os.path.join(folder, "calibration_maps.npz"), mapLx=grid_x, mapLy=grid_y, mapRx=grid_x, mapRy=grid_y)


def _scene_state(scene_index, fps):
    """Compute the drifting scene's 3D marker positions at one true
    underlying scene moment.

    Args:
        scene_index (float): The *true* scene moment, in frames - not
            necessarily equal to either video's own local frame index
            (see `TUTORIAL_SYNC_OFFSET_FRAMES`).
        fps (float): Frames per second, to convert `scene_index` into
            elapsed seconds.

    Returns:
        tuple[float, float, float, float]: `(X1, X2, Y, Z)` - the two
        markers' 3D positions share `Y`/`Z`, differing only in `X`.
    """
    t = scene_index / fps
    drift_x = 40.0 * math.sin(t * 0.5)
    drift_y = 20.0 * math.cos(t * 0.5)
    X1 = -TUTORIAL_OBJECT_SEPARATION_MM / 2.0 + drift_x
    X2 = TUTORIAL_OBJECT_SEPARATION_MM / 2.0 + drift_x
    return X1, X2, drift_y, TUTORIAL_OBJECT_DEPTH_MM


def _render_frame(point_a, point_b, scene_index, fps):
    """Render one synthetic frame: two markers, a connecting line, a
    burned-in clock, and a burned-in frame counter.

    Args:
        point_a (tuple[float, float]): First marker's raw-video pixel
            position (see `_to_raw_pixel`).
        point_b (tuple[float, float]): Second marker's raw-video pixel
            position.
        scene_index (int): This frame's *true* scene moment - drives
            both burned-in readouts. Left and right pass their own,
            deliberately different, values here (see
            `TUTORIAL_SYNC_OFFSET_FRAMES`), not each video's own local
            file-frame position.
        fps (float): The video's frame rate - needed to convert
            `scene_index` into an elapsed duration for the clock.

    Returns:
        numpy.ndarray: A `(TUTORIAL_VIDEO_HEIGHT, TUTORIAL_VIDEO_WIDTH,
        3)` `uint8` BGR frame.
    """
    frame = np.full((TUTORIAL_VIDEO_HEIGHT, TUTORIAL_VIDEO_WIDTH, 3), (40, 30, 20), dtype=np.uint8)

    # A static reference border a few pixels in from each edge - the
    # scene markers alone barely move enough near center to make
    # "Show Rectified"'s crop-in visible against a flat background;
    # this border makes the crop obvious, since it runs almost to the
    # frame's edges and is pulled outside the visible area entirely
    # once the rectification maps' inward crop is applied.
    cv2.rectangle(frame, (4, 4), (TUTORIAL_VIDEO_WIDTH - 5, TUTORIAL_VIDEO_HEIGHT - 5), (90, 90, 90), 2, cv2.LINE_AA)

    pa = (int(round(point_a[0])), int(round(point_a[1])))
    pb = (int(round(point_b[0])), int(round(point_b[1])))
    cv2.line(frame, pa, pb, (60, 180, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, pa, 8, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.circle(frame, pb, 8, (60, 220, 60), -1, cv2.LINE_AA)

    clock_text = (TUTORIAL_CLOCK_START + datetime.timedelta(seconds=scene_index / fps)).strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, clock_text, (10, TUTORIAL_VIDEO_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    # A separate, explicit frame counter (distinct from the clock above) -
    # the Lock/Resync tutorial step's whole point is comparing this
    # number between the two panes, which the clock's 1-second resolution
    # is too coarse to do precisely at `TUTORIAL_SYNC_OFFSET_FRAMES`'s scale.
    frame_text = f"Frame: {scene_index}"
    cv2.putText(
        frame, frame_text, (10, TUTORIAL_VIDEO_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    return frame


def generate_tutorial_videos(left_path, right_path):
    """Write the synthetic left/right stereo video pair.

    Both videos show the same drifting two-marker scene, each frame
    reprojected through the rig's own `PL`/`PR` so the two views are
    genuinely consistent stereo pairs - a point placed on each marker
    in both panes triangulates back to (approximately) the real 3D
    position it was rendered from. The two videos are deliberately out
    of sync with each other by `TUTORIAL_SYNC_OFFSET_FRAMES` - see that
    constant's docstring for why - which the burned-in frame counter
    (see `_render_frame`) makes visible and correctable.

    Args:
        left_path (str): Path to write the left video to (`.mp4`).
        right_path (str): Path to write the right video to (`.mp4`).

    Returns:
        None
    """
    PL, PR = _build_projection_matrices()
    frame_count = int(TUTORIAL_VIDEO_DURATION_SECONDS * TUTORIAL_VIDEO_FPS)
    size = (TUTORIAL_VIDEO_WIDTH, TUTORIAL_VIDEO_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_l = cv2.VideoWriter(left_path, fourcc, TUTORIAL_VIDEO_FPS, size)
    writer_r = cv2.VideoWriter(right_path, fourcc, TUTORIAL_VIDEO_FPS, size)

    try:
        for frame_index in range(frame_count):
            # The right camera's local frame `frame_index` shows the
            # scene TUTORIAL_SYNC_OFFSET_FRAMES *ahead* of the left
            # camera's same local frame_index - i.e. it started
            # recording later. Both are still valid, in-range scene
            # moments (scene_index only needs to be non-negative, which
            # it always is here).
            scene_index_left = frame_index
            scene_index_right = frame_index + TUTORIAL_SYNC_OFFSET_FRAMES

            X1_l, X2_l, Y_l, Z_l = _scene_state(scene_index_left, TUTORIAL_VIDEO_FPS)
            X1_r, X2_r, Y_r, Z_r = _scene_state(scene_index_right, TUTORIAL_VIDEO_FPS)

            # Drawn in raw-video pixel space (`_to_raw_pixel`), not the
            # true `_project(...)` position directly - see that
            # function's docstring for why: it's the inverse of the
            # rectification maps' inward crop, so a click on the marker
            # in RECTIFIED view still triangulates to the exact real
            # position it was rendered from.
            point_a_left = _to_raw_pixel(*_project(PL, X1_l, Y_l, Z_l))
            point_b_left = _to_raw_pixel(*_project(PL, X2_l, Y_l, Z_l))
            point_a_right = _to_raw_pixel(*_project(PR, X1_r, Y_r, Z_r))
            point_b_right = _to_raw_pixel(*_project(PR, X2_r, Y_r, Z_r))

            writer_l.write(_render_frame(point_a_left, point_b_left, scene_index_left, TUTORIAL_VIDEO_FPS))
            writer_r.write(_render_frame(point_a_right, point_b_right, scene_index_right, TUTORIAL_VIDEO_FPS))
    finally:
        writer_l.release()
        writer_r.release()


def generate_tutorial_fixtures():
    """Generate a complete, fresh tutorial video+calibration set.

    Builds a brand new temp directory every call (never reused across
    tutorial runs) holding both videos and the calibration folder, per
    Step 0's "always starts fresh" decision.

    Returns:
        dict: Keys `"left_video_path"`, `"right_video_path"`, and
        `"calibration_folder"` - ready to hand to `main.py`'s
        `_load_left_video_from_path`/`_load_right_video_from_path`/
        `_load_calibration_from_folder`.
    """
    base_dir = tempfile.mkdtemp(prefix="sizeamatic_tutorial_")
    calibration_folder = os.path.join(base_dir, "calibration")
    left_video_path = os.path.join(base_dir, "tutorial_left.mp4")
    right_video_path = os.path.join(base_dir, "tutorial_right.mp4")

    generate_tutorial_calibration(calibration_folder)
    generate_tutorial_videos(left_video_path, right_video_path)

    return {
        "left_video_path": left_video_path,
        "right_video_path": right_video_path,
        "calibration_folder": calibration_folder,
    }
