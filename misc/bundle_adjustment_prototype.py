"""ROADMAP.md Phase 14 research prototype: synthetic-data-only feasibility
check for a photogrammetric bundle-adjustment stereo calibration mode.

This is NOT wired into the app and does not touch the existing
checkerboard/ChArUco calibration path (perform_calibration.py) in any way.
It exists purely to answer one Phase 14 checklist question: "can a joint
bundle adjustment across multiple views of a rigid 3D target actually
recover camera/target geometry, using tools already available to this
project (scipy + OpenCV)?" It does not read or write any real calibration
files, and running it has no effect on the shipped app.

If Phase 14 concludes this is worth building for real, this prototype's
structure (parameter vector layout, residual function, scipy.optimize.
least_squares call) is the starting point for that - but this file itself
is throwaway research code, not production code. It intentionally skips
lens distortion and camera-intrinsic refinement to stay focused on the one
thing this codebase's current pipeline (independent cv2.calibrateCamera per
camera -> cv2.stereoCalibrate with CALIB_FIX_INTRINSIC -> cv2.stereoRectify)
does NOT do: jointly refining a *3D* target's own point geometry and the
stereo extrinsics together across many views, the way SeaGIS's CAL is
described as doing with a 3D calibration cube.

What this simulates:
    - A fixed stereo rig (known intrinsics, unknown-to-the-solver stereo
      extrinsics) observing a rigid "cube" of control points.
    - The cube's true manufactured point positions have small random
      errors relative to its nominal/as-designed geometry - modeling
      realistic 3D-printing/measurement imprecision.
    - The cube is shown to the rig in many different random poses (as a
      real calibration session would move a physical target around).
    - Each view's image points (both cameras) are the TRUE geometry
      projected through the TRUE stereo extrinsics, plus pixel noise.

What the solver is given (its "initial guess", standing in for what a
real calibration session would start from):
    - The cube's NOMINAL (as-designed, not as-manufactured) point
      geometry - deliberately wrong, to see whether the bundle adjustment
      can correct it.
    - A stereo baseline/rotation perturbed away from the truth - standing
      in for an approximate initial stereo estimate.
    - Per-view poses from cv2.solvePnP against the (wrong) nominal
      geometry and the noisy left-camera image points.

What gets jointly refined by scipy.optimize.least_squares:
    - The stereo extrinsics (one shared rotation vector + translation).
    - Every view's board pose relative to the left camera.
    - The cube's own 3D point positions (one shared "release object"
      style set, matching cv2.calibrateCameraRO's philosophy of not
      trusting the nominal target geometry as ground truth).

A real, non-obvious finding surfaced while building this (see
ROADMAP.md's Phase 14 write-up for the full result): letting the object
points float completely freely has a gauge/datum ambiguity - nothing in
the reprojection residuals alone pins down the object-space point
cloud's absolute scale/position/orientation jointly with the stereo
extrinsics, so an unconstrained first attempt recovered a cube geometry
55mm off from the truth despite near-perfect reprojection error. This
script runs BOTH variants back to back for comparison:
    1. Unconstrained - object points fully free (reproduces the
       gauge-ambiguous behavior above).
    2. Softly anchored - an extra residual term pulls solved object
       points back toward the nominal/as-designed geometry, weighted by
       the assumed manufacturing tolerance (`1 / tolerance_mm`) rather
       than treating that nominal geometry as exact. This is what
       actually recovers the true manufactured geometry (to within a
       fraction of a mm), while leaving reprojection RMS and stereo
       baseline recovery just as good as the unconstrained run.

Run directly (`uv run python misc/bundle_adjustment_prototype.py`) to see
both variants' before/after reprojection RMS and parameter-recovery error
printed to the console - no assertions, no pytest wiring, this is a
standalone check.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

# ---- Reproducibility ----
# misc/ scripts aren't covered by AGENTS.md's "no numpy.random without a
# seed" concern for the app itself, but a fixed seed still makes this
# prototype's printed numbers reproducible between runs.
RNG = np.random.default_rng(20260815)

# ---- Ground truth stereo rig ----
# Simple synthetic intrinsics, similar in spirit to tests/conftest.py's
# synthetic_cal fixture, but as plain K matrices (this prototype does not
# rectify anything - it treats the two cameras as an ordinary stereo pair
# with a general rotation/translation between them, not a rectified pair
# with shared PL/PR the way stereo_matching.py's real pipeline does).
FX, FY, CX, CY = 800.0, 800.0, 320.0, 240.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

TRUE_BASELINE_MM = 100.0
TRUE_TOE_IN_DEG = 2.0  # Small inward rotation, not a perfectly parallel rig.
TRUE_STEREO_RVEC = np.array([0.0, np.deg2rad(TRUE_TOE_IN_DEG), 0.0])
TRUE_STEREO_TVEC = np.array([TRUE_BASELINE_MM, 0.0, 0.0])

# ---- Ground truth calibration cube ----
# An 8-corner cube, side length 150mm, centered on its own local origin.
# NOMINAL_CUBE is the "as-designed" geometry a 3D-printed target would be
# designed to. TRUE_CUBE adds small manufacturing/measurement error
# (+/- up to ~0.3mm per axis) - the bundle adjustment never sees TRUE_CUBE
# directly, only noisy image observations generated from it.
CUBE_SIDE_MM = 150.0
NOMINAL_CUBE = (CUBE_SIDE_MM / 2.0) * np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)
MANUFACTURING_ERROR_MM = 0.3
TRUE_CUBE = NOMINAL_CUBE + RNG.uniform(-MANUFACTURING_ERROR_MM, MANUFACTURING_ERROR_MM, NOMINAL_CUBE.shape)

N_VIEWS = 20
PIXEL_NOISE_STD = 0.3  # Assumed click/detection noise, in pixels.


def random_view_pose():
    """Build one random rigid-body pose placing the cube somewhere in
    front of the left camera, roughly matching how a real calibration
    session moves a physical target around during capture.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: `(rvec, tvec)`, a 3-element
        Rodrigues rotation vector and a 3-element translation vector
        (millimeters), both in the left camera's coordinate frame.
    """
    rvec = RNG.uniform(-0.5, 0.5, 3)
    tvec = np.array(
        [
            RNG.uniform(-200.0, 200.0),
            RNG.uniform(-150.0, 150.0),
            RNG.uniform(1200.0, 2500.0),
        ]
    )
    return rvec, tvec


def project(points_3d, rvec, tvec, K_matrix):
    """Project 3D points through one camera pose (no lens distortion).

    Args:
        points_3d (numpy.ndarray): Nx3 object-space points.
        rvec (numpy.ndarray): 3-element Rodrigues rotation vector.
        tvec (numpy.ndarray): 3-element translation vector.
        K_matrix (numpy.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        numpy.ndarray: Nx2 projected pixel coordinates.
    """
    proj, _ = cv2.projectPoints(points_3d, rvec, tvec, K_matrix, None)
    return proj.reshape(-1, 2)


def compose_right_pose(left_rvec, left_tvec, stereo_rvec, stereo_tvec):
    """Compose a view's left-camera pose with the stereo extrinsics to
    get that view's pose relative to the right camera.

    Args:
        left_rvec (numpy.ndarray): View's rotation vector in the left
            camera's frame.
        left_tvec (numpy.ndarray): View's translation in the left
            camera's frame.
        stereo_rvec (numpy.ndarray): Right-camera-relative-to-left
            rotation vector.
        stereo_tvec (numpy.ndarray): Right-camera-relative-to-left
            translation.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: `(right_rvec, right_tvec)`
        for the same view, in the right camera's frame.
    """
    R_left, _ = cv2.Rodrigues(left_rvec)
    R_stereo, _ = cv2.Rodrigues(stereo_rvec)
    R_right = R_stereo @ R_left
    t_right = (R_stereo @ left_tvec.reshape(3, 1) + stereo_tvec.reshape(3, 1)).ravel()
    right_rvec, _ = cv2.Rodrigues(R_right)
    return right_rvec.ravel(), t_right


def simulate_views():
    """Generate synthetic per-view ground-truth poses and noisy left/right
    image observations of TRUE_CUBE.

    Returns:
        tuple: `(true_view_poses, left_obs, right_obs)` - a list of
        `(rvec, tvec)` ground-truth per-view poses, and matching lists of
        Nx2 noisy left/right pixel observations, one array per view.
    """
    true_view_poses = []
    left_obs = []
    right_obs = []

    for _ in range(N_VIEWS):
        rvec, tvec = random_view_pose()
        true_view_poses.append((rvec, tvec))

        right_rvec, right_tvec = compose_right_pose(rvec, tvec, TRUE_STEREO_RVEC, TRUE_STEREO_TVEC)

        pL = project(TRUE_CUBE, rvec, tvec, K)
        pR = project(TRUE_CUBE, right_rvec, right_tvec, K)

        pL_noisy = pL + RNG.normal(0.0, PIXEL_NOISE_STD, pL.shape)
        pR_noisy = pR + RNG.normal(0.0, PIXEL_NOISE_STD, pR.shape)

        left_obs.append(pL_noisy)
        right_obs.append(pR_noisy)

    return true_view_poses, left_obs, right_obs


def pack_params(stereo_rvec, stereo_tvec, view_poses, object_points):
    """Flatten every unknown into scipy.optimize.least_squares's expected
    1D parameter vector.

    Args:
        stereo_rvec (numpy.ndarray): 3-element stereo rotation vector.
        stereo_tvec (numpy.ndarray): 3-element stereo translation vector.
        view_poses (list[tuple[numpy.ndarray, numpy.ndarray]]): Per-view
            `(rvec, tvec)` pairs.
        object_points (numpy.ndarray): Nx3 cube point positions.

    Returns:
        numpy.ndarray: The flattened 1D parameter vector.
    """
    parts = [stereo_rvec, stereo_tvec]
    for rvec, tvec in view_poses:
        parts.append(rvec)
        parts.append(tvec)
    parts.append(object_points.ravel())
    return np.concatenate(parts)


def unpack_params(params, n_views, n_points):
    """Inverse of `pack_params`.

    Args:
        params (numpy.ndarray): The flattened 1D parameter vector.
        n_views (int): Number of calibration views.
        n_points (int): Number of object points (cube corners).

    Returns:
        tuple: `(stereo_rvec, stereo_tvec, view_poses, object_points)`,
        unpacked back into their original shapes.
    """
    idx = 0
    stereo_rvec = params[idx : idx + 3]
    idx += 3
    stereo_tvec = params[idx : idx + 3]
    idx += 3

    view_poses = []
    for _ in range(n_views):
        rvec = params[idx : idx + 3]
        idx += 3
        tvec = params[idx : idx + 3]
        idx += 3
        view_poses.append((rvec, tvec))

    object_points = params[idx : idx + 3 * n_points].reshape(n_points, 3)
    idx += 3 * n_points

    return stereo_rvec, stereo_tvec, view_poses, object_points


def residuals(params, n_views, n_points, left_obs, right_obs, anchor_points=None, anchor_weight=0.0):
    """Bundle-adjustment residual function: stacked left+right reprojection
    errors across every view, for the current parameter guess.

    Args:
        params (numpy.ndarray): Current flattened parameter guess (see
            `pack_params`/`unpack_params`).
        n_views (int): Number of calibration views.
        n_points (int): Number of object points (cube corners).
        left_obs (list[numpy.ndarray]): Per-view Nx2 noisy left-image
            observations.
        right_obs (list[numpy.ndarray]): Per-view Nx2 noisy right-image
            observations.
        anchor_points (numpy.ndarray | None): Nx3 nominal/as-designed
            object points to softly anchor the solved object points to
            (see the module docstring's "gauge/datum ambiguity" finding -
            without this, reprojection error alone does not determine
            absolute object-space scale/position/orientation). None
            disables anchoring entirely (reproduces the unconstrained,
            gauge-ambiguous behavior).
        anchor_weight (float): Weight applied to the anchor residual,
            typically `1 / expected_manufacturing_tolerance_mm` so the
            anchor residual is order-1 at the tolerance boundary and can
            still be outweighed by enough reprojection evidence to
            correct a point beyond that tolerance if warranted.

    Returns:
        numpy.ndarray: The flattened residual vector scipy.optimize.
        least_squares minimizes the sum of squares of.
    """
    stereo_rvec, stereo_tvec, view_poses, object_points = unpack_params(params, n_views, n_points)

    res = []
    for v in range(n_views):
        rvec, tvec = view_poses[v]
        right_rvec, right_tvec = compose_right_pose(rvec, tvec, stereo_rvec, stereo_tvec)

        pL = project(object_points, rvec, tvec, K)
        pR = project(object_points, right_rvec, right_tvec, K)

        res.append((pL - left_obs[v]).ravel())
        res.append((pR - right_obs[v]).ravel())

    if anchor_points is not None:
        # Soft datum constraint: without this, reprojection residuals
        # alone leave the object points' absolute scale/position/
        # orientation and the stereo baseline's absolute scale
        # underdetermined (see module docstring). Anchoring softly to
        # the known nominal/as-designed geometry - not treating it as
        # exact - is what lets the solver still correct real
        # manufacturing error rather than reproducing it verbatim.
        res.append(anchor_weight * (object_points - anchor_points).ravel())

    return np.concatenate(res)


def reprojection_rms(params, n_views, n_points, left_obs, right_obs):
    """Compute the overall pixel-space reprojection RMS for one parameter
    guess, for before/after reporting. Deliberately excludes any anchor
    residual (not pixel-space, and not part of "reprojection error").

    Args:
        params (numpy.ndarray): Flattened parameter guess.
        n_views (int): Number of calibration views.
        n_points (int): Number of object points (cube corners).
        left_obs (list[numpy.ndarray]): Per-view noisy left observations.
        right_obs (list[numpy.ndarray]): Per-view noisy right observations.

    Returns:
        float: RMS pixel reprojection error across every view/camera/point.
    """
    r = residuals(params, n_views, n_points, left_obs, right_obs)
    return float(np.sqrt(np.mean(r**2)))


ANCHOR_WEIGHT = 1.0 / MANUFACTURING_ERROR_MM
"""Soft-anchor weight passed to `residuals` when anchoring is enabled,
chosen so the anchor residual is order-1 at the assumed manufacturing
tolerance (`MANUFACTURING_ERROR_MM`) - see `residuals`'s docstring."""


def main():
    """Run the synthetic bundle-adjustment feasibility check end to end
    and print before/after results to the console.

    Returns:
        None
    """
    true_view_poses, left_obs, right_obs = simulate_views()
    n_points = NOMINAL_CUBE.shape[0]

    # ---- Build the (deliberately imperfect) initial guess ----
    # Stereo extrinsics perturbed away from the truth, standing in for an
    # approximate initial stereo estimate.
    init_stereo_rvec = TRUE_STEREO_RVEC + RNG.uniform(-0.02, 0.02, 3)
    init_stereo_tvec = TRUE_STEREO_TVEC + RNG.uniform(-5.0, 5.0, 3)

    # Per-view initial poses via solvePnP against the WRONG nominal cube
    # geometry and the noisy left-image observations - exactly what a
    # real calibration session would have available before any bundle
    # adjustment runs.
    init_view_poses = []
    for v in range(N_VIEWS):
        ok, rvec0, tvec0 = cv2.solvePnP(NOMINAL_CUBE, left_obs[v], K, None)
        init_view_poses.append((rvec0.ravel(), tvec0.ravel()))

    init_params = pack_params(init_stereo_rvec, init_stereo_tvec, init_view_poses, NOMINAL_CUBE.copy())

    rms_before = reprojection_rms(init_params, N_VIEWS, n_points, left_obs, right_obs)
    baseline_before = float(np.linalg.norm(init_stereo_tvec))
    baseline_true = float(np.linalg.norm(TRUE_STEREO_TVEC))
    cube_error_before = float(np.sqrt(np.mean(np.sum((NOMINAL_CUBE - TRUE_CUBE) ** 2, axis=1))))

    def run_and_report(label, anchor_points, anchor_weight):
        """Run one bundle-adjustment variant and print its results.

        Args:
            label (str): Human-readable label for this variant, printed
                as a section header.
            anchor_points (numpy.ndarray | None): Passed straight through
                to `residuals` - None for the unconstrained variant.
            anchor_weight (float): Passed straight through to `residuals`.

        Returns:
            None
        """
        result = least_squares(
            residuals,
            init_params,
            args=(N_VIEWS, n_points, left_obs, right_obs, anchor_points, anchor_weight),
            method="lm",
        )

        rms_after = reprojection_rms(result.x, N_VIEWS, n_points, left_obs, right_obs)
        _rvec, solved_stereo_tvec, _view_poses, solved_object_points = unpack_params(result.x, N_VIEWS, n_points)
        baseline_after = float(np.linalg.norm(solved_stereo_tvec))
        cube_error_after = float(np.sqrt(np.mean(np.sum((solved_object_points - TRUE_CUBE) ** 2, axis=1))))

        print(f"--- {label} ---")
        print(f"Reprojection RMS  - before: {rms_before:.3f}px   after: {rms_after:.3f}px")
        print(
            f"Stereo baseline   - true: {baseline_true:.3f}mm   before: {baseline_before:.3f}mm   "
            f"after: {baseline_after:.3f}mm   (error before: {abs(baseline_before - baseline_true):.3f}mm, "
            f"after: {abs(baseline_after - baseline_true):.3f}mm)"
        )
        print(
            f"Cube point RMS error vs. true manufactured geometry - "
            f"nominal/assumed: {cube_error_before:.3f}mm   after BA: {cube_error_after:.3f}mm"
        )
        print(f"scipy.optimize.least_squares status: {result.status} ({result.message})")
        print()

    print("=== Phase 14 bundle-adjustment feasibility prototype (synthetic data only) ===")
    print(f"Views simulated: {N_VIEWS}, points per view: {n_points}, pixel noise std: {PIXEL_NOISE_STD}px")
    print()

    run_and_report("Unconstrained (object points fully free - gauge/datum ambiguous)", None, 0.0)
    run_and_report(
        f"Softly anchored to nominal geometry (weight = 1/{MANUFACTURING_ERROR_MM}mm)",
        NOMINAL_CUBE,
        ANCHOR_WEIGHT,
    )


if __name__ == "__main__":
    main()
