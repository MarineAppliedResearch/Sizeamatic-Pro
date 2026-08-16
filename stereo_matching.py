"""Stereo matching and stereo measurement helpers for Sizeamatic Pro.

This module supports matched point selection, stereo triangulation,
reprojection checks, and practical uncertainty estimates for measurements
made from rectified left and right stereo camera views.

Contents:
    - Scanline based mate point guessing for rectified stereo images.
    - Pixel perturbation helpers used for uncertainty estimates.
    - Triangulation from matched left and right image coordinates.
    - Projection of reconstructed 3D points back into image coordinates.
    - Reprojection RMS error calculations.
    - Object-space stereo ray residual calculations.
    - Point depth and range uncertainty estimates (sample-standard-
      deviation and Jacobian/covariance-propagation variants).
    - Segment length and segment uncertainty estimates (sample-standard-
      deviation and Jacobian/covariance-propagation variants).

Design notes:
    Some functions receive the main application object (`app`) so they can
    access current clicked point lists, rectified view state, and loaded
    calibration data. This keeps stereo measurement behavior grouped in one
    file while preserving the application's current state model.

Assumptions:
    - Measurement points are in rectified image coordinates.
    - The loaded calibration dictionary contains rectified projection
      matrices named "PL" and "PR".
    - Calibration translation units determine the output 3D units. In this
      application, those units are normally millimeters.
    - Reprojection and perturbation based uncertainty estimates are
      practical consistency checks, not complete models of total
      measurement error.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# Standard library imports.

# Third-party imports.

# OpenCV is used for stereo triangulation, template matching, projection, and
# other image-space measurement operations.
import cv2

# NumPy is used to build OpenCV-compatible point arrays and perform vector math.
import numpy as np


def guess_mate_point_on_scanline(app, which_src, x_src, y_src, x_hint=None, search_half_width=120):
    """Guess the matching point in the opposite rectified stereo image.

    Uses a small grayscale template around the clicked source point and
    searches for the best matching patch along the same rectified scanline
    in the opposite image. This is only a measurement aid: it assumes
    rectified frames, does not prove the match is correct, and should still
    allow the user to inspect or manually adjust the guessed mate point.

    Note:
        This search assumes the rectified stereo pair is vertically aligned
        well enough that the true mate point is on the same image row as the
        clicked source point. In real footage, small calibration,
        rectification, lens, synchronization, vibration, blur, or
        click-placement errors can leave the best mate point one or more
        pixels above or below the source scanline. Because this helper only
        searches horizontally, it may miss the correct feature or choose a
        weaker match when there is residual vertical error.

    Args:
        app: The main application object, used to read the rectified-view
            flag, loaded calibration, and cached current left/right frames.
        which_src (str): Which pane was clicked, "L" or "R".
        x_src (float): Clicked source image X pixel coordinate.
        y_src (float): Clicked source image Y pixel coordinate.
        x_hint (float | None): Optional X coordinate to center the
            opposite-image search around, used for post-drag refinement
            instead of the default source-X-centered search.
        search_half_width (int): Half-width, in pixels, of the horizontal
            search range in the opposite image.

    Returns:
        tuple[float, float] | None: The guessed opposite-image point as
        (x, y), or None if the guess cannot be made safely (rectified view
        disabled, no calibration, missing frames, or the template/search
        region falls outside the image bounds).
    """

    # NOTE:
    # This search assumes the rectified stereo pair is vertically aligned well
    # enough that the true mate point is on the same image row as the clicked
    # source point. In real footage, small calibration, rectification, lens,
    # synchronization, vibration, blur, or click-placement errors can leave the
    # best mate point one or more pixels above or below the source scanline.
    # Because this helper only searches horizontally, it may miss the correct
    # feature or choose a weaker match when there is residual vertical error.

    # Measurement assistance only works in rectified view.
    if not app.view_rectified.get():
        return None

    # Calibration must be loaded so we know we are using the rectified workflow.
    if app.cal is None:
        return None

    # Select source and target frames based on which pane was clicked.
    if which_src == "L":
        src = app.current_frameL
        dst = app.current_frameR
    else:
        src = app.current_frameR
        dst = app.current_frameL

    # Both cached frames must exist.
    if src is None or dst is None:
        return None

    # Convert to grayscale for patch matching.
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Round the clicked location to integer pixel coordinates.
    x_src = int(round(x_src))
    y_src = int(round(y_src))

    # Define a small square template around the clicked source point.
    patch_r = 7
    x0 = x_src - patch_r
    x1 = x_src + patch_r + 1
    y0 = y_src - patch_r
    y1 = y_src + patch_r + 1

    # Reject clicks too close to the border for a full template patch.
    if x0 < 0 or y0 < 0 or x1 > src_gray.shape[1] or y1 > src_gray.shape[0]:
        return None

    # Extract the source template patch centered on the clicked point.
    templ = src_gray[y0:y1, x0:x1]

    # Choose the target X position around which we search.
    # For an initial auto-match, this defaults to the source X.
    # For post-drag refinement, the caller can provide the current mate X instead.
    if x_hint is None:
        x_center = x_src
    else:
        x_center = int(round(x_hint))

    # Search only along the same rectified scanline neighborhood in the target image.
    # A wide window is useful for initial guessing, while a narrow window is useful
    # for refining a point the user already dragged near the correct feature.
    sx0 = max(0, x_center - int(search_half_width))
    sx1 = min(dst_gray.shape[1], x_center + int(search_half_width) + 1)

    # Build a target strip centered on the same Y row.
    # Keep the strip tall enough for the template to slide across it.
    tx0 = sx0
    tx1 = sx1
    ty0 = y_src - patch_r
    ty1 = y_src + patch_r + 1

    # Reject if the target strip would fall outside the image.
    if ty0 < 0 or ty1 > dst_gray.shape[0]:
        return None

    # The target strip must be at least as wide as the template.
    if (tx1 - tx0) < templ.shape[1]:
        return None

    # Extract the target strip from the opposite image.
    strip = dst_gray[ty0:ty1, tx0:tx1]

    # Match the template against the strip and look for the best score.
    result = cv2.matchTemplate(strip, templ, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(result)

    # Read the best integer match location in strip coordinates.
    best_ix = int(max_loc[0])

    # Start with no subpixel offset from the integer winner.
    sub_dx = 0.0

    # Refine only if the winner has one score sample on each side.
    if best_ix > 0 and best_ix < (result.shape[1] - 1):
        # Read the local correlation scores around the winning position.
        s0 = float(result[0, best_ix - 1])
        s1 = float(result[0, best_ix])
        s2 = float(result[0, best_ix + 1])

        # Fit a local parabola and estimate the fractional peak location.
        denom = (s0 - 2.0 * s1 + s2)
        if abs(denom) > 1e-12:
            sub_dx = 0.5 * (s0 - s2) / denom

            # Clamp the refinement so noisy scores cannot jump too far.
            if sub_dx < -1.0:
                sub_dx = -1.0
            elif sub_dx > 1.0:
                sub_dx = 1.0

    # Recover the matched X position in full-image coordinates.
    # matchTemplate returns the template's top-left corner, so shift back to center.
    best_x = tx0 + best_ix + sub_dx + patch_r

    # Keep the matched point on the same rectified scanline as the source point.
    best_y = y_src

    # Return the guessed mate point in image pixel coordinates.
    return (float(best_x), float(best_y))


def triangulate_from_pixels(app, xL, yL, xR, yR):
    """Triangulate one matched stereo pixel pair into a 3D point.

    Uses the rectified left and right projection matrices from the loaded
    calibration to reconstruct one matched stereo point in 3D. The left and
    right Y values are averaged before triangulation so small manual
    vertical click differences do not directly enter the 3D solve. This
    assumes the input pixels are already in the coordinate space expected by
    PL and PR.

    Args:
        app: The main application object, used to read the loaded
            calibration's "PL"/"PR" rectified projection matrices.
        xL (float): Left rectified image X pixel coordinate.
        yL (float): Left rectified image Y pixel coordinate.
        xR (float): Right rectified image X pixel coordinate.
        yR (float): Right rectified image Y pixel coordinate.

    Returns:
        tuple[float, float, float] | None: The triangulated 3D point as
        (X, Y, Z) in calibration units, or None if the homogeneous result
        cannot be safely normalized (W too close to zero).
    """

    # In a rectified stereo pair, corresponding points should lie on the same scanline.
    # Manual clicks may differ slightly in Y between left and right, even when the user
    # picked the same physical feature. Average the two Y values so we do not feed that
    # vertical click mismatch directly into triangulation.
    y = 0.5 * (float(yL) + float(yR))

    # Build 2x1 pixel coordinate arrays for left and right using the shared rectified Y.
    ptsL = np.array([[xL], [y]], dtype=np.float64)
    ptsR = np.array([[xR], [y]], dtype=np.float64)


    # Use the left and right rectified projection matrices to reconstruct the
    # 3D point from the matched left/right image coordinates.

    # OpenCV returns the result in homogeneous coordinates, meaning the first
    # three values still need to be divided by the fourth value, W.
    Xh = cv2.triangulatePoints(app.cal["PL"], app.cal["PR"], ptsL, ptsR)

    # Read the homogeneous scale value used to normalize the 3D point.
    W = float(Xh[3, 0])

    # Reject degenerate results where W is too close to zero, because dividing
    # by it would produce an unstable or invalid 3D position.
    if abs(W) < 1e-9:
        return None

    # Convert from homogeneous coordinates into normal Cartesian coordinates.
    X = float(Xh[0, 0]) / W
    Y = float(Xh[1, 0]) / W
    Z = float(Xh[2, 0]) / W

    # Return the reconstructed 3D point in the calibration coordinate system.
    return (X, Y, Z)


def triangulate_point_pair(app, index):
    """Triangulate one clicked left/right point pair by index.

    Validates that rectified stereo measurement is currently available,
    confirms that the requested left/right point pair exists, reads the
    matched clicked pixels from the app state, and passes those pixels to
    `triangulate_from_pixels`. This keeps point-list validation separate
    from the lower-level triangulation math.

    Args:
        app: The main application object, used to read the rectified-view
            flag, loaded calibration, and clicked left/right point lists
            (`app.ptsL`, `app.ptsR`).
        index (int): Index of the matched left/right point pair to
            triangulate.

    Returns:
        tuple: `((X, Y, Z), None)` on success, or `(None, error_message)` if
        the point pair cannot be triangulated safely.
    """

    # Measurements require rectified image coordinates and rectified projection
    # matrices, so do not triangulate while the app is showing raw camera frames.
    if not app.view_rectified.get():
        return None, "Enable rectified view to measure"

    # Calibration must be loaded before triangulation because the projection
    # matrices define how left/right image points map into 3D space.
    if app.cal is None:
        return None, "Load calibration to measure"

    # The calibration dictionary must contain the rectified left and right
    # projection matrices used by OpenCV triangulation.
    if "PL" not in app.cal or "PR" not in app.cal:
        return None, "Calibration missing PL/PR projection matrices"

    # Reject negative indexes before reading the clicked point lists.
    if index < 0:
        return None, "Invalid point index"

    # Require a clicked point on both the left and right side at the same index.
    if index >= len(app.ptsL) or index >= len(app.ptsR):
        return None, "Point pair incomplete"

    # Read the matched left and right clicked points in image pixel coordinates.
    # These points are expected to belong to the rectified view and to match the
    # loaded rectified projection matrices.
    xL, yL = app.ptsL[index]
    xR, yR = app.ptsR[index]

    # Use the lower-level triangulation helper to reconstruct the clicked pixel
    # pair into one 3D point.
    P = triangulate_from_pixels(app, xL, yL, xR, yR)

    # If the lower-level triangulation failed, return a user-readable reason.
    if P is None:
        return None, "Triangulation unstable (W≈0)"

    # Return the reconstructed 3D point and no error message.
    return P, None


def project_point(P, X, Y, Z):
    """Project one 3D point into image pixel space.

    This is mainly used for reprojection checks, where a triangulated 3D
    point is projected back into the left or right image and compared
    against the original clicked image point.

    Args:
        P (numpy.ndarray): A 3x4 camera projection matrix.
        X (float): 3D point X coordinate, in the calibration coordinate
            system expected by `P`.
        Y (float): 3D point Y coordinate, in the calibration coordinate
            system expected by `P`.
        Z (float): 3D point Z coordinate, in the calibration coordinate
            system expected by `P`.

    Returns:
        tuple[float, float] | None: The projected image coordinates as
        (u, v), or None if the homogeneous projection cannot be safely
        normalized (w too close to zero).
    """

    # Build the 3D point in homogeneous form so it can be multiplied by the
    # 3x4 camera projection matrix.
    Xh = np.array([[X], [Y], [Z], [1.0]], dtype=np.float64)

    # Project the 3D point into homogeneous image coordinates.
    ph = P @ Xh

    # Read the homogeneous image scale value used to normalize the pixel point.
    w = float(ph[2, 0])

    # Reject degenerate projections where w is too close to zero, because dividing
    # by it would produce unstable or invalid image coordinates.
    if abs(w) < 1e-12:
        return None

    # Convert from homogeneous image coordinates into normal pixel coordinates.
    u = float(ph[0, 0]) / w
    v = float(ph[1, 0]) / w

    # Return the projected image-space point.
    return (u, v)


def reprojection_rms_px(app, index):
    """Compute the pixel-space reprojection RMS error for a clicked point pair.

    Triangulates one clicked stereo point pair into 3D, projects that 3D
    point back into both rectified camera images, and compares the
    projected pixels against the original clicked pixels. This gives a
    pixel-space consistency check for the selected point pair.

    Args:
        app: The main application object, used to read clicked point lists
            and loaded calibration data.
        index (int): Index of the clicked left/right point pair to
            evaluate.

    Returns:
        float | None: The combined left/right reprojection RMS error in
        pixels, or None if the point cannot be triangulated or projected
        safely.
    """

    # Triangulate the selected clicked left/right point pair into one 3D point.
    P, err = triangulate_point_pair(app, index)

    # If triangulation failed, there is no reliable 3D point to project back into
    # the images, so reprojection error cannot be computed.
    if err is not None:
        return None

    # Split the reconstructed 3D point into named coordinates for projection.
    X, Y, Z = P

    # Read the original clicked pixel coordinates for this point pair.
    xL, yL = app.ptsL[index]
    xR, yR = app.ptsR[index]

    # Project the reconstructed 3D point back into the left rectified image.
    pL = project_point(app.cal["PL"], X, Y, Z)

    # Project the reconstructed 3D point back into the right rectified image.
    pR = project_point(app.cal["PR"], X, Y, Z)

    # If either projection failed, the reprojection error cannot be trusted.
    if pL is None or pR is None:
        return None

    # Split the projected image coordinates into left and right pixel values.
    uL, vL = pL
    uR, vR = pR

    # Measure the 2D pixel distance between the left clicked point and the left
    # projected point.
    eL = ((uL - xL) ** 2 + (vL - yL) ** 2) ** 0.5

    # Measure the 2D pixel distance between the right clicked point and the right
    # projected point.
    eR = ((uR - xR) ** 2 + (vR - yR) ** 2) ** 0.5

    # Combine the left and right residuals into one RMS pixel error.
    erms = ((eL * eL + eR * eR) / 2.0) ** 0.5

    # Return the reprojection consistency error as a plain float.
    return float(erms)


def camera_center_and_ray_direction(P, x, y):
    """Recover one camera's optical center and viewing-ray direction for a
    clicked image pixel, directly from its 3x4 projection matrix.

    Splits P into its leading 3x3 submatrix M and its last column p4
    (P = [M | p4], so for a real calibrated camera P = K[R|t], M = KR and
    p4 = Kt). For a finite camera M is invertible, which gives both
    quantities directly with no special-case analytical camera model
    needed - the same "use the calibrated projection matrices directly"
    approach `triangulate_from_pixels` already uses for triangulation:

    - Camera center (world coordinates): `C = -M^-1 @ p4`, since the
      center is the point P projects to zero (`M @ C + p4 == 0`).
    - Viewing-ray direction for pixel (x, y): `M^-1 @ [x, y, 1]` (up to
      scale) - the standard back-projection formula, equivalent to
      `R^-1 @ K^-1 @ [x, y, 1]` without decomposing M into R and K
      separately.

    An earlier version of this function instead used P's homogeneous
    null space (via SVD) for the camera center and a pseudoinverse
    solution for a point on the ray - mathematically valid in general,
    but the pseudoinverse's minimum-norm solution lands exactly at a
    point at infinity whenever a camera sits exactly at the world origin
    (`p4` all zero), which is the common case for a rectified left
    camera used as the reference frame. That's not a rare edge case for
    this codebase's actual rectified calibrations, so this M/p4 form is
    used instead - it has no such degeneracy for any real finite camera.

    Args:
        P (numpy.ndarray): A 3x4 camera projection matrix (PL or PR).
        x (float): Clicked image X pixel coordinate.
        y (float): Clicked image Y pixel coordinate.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray] | None: `(center, direction)`,
        each a length-3 float64 array in the calibration's 3D coordinate
        system, with `direction` normalized to unit length - or None if
        P's leading 3x3 submatrix isn't invertible (a degenerate,
        non-finite camera, not expected for a real calibrated projection
        matrix).
    """

    M = P[:, :3]
    p4 = P[:, 3]

    # A real calibrated finite camera's M (= K @ R) is always invertible;
    # reject a degenerate/affine projection matrix rather than raising.
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None

    center = -M_inv @ p4

    # Ray direction is only defined up to scale, so normalize it to unit
    # length here rather than downstream.
    direction = M_inv @ np.array([x, y, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    return center, direction / norm


def ray_residual_mm(PL, PR, xL, yL, xR, yR):
    """Compute the closest-approach distance between two stereo viewing rays.

    This is the object-space counterpart to `reprojection_rms_px`'s
    pixel-space consistency check, and the quantity SeaGIS's EventMeasure
    documentation calls "RMS": the shortest 3D distance separating the
    left and right camera's original viewing rays for one clicked point,
    in the calibration's real-world units (normally millimeters). Unlike
    `triangulate_from_pixels`, this does not average the left/right Y
    pixel coordinates first - it uses each clicked point exactly as
    given, so a vertical click mismatch between the two images shows up
    directly here even though `triangulate_from_pixels` folds it away by
    design. `ReprojRMS(px)` and this value are not the same quantity and
    are not on the same scale (pixels vs. millimeters) - see
    `stereo_matching.py`'s module docstring.

    For two skew lines defined by point/direction pairs (C_L, d_L) and
    (C_R, d_R), the shortest distance between them is:

        |(C_R - C_L) . (d_L x d_R)| / |d_L x d_R|

    which only degenerates (division by zero) when the two rays are
    parallel - not expected for a real stereo rig with two distinct
    camera centers, but guarded against below regardless.

    Args:
        PL (numpy.ndarray): Left camera's 3x4 rectified projection matrix.
        PR (numpy.ndarray): Right camera's 3x4 rectified projection matrix.
        xL (float): Left clicked image X pixel coordinate.
        yL (float): Left clicked image Y pixel coordinate.
        xR (float): Right clicked image X pixel coordinate.
        yR (float): Right clicked image Y pixel coordinate.

    Returns:
        float | None: The shortest distance between the two viewing rays,
        in calibration units (normally millimeters), or None if either
        camera's center/ray cannot be recovered safely or the two rays
        are (numerically) parallel.
    """
    left_ray = camera_center_and_ray_direction(PL, xL, yL)
    right_ray = camera_center_and_ray_direction(PR, xR, yR)

    if left_ray is None or right_ray is None:
        return None

    center_l, dir_l = left_ray
    center_r, dir_r = right_ray

    # The cross product of the two ray directions is perpendicular to
    # both rays; its length is also the denominator of the
    # closest-distance formula below.
    cross = np.cross(dir_l, dir_r)
    cross_norm = float(np.linalg.norm(cross))

    # Parallel (or anti-parallel) rays have no unique closest-approach
    # distance via this formula - not expected in practice for two
    # distinct camera centers, but reject rather than divide by ~0.
    if cross_norm < 1e-9:
        return None

    distance = abs(float(np.dot(center_r - center_l, cross))) / cross_norm
    return distance


def stereo_ray_residual_mm(app, index):
    """Compute the object-space ray residual for one clicked point pair.

    Thin app-state wrapper around `ray_residual_mm`, mirroring
    `reprojection_rms_px`'s validation pattern: requires rectified view,
    a loaded calibration with PL/PR, and a complete point pair at
    `index`. See `ray_residual_mm`'s docstring for what this quantity
    means and how it differs from `ReprojRMS(px)`.

    Args:
        app: The main application object, used to read the
            rectified-view flag, loaded calibration, and clicked
            left/right point lists (`app.ptsL`, `app.ptsR`).
        index (int): Index of the clicked left/right point pair to
            evaluate.

    Returns:
        float | None: The ray residual in calibration units (normally
        millimeters), or None if the point pair isn't available, the
        calibration is missing PL/PR, or the residual can't be computed
        safely.
    """

    # Ray residuals are only meaningful in the same rectified pixel space
    # PL/PR were calibrated against.
    if not app.view_rectified.get():
        return None

    if app.cal is None:
        return None

    if "PL" not in app.cal or "PR" not in app.cal:
        return None

    if index < 0:
        return None

    if index >= len(app.ptsL) or index >= len(app.ptsR):
        return None

    # Read the matched left and right clicked points exactly as given -
    # deliberately not the Y-averaged pixels `triangulate_from_pixels`
    # uses, since the whole point of this metric is to see the vertical
    # mismatch triangulation otherwise folds away.
    xL, yL = app.ptsL[index]
    xR, yR = app.ptsR[index]

    return ray_residual_mm(app.cal["PL"], app.cal["PR"], xL, yL, xR, yR)


def endpoint_perturbs(app, idx, sigma_px):
    """Build the eight single-coordinate perturbations for a clicked endpoint.

    Each perturbation moves only one of xL, yL, xR, or yR by ±sigma_px while
    leaving the other image coordinates unchanged. This helper is used by
    segment uncertainty estimation to test how endpoint click error affects
    measured length.

    Args:
        app: The main application object, used to read clicked left/right
            point lists (`app.ptsL`, `app.ptsR`).
        idx (int): Index of the clicked endpoint to perturb.
        sigma_px (float): Image-space perturbation amount, in pixels.

    Returns:
        list[tuple[float, float, float, float]]: A list of eight perturbed
        (xL, yL, xR, yR) pixel coordinate tuples.
    """

    # Read the clicked left and right image coordinates for this endpoint.
    xL, yL = app.ptsL[idx]
    xR, yR = app.ptsR[idx]

    # Return one positive and one negative perturbation for each image coordinate.
    return [
        (xL + sigma_px, yL, xR, yR),
        (xL - sigma_px, yL, xR, yR),
        (xL, yL + sigma_px, xR, yR),
        (xL, yL - sigma_px, xR, yR),
        (xL, yL, xR + sigma_px, yR),
        (xL, yL, xR - sigma_px, yR),
        (xL, yL, xR, yR + sigma_px),
        (xL, yL, xR, yR - sigma_px),
    ]


def estimate_point_sigma_mm(app, index, sigma_px):
    """Estimate local depth/range uncertainty for one triangulated point.

    Estimates how sensitive one triangulated 3D point is to small
    image-space click errors. The function perturbs each left/right pixel
    coordinate by ±sigma_px, retriangulates each perturbed point pair, and
    uses the spread in resulting Z and 3D range values as a practical local
    uncertainty estimate.

    Args:
        app: The main application object, used to read clicked point lists
            and calibration data.
        index (int): Index of the matched point pair to test.
        sigma_px (float): Assumed click uncertainty, in image pixels.

    Returns:
        tuple[float, float] | None: `(sigma_Z, sigma_range)` in calibration
        units (normally millimeters), or None if the uncertainty estimate
        cannot be computed safely (baseline triangulation fails, or fewer
        than 4 perturbations triangulate successfully).
    """

    # Read the matched left and right clicked points in image pixel coordinates.
    xL, yL = app.ptsL[index]
    xR, yR = app.ptsR[index]

    # Make sure the unmodified point pair can be triangulated before estimating
    # how nearby click perturbations affect the result.
    P0 = triangulate_from_pixels(app, xL, yL, xR, yR)
    if P0 is None:
        return None

     # Build the same eight single-coordinate perturbations used by the segment
    # uncertainty estimate. Each perturbation moves only one clicked coordinate
    # by ±sigma_px while leaving the other coordinates unchanged.
    perturbs = endpoint_perturbs(app, index, sigma_px)

    # Store the perturbed depth and range values so their spread can be measured.
    Zs = []
    Rs = []

    # Triangulate each perturbed point pair and collect the resulting Z and range.
    for (pxL, pyL, pxR, pyR) in perturbs:

        # Reconstruct the 3D point from the perturbed image coordinates.
        Pp = triangulate_from_pixels(app, pxL, pyL, pxR, pyR)

        # Skip failed perturbations instead of failing the whole estimate
        # immediately.
        if Pp is None:
            continue

        # Split the perturbed 3D point into named coordinates.
        Xp, Yp, Zp = Pp

        # Compute the 3D range from the stereo coordinate origin to this point.
        Rp = (Xp * Xp + Yp * Yp + Zp * Zp) ** 0.5

        # Save the perturbed depth and range results.
        Zs.append(Zp)
        Rs.append(Rp)

    # Require enough successful perturbations for a minimally meaningful sample
    # standard deviation.
    if len(Zs) < 4:
        return None

    # Estimate depth uncertainty from the sample standard deviation of perturbed Z.
    sZ = float(np.std(np.array(Zs, dtype=np.float64), ddof=1))

    # Estimate range uncertainty from the sample standard deviation of perturbed
    # 3D range.
    sR = float(np.std(np.array(Rs, dtype=np.float64), ddof=1))

    # Return the local sensitivity estimates in calibration units.
    return (sZ, sR)


def estimate_point_sigma_mm_jacobian(app, index, sigma_px):
    """Estimate depth/range uncertainty via numerical Jacobian propagation.

    This is a more statistically formal alternative to
    `estimate_point_sigma_mm` (ROADMAP.md Phase 13), reusing the exact
    same eight perturbed pixel coordinates from `endpoint_perturbs`
    rather than a separate calculation. Where `estimate_point_sigma_mm`
    takes the sample standard deviation of all eight perturbed results
    together, this instead treats each coordinate's +/- pair as one
    central-difference partial derivative, then combines the four
    partial derivatives as an explicit propagated variance assuming
    independent, identically distributed image-coordinate noise:

        sigma_g^2 = sum_i (dg/dq_i)^2 * sigma_px^2

    `endpoint_perturbs` always perturbs by exactly +/-sigma_px, so the
    central-difference partial derivative for coordinate q_i is
    `(g(q_i + sigma_px) - g(q_i - sigma_px)) / (2 * sigma_px)`. Squaring
    that and multiplying by `sigma_px^2` cancels the `(2 * sigma_px)^2`
    denominator down to a simple `((g_plus - g_minus) / 2) ** 2` term per
    coordinate - so this reuses `endpoint_perturbs`'s existing
    perturbations, just combined arithmetically differently than the
    sample-standard-deviation version above.

    Args:
        app: The main application object, used to read clicked point
            lists and calibration data.
        index (int): Index of the matched point pair to test.
        sigma_px (float): Assumed click uncertainty, in image pixels.

    Returns:
        tuple[float, float] | None: `(sigma_Z, sigma_range)` in
        calibration units (normally millimeters), or None if the
        uncertainty estimate cannot be computed safely (baseline
        triangulation fails, or fewer than 2 of the 4 coordinate pairs
        triangulate successfully on both sides).
    """

    # Read the matched left and right clicked points in image pixel coordinates.
    xL, yL = app.ptsL[index]
    xR, yR = app.ptsR[index]

    # Make sure the unmodified point pair can be triangulated before estimating
    # how nearby click perturbations affect the result.
    P0 = triangulate_from_pixels(app, xL, yL, xR, yR)
    if P0 is None:
        return None

    # endpoint_perturbs returns eight perturbations as four consecutive
    # (+sigma_px, -sigma_px) pairs, one pair per image coordinate
    # (xL, yL, xR, yR in that order).
    perturbs = endpoint_perturbs(app, index, sigma_px)

    # Store each coordinate's central-difference half-delta for Z and
    # range, so their propagated variance can be summed below.
    z_half_deltas = []
    r_half_deltas = []

    for k in range(0, len(perturbs), 2):
        plus_px = perturbs[k]
        minus_px = perturbs[k + 1]

        Pp = triangulate_from_pixels(app, *plus_px)
        Pm = triangulate_from_pixels(app, *minus_px)

        # Skip a coordinate pair where either side failed to triangulate,
        # rather than failing the whole estimate immediately.
        if Pp is None or Pm is None:
            continue

        Xp, Yp, Zp = Pp
        Xm, Ym, Zm = Pm
        Rp = (Xp * Xp + Yp * Yp + Zp * Zp) ** 0.5
        Rm = (Xm * Xm + Ym * Ym + Zm * Zm) ** 0.5

        z_half_deltas.append((Zp - Zm) / 2.0)
        r_half_deltas.append((Rp - Rm) / 2.0)

    # Require at least half the coordinate pairs to have triangulated
    # successfully on both sides for a minimally meaningful propagation.
    if len(z_half_deltas) < 2:
        return None

    sigma_Z = float(np.sqrt(sum(d * d for d in z_half_deltas)))
    sigma_R = float(np.sqrt(sum(d * d for d in r_half_deltas)))

    return (sigma_Z, sigma_R)


def estimate_segment_sigma_len_mm_jacobian(app, i0, i1, sigma_px):
    """Estimate segment length uncertainty via numerical Jacobian propagation.

    Jacobian/covariance-propagation counterpart to
    `estimate_segment_sigma_len_mm` (ROADMAP.md Phase 13) - see that
    function's docstring for the perturbation setup, and
    `estimate_point_sigma_mm_jacobian`'s docstring for why reusing
    `endpoint_perturbs`'s existing +/-sigma_px pairs reduces the
    propagated-variance formula down to a simple sum of squared
    half-deltas. The two endpoints' eight coordinates (four from each of
    `i0`/`i1`) are treated as independent, so their propagated variances
    just add.

    Args:
        app: The main application object, used to read clicked point
            lists and calibration data.
        i0 (int): Index of the first endpoint.
        i1 (int): Index of the second endpoint.
        sigma_px (float): Assumed click uncertainty, in image pixels.

    Returns:
        tuple[float, float] | None: `(length, sigma_length)` in
        calibration units (normally millimeters), or None if the segment
        uncertainty cannot be computed safely (invalid indexes, baseline
        triangulation fails, or fewer than 4 of the 8 coordinate pairs
        triangulate successfully on both sides).
    """

    # Reject invalid negative endpoint indexes before reading point lists.
    if i0 < 0 or i1 < 0:
        return None

    # Require both endpoint indexes to exist in the left clicked point list.
    if i0 >= len(app.ptsL) or i1 >= len(app.ptsL):
        return None

    # Require both endpoint indexes to exist in the right clicked point list.
    if i0 >= len(app.ptsR) or i1 >= len(app.ptsR):
        return None

    # Triangulate both endpoints from their matched left/right clicked pixels.
    P0 = triangulate_from_pixels(app, *app.ptsL[i0], *app.ptsR[i0])
    P1 = triangulate_from_pixels(app, *app.ptsL[i1], *app.ptsR[i1])

    # If either endpoint cannot be triangulated, the segment length is invalid.
    if P0 is None or P1 is None:
        return None

    X0, Y0, Z0 = P0
    X1, Y1, Z1 = P1

    # Compute the baseline 3D segment length.
    dX = X1 - X0
    dY = Y1 - Y0
    dZ = Z1 - Z0
    L0 = (dX * dX + dY * dY + dZ * dZ) ** 0.5

    # Store each coordinate's central-difference half-delta in segment
    # length, across both endpoints' perturbations, so their propagated
    # variance can be summed below.
    half_deltas = []

    # Perturb endpoint i0's four coordinates, keeping endpoint i1 fixed at
    # its baseline 3D point.
    perturbs0 = endpoint_perturbs(app, i0, sigma_px)
    for k in range(0, len(perturbs0), 2):
        P0p = triangulate_from_pixels(app, *perturbs0[k])
        P0m = triangulate_from_pixels(app, *perturbs0[k + 1])
        if P0p is None or P0m is None:
            continue

        X0p, Y0p, Z0p = P0p
        Lp = ((X1 - X0p) ** 2 + (Y1 - Y0p) ** 2 + (Z1 - Z0p) ** 2) ** 0.5

        X0m, Y0m, Z0m = P0m
        Lm = ((X1 - X0m) ** 2 + (Y1 - Y0m) ** 2 + (Z1 - Z0m) ** 2) ** 0.5

        half_deltas.append((Lp - Lm) / 2.0)

    # Perturb endpoint i1's four coordinates, keeping endpoint i0 fixed at
    # its baseline 3D point.
    perturbs1 = endpoint_perturbs(app, i1, sigma_px)
    for k in range(0, len(perturbs1), 2):
        P1p = triangulate_from_pixels(app, *perturbs1[k])
        P1m = triangulate_from_pixels(app, *perturbs1[k + 1])
        if P1p is None or P1m is None:
            continue

        X1p, Y1p, Z1p = P1p
        Lp = ((X1p - X0) ** 2 + (Y1p - Y0) ** 2 + (Z1p - Z0) ** 2) ** 0.5

        X1m, Y1m, Z1m = P1m
        Lm = ((X1m - X0) ** 2 + (Y1m - Y0) ** 2 + (Z1m - Z0) ** 2) ** 0.5

        half_deltas.append((Lp - Lm) / 2.0)

    # Require at least half of the eight coordinate pairs to have
    # triangulated successfully on both sides for a minimally meaningful
    # propagation.
    if len(half_deltas) < 4:
        return None

    sigma_L = float(np.sqrt(sum(d * d for d in half_deltas)))

    return (L0, sigma_L)


def estimate_segment_sigma_len_mm(app, i0, i1, sigma_px):
    """Estimate local length uncertainty for a segment between two endpoints.

    Triangulates the two selected stereo endpoints into 3D, computes the
    baseline segment length, then perturbs each endpoint independently to
    estimate how much click uncertainty affects the measured length. This
    estimates local sensitivity to endpoint click error, not total
    measurement uncertainty.

    Args:
        app: The main application object, used to read clicked point lists
            and calibration data.
        i0 (int): Index of the first endpoint.
        i1 (int): Index of the second endpoint.
        sigma_px (float): Assumed click uncertainty, in image pixels.

    Returns:
        tuple[float, float] | None: `(length, sigma_length)` in calibration
        units (normally millimeters), or None if the segment uncertainty
        cannot be computed safely (invalid indexes, baseline triangulation
        fails, or fewer than 6 perturbations triangulate successfully).
    """

    # Reject invalid negative endpoint indexes before reading point lists.
    if i0 < 0 or i1 < 0:
        return None

    # Require both endpoint indexes to exist in the left clicked point list.
    if i0 >= len(app.ptsL) or i1 >= len(app.ptsL):
        return None

    # Require both endpoint indexes to exist in the right clicked point list.
    if i0 >= len(app.ptsR) or i1 >= len(app.ptsR):
        return None

    # Triangulate endpoint 0 from its matched left/right clicked pixels.
    P0 = triangulate_from_pixels(app, *app.ptsL[i0], *app.ptsR[i0])

    # Triangulate endpoint 1 from its matched left/right clicked pixels.
    P1 = triangulate_from_pixels(app, *app.ptsL[i1], *app.ptsR[i1])

    # If either endpoint cannot be triangulated, the segment length is invalid.
    if P0 is None or P1 is None:
        return None

    # Split the baseline endpoint coordinates into named values.
    X0, Y0, Z0 = P0
    X1, Y1, Z1 = P1

    # Compute the baseline 3D segment vector from endpoint 0 to endpoint 1.
    dX = X1 - X0
    dY = Y1 - Y0
    dZ = Z1 - Z0

    # Compute the baseline 3D segment length.
    L0 = (dX * dX + dY * dY + dZ * dZ) ** 0.5

    # Store perturbed segment lengths so their spread can be measured.
    Ls = []

    # Perturb endpoint 0 while keeping endpoint 1 fixed at its baseline 3D point.
    for (pxL, pyL, pxR, pyR) in endpoint_perturbs(app, i0, sigma_px):

        # Reconstruct endpoint 0 from the perturbed image coordinates.
        P0p = triangulate_from_pixels(app, pxL, pyL, pxR, pyR)

        # Skip failed perturbations instead of failing immediately.
        if P0p is None:
            continue

        # Split the perturbed endpoint 0 coordinates into named values.
        X0p, Y0p, Z0p = P0p

        # Recompute the segment vector using perturbed endpoint 0 and baseline
        # endpoint 1.
        dX = X1 - X0p
        dY = Y1 - Y0p
        dZ = Z1 - Z0p

        # Save the resulting perturbed segment length.
        Lp = (dX * dX + dY * dY + dZ * dZ) ** 0.5
        Ls.append(Lp)

    # Perturb endpoint 1 while keeping endpoint 0 fixed at its baseline 3D point.
    for (pxL, pyL, pxR, pyR) in endpoint_perturbs(app, i1, sigma_px):

        # Reconstruct endpoint 1 from the perturbed image coordinates.
        P1p = triangulate_from_pixels(app, pxL, pyL, pxR, pyR)

        # Skip failed perturbations instead of failing immediately.
        if P1p is None:
            continue

        # Split the perturbed endpoint 1 coordinates into named values.
        X1p, Y1p, Z1p = P1p

        # Recompute the segment vector using baseline endpoint 0 and perturbed
        # endpoint 1.
        dX = X1p - X0
        dY = Y1p - Y0
        dZ = Z1p - Z0

        # Save the resulting perturbed segment length.
        Lp = (dX * dX + dY * dY + dZ * dZ) ** 0.5
        Ls.append(Lp)

    # Require enough successful perturbations for a minimally useful sample
    # standard deviation.
    if len(Ls) < 6:
        return None

    # Estimate segment length uncertainty from the spread of perturbed lengths.
    sL = float(np.std(np.array(Ls, dtype=np.float64), ddof=1))

    # Return the baseline length and the estimated length uncertainty.
    return (L0, sL)
