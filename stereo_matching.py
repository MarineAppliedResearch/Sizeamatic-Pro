# -----------------------------------------------------------------------------
# stereo_matching.py
#
# Author: Isaac Travers
# Created: 2026-05-18
# Project: Sizeamatic Pro
#
# Purpose:
#   Provides stereo matching and stereo measurement helper functions for the
#   Sizeamatic Pro application.
#
#   This module supports matched point selection, stereo triangulation,
#   reprojection checks, and practical uncertainty estimates for measurements
#   made from rectified left and right stereo camera views.
#
# Contents:
#   - Scanline based mate point guessing for rectified stereo images.
#   - Pixel perturbation helpers used for uncertainty estimates.
#   - Triangulation from matched left and right image coordinates.
#   - Projection of reconstructed 3D points back into image coordinates.
#   - Reprojection RMS error calculations.
#   - Point depth and range uncertainty estimates.
#   - Segment length and segment uncertainty estimates.
#   - Formatting helpers for measurement display.
#
# Design Notes:
#   Some functions receive the main application object so they can access current
#   clicked point lists, rectified view state, and loaded calibration data. This
#   keeps stereo measurement behavior grouped in one file while preserving the
#   application's current state model.
#
# Assumptions:
#   - Measurement points are in rectified image coordinates.
#   - The loaded calibration dictionary contains rectified projection matrices
#     named "PL" and "PR".
#   - Calibration translation units determine the output 3D units. In this
#     application, those units are normally millimeters.
#   - Reprojection and perturbation based uncertainty estimates are practical
#     consistency checks, not complete models of total measurement error.
#
# Dependencies:
#   - OpenCV is used for template matching, triangulation, and projection related
#     operations.
#   - NumPy is used for OpenCV compatible arrays and numeric calculations.
# -----------------------------------------------------------------------------

# Standard library imports.

# Third-party imports.

# OpenCV is used for stereo triangulation, template matching, projection, and
# other image-space measurement operations.
import cv2

# NumPy is used to build OpenCV-compatible point arrays and perform vector math.
import numpy as np


# -------------------------------------------------------------------------
    # guess_mate_point_on_scanline
    #
    # Inputs: which_src identifies the clicked pane ("L" or "R"), x_src/y_src are
    # source image pixel coordinates, x_hint optionally centers the opposite-image
    # search, and search_half_width controls the horizontal search range.
    # Outputs: returns a guessed opposite-image point as (x, y) floats, or None if
    # the guess cannot be made safely.
    #
    # Uses a small grayscale template around the clicked source point and searches
    # for the best matching patch along the same rectified scanline in the opposite
    # image. This is only a measurement aid: it assumes rectified frames, does not
    # prove the match is correct, and should still allow the user to inspect or
    # manually adjust the guessed mate point.
    # -------------------------------------------------------------------------
def guess_mate_point_on_scanline(app, which_src, x_src, y_src, x_hint=None, search_half_width=120):
        
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


# -------------------------------------------------------------------------
# triangulate_from_pixels
#
# Inputs: xL/yL are the left rectified image pixel coordinates, and xR/yR are
# the right rectified image pixel coordinates for the same physical point.
# Outputs: returns the triangulated 3D point as (X, Y, Z) in calibration units,
# or None if the homogeneous result cannot be safely normalized.
#
# Uses the rectified left and right projection matrices from the loaded
# calibration to reconstruct one matched stereo point in 3D. The left and
# right Y values are averaged before triangulation so small manual vertical
# click differences do not directly enter the 3D solve. This assumes the
# input pixels are already in the coordinate space expected by PL and PR.
# -------------------------------------------------------------------------
def triangulate_from_pixels(app, xL, yL, xR, yR):

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


# -----------------------------------------------------------------------------
# triangulate_point_pair
#
# Inputs: app provides the rectified-view state, loaded calibration, and clicked
# left/right point lists; index selects the matched point pair to triangulate.
# Outputs: returns ((X, Y, Z), None) on success, or (None, error_message) if the
# point pair cannot be triangulated safely.
#
# Validates that rectified stereo measurement is currently available, confirms
# that the requested left/right point pair exists, reads the matched clicked
# pixels from the app state, and passes those pixels to triangulate_from_pixels.
# This keeps point-list validation separate from the lower-level triangulation
# math.
# -----------------------------------------------------------------------------
def triangulate_point_pair(app, index):

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


# -----------------------------------------------------------------------------
# project_point
#
# Inputs: P is a 3x4 projection matrix, and X/Y/Z are the 3D point coordinates in
# the same calibration coordinate system expected by that projection matrix.
# Outputs: returns projected image coordinates as (u, v) floats, or None if the
# homogeneous projection cannot be safely normalized.
#
# Projects one 3D point back into image pixel space. This is mainly used for
# reprojection checks, where a triangulated 3D point is projected back into the
# left or right image and compared against the original clicked image point.
# -----------------------------------------------------------------------------
def project_point(P, X, Y, Z):

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


# -----------------------------------------------------------------------------
# reprojection_rms_px
#
# Inputs: app provides the clicked point lists and loaded calibration data; index
# selects which clicked left/right point pair to evaluate.
# Outputs: returns the left/right reprojection RMS error in pixels, or None if the
# point cannot be triangulated or projected safely.
#
# Triangulates one clicked stereo point pair into 3D, projects that 3D point back
# into both rectified camera images, and compares the projected pixels against the
# original clicked pixels. This gives a pixel-space consistency check for the
# selected point pair.
# -----------------------------------------------------------------------------
def reprojection_rms_px(app, index):

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


# -----------------------------------------------------------------------------
# endpoint_perturbs
#
# Inputs: app provides the clicked left/right point lists, idx selects the endpoint
# point to perturb, and sigma_px is the image-space perturbation amount in pixels.
# Outputs: returns a list of perturbed left/right pixel coordinate tuples.
#
# Builds the eight single-coordinate perturbations for one clicked stereo endpoint.
# Each perturbation moves only one of xL, yL, xR, or yR by ±sigma_px while leaving
# the other image coordinates unchanged. This helper is used by segment uncertainty
# estimation to test how endpoint click error affects measured length.
# -----------------------------------------------------------------------------
def endpoint_perturbs(app, idx, sigma_px):

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


# -----------------------------------------------------------------------------
# estimate_point_sigma_mm
#
# Inputs: app provides clicked point lists and calibration data, index selects the
# matched point pair to test, and sigma_px is the assumed click uncertainty in
# image pixels.
# Outputs: returns (sigma_Z, sigma_range) in calibration units, normally
# millimeters, or None if the uncertainty estimate cannot be computed safely.
#
# Estimates how sensitive one triangulated 3D point is to small image-space click
# errors. The function perturbs each left/right pixel coordinate by ±sigma_px,
# retriangulates each perturbed point pair, and uses the spread in resulting Z
# and 3D range values as a practical local uncertainty estimate.
# -----------------------------------------------------------------------------
def estimate_point_sigma_mm(app, index, sigma_px):

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


# -----------------------------------------------------------------------------
# estimate_segment_sigma_len_mm
#
# Inputs: app provides clicked point lists and calibration data, i0/i1 select the
# two matched stereo endpoints, and sigma_px is the assumed click uncertainty in
# image pixels.
# Outputs: returns (length, sigma_length) in calibration units, normally
# millimeters, or None if the segment uncertainty cannot be computed safely.
#
# Triangulates the two selected stereo endpoints into 3D, computes the baseline
# segment length, then perturbs each endpoint independently to estimate how much
# click uncertainty affects the measured length. This estimates local sensitivity
# to endpoint click error, not total measurement uncertainty.
# -----------------------------------------------------------------------------
def estimate_segment_sigma_len_mm(app, i0, i1, sigma_px):

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