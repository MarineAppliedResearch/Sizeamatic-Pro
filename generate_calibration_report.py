# OpenCV Stereo Calibration Report Generator
#
# Author:
# Date:
#
# Purpose:
#   Read OpenCV stereo calibration .npz files and generate a responsive HTML report
#   with tables, graphs, and layman friendly explanations.
#
# Output:
#   Creates a new timestamped report folder containing:
#     index.html
#     calibration_report.txt
#     assets/
#       report_data.json
#       plot_*.png
#       calibration_*.npz (copies of inputs)
#       chart.umd.min.js (optional local Chart.js, if you place it there)
#
# Intended input files (in one directory):
#   calibration_intrinsics.npz
#   calibration_extrinsics.npz
#   calibration_rectification.npz
#   calibration_maps.npz (optional)
# python generate_calibration_report.py --calib_dir "examples/calibration" --out_root "./examples/calibration"


import os
import re
import json
import math
import time
import shutil
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Return current timestamp string for folder naming
def get_timestamp_string():

    # Use local time for human readability
    return time.strftime("%Y%m%d_%H%M%S")


# Clamp a number into a safe range
def clamp(value, min_value, max_value):

    # Prevent numeric issues in acos and related operations
    return max(min_value, min(max_value, value))


# Convert radians to degrees
def rad_to_deg(radians):

    # Convert angle units for humans
    return float(radians) * 180.0 / math.pi


# Load a .npz file into a standard python dict
def load_npz_dict(npz_path):

    # Ensure we fail loudly if a required file is missing
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    # Load arrays from disk
    data = np.load(str(npz_path), allow_pickle=True)

    # Convert to a normal dict
    return {k: data[k] for k in data.files}


# Write a dict as pretty JSON
def write_json(path, payload):

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON in a readable format
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Write a plain text file
def write_text(path, text):

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write text with utf-8 encoding
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# Save a matplotlib figure to a file
def save_plot(fig, out_path):

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with decent resolution for reports
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")

    # Close to avoid memory creep across multiple plots
    plt.close(fig)


# Compute field of view from focal length in pixels and image dimension in pixels
def compute_fov_degrees(f_pixels, size_pixels):

    # Use pinhole model approximation
    f_pixels = float(f_pixels)
    size_pixels = float(size_pixels)

    # Guard against division by zero
    if f_pixels <= 0.0:
        return float("nan")

    # FOV = 2*atan((size/2)/f)
    return rad_to_deg(2.0 * math.atan((size_pixels * 0.5) / f_pixels))


# Compute diagonal field of view from focal length and image width and height
def compute_diag_fov_degrees(f_pixels, width_pixels, height_pixels):

    # Compute half diagonal in pixels
    half_diag = math.sqrt((float(width_pixels) * 0.5) ** 2 + (float(height_pixels) * 0.5) ** 2)

    # Guard against division by zero
    if float(f_pixels) <= 0.0:
        return float("nan")

    # FOV = 2*atan((diag/2)/f)
    return rad_to_deg(2.0 * math.atan(half_diag / float(f_pixels)))


# Compute rotation angle between cameras from rotation matrix
def compute_rotation_angle_degrees(R):

    # Extract trace for angle
    tr = float(np.trace(R))

    # Convert trace to cos(theta)
    c = (tr - 1.0) * 0.5

    # Clamp for numerical stability
    c = clamp(c, -1.0, 1.0)

    # Convert to angle
    return rad_to_deg(math.acos(c))


# Compute baseline length from translation vector
def compute_baseline_units(T):

    # Baseline is the norm of the translation vector
    return float(np.linalg.norm(T.reshape(-1)))


# Convert units to meters using a provided scale
def units_to_meters(value_units, mm_per_unit):

    # Convert to millimeters then to meters
    mm = float(value_units) * float(mm_per_unit)

    # Convert mm to meters
    return mm / 1000.0


# Convert meters to calibration units using a provided scale
def meters_to_units(value_m, mm_per_unit):

    # Convert meters to millimeters
    mm = float(value_m) * 1000.0

    # Convert millimeters to calibration units
    return mm / float(mm_per_unit)


# Compute stereo depth and error curves using rectified focal length and baseline
def compute_depth_error_curves(depth_m_array, f_rect_px, baseline_units, mm_per_unit, sigma_disp_px):

    # Convert baseline to meters for formula usage
    baseline_m = units_to_meters(baseline_units, mm_per_unit)

    # Use rectified focal length in pixels
    f = float(f_rect_px)

    # Convert sigma disparity to float
    sigma_d = float(sigma_disp_px)

    # Prepare output arrays
    z_m = depth_m_array.astype(np.float64)

    # Prevent divide by zero and negative depths
    z_m = np.maximum(z_m, 1e-9)

    # Compute disparity in pixels: d = f*B/Z
    disparity_px = (f * baseline_m) / z_m

    # Compute absolute depth uncertainty: sigma_Z = (Z^2/(f*B)) * sigma_d
    sigma_z_m = (z_m * z_m / (f * baseline_m)) * sigma_d

    # Compute relative depth error: sigma_Z / Z
    rel_sigma = sigma_z_m / z_m

    # Return a packed dict
    return {
        "depth_m": z_m,
        "disparity_px": disparity_px,
        "sigma_z_m": sigma_z_m,
        "rel_sigma": rel_sigma,
    }


# Compute length measurement uncertainty curves at depth
def compute_length_error_curves(depth_m_array, f_rect_px, sigma_len_px, example_length_mm):

    # Use rectified focal length in pixels
    f = float(f_rect_px)

    # Convert pixel selection uncertainty to float
    sigma_px = float(sigma_len_px)

    # Convert example length to meters
    example_length_m = float(example_length_mm) / 1000.0

    # Prepare output arrays
    z_m = depth_m_array.astype(np.float64)

    # Prevent negative depths
    z_m = np.maximum(z_m, 1e-9)

    # Compute mm per pixel at depth: meters per pixel = Z / f
    m_per_px = z_m / f

    # Convert to mm per pixel for a nicer human unit
    mm_per_px = m_per_px * 1000.0

    # Compute length uncertainty in meters: sigma_L ≈ (Z/f) * sigma_px
    sigma_L_m = m_per_px * sigma_px

    # Compute percent error for the example object
    percent_error = (sigma_L_m / example_length_m) * 100.0

    # Return a packed dict
    return {
        "depth_m": z_m,
        "mm_per_px": mm_per_px,
        "sigma_L_mm": sigma_L_m * 1000.0,
        "percent_error_example": percent_error,
        "example_length_mm": float(example_length_mm),
    }


# Classify depth bands based on relative depth error thresholds
def compute_depth_bands(depth_m_array, rel_sigma_array, thresholds):

    # Sort thresholds so the output is stable
    thresholds_sorted = sorted([float(t) for t in thresholds])

    # Prepare band results
    bands = []

    # For each threshold, compute the max depth that still satisfies it
    for t in thresholds_sorted:

        # Find depths where we meet the criterion
        ok = rel_sigma_array <= (t / 100.0)

        # If none meet the threshold, report empty range
        if not np.any(ok):
            bands.append({
                "threshold_percent": t,
                "max_depth_m": None,
                "range_m": [None, None],
            })
            continue

        # Compute contiguous range from the start of the depth grid
        max_depth = float(np.max(depth_m_array[ok]))

        # Report as 0 to max_depth for simple interpretation
        bands.append({
            "threshold_percent": t,
            "max_depth_m": max_depth,
            "range_m": [float(np.min(depth_m_array[ok])), max_depth],
        })

    # Return band descriptors
    return bands


# Compute point estimates at specific distances
def compute_point_estimates(distances_m, curve_pack):

    # Use arrays from the curve pack
    z = curve_pack["depth_m"]
    disparity = curve_pack["disparity_px"]
    sigma_z = curve_pack["sigma_z_m"]
    rel = curve_pack["rel_sigma"]

    # Prepare table rows
    rows = []

    # For each distance, interpolate from curve arrays
    for d in distances_m:

        # Clamp requested distance into the curve range
        d = float(d)
        d = clamp(d, float(np.min(z)), float(np.max(z)))

        # Interpolate values
        disp_i = float(np.interp(d, z, disparity))
        sig_i = float(np.interp(d, z, sigma_z))
        rel_i = float(np.interp(d, z, rel)) * 100.0

        # Append a row
        rows.append({
            "depth_m": float(d),
            "disparity_px": disp_i,
            "sigma_z_m": sig_i,
            "rel_sigma_percent": rel_i,
        })

    # Return rows
    return rows


# Compute warp statistics and images from undistort rectify maps
def compute_warp_products(maps_npz, image_width, image_height):

    # Extract maps
    mapLx = maps_npz["mapLx"]
    mapLy = maps_npz["mapLy"]
    mapRx = maps_npz["mapRx"]
    mapRy = maps_npz["mapRy"]

    # Build pixel coordinate grids for displacement calculation
    xs = np.arange(image_width, dtype=np.float32)
    ys = np.arange(image_height, dtype=np.float32)

    # Make full grids
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Compute displacement magnitude for left
    dxL = mapLx - grid_x
    dyL = mapLy - grid_y
    magL = np.sqrt(dxL * dxL + dyL * dyL)

    # Compute displacement magnitude for right
    dxR = mapRx - grid_x
    dyR = mapRy - grid_y
    magR = np.sqrt(dxR * dxR + dyR * dyR)

    # Compute validity masks where the remap samples inside the source image
    validL = (mapLx >= 0.0) & (mapLx < (image_width - 1)) & (mapLy >= 0.0) & (mapLy < (image_height - 1))
    validR = (mapRx >= 0.0) & (mapRx < (image_width - 1)) & (mapRy >= 0.0) & (mapRy < (image_height - 1))

    # Compute usable fraction
    usableL = float(np.mean(validL))
    usableR = float(np.mean(validR))

    # Compute histograms for magnitude using only valid pixels
    magL_valid = magL[validL]
    magR_valid = magR[validR]

    # Choose histogram bins in pixels
    bins = 80
    max_mag = float(max(np.max(magL_valid) if magL_valid.size else 0.0, np.max(magR_valid) if magR_valid.size else 0.0))

    # Guard against empty or degenerate cases
    if max_mag <= 1e-9:
        max_mag = 1.0

    # Build histogram edges
    hist_edges = np.linspace(0.0, max_mag, bins + 1)

    # Compute hist counts
    histL, _ = np.histogram(magL_valid, bins=hist_edges)
    histR, _ = np.histogram(magR_valid, bins=hist_edges)

    # Compute bin centers for charting
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    # Return packed products
    return {
        "magL": magL,
        "magR": magR,
        "validL": validL.astype(np.uint8),
        "validR": validR.astype(np.uint8),
        "usable_fraction_L": usableL,
        "usable_fraction_R": usableR,
        "warp_hist_bin_centers": centers,
        "warp_hist_counts_L": histL.astype(np.int64),
        "warp_hist_counts_R": histR.astype(np.int64),
        "warp_mag_max": max_mag,
    }


# Generate core PNG plots (static figures)
def generate_static_plots(out_assets_dir, depth_pack, length_pack, depth_bands, warp_pack_or_none):

    # Plot relative depth error vs depth
    fig = plt.figure()
    plt.plot(depth_pack["depth_m"], depth_pack["rel_sigma"] * 100.0)

    plt.xlabel("Depth (m)")
    plt.ylabel("Relative depth error estimate (%)")

    plt.title("Relative depth error vs depth")

    # Add threshold lines for bands
    for band in depth_bands:
        t = float(band["threshold_percent"])
        plt.axhline(y=t, linestyle="--")

    save_plot(fig, out_assets_dir / "plot_depth_relative_error.png")

    # Plot absolute depth error vs depth
    fig = plt.figure()
    plt.plot(depth_pack["depth_m"], depth_pack["sigma_z_m"])

    plt.xlabel("Depth (m)")
    plt.ylabel("Absolute depth error estimate (m)")

    plt.title("Absolute depth error vs depth")

    save_plot(fig, out_assets_dir / "plot_depth_absolute_error.png")

    # Plot disparity vs depth
    fig = plt.figure()
    plt.plot(depth_pack["depth_m"], depth_pack["disparity_px"])

    plt.xlabel("Depth (m)")
    plt.ylabel("Disparity (px)")

    plt.title("Disparity vs depth")

    save_plot(fig, out_assets_dir / "plot_disparity_vs_depth.png")

    # Plot mm per pixel vs depth
    fig = plt.figure()
    plt.plot(length_pack["depth_m"], length_pack["mm_per_px"])

    plt.xlabel("Depth (m)")
    plt.ylabel("mm per pixel")

    plt.title("Scale (mm per pixel) vs depth")

    save_plot(fig, out_assets_dir / "plot_mm_per_pixel_vs_depth.png")

    # Plot length uncertainty vs depth
    fig = plt.figure()
    plt.plot(length_pack["depth_m"], length_pack["sigma_L_mm"])

    plt.xlabel("Depth (m)")
    plt.ylabel("Estimated length error (mm)")

    plt.title("Estimated length error vs depth")

    save_plot(fig, out_assets_dir / "plot_length_error_mm_vs_depth.png")

    # Plot percent error for example length vs depth
    fig = plt.figure()
    plt.plot(length_pack["depth_m"], length_pack["percent_error_example"])

    plt.xlabel("Depth (m)")
    plt.ylabel("Estimated length error (% of example)")

    plt.title(f"Estimated length percent error vs depth (example = {length_pack['example_length_mm']:.0f} mm)")

    save_plot(fig, out_assets_dir / "plot_length_percent_error_vs_depth.png")

    # If maps were available, plot warp visuals
    if warp_pack_or_none is not None:

        # Plot left warp heatmap
        fig = plt.figure()
        plt.imshow(warp_pack_or_none["magL"])
        plt.colorbar()

        plt.title("Left warp magnitude heatmap (px)")

        save_plot(fig, out_assets_dir / "plot_warp_heatmap_left.png")

        # Plot right warp heatmap
        fig = plt.figure()
        plt.imshow(warp_pack_or_none["magR"])
        plt.colorbar()

        plt.title("Right warp magnitude heatmap (px)")

        save_plot(fig, out_assets_dir / "plot_warp_heatmap_right.png")

        # Plot left valid mask
        fig = plt.figure()
        plt.imshow(warp_pack_or_none["validL"], vmin=0, vmax=1)

        plt.title("Left valid remap region (1=valid)")

        save_plot(fig, out_assets_dir / "plot_valid_mask_left.png")

        # Plot right valid mask
        fig = plt.figure()
        plt.imshow(warp_pack_or_none["validR"], vmin=0, vmax=1)

        plt.title("Right valid remap region (1=valid)")

        save_plot(fig, out_assets_dir / "plot_valid_mask_right.png")

        # Plot warp histogram
        fig = plt.figure()
        plt.plot(warp_pack_or_none["warp_hist_bin_centers"], warp_pack_or_none["warp_hist_counts_L"], label="Left")
        plt.plot(warp_pack_or_none["warp_hist_bin_centers"], warp_pack_or_none["warp_hist_counts_R"], label="Right")

        plt.xlabel("Warp magnitude (px)")
        plt.ylabel("Pixel count")

        plt.title("Warp magnitude histogram (valid pixels only)")

        plt.legend()

        save_plot(fig, out_assets_dir / "plot_warp_histogram.png")


# Create a human readable text report
def build_text_report(report_data):

    # Start with a compact header
    lines = []
    lines.append("Stereo Calibration Report")
    lines.append("")
    lines.append(f"Generated: {report_data['meta']['generated_local']}")
    lines.append(f"Input directory: {report_data['meta']['input_directory']}")
    lines.append("")

    # Summarize key numbers
    lines.append("Summary")
    lines.append(f"  Stereo RMS: {report_data['summary']['stereo_rms']:.6f}")
    lines.append(f"  Baseline: {report_data['summary']['baseline_units']:.6f} units ({report_data['summary']['baseline_m']:.6f} m)")
    lines.append(f"  Camera rotation angle: {report_data['summary']['rotation_deg']:.6f} deg")
    lines.append(f"  Rectified focal length fx: {report_data['summary']['f_rect_px']:.6f} px")
    lines.append("")

    # Add point estimates table
    lines.append("Depth error point estimates")
    lines.append("  depth_m    disparity_px    sigma_z_m    rel_sigma_percent")
    for row in report_data["depth_point_estimates"]:
        lines.append(f"  {row['depth_m']:6.2f}    {row['disparity_px']:11.3f}    {row['sigma_z_m']:9.6f}    {row['rel_sigma_percent']:15.3f}")
    lines.append("")

    # Add intrinsics
    lines.append("Left intrinsics")
    lines.append(f"  fx fy cx cy: {report_data['left']['fx']:.6f}  {report_data['left']['fy']:.6f}  {report_data['left']['cx']:.6f}  {report_data['left']['cy']:.6f}")
    lines.append(f"  FOV (h v d): {report_data['left']['fov_h_deg']:.3f}  {report_data['left']['fov_v_deg']:.3f}  {report_data['left']['fov_d_deg']:.3f}")
    lines.append(f"  distortion: {', '.join(str(x) for x in report_data['left']['distortion'])}")
    lines.append("")

    lines.append("Right intrinsics")
    lines.append(f"  fx fy cx cy: {report_data['right']['fx']:.6f}  {report_data['right']['fy']:.6f}  {report_data['right']['cx']:.6f}  {report_data['right']['cy']:.6f}")
    lines.append(f"  FOV (h v d): {report_data['right']['fov_h_deg']:.3f}  {report_data['right']['fov_v_deg']:.3f}  {report_data['right']['fov_d_deg']:.3f}")
    lines.append(f"  distortion: {', '.join(str(x) for x in report_data['right']['distortion'])}")
    lines.append("")

    # Add depth bands
    lines.append("Depth bands (relative depth error thresholds)")
    for band in report_data["depth_bands"]:
        t = band["threshold_percent"]
        max_d = band["max_depth_m"]
        if max_d is None:
            lines.append(f"  <= {t:.1f}% : no depths in evaluated range meet this threshold")
        else:
            lines.append(f"  <= {t:.1f}% : 0.0 m to {max_d:.3f} m")
    lines.append("")

    # Add assumptions
    lines.append("Assumptions")
    lines.append(f"  sigma disparity (px): {report_data['assumptions']['sigma_disp_px']}")
    lines.append(f"  sigma length pick (px): {report_data['assumptions']['sigma_len_px']}")
    lines.append(f"  example length (mm): {report_data['assumptions']['example_length_mm']}")
    lines.append(f"  mm per unit: {report_data['assumptions']['mm_per_unit']}")
    lines.append("")

    # Add maps summary if present
    if report_data.get("warp") is not None:
        lines.append("Warp / rectification map summary")
        lines.append(f"  usable fraction left: {report_data['warp']['usable_fraction_L']:.4f}")
        lines.append(f"  usable fraction right: {report_data['warp']['usable_fraction_R']:.4f}")
        lines.append(f"  warp magnitude max (px): {report_data['warp']['warp_mag_max']:.3f}")
        lines.append("")

    # Return final report text
    return "\n".join(lines)


# Build the HTML report body with Bootstrap, tooltips, and Chart.js
def build_html(report_data):

    # Use local Chart.js if present, otherwise fallback to CDN
    chart_js_local = "assets/chart.umd.min.js"
    chart_js_cdn = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

    # Convert report data to a JSON string we can inline into the HTML
    report_data_json = json.dumps(report_data)

    # Provide a short tooltip dictionary
    tooltip_defs = {
        "RMS": "Root Mean Square reprojection error from calibration. Lower is generally better.",
        "Baseline": "Distance between camera centers. Larger baseline improves depth accuracy but can increase occlusion.",
        "Disparity": "Horizontal pixel shift between left and right views in rectified images.",
        "Rectification": "Transforms that align stereo images so epipolar lines are horizontal.",
        "Intrinsics": "Camera internal parameters: focal lengths and optical center (fx, fy, cx, cy) and distortion.",
        "Extrinsics": "Camera to camera pose: rotation R and translation T.",
        "fx": "Focal length in pixels (horizontal). Larger fx means smaller mm per pixel at a given depth.",
        "Distortion": "Lens and port distortion coefficients used to undistort the image.",
        "Q": "Reprojection matrix used to convert disparity to 3D coordinates in rectified stereo.",
        "RelativeDepthError": "Estimated depth uncertainty divided by depth. Lower is better.",
    }

    # Serialize tooltip defs into JS
    tooltip_json = json.dumps(tooltip_defs)

    # Build HTML with embedded JS to render tables and charts from report_data.json
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>Stereo Calibration Report</title>

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

  <style>
    body {{ background: #0b1220; color: #e8eefc; }}
    .card {{ background: #111a2e; border: 1px solid rgba(255,255,255,0.08); }}
    .muted {{ color: rgba(232,238,252,0.75); }}
    .term {{ text-decoration: underline dotted; cursor: help; }}
    .figure-img {{ border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }}
    pre {{ background: #0f172a; padding: 12px; border-radius: 12px; color: #e8eefc; }}
    a {{ color: #93c5fd; }}
  </style>
</head>

<body>
  <div class="container-fluid py-4 px-3 px-md-5">

    <div class="d-flex flex-column flex-md-row align-items-md-end justify-content-between mb-3 text-light">
      <div>
        <h1 class="mb-1">Stereo Calibration Report</h1>
        <div class="muted">Generated: <span id="generatedTime"></span></div>
        <div class="muted">Input directory: <span id="inputDir"></span></div>
      </div>

      <div class="mt-3 mt-md-0">
        <a class="btn btn-outline-light btn-sm" href="assets/report_data.json">Download report_data.json</a>
      </div>
    </div>

    <div class="row g-3 mb-3 text-light">
      <div class="col-12 col-md-3">
        <div class="card p-3 h-100">
          <div class="muted">
            <span class="term" data-term="RMS">Stereo RMS</span>
          </div>
          <div class="fs-4 fw-semibold text-light" id="stereoRms"></div>
          <div class="muted mt-2">Overall reprojection fit of the stereo solve.</div>
        </div>
      </div>

      <div class="col-12 col-md-3">
        <div class="card p-3 h-100">
          <div class="muted">
            <span class="term" data-term="Baseline">Baseline</span>
          </div>
          <div class="fs-4 fw-semibold text-light" id="baselineM"></div>
          <div class="muted mt-2">Primary lever for depth accuracy in stereo.</div>
        </div>
      </div>

      <div class="col-12 col-md-3">
        <div class="card p-3 h-100">
          <div class="muted">Camera rotation</div>
          <div class="fs-4 fw-semibold text-light" id="rotationDeg"></div>
          <div class="muted mt-2">Angle between camera orientations from <span class="term" data-term="Extrinsics">extrinsics</span>.</div>
        </div>
      </div>

      <div class="col-12 col-md-3">
        <div class="card p-3 h-100">
          <div class="muted">Rectified fx</div>
          <div class="fs-4 fw-semibold text-light" id="fRect"></div>
          <div class="muted mt-2">Effective focal length after <span class="term" data-term="Rectification">rectification</span>.</div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-3 text-light">
      <div class="col-12 col-lg-7">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Depth accuracy overview</h3>
          <div class="muted mb-3 text-light">
            This chart shows estimated <span class="term" data-term="RelativeDepthError">relative depth error</span> vs distance, using your baseline and focal length.
          </div>

          <canvas id="chartRelDepthError"></canvas>

          <div class="muted mt-3">
            Band lines represent thresholds of 1%, 3%, and 10% relative error (configurable in the generator).
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-5">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Point estimates</h3>
          <div class="muted mb-3 text-light">
            Estimated depth error at 1 m, 5 m, 10 m, and 20 m.
          </div>

          <div class="table-responsive">
            <table class="table table-dark table-sm align-middle mb-0">
              <thead>
                <tr>
                  <th>Depth (m)</th>
                  <th><span class="term" data-term="Disparity">Disparity</span> (px)</th>
                  <th>σZ (m)</th>
                  <th>σZ/Z (%)</th>
                </tr>
              </thead>
              <tbody id="pointTableBody"></tbody>
            </table>
          </div>

          <div class="muted mt-3">
            These numbers assume a disparity uncertainty of σd set in the report assumptions.
          </div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-3 text-light">
      <div class="col-12 col-lg-6">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Intrinsics</h3>
          <div class="muted mb-3 text-light">
            <span class="term" data-term="Intrinsics">Intrinsics</span> summarize the pinhole camera model and distortion fit.
          </div>

          <div class="row g-3">
            <div class="col-12 col-md-6">
              <div class="card p-3">
                <div class="fw-semibold text-light mb-2 text-light">Left</div>
                <div class="table-responsive">
                  <table class="table table-dark table-sm mb-0">
                    <tbody id="intrinsicsLeft"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div class="col-12 col-md-6">
              <div class="card p-3">
                <div class="fw-semibold text-light mb-2 text-light">Right</div>
                <div class="table-responsive">
                  <table class="table table-dark table-sm mb-0">
                    <tbody id="intrinsicsRight"></tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div class="muted mt-3">
            FOV values are derived from focal length and image size under a pinhole approximation.
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-6">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Length measurement estimates</h3>
          <div class="muted mb-3 text-light">
            These curves estimate how pixel measurement error maps into millimeters at a given depth.
          </div>

          <canvas id="chartMmPerPixel"></canvas>

          <div class="muted mt-3">
            Assumes a human endpoint picking uncertainty σpx from the assumptions section.
          </div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-3 text-light">
      <div class="col-12">
        <div class="card p-3">
          <h3 class="mb-2 text-light">Figures (static PNGs)</h3>
          <div class="muted mb-3 text-light">
            These are saved to assets/ so the report is easy to archive and share.
          </div>

          <div class="row g-3" id="figureGrid"></div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-3 text-light">
      <div class="col-12 col-lg-6">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Assumptions</h3>
          <div class="muted mb-3 text-light">
            Some accuracy estimates depend on assumed matching and measurement noise.
          </div>

          <div class="table-responsive">
            <table class="table table-dark table-sm mb-0">
              <tbody id="assumptionsBody"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-6">
        <div class="card p-3 h-100">
          <h3 class="mb-2 text-light">Calibration files</h3>
          <div class="muted mb-3 text-light text-light">
            Copied into assets/ so this report is self contained.
          </div>

          <ul id="calFiles" class="mb-0"></ul>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-5">
      <div class="col-12">
        <div class="card p-3">
          <h3 class="mb-2 text-light">Engineer details (raw matrices)</h3>
          <div class="muted mb-3 text-light">
            Expand to see raw matrices for debugging or downstream processing.
          </div>

          <details>
            <summary class="muted">Show matrices</summary>
            <pre id="matrixDump" class="mt-3 mb-0"></pre>
          </details>
        </div>
      </div>
    </div>

  </div>

  <script>
    const TOOLTIP_DEFS = {tooltip_json};
     // Report data is embedded so this HTML file works offline with file://
    window.REPORT_DATA = {report_data_json};

    function termTooltipText(term) {{

      return TOOLTIP_DEFS[term] || "No tooltip definition found.";
    }}

    function initTermTooltips() {{

      const elems = document.querySelectorAll(".term");

      elems.forEach(el => {{

        const term = el.getAttribute("data-term");
        el.setAttribute("data-bs-toggle", "tooltip");
        el.setAttribute("data-bs-title", termTooltipText(term));
      }});

      const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipTriggerList.map(function (tooltipTriggerEl) {{
        return new bootstrap.Tooltip(tooltipTriggerEl);
      }});
    }}

        // Load report data from an inline JSON object so file:// works without CORS issues
    async function loadReportData() {{

      return window.REPORT_DATA;
    }}

    function addKeyValueRow(tbody, key, value) {{

      const tr = document.createElement("tr");

      const tdKey = document.createElement("td");
      tdKey.textContent = key;

      const tdVal = document.createElement("td");
      tdVal.textContent = value;

      tr.appendChild(tdKey);
      tr.appendChild(tdVal);

      tbody.appendChild(tr);
    }}

    function formatNumber(value, digits=6) {{

      if (value === null || value === undefined) {{
        return "n/a";
      }}

      if (typeof value !== "number") {{
        return String(value);
      }}

      return value.toFixed(digits);
    }}

    function insertIntrinsics(tbody, intr) {{

      addKeyValueRow(tbody, "fx (px)", formatNumber(intr.fx, 3));
      addKeyValueRow(tbody, "fy (px)", formatNumber(intr.fy, 3));
      addKeyValueRow(tbody, "cx (px)", formatNumber(intr.cx, 3));
      addKeyValueRow(tbody, "cy (px)", formatNumber(intr.cy, 3));

      addKeyValueRow(tbody, "FOV h (deg)", formatNumber(intr.fov_h_deg, 3));
      addKeyValueRow(tbody, "FOV v (deg)", formatNumber(intr.fov_v_deg, 3));
      addKeyValueRow(tbody, "FOV d (deg)", formatNumber(intr.fov_d_deg, 3));

      addKeyValueRow(tbody, "distortion", intr.distortion.join(", "));
    }}

    function insertPointTable(tbody, rows) {{

      rows.forEach(r => {{

        const tr = document.createElement("tr");

        const tdD = document.createElement("td");
        tdD.textContent = r.depth_m.toFixed(2);

        const tdDisp = document.createElement("td");
        tdDisp.textContent = r.disparity_px.toFixed(3);

        const tdSig = document.createElement("td");
        tdSig.textContent = r.sigma_z_m.toFixed(4);

        const tdRel = document.createElement("td");
        tdRel.textContent = r.rel_sigma_percent.toFixed(2);

        tr.appendChild(tdD);
        tr.appendChild(tdDisp);
        tr.appendChild(tdSig);
        tr.appendChild(tdRel);

        tbody.appendChild(tr);
      }});
    }}

    function insertAssumptions(tbody, a) {{

      addKeyValueRow(tbody, "σ disparity (px)", a.sigma_disp_px);
      addKeyValueRow(tbody, "σ length pick (px)", a.sigma_len_px);
      addKeyValueRow(tbody, "Example length (mm)", a.example_length_mm);
      addKeyValueRow(tbody, "mm per unit", a.mm_per_unit);
      addKeyValueRow(tbody, "Depth range (m)", `${{a.depth_min_m}} to ${{a.depth_max_m}}`);
    }}

    function insertCalibrationFileLinks(listEl, files) {{

      files.forEach(f => {{
        const li = document.createElement("li");
        const a = document.createElement("a");

        a.href = "assets/" + f;
        a.textContent = f;

        li.appendChild(a);
        listEl.appendChild(li);
      }});
    }}

    function insertFigureGrid(gridEl, figures) {{

      figures.forEach(fig => {{

        const col = document.createElement("div");
        col.className = "col-12 col-md-6 col-lg-4";

        const card = document.createElement("div");
        card.className = "card p-3 h-100";

        const title = document.createElement("div");
        title.className = "fw-semibold text-light mb-2 text-light";
        title.textContent = fig.title;

        const img = document.createElement("img");
        img.className = "img-fluid figure-img";
        img.src = "assets/" + fig.file;
        img.alt = fig.title;

        const cap = document.createElement("div");
        cap.className = "muted mt-2";
        cap.textContent = fig.caption;

        card.appendChild(title);
        card.appendChild(img);
        card.appendChild(cap);

        col.appendChild(card);
        gridEl.appendChild(col);
      }});
    }}

    function dumpMatrices(preEl, m) {{

      preEl.textContent = JSON.stringify(m, null, 2);
    }}

    function buildRelDepthErrorChart(ctx, data) {{

      const xs = data.depth_curves.depth_m;
      const ys = data.depth_curves.rel_sigma_percent;

      const thresholdLines = data.depth_bands.map(b => b.threshold_percent);

      const datasets = [
        {{
          label: "Relative depth error (%)",
          data: ys.map((y, i) => ({{x: xs[i], y: y}})),
          tension: 0.1,
        }}
      ];

      thresholdLines.forEach(t => {{

        datasets.push({{
          label: `Threshold ${{t}}%`,
          data: xs.map(x => ({{x: x, y: t}})),
          borderDash: [6, 6],
          pointRadius: 0,
        }});
      }});

      return new Chart(ctx, {{
        type: "line",
        data: {{
          datasets: datasets
        }},
        options: {{
          responsive: true,
          parsing: false,
          plugins: {{
            legend: {{
              labels: {{
                color: "#e8eefc"
              }}
            }},
            tooltip: {{
              callbacks: {{
                label: (ctx) => {{
                  const y = ctx.parsed.y;
                  return `${{ctx.dataset.label}}: ${{y.toFixed(3)}}`;
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              type: "linear",
              title: {{
                display: true,
                text: "Depth (m)",
                color: "#e8eefc"
              }},
              ticks: {{
                color: "#e8eefc"
              }},
              grid: {{
                color: "rgba(232,238,252,0.08)"
              }}
            }},
            y: {{
              title: {{
                display: true,
                text: "Relative depth error (%)",
                color: "#e8eefc"
              }},
              ticks: {{
                color: "#e8eefc"
              }},
              grid: {{
                color: "rgba(232,238,252,0.08)"
              }}
            }}
          }}
        }}
      }});
    }}

    function buildMmPerPixelChart(ctx, data) {{

      const xs = data.length_curves.depth_m;
      const ys = data.length_curves.mm_per_px;

      return new Chart(ctx, {{
        type: "line",
        data: {{
          datasets: [
            {{
              label: "mm per pixel",
              data: ys.map((y, i) => ({{x: xs[i], y: y}})),
              tension: 0.1,
            }}
          ]
        }},
        options: {{
          responsive: true,
          parsing: false,
          plugins: {{
            legend: {{
              labels: {{
                color: "#e8eefc"
              }}
            }}
          }},
          scales: {{
            x: {{
              type: "linear",
              title: {{
                display: true,
                text: "Depth (m)",
                color: "#e8eefc"
              }},
              ticks: {{
                color: "#e8eefc"
              }},
              grid: {{
                color: "rgba(232,238,252,0.08)"
              }}
            }},
            y: {{
              title: {{
                display: true,
                text: "mm per pixel",
                color: "#e8eefc"
              }},
              ticks: {{
                color: "#e8eefc"
              }},
              grid: {{
                color: "rgba(232,238,252,0.08)"
              }}
            }}
          }}
        }}
      }});
    }}

    async function main() {{

      const data = await loadReportData();

      document.getElementById("generatedTime").textContent = data.meta.generated_local;
      document.getElementById("inputDir").textContent = data.meta.input_directory;

      document.getElementById("stereoRms").textContent = data.summary.stereo_rms.toFixed(6);
      document.getElementById("baselineM").textContent = data.summary.baseline_m.toFixed(4) + " m";
      document.getElementById("rotationDeg").textContent = data.summary.rotation_deg.toFixed(3) + " deg";
      document.getElementById("fRect").textContent = data.summary.f_rect_px.toFixed(2) + " px";

      insertPointTable(document.getElementById("pointTableBody"), data.depth_point_estimates);

      insertIntrinsics(document.getElementById("intrinsicsLeft"), data.left);
      insertIntrinsics(document.getElementById("intrinsicsRight"), data.right);

      insertAssumptions(document.getElementById("assumptionsBody"), data.assumptions);

      insertCalibrationFileLinks(document.getElementById("calFiles"), data.meta.copied_calibration_files);

      insertFigureGrid(document.getElementById("figureGrid"), data.figures);

      dumpMatrices(document.getElementById("matrixDump"), data.matrices);

      initTermTooltips();

      const relCtx = document.getElementById("chartRelDepthError");
      const mmCtx = document.getElementById("chartMmPerPixel");

      buildRelDepthErrorChart(relCtx, data);
      buildMmPerPixelChart(mmCtx, data);
    }}

   // Wait until Chart.js is available before running main
    function waitForChartAndRun() {{

      // If Chart is ready, run the report setup now
      if (typeof Chart !== "undefined") {{
        main();
        return;
      }}

      // Otherwise retry until Chart is available
      setTimeout(waitForChartAndRun, 50);
    }}

    // Start the wait loop
    waitForChartAndRun();
  </script>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <script>
    // Load Chart.js in a way that works on file://
    //
    // Try local first using a normal script tag.
    //
    // If Chart is still undefined after that, load from CDN as a fallback.
    (function() {{

      const local = "{chart_js_local}";
      const cdn = "{chart_js_cdn}";

      function loadScript(src, onload) {{

        const s = document.createElement("script");
        s.src = src;
        s.onload = onload;

        s.onerror = () => {{

          console.warn("Failed to load script:", src);
        }};

        document.head.appendChild(s);
      }}

      // Try loading local Chart.js first
      loadScript(local, () => {{

        // If local worked, we are done
        if (typeof Chart !== "undefined") {{
          return;
        }}

        // If local didn't define Chart, try CDN
        loadScript(cdn, () => {{}});
      }});

      // If local fails to load at all, onerror will fire but we still want CDN
      //
      // This fallback runs if Chart remains undefined after a short delay
      setTimeout(() => {{

        if (typeof Chart === "undefined") {{
          loadScript(cdn, () => {{}});
        }}
      }}, 250);
    }})();
  </script>

</body>
</html>
"""
    return html


# Build figure metadata for the HTML report
def build_figure_list(has_maps):

    # Always include core plots
    figs = [
        {
            "file": "plot_depth_relative_error.png",
            "title": "Relative depth error vs depth",
            "caption": "Estimated depth error divided by distance. Lower is better. Grows with range.",
        },
        {
            "file": "plot_depth_absolute_error.png",
            "title": "Absolute depth error vs depth",
            "caption": "Estimated absolute depth uncertainty in meters vs distance.",
        },
        {
            "file": "plot_disparity_vs_depth.png",
            "title": "Disparity vs depth",
            "caption": "Disparity decreases with distance. When disparity gets very small, depth becomes noisy.",
        },
        {
            "file": "plot_mm_per_pixel_vs_depth.png",
            "title": "Scale vs depth",
            "caption": "How many millimeters each pixel corresponds to at a given depth (approx).",
        },
        {
            "file": "plot_length_error_mm_vs_depth.png",
            "title": "Length error vs depth",
            "caption": "Estimated length uncertainty in millimeters vs depth, based on σpx picking error.",
        },
        {
            "file": "plot_length_percent_error_vs_depth.png",
            "title": "Length percent error vs depth",
            "caption": "Estimated percent error for a chosen example length vs depth.",
        },
    ]

    # Add map related plots if present
    if has_maps:
        figs.extend([
            {
                "file": "plot_warp_heatmap_left.png",
                "title": "Left warp heatmap",
                "caption": "How far pixels move during undistort + rectify. Larger near edges is normal.",
            },
            {
                "file": "plot_warp_heatmap_right.png",
                "title": "Right warp heatmap",
                "caption": "Same as left, for the right camera.",
            },
            {
                "file": "plot_valid_mask_left.png",
                "title": "Left valid region",
                "caption": "Pixels that map to valid source coordinates. Shows usable coverage.",
            },
            {
                "file": "plot_valid_mask_right.png",
                "title": "Right valid region",
                "caption": "Valid remap coverage for the right camera.",
            },
            {
                "file": "plot_warp_histogram.png",
                "title": "Warp magnitude histogram",
                "caption": "Distribution of pixel motion magnitudes due to the rectification map.",
            },
        ])

    # Return figure descriptors
    return figs


# Create a report folder and return paths
def create_report_folder(out_root):

    # Create a timestamped folder name
    ts = get_timestamp_string()
    report_dir = Path(out_root) / f"calibration_report_{ts}"

    # Create assets directory
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Return both paths
    return report_dir, assets_dir


# Copy calibration input files into the report assets folder
def copy_calibration_files(calib_dir, assets_dir):

    # List candidate file names
    candidates = [
        "calibration_intrinsics.npz",
        "calibration_extrinsics.npz",
        "calibration_rectification.npz",
        "calibration_maps.npz",
    ]

    # Track what we copied
    copied = []

    # Copy any file that exists
    for name in candidates:

        src = Path(calib_dir) / name
        if not src.exists():
            continue

        dst = Path(assets_dir) / name
        shutil.copy2(src, dst)

        copied.append(name)

    # Return list of copied filenames
    return copied


# Build full report_data.json payload from calibration files
def build_report_data(calib_dir, args, copied_files):

    # Resolve file paths
    intr_path = Path(calib_dir) / "calibration_intrinsics.npz"
    extr_path = Path(calib_dir) / "calibration_extrinsics.npz"
    rect_path = Path(calib_dir) / "calibration_rectification.npz"
    maps_path = Path(calib_dir) / "calibration_maps.npz"

    # Load required files
    intr = load_npz_dict(intr_path)
    extr = load_npz_dict(extr_path)
    rect = load_npz_dict(rect_path)

    # Load maps if present
    maps = None
    if maps_path.exists():
        maps = load_npz_dict(maps_path)

    # Read image dimensions
    w = int(intr["image_width"])
    h = int(intr["image_height"])

    # Extract left intrinsics
    mtxL = intr["mtxL"]
    distL = intr["distL"].reshape(-1)

    # Extract right intrinsics
    mtxR = intr["mtxR"]
    distR = intr["distR"].reshape(-1)

    # Extract stereo extrinsics
    R = extr["R"]
    T = extr["T"].reshape(-1)
    stereo_rms = float(extr.get("stereo_rms", np.nan))

    # Extract rectification
    PL = rect["PL"]
    PR = rect["PR"]
    Q = rect["Q"]

    # Compute basic camera parameters
    fxL = float(mtxL[0, 0])
    fyL = float(mtxL[1, 1])
    cxL = float(mtxL[0, 2])
    cyL = float(mtxL[1, 2])

    fxR = float(mtxR[0, 0])
    fyR = float(mtxR[1, 1])
    cxR = float(mtxR[0, 2])
    cyR = float(mtxR[1, 2])

    # Compute FOVs
    left_fov_h = compute_fov_degrees(fxL, w)
    left_fov_v = compute_fov_degrees(fyL, h)
    left_fov_d = compute_diag_fov_degrees(fxL, w, h)

    right_fov_h = compute_fov_degrees(fxR, w)
    right_fov_v = compute_fov_degrees(fyR, h)
    right_fov_d = compute_diag_fov_degrees(fxR, w, h)

    # Compute stereo geometry
    baseline_units = compute_baseline_units(T)
    baseline_m = units_to_meters(baseline_units, args.mm_per_unit)
    rotation_deg = compute_rotation_angle_degrees(R)

    # Use rectified focal length
    f_rect_px = float(PL[0, 0])

    # Build depth grid
    depth_min_m = float(args.depth_min_m)
    depth_max_m = float(args.depth_max_m)

    depth_m = np.linspace(depth_min_m, depth_max_m, int(args.depth_samples)).astype(np.float64)

    # Compute depth error curves
    depth_pack = compute_depth_error_curves(
        depth_m,
        f_rect_px,
        baseline_units,
        args.mm_per_unit,
        args.sigma_disp_px
    )

    # Compute length error curves
    length_pack = compute_length_error_curves(
        depth_m,
        f_rect_px,
        args.sigma_len_px,
        args.example_length_mm
    )

    # Compute depth bands
    bands = compute_depth_bands(depth_pack["depth_m"], depth_pack["rel_sigma"], args.depth_band_thresholds)

    # Compute point estimates
    point_estimates = compute_point_estimates([1.0, 5.0, 10.0, 20.0], depth_pack)

    # Compute warp pack if maps exist
    warp_pack = None
    if maps is not None:
        warp_pack = compute_warp_products(maps, w, h)

    # Build curve arrays for JSON
    depth_curves = {
        "depth_m": depth_pack["depth_m"].tolist(),
        "disparity_px": depth_pack["disparity_px"].tolist(),
        "sigma_z_m": depth_pack["sigma_z_m"].tolist(),
        "rel_sigma_percent": (depth_pack["rel_sigma"] * 100.0).tolist(),
    }

    length_curves = {
        "depth_m": length_pack["depth_m"].tolist(),
        "mm_per_px": length_pack["mm_per_px"].tolist(),
        "sigma_L_mm": length_pack["sigma_L_mm"].tolist(),
        "percent_error_example": length_pack["percent_error_example"].tolist(),
        "example_length_mm": length_pack["example_length_mm"],
    }

    # Pack matrices for optional dump
    matrices = {
        "mtxL": mtxL.tolist(),
        "distL": distL.tolist(),
        "mtxR": mtxR.tolist(),
        "distR": distR.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "PL": PL.tolist(),
        "PR": PR.tolist(),
        "Q": Q.tolist(),
    }

    # Pack warp data summary for JSON
    warp_summary = None
    if warp_pack is not None:
        warp_summary = {
            "usable_fraction_L": warp_pack["usable_fraction_L"],
            "usable_fraction_R": warp_pack["usable_fraction_R"],
            "warp_mag_max": warp_pack["warp_mag_max"],
            "warp_hist_bin_centers": warp_pack["warp_hist_bin_centers"].tolist(),
            "warp_hist_counts_L": warp_pack["warp_hist_counts_L"].tolist(),
            "warp_hist_counts_R": warp_pack["warp_hist_counts_R"].tolist(),
        }

    # Build report payload
    report_data = {
        "meta": {
            "generated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_directory": str(Path(calib_dir).resolve()),
            "copied_calibration_files": copied_files,
            "image_width": w,
            "image_height": h,
        },
        "summary": {
            "stereo_rms": stereo_rms,
            "baseline_units": baseline_units,
            "baseline_m": baseline_m,
            "rotation_deg": rotation_deg,
            "f_rect_px": f_rect_px,
        },
        "assumptions": {
            "sigma_disp_px": float(args.sigma_disp_px),
            "sigma_len_px": float(args.sigma_len_px),
            "example_length_mm": float(args.example_length_mm),
            "mm_per_unit": float(args.mm_per_unit),
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
        },
        "left": {
            "fx": fxL,
            "fy": fyL,
            "cx": cxL,
            "cy": cyL,
            "fov_h_deg": left_fov_h,
            "fov_v_deg": left_fov_v,
            "fov_d_deg": left_fov_d,
            "distortion": [float(x) for x in distL.tolist()],
        },
        "right": {
            "fx": fxR,
            "fy": fyR,
            "cx": cxR,
            "cy": cyR,
            "fov_h_deg": right_fov_h,
            "fov_v_deg": right_fov_v,
            "fov_d_deg": right_fov_d,
            "distortion": [float(x) for x in distR.tolist()],
        },
        "depth_curves": depth_curves,
        "length_curves": length_curves,
        "depth_bands": bands,
        "depth_point_estimates": point_estimates,
        "warp": warp_summary,
        "matrices": matrices,
    }

    # Return report payload plus raw warp pack for plotting
    return report_data, warp_pack


# Parse comma separated thresholds like "1,3,10"
def parse_thresholds_csv(text):

    # Split by commas and whitespace
    parts = re.split(r"[,\s]+", text.strip())

    # Convert to floats
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))

    # Return list
    return vals



# Main entry point for the report generator
def main():

    parser = argparse.ArgumentParser(description="Generate an HTML + PNG report from OpenCV stereo calibration files.")

    parser.add_argument("--calib_dir", required=True, help="Directory containing calibration_*.npz files")
    parser.add_argument("--out_root", default=".", help="Directory to place generated report folders")

    parser.add_argument("--mm_per_unit", type=float, default=1.0, help="Millimeters per calibration unit (default assumes calibration units are mm)")

    parser.add_argument("--sigma_disp_px", type=float, default=0.25, help="Assumed disparity uncertainty in pixels")
    parser.add_argument("--sigma_len_px", type=float, default=1.0, help="Assumed endpoint picking uncertainty in pixels")

    parser.add_argument("--example_length_mm", type=float, default=300.0, help="Example object length for percent error curve")

    parser.add_argument("--depth_min_m", type=float, default=0.5, help="Minimum depth for curves (meters)")
    parser.add_argument("--depth_max_m", type=float, default=25.0, help="Maximum depth for curves (meters)")
    parser.add_argument("--depth_samples", type=int, default=250, help="Number of samples for depth curves")

    parser.add_argument("--depth_band_thresholds", type=str, default="1,3,10", help="Relative depth error thresholds in percent, CSV")

    args = parser.parse_args()

    # Parse depth band thresholds
    args.depth_band_thresholds = parse_thresholds_csv(args.depth_band_thresholds)

    # Create output folder structure
    report_dir, assets_dir = create_report_folder(args.out_root)

    # Copy calibration files into assets
    copied_files = copy_calibration_files(args.calib_dir, assets_dir)

    # Build report_data.json payload
    report_data, warp_pack = build_report_data(args.calib_dir, args, copied_files)

    # Generate static plots
    depth_pack_for_plot = {
        "depth_m": np.array(report_data["depth_curves"]["depth_m"], dtype=np.float64),
        "rel_sigma": np.array(report_data["depth_curves"]["rel_sigma_percent"], dtype=np.float64) / 100.0,
        "sigma_z_m": np.array(report_data["depth_curves"]["sigma_z_m"], dtype=np.float64),
        "disparity_px": np.array(report_data["depth_curves"]["disparity_px"], dtype=np.float64),
    }

    length_pack_for_plot = {
        "depth_m": np.array(report_data["length_curves"]["depth_m"], dtype=np.float64),
        "mm_per_px": np.array(report_data["length_curves"]["mm_per_px"], dtype=np.float64),
        "sigma_L_mm": np.array(report_data["length_curves"]["sigma_L_mm"], dtype=np.float64),
        "percent_error_example": np.array(report_data["length_curves"]["percent_error_example"], dtype=np.float64),
        "example_length_mm": float(report_data["length_curves"]["example_length_mm"]),
    }

    # Create figure list now that we know whether we have maps
    figures = build_figure_list(has_maps=(warp_pack is not None))
    report_data["figures"] = figures

    generate_static_plots(
        assets_dir,
        depth_pack_for_plot,
        length_pack_for_plot,
        report_data["depth_bands"],
        warp_pack
    )

    # Write report_data.json
    write_json(assets_dir / "report_data.json", report_data)

    # Write calibration_report.txt
    report_text = build_text_report(report_data)
    write_text(report_dir / "calibration_report.txt", report_text)

    # Write index.html
    html = build_html(report_data)
    write_text(report_dir / "index.html", html)

    # Show final location
    print("Report generated at:")
    print(f"  {report_dir.resolve()}")
    print("")
    print("Open:")
    print(f"  {report_dir.resolve() / 'index.html'}")
    print("")
    print("Optional:")
    print("  If you want fully offline interactive charts, place Chart.js at:")
    print(f"    {assets_dir.resolve() / 'chart.umd.min.js'}")


if __name__ == "__main__":
    main()