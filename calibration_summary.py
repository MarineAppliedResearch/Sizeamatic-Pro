"""Calibration summary window for Sizeamatic Pro.

This module creates and updates a Tkinter summary window for loaded stereo
calibration data. The summary is intended to help users inspect important
calibration values, identify obvious calibration problems, and copy useful
diagnostics into notes, reports, or spreadsheets.

Contents:
    - `CalibrationSummaryWindow` — owns the summary window and its widgets.
    - `map_oob_percent` — standalone remap out-of-bounds calculation, used
      by the summary and independently unit-tested.

Design notes:
    `CalibrationSummaryWindow` is a plain class instance owned by the main
    application (`app.cal_summary_window`), not a module-level singleton.
    An earlier version of this module stored the window/widget references
    as module-level globals instead, which caused a real bug: a nested
    `_on_close` closure that assigned to those globals needed (but
    initially missed) its own `global` declaration — a `global` statement
    in the enclosing function doesn't extend into a nested one — leaving
    stale references to destroyed widgets after the window closed and
    crashing the next update (see `FINDINGS.md` #1). A class instance
    removes that whole failure mode: `self.tree` is just an attribute: no
    `global` statement anywhere, and nothing to get wrong.

Assumptions:
    - The main application stores the loaded calibration dictionary as
      `app.cal`.
    - The calibration dictionary contains the camera matrices, distortion
      coefficients, stereo transform, rectification matrices, projection
      matrices, and remap arrays expected by the summary.
    - Calibration translation units determine the baseline units. In this
      application, those units are normally millimeters.
    - Warning checks in this file are simple diagnostic checks, not a
      complete validation of calibration quality.

Author:
    Isaac Travers

Created:
    2026-05-18
"""

# Standard library imports.

# math is used for angle and field of view calculations.
import math

# Third Party Imports

# tkinter provides the calibration summary Toplevel window and text widget.
import tkinter as tk

# ttk provides themed Tkinter widgets such as Frame, Label, and Treeview.
from tkinter import ttk

# NumPy is used to build OpenCV-compatible point arrays and perform vector math.
import numpy as np

# OpenCV is used for stereo triangulation, template matching, projection, and
# other image-space measurement operations.
import cv2


class CalibrationSummaryWindow:
    """Owns the calibration summary Toplevel window and its widgets.

    One instance lives on the main application (`app.cal_summary_window`),
    created once and reused for the lifetime of the app — building the
    window (`ensure_window`) and refreshing its contents (`update_window`)
    are separate steps, matching how the app calls them (build lazily on
    first open, refresh whenever calibration changes).
    """

    def __init__(self, app):
        """Store the owning app and initialize widget references to None.

        Args:
            app: The main application object, used for the Tk root window
                that owns the summary Toplevel, and for the currently
                loaded calibration dictionary (`app.cal`).

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The summary Toplevel window, or None if it hasn't been opened
        yet (or was closed)."""

        self.tree = None
        """The item/value Treeview table widget, or None if the window
        hasn't been built yet. `update_window` treats `None` here as its
        signal that there's nothing safe to update."""

        self.copy_text = None
        """The copyable summary Text widget, or None if the window hasn't
        been built yet."""

    def _on_close(self):
        """Handle the user manually closing the summary window.

        Destroys the Tkinter window and clears the stored widget
        references. Clearing these references is important because the
        next `update_window` call needs to know the widgets no longer
        exist and must be rebuilt via `ensure_window` first.

        Returns:
            None
        """

        # Destroy the Tkinter window.
        self.win.destroy()

        # Clear the stored references because the widgets were destroyed.
        self.win = None
        self.tree = None
        self.copy_text = None

    def ensure_window(self):
        """Create the summary window, or raise it if it already exists.

        Creates the summary window, including the item/value table and
        copyable text area. If the window already exists, brings it to
        the front instead of creating a duplicate. Only builds the UI
        widgets; calibration values are filled in later by
        `update_window`.

        Returns:
            None
        """

        # If the window already exists, bring it to the front and reuse it
        # instead of creating a duplicate window.
        if self.win is not None:
            try:
                self.win.lift()
                return

            # If the stored window reference is stale, clear it so a new
            # window can be created below.
            except Exception:
                self.win = None

        # Create a separate top level window owned by the main application root.
        win = tk.Toplevel(self.app.root)

        # Set the user visible title for the calibration summary window.
        win.title("Calibration Summary")

        # Give the window an initial size large enough for the table and copy box.
        win.geometry("700x600")

        # Use the cleanup callback when the user closes the calibration summary window.
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create one padded outer frame to hold all calibration summary content.
        outer = ttk.Frame(win, padding=(10, 10))
        outer.grid(row=0, column=0, sticky="nsew")

        # Let the outer frame expand with the calibration summary window.
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        # Let the table row and main content column expand when the window resizes.
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        # Add the title label for the calibration summary content.
        ttk.Label(
            outer,
            text="Calibration Summary",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # Define the summary table columns.
        cols = ("Item", "Value")

        # Create the calibration summary table widget using heading only columns.
        tree = ttk.Treeview(
            outer,
            columns=cols,
            show="headings",
            height=18,
        )
        tree.grid(row=1, column=0, sticky="nsew", pady=(8, 10))

        # Label the table columns.
        tree.heading("Item", text="Item")
        tree.heading("Value", text="Value")

        # Set table column widths and alignment.
        tree.column("Item", width=320, anchor="w")
        tree.column("Value", width=340, anchor="w")

        # Add the section label for copyable calibration summary output.
        ttk.Label(
            outer,
            text="Copy",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=2, column=0, sticky="w")

        # Create a disabled text widget that later receives copyable summary text.
        txt = tk.Text(outer, height=10, width=1, wrap="none")
        txt.grid(row=3, column=0, sticky="nsew")

        # Start disabled so users do not accidentally edit generated summary text.
        txt.configure(state="disabled")

        # Keep the copy box from consuming extra vertical stretch by default.
        outer.grid_rowconfigure(3, weight=0)

        # Store the window and widgets for later update calls.
        self.win = win
        self.tree = tree
        self.copy_text = txt

    def update_window(self):
        """Refresh the summary window from the currently loaded calibration.

        Refreshes the summary window from `self.app.cal`. Displays image
        size, baseline, relative rotation, intrinsics, field of view
        estimates, distortion coefficients, rectification details, ROI
        overlap, remap validity, and simple warning checks. Also builds a
        tab separated copy block for spreadsheet or report use.

        Returns:
            None
        """

        # Calibration data and window widgets must both exist before the
        # summary can be updated.
        if self.app.cal is None or self.tree is None:
            return

        # Clear existing summary rows from the previous update.
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Keep a shorter local name for the calibration dictionary.
        c = self.app.cal

        # Read the calibrated image size.
        w = int(c["w"])
        h = int(c["h"])

        # Store tab separated output lines for the copy text box.
        lines = []

        # -------------------------------------------------------------------------
        # Overview.
        # -------------------------------------------------------------------------

        # Show the calibrated image size.
        self.add_row("Image size", f"{w}×{h}")
        lines.append(f"Image size:\t{w}x{h}")

        # Read the stereo translation vector and compute the baseline length.
        T = np.array(c["T"], dtype=np.float64).reshape(-1)
        Tx, Ty, Tz = float(T[0]), float(T[1]), float(T[2])
        baseline = float(np.linalg.norm(T))

        # Display baseline magnitude and individual translation components.
        self.add_row("Baseline ||T|| (mm)", f"{baseline:.2f}")
        self.add_row("T (mm)", f"Tx {Tx:.2f}  Ty {Ty:.2f}  Tz {Tz:.2f}")
        lines.append(f"Baseline_mm:\t{baseline:.2f}")
        lines.append(f"T_mm:\t{Tx:.2f}\t{Ty:.2f}\t{Tz:.2f}")

        # Read the relative rotation matrix between the cameras.
        Rm = np.array(c["R"], dtype=np.float64)

        # Estimate the rotation angle from the rotation matrix trace.
        tr = float(np.trace(Rm))
        cosang = (tr - 1.0) / 2.0

        # Clamp the cosine value so small floating point errors cannot push acos()
        # outside its valid input range.
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))

        # Convert the rotation matrix to Rodrigues vector form so the rotation axis
        # can be displayed.
        rvec, _ = cv2.Rodrigues(Rm)
        rvec = rvec.reshape(-1)
        rmag = float(np.linalg.norm(rvec))

        # Normalize the Rodrigues vector into a unit rotation axis when possible.
        if rmag > 1e-12:
            axis = rvec / rmag
            axis_str = f"{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}"
        else:
            axis_str = "0, 0, 0"

        # Display the relative camera rotation summary.
        self.add_row("Relative rotation angle (deg)", f"{ang:.4f}")
        self.add_row("Rotation axis (unit)", axis_str)
        lines.append(f"Rot_angle_deg:\t{ang:.4f}")
        lines.append(f"Rot_axis:\t{axis_str}")

        # Display stereo calibration RMS if it was saved in the calibration data.
        if c.get("stereo_rms", None) is not None:
            self.add_row("Stereo RMS", f"{float(c['stereo_rms']):.6f}")
            lines.append(f"Stereo_RMS:\t{float(c['stereo_rms']):.6f}")

        # -------------------------------------------------------------------------
        # Intrinsics.
        # -------------------------------------------------------------------------

        # Read left and right camera intrinsic matrices.
        mtxL = np.array(c["mtxL"], dtype=np.float64)
        mtxR = np.array(c["mtxR"], dtype=np.float64)

        # Extract focal lengths and principal points from the left camera matrix.
        fxL, fyL = float(mtxL[0, 0]), float(mtxL[1, 1])
        cxL, cyL = float(mtxL[0, 2]), float(mtxL[1, 2])

        # Extract focal lengths and principal points from the right camera matrix.
        fxR, fyR = float(mtxR[0, 0]), float(mtxR[1, 1])
        cxR, cyR = float(mtxR[0, 2]), float(mtxR[1, 2])

        # Display left and right intrinsic summaries.
        self.add_row("Left intrinsics", f"fx {fxL:.2f}  fy {fyL:.2f}  cx {cxL:.2f}  cy {cyL:.2f}")
        self.add_row("Right intrinsics", f"fx {fxR:.2f}  fy {fyR:.2f}  cx {cxR:.2f}  cy {cyR:.2f}")
        lines.append(f"L_fx_fy_cx_cy:\t{fxL:.2f}\t{fyL:.2f}\t{cxL:.2f}\t{cyL:.2f}")
        lines.append(f"R_fx_fy_cx_cy:\t{fxR:.2f}\t{fyR:.2f}\t{cxR:.2f}\t{cyR:.2f}")

        # Estimate horizontal and vertical field of view from image size and focal
        # length in pixels.
        fovxL = math.degrees(2.0 * math.atan(w / (2.0 * fxL)))
        fovyL = math.degrees(2.0 * math.atan(h / (2.0 * fyL)))
        fovxR = math.degrees(2.0 * math.atan(w / (2.0 * fxR)))
        fovyR = math.degrees(2.0 * math.atan(h / (2.0 * fyR)))

        # Display field of view estimates.
        self.add_row("Left FOV (deg)", f"FOVx {fovxL:.2f}  FOVy {fovyL:.2f}")
        self.add_row("Right FOV (deg)", f"FOVx {fovxR:.2f}  FOVy {fovyR:.2f}")
        lines.append(f"L_FOVx_FOVy_deg:\t{fovxL:.2f}\t{fovyL:.2f}")
        lines.append(f"R_FOVx_FOVy_deg:\t{fovxR:.2f}\t{fovyR:.2f}")

        # Display all saved distortion coefficients for each camera.
        distL = np.array(c["distL"], dtype=np.float64).reshape(-1)
        distR = np.array(c["distR"], dtype=np.float64).reshape(-1)
        self.add_row("Left distortion", " ".join([f"{v:.6g}" for v in distL]))
        self.add_row("Right distortion", " ".join([f"{v:.6g}" for v in distR]))

        # -------------------------------------------------------------------------
        # Rectification.
        # -------------------------------------------------------------------------

        # Read the rectified projection matrices.
        PL = np.array(c["PL"], dtype=np.float64)
        PR = np.array(c["PR"], dtype=np.float64)

        # Extract the rectified focal lengths from the projection matrices.
        fx_rect_L = float(PL[0, 0])
        fx_rect_R = float(PR[0, 0])
        self.add_row("Rectified fx (PL, PR)", f"{fx_rect_L:.2f}, {fx_rect_R:.2f}")

        # Read valid rectification ROIs if they were saved in the calibration data.
        roiL = c.get("roiL", None)
        roiR = c.get("roiR", None)

        # Display ROI details and compute overlap when both ROIs are available.
        if roiL is not None and roiR is not None:
            roiL = np.array(roiL).reshape(-1).astype(int)
            roiR = np.array(roiR).reshape(-1).astype(int)
            self.add_row("roiL", f"{tuple(roiL)}")
            self.add_row("roiR", f"{tuple(roiR)}")

            # Compute the intersection rectangle between the two valid ROIs.
            x0 = max(roiL[0], roiR[0])
            y0 = max(roiL[1], roiR[1])
            x1 = min(roiL[0] + roiL[2], roiR[0] + roiR[2])
            y1 = min(roiL[1] + roiL[3], roiR[1] + roiR[3])

            # Convert the intersection rectangle into an area percentage of the full
            # image.
            iw = max(0, x1 - x0)
            ih = max(0, y1 - y0)
            inter = iw * ih
            pct = 100.0 * float(inter) / float(w * h)

            # Display and copy the ROI overlap percentage.
            self.add_row("ROI overlap (% image)", f"{pct:.2f}%")
            lines.append(f"ROI_overlap_pct:\t{pct:.2f}")

        # -------------------------------------------------------------------------
        # Map validity.
        # -------------------------------------------------------------------------

        # Estimate how much of each remap points outside the source image.
        oobL = map_oob_percent(c["mapLx"], c["mapLy"], w, h)
        oobR = map_oob_percent(c["mapRx"], c["mapRy"], w, h)

        # Display map out of bounds percentages.
        self.add_row("Map out-of-bounds L", f"{oobL:.3f}%")
        self.add_row("Map out-of-bounds R", f"{oobR:.3f}%")
        lines.append(f"Map_OOB_L_pct:\t{oobL:.3f}")
        lines.append(f"Map_OOB_R_pct:\t{oobR:.3f}")

        # -------------------------------------------------------------------------
        # Warnings.
        # -------------------------------------------------------------------------

        # Build simple warnings for calibration values that look suspicious.
        warnings = []

        # A near zero baseline usually means the stereo calibration is unusable for
        # depth measurement.
        if baseline < 1.0:
            warnings.append("Baseline is very small")

        # Rectified projection matrices normally should have matching focal lengths.
        if abs(fx_rect_L - fx_rect_R) > 1e-3:
            warnings.append("Rectified fx differs between PL and PR")

        # Large out of bounds remap percentages may indicate poor rectification
        # coverage or an image size mismatch.
        if oobL > 1.0 or oobR > 1.0:
            warnings.append("High map out-of-bounds percentage")

        # Display warning text if any warning checks were triggered.
        if warnings:
            self.add_row("Warnings", "; ".join(warnings))
            lines.append(f"Warnings:\t{'; '.join(warnings)}")

        # -------------------------------------------------------------------------
        # Copy box update.
        # -------------------------------------------------------------------------

        # Join the tab separated summary lines into one copyable text block.
        copy_block = "\n".join(lines)

        # Temporarily enable the text widget so the generated output can be replaced.
        self.copy_text.configure(state="normal")

        # Clear the previous copy block.
        self.copy_text.delete("1.0", "end")

        # Insert the latest calibration summary text.
        self.copy_text.insert("1.0", copy_block)

        # Disable editing again so users do not accidentally modify generated output.
        self.copy_text.configure(state="disabled")

    def add_row(self, label, value):
        """Add one item/value row to the summary table.

        Args:
            label (str): Row name to display in the "Item" column.
            value (str): Row value to display in the "Value" column.

        Returns:
            None
        """

        # If the table does not exist, there is nowhere safe to insert the row.
        if self.tree is None:
            return

        # Insert the label and value into the summary table.
        self.tree.insert("", "end", values=(label, value))


def map_oob_percent(mapx, mapy, w, h):
    """Compute the percentage of a rectification map that samples out of bounds.

    Computes how much of a rectification map samples outside the valid
    source image. A high percentage can indicate poor rectification
    coverage, mismatched image size, or calibration data that does not
    match the loaded video dimensions.

    Args:
        mapx (numpy.ndarray): Remap array of source image X coordinates,
            indexed by rectified pixel position.
        mapy (numpy.ndarray): Remap array of source image Y coordinates,
            indexed by rectified pixel position.
        w (int): Source image width.
        h (int): Source image height.

    Returns:
        float: The percentage of remap samples that point outside the
        source image bounds.
    """

    # Mark every remap coordinate that samples outside the valid source image.
    # The upper bound uses w - 1 and h - 1 because interpolation needs neighboring
    # source pixels and values at the final edge can be unsafe.
    oob = (
        (mapx < 0)
        | (mapx >= (w - 1))
        | (mapy < 0)
        | (mapy >= (h - 1))
    )

    # Convert the number of out of bounds samples into a percentage of the full
    # remap grid.
    return 100.0 * float(np.count_nonzero(oob)) / float(oob.size)
