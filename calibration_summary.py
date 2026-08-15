"""PySide6 calibration summary window for Sizeamatic Pro.

This module creates and updates a summary window for loaded stereo
calibration data. The summary is intended to help users inspect important
calibration values, identify obvious calibration problems, and copy useful
diagnostics into notes, reports, or spreadsheets.

This is a PySide6 port of the original Tkinter module (ROADMAP.md Phase
11) - see `main.py`'s module docstring for why the app switched
frameworks. `map_oob_percent` and the rest of the calibration math below
are unchanged from the original; only the widgets (`ttk.Treeview` ->
`QTableWidget`, `tk.Text` -> `QPlainTextEdit`, `tk.Toplevel` ->
`qt_helpers.ClosableDialog`) changed.

Contents:
    - `CalibrationSummaryWindow` — owns the summary window and its widgets.
    - `map_oob_percent` — standalone remap out-of-bounds calculation, used
      by the summary and independently unit-tested.

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

# math is used for angle and field of view calculations.
import math

# NumPy is used to build OpenCV-compatible point arrays and perform vector math.
import numpy as np

# OpenCV is used for stereo triangulation, template matching, projection, and
# other image-space measurement operations.
import cv2

from PySide6.QtWidgets import QHeaderView, QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout

from qt_helpers import ClosableDialog, move_to_same_screen_as


class CalibrationSummaryWindow:
    """Owns the calibration summary dialog and its widgets.

    One instance lives on the main application (`app.cal_summary_window`),
    created once and reused for the lifetime of the app — building the
    window (`ensure_window`) and refreshing its contents (`update_window`)
    are separate steps, matching how the app calls them (build lazily on
    first open, refresh whenever calibration changes).
    """

    def __init__(self, app):
        """Store the owning app and initialize widget references to None.

        Args:
            app: The main application object, used as the dialog's
                parent, for the app window title, and for the currently
                loaded calibration dictionary (`app.cal`).

        Returns:
            None
        """
        self.app = app

        self.win = None
        """The summary dialog, or None if it hasn't been opened yet (or
        was closed)."""

        self.table = None
        """The item/value `QTableWidget`, or None if the window hasn't
        been built yet. `update_window` treats `None` here as its signal
        that there's nothing safe to update."""

        self.copy_text = None
        """The copyable summary `QPlainTextEdit`, or None if the window
        hasn't been built yet."""

    def _on_close(self):
        """Handle the user manually closing the summary window.

        Clears the stored widget references. Clearing these is
        important because the next `update_window` call needs to know
        the widgets no longer exist and must be rebuilt via
        `ensure_window` first.

        Returns:
            None
        """
        self.win = None
        self.table = None
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
        if self.win is not None:
            self.win.show()
            self.win.raise_()
            self.win.activateWindow()
            return

        win = ClosableDialog(self._on_close)
        win.setWindowTitle(self.app._app_window_title())
        win.resize(700, 600)

        outer = QVBoxLayout(win)

        title_label = QLabel("Calibration Summary")
        title_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(title_label)

        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Item", "Value"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        outer.addWidget(table, stretch=1)

        copy_label = QLabel("Copy")
        copy_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(copy_label)

        copy_text = QPlainTextEdit()
        copy_text.setReadOnly(True)
        copy_text.setFixedHeight(150)
        copy_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(copy_text)

        self.win = win
        self.table = table
        self.copy_text = copy_text

        win.show()
        # Positioned only after show(), which finalizes the window's real
        # layout-driven size rather than the initial resize() hint.
        move_to_same_screen_as(win, self.app)

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
        if self.app.cal is None or self.table is None:
            return

        self.table.setRowCount(0)

        c = self.app.cal

        w = int(c["w"])
        h = int(c["h"])

        lines = []

        # -------------------------------------------------------------------------
        # Overview.
        # -------------------------------------------------------------------------

        self.add_row("Image size", f"{w}×{h}")
        lines.append(f"Image size:\t{w}x{h}")

        T = np.array(c["T"], dtype=np.float64).reshape(-1)
        Tx, Ty, Tz = float(T[0]), float(T[1]), float(T[2])
        baseline = float(np.linalg.norm(T))

        self.add_row("Baseline ||T|| (mm)", f"{baseline:.2f}")
        self.add_row("T (mm)", f"Tx {Tx:.2f}  Ty {Ty:.2f}  Tz {Tz:.2f}")
        lines.append(f"Baseline_mm:\t{baseline:.2f}")
        lines.append(f"T_mm:\t{Tx:.2f}\t{Ty:.2f}\t{Tz:.2f}")

        Rm = np.array(c["R"], dtype=np.float64)

        tr = float(np.trace(Rm))
        cosang = (tr - 1.0) / 2.0
        cosang = max(-1.0, min(1.0, cosang))
        ang = math.degrees(math.acos(cosang))

        rvec, _ = cv2.Rodrigues(Rm)
        rvec = rvec.reshape(-1)
        rmag = float(np.linalg.norm(rvec))

        if rmag > 1e-12:
            axis = rvec / rmag
            axis_str = f"{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}"
        else:
            axis_str = "0, 0, 0"

        self.add_row("Relative rotation angle (deg)", f"{ang:.4f}")
        self.add_row("Rotation axis (unit)", axis_str)
        lines.append(f"Rot_angle_deg:\t{ang:.4f}")
        lines.append(f"Rot_axis:\t{axis_str}")

        if c.get("stereo_rms", None) is not None:
            self.add_row("Stereo RMS", f"{float(c['stereo_rms']):.6f}")
            lines.append(f"Stereo_RMS:\t{float(c['stereo_rms']):.6f}")

        # -------------------------------------------------------------------------
        # Intrinsics.
        # -------------------------------------------------------------------------

        mtxL = np.array(c["mtxL"], dtype=np.float64)
        mtxR = np.array(c["mtxR"], dtype=np.float64)

        fxL, fyL = float(mtxL[0, 0]), float(mtxL[1, 1])
        cxL, cyL = float(mtxL[0, 2]), float(mtxL[1, 2])

        fxR, fyR = float(mtxR[0, 0]), float(mtxR[1, 1])
        cxR, cyR = float(mtxR[0, 2]), float(mtxR[1, 2])

        self.add_row("Left intrinsics", f"fx {fxL:.2f}  fy {fyL:.2f}  cx {cxL:.2f}  cy {cyL:.2f}")
        self.add_row("Right intrinsics", f"fx {fxR:.2f}  fy {fyR:.2f}  cx {cxR:.2f}  cy {cyR:.2f}")
        lines.append(f"L_fx_fy_cx_cy:\t{fxL:.2f}\t{fyL:.2f}\t{cxL:.2f}\t{cyL:.2f}")
        lines.append(f"R_fx_fy_cx_cy:\t{fxR:.2f}\t{fyR:.2f}\t{cxR:.2f}\t{cyR:.2f}")

        fovxL = math.degrees(2.0 * math.atan(w / (2.0 * fxL)))
        fovyL = math.degrees(2.0 * math.atan(h / (2.0 * fyL)))
        fovxR = math.degrees(2.0 * math.atan(w / (2.0 * fxR)))
        fovyR = math.degrees(2.0 * math.atan(h / (2.0 * fyR)))

        self.add_row("Left FOV (deg)", f"FOVx {fovxL:.2f}  FOVy {fovyL:.2f}")
        self.add_row("Right FOV (deg)", f"FOVx {fovxR:.2f}  FOVy {fovyR:.2f}")
        lines.append(f"L_FOVx_FOVy_deg:\t{fovxL:.2f}\t{fovyL:.2f}")
        lines.append(f"R_FOVx_FOVy_deg:\t{fovxR:.2f}\t{fovyR:.2f}")

        distL = np.array(c["distL"], dtype=np.float64).reshape(-1)
        distR = np.array(c["distR"], dtype=np.float64).reshape(-1)
        self.add_row("Left distortion", " ".join([f"{v:.6g}" for v in distL]))
        self.add_row("Right distortion", " ".join([f"{v:.6g}" for v in distR]))

        # -------------------------------------------------------------------------
        # Rectification.
        # -------------------------------------------------------------------------

        PL = np.array(c["PL"], dtype=np.float64)
        PR = np.array(c["PR"], dtype=np.float64)

        fx_rect_L = float(PL[0, 0])
        fx_rect_R = float(PR[0, 0])
        self.add_row("Rectified fx (PL, PR)", f"{fx_rect_L:.2f}, {fx_rect_R:.2f}")

        roiL = c.get("roiL", None)
        roiR = c.get("roiR", None)

        if roiL is not None and roiR is not None:
            roiL = np.array(roiL).reshape(-1).astype(int)
            roiR = np.array(roiR).reshape(-1).astype(int)
            self.add_row("roiL", f"{tuple(roiL)}")
            self.add_row("roiR", f"{tuple(roiR)}")

            x0 = max(roiL[0], roiR[0])
            y0 = max(roiL[1], roiR[1])
            x1 = min(roiL[0] + roiL[2], roiR[0] + roiR[2])
            y1 = min(roiL[1] + roiL[3], roiR[1] + roiR[3])

            iw = max(0, x1 - x0)
            ih = max(0, y1 - y0)
            inter = iw * ih
            pct = 100.0 * float(inter) / float(w * h)

            self.add_row("ROI overlap (% image)", f"{pct:.2f}%")
            lines.append(f"ROI_overlap_pct:\t{pct:.2f}")

        # -------------------------------------------------------------------------
        # Map validity.
        # -------------------------------------------------------------------------

        oobL = map_oob_percent(c["mapLx"], c["mapLy"], w, h)
        oobR = map_oob_percent(c["mapRx"], c["mapRy"], w, h)

        self.add_row("Map out-of-bounds L", f"{oobL:.3f}%")
        self.add_row("Map out-of-bounds R", f"{oobR:.3f}%")
        lines.append(f"Map_OOB_L_pct:\t{oobL:.3f}")
        lines.append(f"Map_OOB_R_pct:\t{oobR:.3f}")

        # -------------------------------------------------------------------------
        # Warnings.
        # -------------------------------------------------------------------------

        warnings = []

        if baseline < 1.0:
            warnings.append("Baseline is very small")

        if abs(fx_rect_L - fx_rect_R) > 1e-3:
            warnings.append("Rectified fx differs between PL and PR")

        if oobL > 1.0 or oobR > 1.0:
            warnings.append("High map out-of-bounds percentage")

        if warnings:
            self.add_row("Warnings", "; ".join(warnings))
            lines.append(f"Warnings:\t{'; '.join(warnings)}")

        # -------------------------------------------------------------------------
        # Copy box update.
        # -------------------------------------------------------------------------

        self.copy_text.setPlainText("\n".join(lines))

    def add_row(self, label, value):
        """Add one item/value row to the summary table.

        Args:
            label (str): Row name to display in the "Item" column.
            value (str): Row value to display in the "Value" column.

        Returns:
            None
        """
        if self.table is None:
            return

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(label))
        self.table.setItem(row, 1, QTableWidgetItem(value))


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
