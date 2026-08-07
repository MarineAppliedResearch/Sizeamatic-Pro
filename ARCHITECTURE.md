# Architecture

> **This file describes the system as it exists right now.** It is not a
> history of how it got this way and not a plan for how it should look in the
> future — that's what `git log` and `ROADMAP.md` are for. Whenever you change
> the code's structure, update this file in the same change so it doesn't go
> stale. If you're reading this to understand the codebase before making a
> change, trust this file over your assumptions; if you find it's wrong,
> fix it.

## Current shape: mid-restructure (Phase 5 in progress)

Target decided (see `ROADMAP.md` Phase 5): replace the module-level-global
state pattern with real classes owned by the app, one module at a time,
running the test suite after each step. **Staying on Tkinter** — the
"unacceptably slow" rendering that partly motivated considering a framework
migration turned out to be two self-inflicted bottlenecks in `main.py`, both
fixed directly: a PNG-encode/base64 round trip in `_display_bgr_on_canvas`,
and an unconditional `cap.set(CAP_PROP_POS_FRAMES)` seek on every frame read
in `_read_frame_at` (bigger of the two — 19x slower than sequential reads).

`main.py` (~1,750 lines) still holds the core `SizeamaticProApp` class — GUI
construction, window/menu setup, playback/timeline state, video I/O (two
OpenCV `VideoCapture` objects held open for the app's lifetime), and the
top-level event wiring. Calibration file loading/validation has been pulled
out into `calibration_io.py`. It owns three converted pieces so far:
`self.cal_summary_window` (a `calibration_summary.CalibrationSummaryWindow`
instance), `self.measurement_window` (a
`measurement_window.MeasurementWindow` instance), and
`self.anaglyph_preview` (an `anaglyph_preview.AnaglyphPreview` instance).

Several concerns have been pulled out of `main.py` into separate files.
**Three have been converted to classes; one is still plain functions** that
take the `SizeamaticProApp` instance (`app`) as their first argument and
read/mutate its attributes directly, or (worse) keep their own state as
module-level globals:

- **`stereo_matching.py`** — stereo point matching and triangulation math
  (scanline mate-point search, triangulation, reprojection error,
  uncertainty estimation). Pure functions, no GUI code, no restructuring
  needed here — this one's fine as-is.
- **`calibration_io.py`** — loads and validates calibration NPZ files, no
  GUI code, no restructuring needed here either. Pulled out of `main.py`'s
  `on_load_calibration_folder` so it's testable without a directory-chooser
  dialog.
- **`calibration_summary.py`** — `CalibrationSummaryWindow` class (done).
  Owns the Tkinter calibration-summary window and its update logic as
  instance attributes/methods instead of module-level globals — this
  conversion is what removed the fragility that caused `FINDINGS.md` #1
  (a nested closure needing, but missing, its own `global` declaration).
- **`measurement_window.py`** — `MeasurementWindow` class (done). Same
  conversion as `calibration_summary.py`.
- **`anaglyph_preview.py`** — `AnaglyphPreview` class (done). Same
  conversion again — this one also structurally eliminates `FINDINGS.md`
  #2 (an uninitialized module global) and #3 (a bound method can't be
  called with a missing `app` argument the way a free function could).
- **`video_overlay.py`** — still plain functions + module-level globals
  (`drag_active`, `left_overlay_canvas`, etc.). Not yet converted — last
  one left.

Once fully converted, each piece will be independently unit-testable without
constructing a real Tk root — `tests/conftest.py`'s `FakeApp` stand-in
already anticipates this for the pure-logic functions, and all three
converted classes can now be tested by constructing an instance directly
(see `tests/test_regressions.py`).

## Other first-party scripts

These are standalone tools, not part of the `main.py` app:

- `generate_calibration_report.py` — generates a calibration QA report
  (reads calibration NPZ files + checkerboard/charuco captures, produces a
  report)
- `create_charuco_calibration_target.py` — generates a printable ChArUco
  calibration target PDF/PNG, writes to gitignored `output/`

## Non-app directories

- `misc/` — unrelated dev-scratch scripts, not part of the app; also
  `misc/AprilCalibration1/` (the 4 calibration NPZ files, tracked in git —
  the 82 source checkerboard JPGs are gitignored), a real calibration
  fixture used by `tests/test_calibration_io.py` and
  `tests/test_rendering_performance.py`
- `examples/` — real stereo video + calibration fixtures, gitignored,
  local-only
- `output/` — gitignored, generated artifacts land here (e.g. the ChArUco
  PDF/PNG)
