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
migration turned out to be a self-inflicted PNG-encode/base64 bottleneck in
`main.py`'s render path, fixed directly (see `_display_bgr_on_canvas`).

`main.py` (~1,900 lines) still holds the core `SizeamaticProApp` class — GUI
construction, window/menu setup, playback/timeline state, video I/O (two
OpenCV `VideoCapture` objects held open for the app's lifetime), calibration
loading, and the top-level event wiring. It owns one converted piece so far:
`self.cal_summary_window`, a `calibration_summary.CalibrationSummaryWindow`
instance.

Several concerns have been pulled out of `main.py` into separate files.
**One has been converted to a class; the rest are still plain functions**
that take the `SizeamaticProApp` instance (`app`) as their first argument
and read/mutate its attributes directly, or (worse) keep their own state as
module-level globals:

- **`stereo_matching.py`** — stereo point matching and triangulation math
  (scanline mate-point search, triangulation, reprojection error,
  uncertainty estimation). Pure functions, no GUI code, no restructuring
  needed here — this one's fine as-is.
- **`calibration_summary.py`** — `CalibrationSummaryWindow` class (done).
  Owns the Tkinter calibration-summary window and its update logic as
  instance attributes/methods instead of module-level globals — this
  conversion is what removed the fragility that caused `FINDINGS.md` #1
  (a nested closure needing, but missing, its own `global` declaration).
- **`measurement_window.py`** — still plain functions + `app.meas_win`/etc.
  attributes on the app. Next in line for the same class conversion.
- **`anaglyph_preview.py`** — still plain functions + module-level globals
  (`anaglyph_active`, `anaglyph_after_id`, etc.). Not yet converted.
- **`video_overlay.py`** — still plain functions + module-level globals
  (`drag_active`, `left_overlay_canvas`, etc.). Not yet converted.

Once fully converted, each piece will be independently unit-testable without
constructing a real Tk root — `tests/conftest.py`'s `FakeApp` stand-in
already anticipates this for the pure-logic functions, and
`CalibrationSummaryWindow` can now be tested by constructing an instance
directly (see `tests/test_regressions.py`).

## Other first-party scripts

These are standalone tools, not part of the `main.py` app:

- `generate_calibration_report.py` — generates a calibration QA report
  (reads calibration NPZ files + checkerboard/charuco captures, produces a
  report)
- `create_charuco_calibration_target.py` — generates a printable ChArUco
  calibration target PDF/PNG, writes to gitignored `output/`

## Non-app directories

- `misc/` — unrelated dev-scratch scripts, not part of the app or a real test
  suite
- `examples/` — real stereo video + calibration fixtures, gitignored,
  local-only (may become the source of trimmed real test fixtures in Phase 4)
- `output/` — gitignored, generated artifacts land here (e.g. the ChArUco
  PDF/PNG)
