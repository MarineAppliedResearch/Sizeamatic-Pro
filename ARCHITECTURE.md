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

All four supporting modules that used to mix GUI state into plain functions
(or worse, module-level globals) are now real classes, each owned by an
instance on the app:

- **`calibration_summary.py`** — `CalibrationSummaryWindow`
  (`app.cal_summary_window`). This conversion is what removed the
  fragility that caused finding 1 in `FINDINGS.md` (a nested closure
  needing, but missing, its own `global` declaration).
- **`measurement_window.py`** — `MeasurementWindow`
  (`app.measurement_window`). Same conversion.
- **`anaglyph_preview.py`** — `AnaglyphPreview` (`app.anaglyph_preview`).
  Same conversion again — this one also structurally eliminates findings
  2 (an uninitialized module global) and 3 (a bound method can't be
  called with a missing `app` argument the way a free function could).
- **`video_overlay.py`** — `VideoOverlay` (`app.video_overlay`). Same
  conversion, and the biggest of the four (16 methods, the click/drag/
  refine state machine). Constructed earlier than the other three in
  `SizeamaticProApp.__init__`, since `_build_viewers` calls
  `self.video_overlay.create_canvases()` directly — the other three only
  need to exist before a user action first opens them.

Each is independently unit-testable without constructing a real Tk root for
the pure-logic parts — `tests/conftest.py`'s `FakeApp` stand-in — and can be
tested by constructing an instance directly for anything that does need a
real canvas/window (see `tests/test_regressions.py`,
`tests/test_video_overlay.py`).

`stereo_matching.py` (stereo point matching and triangulation math) and the
new `calibration_io.py` (calibration NPZ loading/validation, pulled out of
`main.py`'s `on_load_calibration_folder`) are pure functions with no GUI
code — no restructuring needed for either.

**Still open — not yet done:** `main.py` (~1,750 lines) itself hasn't been
"thinned." It still holds the core `SizeamaticProApp` class with window/menu
construction, video I/O (two OpenCV `VideoCapture` objects held open for the
app's lifetime), playback/timeline/slider logic, rendering, and zoom/pan
state all bundled together, wiring the four classes above together via
composition (`self.cal_summary_window`, `self.measurement_window`,
`self.anaglyph_preview`, `self.video_overlay`). Whether/how to split
`main.py`'s own remaining concerns further is an open question for the rest
of Phase 5.

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
