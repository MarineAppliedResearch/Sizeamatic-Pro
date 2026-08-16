# Architecture

> **This file describes the system as it exists right now.** It is not a
> history of how it got this way and not a plan for how it should look in the
> future — that's what `git log` and `ROADMAP.md` are for. Whenever you change
> the code's structure, update this file in the same change so it doesn't go
> stale. If you're reading this to understand the codebase before making a
> change, trust this file over your assumptions; if you find it's wrong,
> fix it.

## Current shape: post-restructure, into Phase 11 (look and feel polish)

Phase 5's target (real classes owned by the app, replacing the
module-level-global state pattern) is done — see `ROADMAP.md` Phase 5 for
how that decision was reached. Phase 5 itself decided to stay on Tkinter,
since the "unacceptably slow" rendering that partly motivated considering
a framework migration turned out to be two self-inflicted bottlenecks in
`main.py`: a PNG-encode/base64 round trip in `_display_bgr_on_canvas`,
and an unconditional `cap.set(CAP_PROP_POS_FRAMES)` seek on every frame
read in `_read_frame_at` (bigger of the two — 19x slower than sequential
reads). **That decision was later revisited mid-Phase-11**: Tkinter's
native menu bar can't be dark-themed on Windows, which blocked the
look-and-feel work this phase is actually about, so the whole app was
ported to PySide6/Qt instead — every module below is now Qt-based
(`QDialog`/`QTableWidget`/`QLineEdit`/etc.), not Tkinter.

All four supporting modules that used to mix GUI state into plain functions
(or worse, module-level globals) are real classes, each owned by an
instance on the app. Two more classes were added during the Phase 11 Qt
migration for the in-app calibration workflow (Phase 10) — same pattern,
listed alongside the original four below:

- **`calibration_summary.py`** — `CalibrationSummaryWindow`
  (`app.cal_summary_window`). This conversion is what removed the
  fragility that caused finding 1 in `FINDINGS.md` (a nested closure
  needing, but missing, its own `global` declaration).
- **`measurement_window.py`** — `MeasurementWindow`
  (`app.measurement_window`). Same conversion. Its results table,
  "current measurement" copy box, and the (Phase 7) explicit-"Record"
  accumulating log all share one row format — a single flat table tagged
  by a `Type` column ("Point"/"Segment"/"Total") with leading
  Video/Frame/Timestamp columns on every row — defined once as
  `RESULT_COLUMNS`/`RESULT_HEADERS` at module level so the table, the
  copy block, and the log can't drift apart from each other.
- **`anaglyph_preview.py`** — `AnaglyphPreview` (`app.anaglyph_preview`).
  Same conversion again — this one also structurally eliminates findings
  2 (an uninitialized module global) and 3 (a bound method can't be
  called with a missing `app` argument the way a free function could).
- **`video_overlay.py`** — `VideoOverlay` (`app.video_overlay`). Same
  conversion, and the biggest of the original four (16 methods, the
  click/drag/refine state machine). Constructed earlier than the others
  in `SizeamaticProApp.__init__`, since `_build_viewers` calls
  `self.video_overlay.create_canvases()` directly — the rest only need
  to exist before a user action first opens them.
- **`perform_calibration.py`** — `PerformCalibrationWindow`
  (`app.perform_calibration_window`). The Phase 10 frame-pair capture +
  auto-scan + run-calibration workflow. Its background auto-scan keeps
  the original `threading.Thread` + `queue.Queue` producer design
  unchanged (pure cv2/numpy, no GUI dependency) — only the consumer side
  swapped Tkinter's `after()` polling for a `QTimer`.
- **`generate_calibration_target.py`** — `GenerateCalibrationTargetWindow`
  (`app.generate_calibration_target_window`). The Phase 10 printable
  checkerboard/ChArUco board generator with a live preview. Despite its
  filename suggesting a standalone script (like
  `create_checkerboard_calibration_target.py`/
  `create_charuco_calibration_target.py` below, which it calls into), it
  has no `__main__` entry point of its own — it's exclusively a
  `QDialog` wired into the app, not a standalone tool.

Each is independently unit-testable without constructing a real Qt
`QApplication` for the pure-logic parts — `tests/conftest.py`'s `FakeApp`/
`FakeAppWidget` stand-ins — and can be tested by constructing a real
instance for anything that does need an actual widget/dialog (see
`tests/test_regressions.py`, `tests/test_video_overlay.py`).

`stereo_matching.py` (stereo point matching and triangulation math —
including two distinct point-quality diagnostics: `reprojection_rms_px`,
a pixel-space check on the Y-averaged triangulated point, and (ROADMAP.md
Phase 12) `stereo_ray_residual_mm`, an object-space check on the two
original un-averaged left/right viewing rays — see the latter's
docstring for why these are deliberately not the same quantity),
`calibration_io.py` (calibration NPZ loading/validation, pulled out of
`main.py`'s `on_load_calibration_folder`), and the Phase 7 addition
`project_io.py` (saves/loads a small JSON manifest of video paths,
calibration folder, and resync offset) are all pure functions with no GUI
code — no class needed for any of them.

`main.py`'s own video/calibration loaders each follow the same
dialog-vs-logic split as `calibration_io.py`'s extraction: `on_load_left_
video`/`on_load_right_video`/`on_load_calibration_folder` only handle the
file/folder picker, delegating the actual loading and UI-state update to
a dialog-free `_load_left_video_from_path`/`_load_right_video_from_path`/
`_load_calibration_from_folder`. `on_open_project` drives those same
dialog-free helpers directly with paths read from a project file, so a
project load surfaces the exact same per-stage error handling a manual
reload would.

**Still open — not yet done:** `main.py` (~2,000 lines and growing) itself
hasn't been "thinned." It still holds the core `SizeamaticProApp` class
with window/menu construction, video I/O (two OpenCV `VideoCapture`
objects held open for the app's lifetime), playback/timeline/slider logic
(including the resync offset and middle-mouse pan state added in Phase 7),
rendering, and zoom/pan state all bundled together, wiring the classes
above together via composition (`self.cal_summary_window`,
`self.measurement_window`, `self.anaglyph_preview`, `self.video_overlay`,
`self.perform_calibration_window`, `self.generate_calibration_target_window`).
Whether/how to split `main.py`'s own remaining concerns further is still
an open, deliberately deferred design question — see `ROADMAP.md` Phase
5's scope note; nothing since has revisited it.

## Other first-party scripts

These are standalone tools, not part of the `main.py` app:

- `generate_calibration_report.py` — generates a calibration QA report
  (reads calibration NPZ files + checkerboard/charuco captures, produces a
  report)
- `create_charuco_calibration_target.py` — the ChArUco board-image/PDF
  rendering functions `generate_calibration_target.py` calls into (also
  runnable standalone as a dev-convenience script)
- `create_checkerboard_calibration_target.py` — the checkerboard
  counterpart to `create_charuco_calibration_target.py`

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
