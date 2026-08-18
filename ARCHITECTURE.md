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
docstring for why these are deliberately not the same quantity; also
(ROADMAP.md Phase 13) two uncertainty estimators shown side by side per
the project owner's request during that phase: the original sample-
standard-deviation-of-perturbations `estimate_point_sigma_mm`/
`estimate_segment_sigma_len_mm`, and a Jacobian/covariance-propagation
alternative, `estimate_point_sigma_mm_jacobian`/
`estimate_segment_sigma_len_mm_jacobian`, reusing the same perturbed
coordinates but combining them as an explicit propagated variance
instead of a sample statistic),
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
`self.perform_calibration_window`, `self.generate_calibration_target_window`,
`self.tutorial_window` - see "Tutorial mode" below).
Whether/how to split `main.py`'s own remaining concerns further is still
an open, deliberately deferred design question — see `ROADMAP.md` Phase
5's scope note; nothing since has revisited it.

## Tutorial mode (ROADMAP.md Phase 15)

Four new modules, all at the repo root (no subpackages used anywhere in
this project), splitting the same way the rest of the codebase does:
pure logic with no Qt dependency vs. real Qt widgets.

- **`tutorial_engine.py`** — pure step-tracking data model, no Qt import
  at all (matches `stereo_matching.py`/`calibration_io.py`/
  `project_io.py`'s "pure functions/classes, no GUI code" convention).
  `TutorialStep` holds one step's static content (`step_id`, `title`,
  `description`, `details`, a `(host, kind, ref)` `target` tuple naming
  which real widget to highlight and on which window, and a
  `completion_action` key). `Tutorial` tracks an ordered list of steps
  plus runtime progress (`current_index`, `done`) - `advance()` (linear
  default), `jump_to(index)` (the checklist's skip/jump escape hatch,
  no locking), and `mark_action_done(action_name)` (the real
  completion-detection entry point every wired-up handler calls into,
  via `TutorialController.notify_action`). Deliberately decoupled from
  any specific track's *content*, per Phase 15's "explicitly built to
  extend later" requirement - a future calculations-tutorial track
  reuses this exact engine, unmodified, with its own step list.
- **`tutorial_content_operational.py`** — the actual v1 "Getting
  Started" step list: a plain Python list of `TutorialStep(...)` calls
  covering the operational workflow (load video/calibration, view/sync,
  place/record measurements, save/load projects). Kept as its own
  module, separate from a future `tutorial_content_calculations.py`, so
  both content sets plug into `tutorial_engine.Tutorial` unmodified. The
  `point_quality_metrics`/`total_row_and_chain_sigma` steps' explanation
  text (covering `ReprojRMS(px)`/`RayResidual(mm)`/sigma columns) went
  through the project owner's scientific-accuracy sign-off before this
  phase closed, per Phase 12/13's precedent for how subtle those
  distinctions are.
- **`tutorial_window.py`** — the Qt-facing half: `TutorialController`
  (`app.tutorial_window`, same ownership pattern as
  `app.measurement_window` etc.) owns a `tutorial_engine.Tutorial` built
  from `tutorial_content_operational.STEPS`, plus every widget below,
  and is the single object real handlers call into
  (`notify_action(action_name)`). `HighlightOverlay` dims a host window
  except a pulsing-bordered cutout around the current step's target
  widget - one instance per host window (the main app, and lazily the
  measurement window once it exists), since an overlay is a child
  widget of one specific top-level window. `DraggablePanel` is the
  shared base for `TutorialStepBubble` (the roaming current-step panel:
  title/description/expandable Details/Next button) and `ChecklistPanel`
  (the fixed-corner, scrollable full step list with completion marks) -
  both floating, non-modal, `Tool | FramelessWindowHint |
  WindowStaysOnTopHint` windows with custom drag-to-move (no native
  title bar) and their own "x", which postpones the whole tutorial
  (`TutorialController._postpone`) rather than dismissing either panel
  independently - clicking Help > Start Tutorial… again resumes exactly
  where it left off. `WindowTracker` is a `QObject` event filter
  (`installEventFilter`) that catches the main window's Move/Resize/
  WindowStateChange - instance-patching `moveEvent` directly on someone
  else's class doesn't reliably fire in PySide6.
- **`tutorial_fixtures.py`** — the synthetic tutorial video+calibration
  generator, pure logic, no Qt. Builds a small rectified stereo rig
  (mirroring `tests/conftest.py`'s `synthetic_cal` fixture's approach)
  and renders a short left/right MP4 pair directly from that rig's own
  `PL`/`PR` projection matrices, with a burned-in on-screen clock and a
  separate burned-in frame counter. The two videos are deliberately
  generated out of sync with each other
  (`TUTORIAL_SYNC_OFFSET_FRAMES`), so the tutorial's Lock/Resync step
  has a real offset to find and correct, not a pair that already lines
  up by construction. The calibration's rectification maps apply a mild
  inward crop (`TUTORIAL_RECTIFICATION_ZOOM`) rather than a pure
  identity grid, purely so toggling "Show Rectified" visibly
  zooms/crops like a real calibration's rectified view does - marker
  positions are drawn through the inverse of that same crop
  (`_to_raw_pixel`) so a click on the displayed marker still
  triangulates to the exact real position it was rendered from,
  regardless of which view mode it's clicked in.
  `TutorialController.start()` generates these fixtures but
  deliberately does NOT load them automatically - the user still goes
  through the real File > Load Left/Right Video…/Calibration > Load
  Calibration… actions themselves, so the tutorial's early "load" steps
  are genuine practice, not a shortcut past them.

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
