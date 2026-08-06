# Architecture

> **This file describes the system as it exists right now.** It is not a
> history of how it got this way and not a plan for how it should look in the
> future — that's what `git log` and `ROADMAP.md` are for. Whenever you change
> the code's structure, update this file in the same change so it never goes
> stale. If you're reading this to understand the codebase before making a
> change, trust this file over your assumptions; if you find it's wrong,
> fix it.

## Current shape: single-file monolith

`main.py` (~3,000 lines) contains one class, `SizeamaticProApp`, which mixes
together:

- Tkinter GUI construction (widgets, layout, menus)
- Event handling (mouse clicks for point selection, playback controls,
  scrubbing)
- Video I/O — two OpenCV `VideoCapture` objects (`capL`/`capR`) held open for
  the app's lifetime for fast seeking
- Calibration loading (NPZ files: intrinsics, extrinsics, rectification, maps)
- Stereo rectification
- Stereo measurement math (triangulation for depth/size)

There is no separation between GUI code and business logic — measurement math
runs directly inside UI event callbacks. This is why Phase 4 (regression
testing) can't be meaningfully designed yet: there's no clean logic boundary
to write unit tests against without either testing through the GUI or
extracting logic first.

No target module structure has been decided yet. That decision is explicitly
deferred to Phase 5 in `ROADMAP.md`, once Phase 3 (documentation) and Phase 4
(testing) give enough real understanding of the code to make that call well.

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
