# Architecture

> **This file describes the system as it exists right now.** It is not a
> history of how it got this way and not a plan for how it should look in the
> future — that's what `git log` and `ROADMAP.md` are for. Whenever you change
> the code's structure, update this file in the same change so it doesn't go
> stale. If you're reading this to understand the codebase before making a
> change, trust this file over your assumptions; if you find it's wrong,
> fix it.

## Current shape: one core class + function modules operating on it

`main.py` (~1,900 lines) still holds the core `SizeamaticProApp` class — GUI
construction, window/menu setup, playback/timeline state, video I/O (two
OpenCV `VideoCapture` objects held open for the app's lifetime), calibration
loading, and the top-level event wiring.

Several concerns have been pulled out of `main.py` into separate files, but
**not as classes or a layered architecture** — each is a module of plain
functions that take the `SizeamaticProApp` instance (`app`) as their first
argument and read/mutate its attributes directly:

- **`stereo_matching.py`** — stereo point matching and triangulation math
  (scanline mate-point search, triangulation, reprojection error,
  uncertainty estimation)
- **`measurement_window.py`** — the Tkinter measurement-results window and
  its update logic
- **`calibration_summary.py`** — the Tkinter calibration-summary window and
  its update logic
- **`anaglyph_preview.py`** — anaglyph (red/cyan) stereo preview generation
  and its start/stop/tick lifecycle
- **`video_overlay.py`** — overlay canvases, point-handle drag/drop
  interaction, and overlay redraw logic

This means the *file* boundaries now roughly track feature areas, but there's
still no real encapsulation — every module reaches back into `app`'s
attributes rather than being handed the specific state it needs, so nothing
here is unit-testable in isolation yet. `main.py` imports all five modules
and calls into them from its event handlers.

No target architecture (classes/layers, or whether module-per-feature is the
end state) has been decided yet. That decision is explicitly deferred to
Phase 5 in `ROADMAP.md`.

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
