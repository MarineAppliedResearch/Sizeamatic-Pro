# Sizeamatic Pro Roadmap

This roadmap tracks the multi-phase plan for turning Sizeamatic Pro from a working
single-file prototype into a maintainable, documented, testable, agent-friendly
project. Phases are sequential by intent, but scope within a phase can shift as we
learn more.

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done.

## Phase 1 — Conventions, architecture notes, and agent instructions `[x]`

Establish what future contributors (human or agent) need to know before touching
this codebase: coding conventions, current architecture state, git workflow,
dependency management, and this roadmap itself.

- [x] Install `uv`, create `pyproject.toml` + `uv.lock`
- [x] Repo hygiene: rename `tests/` → `misc/`, relocate generated report outputs,
      move charuco outputs into gitignored `output/`, gitignore `examples/`
- [x] `AGENTS.md` — durable conventions, git workflow (points to
      `ARCHITECTURE.md`/`ROADMAP.md` for anything that changes over time)
- [x] `ARCHITECTURE.md` — living doc describing the code's current structure;
      keep it updated whenever the structure changes
- [x] `CLAUDE.md` — pointer to `AGENTS.md`
- [x] `ROADMAP.md` (this file)

## Phase 2 — Documentation generation system `[x]`

Install and configure MkDocs + mkdocstrings so developer docs generate from
docstrings, per the conventions established in Phase 1.

- [x] Add `mkdocs`, `mkdocstrings[python]` as dev dependencies
- [x] `mkdocs.yml` site config, nav structure
- [x] Verify `mkdocs build` renders a working API reference page from at
      least one real docstring (the `SizeamaticProApp` docstring renders
      correctly). Note: verify via `mkdocs build` + opening the static
      HTML, not `mkdocs serve` — see `FINDINGS.md` item D for why.
- [x] `filters: []` set in `mkdocs.yml` so private (`_`-prefixed) members
      actually render — the default filter was hiding most of
      `SizeamaticProApp`'s methods; see `FINDINGS.md` "Documentation
      tooling gaps" section B (caught during Phase 3, not at initial setup)

## Phase 3 — Document and analyze the existing system as-is `[x]`

Go through `main.py` and its supporting modules (see `ARCHITECTURE.md` for
the current file breakdown) and add docstrings to everything per the Phase 1
convention (all public and private classes, functions, methods, and notable
variables/objects). While going through the code this closely, do a
deliberate pass looking for obvious bugs, flaws, and risky patterns
(measurement math, rectification, calibration loading, UI state edge cases).
Also stand up whatever tooling an agent needs to be able to exercise and run
the app itself for testing purposes — this is documentation + analysis, not a
refactor. The goal is a fully-documented, well-understood codebase before we
decide how to restructure it further.

- [x] `main.py`
- [x] `stereo_matching.py`
- [x] `measurement_window.py`
- [x] `calibration_summary.py`
- [x] `anaglyph_preview.py`
- [x] `video_overlay.py`
- [x] `generate_calibration_report.py`
- [x] `create_charuco_calibration_target.py`
- [x] Bugs/flaws findings written up somewhere durable (an issue list, or a
      findings doc) — see `FINDINGS.md`
- [x] Tooling so an agent can run/exercise the app for testing purposes —
      see `smoke_test.py`
- [x] Pass over the rendered docs site to sanity-check the output — built
      clean (`mkdocs build`), spot-checked rendered docstrings across
      several modules. `--strict` mode reports ~163 warnings for missing
      type annotations on the untyped `app` parameter/returns — expected
      given the "type hints on new/touched code only" convention and that
      typing `app: SizeamaticProApp` would require a circular import back
      into `main.py`; not worth chasing down right now.

## Phase 4 — Regression testing system `[x]`

Design and build the actual test suite (unit + any feasible integration
tests), now that the codebase is documented and understood well enough to know
what's testable and how. Usable by both a developer and an agent to catch
regressions before/after changes.

- [x] Decide the testing framework/approach given the GUI+OpenCV shape of the
      code — `pytest`, with a `FakeApp` stand-in (see `AGENTS.md` §Testing)
      for the app-parameter functions instead of a real Tk GUI wherever
      possible
- [x] Real test fixtures — used **synthetic** fixtures with known-correct
      expected values instead of a trimmed subset of `examples/`: real
      calibration files have no "known correct answer" to assert against,
      and the one large real file (`calibration_maps.npz`, 33MB) isn't
      needed when a small synthetic remap array tests the same code path
- [x] Test suite covering core measurement/calibration logic — 43 tests
      across `stereo_matching.py`, `generate_calibration_report.py`,
      `calibration_summary.py`, `create_charuco_calibration_target.py`,
      plus one regression test per bug fixed in `FINDINGS.md`, plus
      `smoke_test.py` folded in as an integration test
- [x] Document how to run tests (for humans and agents) in `AGENTS.md`

Deliberately out of scope: simulated GUI interaction testing (clicks/drags
in `video_overlay.py`) — its module-level mutable state makes it awkward
to test in isolation before the Phase 5 restructure.

## Phase 5 — Architecture restructure `[x]`

Using everything learned in Phases 3-4, break `main.py` out of the monolith
into real classes, with the Phase 4 test suite as a safety net against
regressions. Executed incrementally, one module at a time, with the test
suite run after each step.

**Scope note:** this phase closes with the concrete, identified goal done
(the four supporting modules converted from module-globals to classes,
plus the two performance fixes found along the way) — not with `main.py`
itself fully "thinned." `main.py` (~1,750 lines) still bundles window/menu
construction, video I/O, playback/timeline/slider logic, rendering, and
zoom/pan state in one class. Talked through the breakdown (see
`ARCHITECTURE.md`'s "Still open" note) and deliberately deferred deciding
whether/how to split it further to a future phase, rather than deciding
that open-ended design question as an afterthought at the tail of this
one — it deserves its own dedicated planning round the way Phase 5 itself
got at the start.

- [x] Decide target architecture/module layout — real classes replacing the
      module-level-global pattern in `video_overlay.py`,
      `calibration_summary.py`, `anaglyph_preview.py`, and
      `measurement_window.py` (removes the fragility that caused
      `FINDINGS.md` #1)
- [x] Decide whether to keep Tkinter or migrate GUI frameworks — **staying
      on Tkinter**. The "unacceptably slow" rectified rendering (one of two
      migration motivations) turned out to be a self-inflicted bottleneck,
      not a Tkinter limitation: the old per-frame PNG-encode + base64 +
      `tk.PhotoImage`-parses-base64 round trip in `_display_bgr_on_canvas`
      benchmarked at 42ms/frame against a real captured frame from
      `examples/`; switching to Pillow's `ImageTk.PhotoImage` (wraps the
      numpy array directly, no encoding step) dropped that to 2.36ms/frame
      — a 17.9x speedup, confirmed live. The other motivation (visual
      polish) remains a standing, separate consideration for later if it
      still matters once the rest of the restructure is done.
- [x] Execute the restructure:
  - [x] Fix the render-path performance bottleneck (`main.py`,
        `_display_bgr_on_canvas`) — see above
  - [x] Fix the video-seek performance bottleneck (`main.py`,
        `_read_frame_at`) — found while writing an end-to-end rendering
        test against real footage (`examples/left_20260309_171631.mp4`)
        and the real `misc/AprilCalibration1/` calibration fixture: even
        after the render-path fix, full rectified playback only achieved
        10.2 fps. `cap.set(CAP_PROP_POS_FRAMES)` before every read (even
        for a one-frame sequential advance) forces an expensive keyframe
        seek — benchmarked at 54ms/frame vs. 2.8ms/frame for a plain
        sequential `cap.read()`, a ~19x difference, bigger than the
        render-path fix itself. Fixed by checking the capture's own
        reported position (`cap.get(CAP_PROP_POS_FRAMES)`, ~0.0002ms) and
        only seeking when it doesn't already match the requested index —
        this is correct (not just "assume sequential") even when
        `anaglyph_preview.py`'s independent preview tick reads from the
        same capture in between. End-to-end fps went from 10.2 to 110.7.
        See `tests/test_rendering_performance.py` and
        `tests/test_main.py`'s seek-correctness test.
  - [x] `calibration_summary.py` → `CalibrationSummaryWindow` class
  - [x] Pull calibration file loading/validation out of `main.py` into its
        own module (`calibration_io.py`) — pulled forward from its own
        step since testing the new performance test needed a dialog-free
        load path anyway
  - [x] `measurement_window.py` → `MeasurementWindow` class
  - [x] `anaglyph_preview.py` → `AnaglyphPreview` class
  - [x] `video_overlay.py` → `VideoOverlay` class (the biggest of the
        four — 16 methods, the click/drag/refine state machine; also
        removed `set_overlay_canvases`, confirmed dead code with zero
        callers)
  - [x] Update `tests/` as each piece becomes a class; update
        `ARCHITECTURE.md` to describe the new shape
  - **Deferred, not done:** `main.py` ends up as a thin `SizeamaticProApp`
        wiring the pieces together — see the scope note above. `main.py`
        itself still holds its own substantial logic (playback/timeline,
        rendering, video I/O); splitting that further is an open design
        question for a future phase.

## Phase 6 — Open source readiness `[x]`

Groundwork for outside contributors (issue #7). Flipping the GitHub repo's
visibility to public is a separate, deliberately deferred decision — not
part of this phase.

- [x] Choose a license — Apache License 2.0 (`LICENSE`), copyright held by
      Marine Applied Research & Exploration
- [x] `CONTRIBUTING.md` — branching model, commit conventions, pointer to
      `AGENTS.md` for code/testing conventions, issue-first workflow
- [x] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1, with the
      Enforcement Responsibilities/Enforcement/Consequences sections
      adapted for a single-maintainer project rather than used verbatim
- [x] `license`/`authors` metadata added to `pyproject.toml`
- [x] License/Contributing section added to `README.md`
- [x] GitHub issue/PR templates under `.github/`
- [x] Audit `misc/`, `examples/`, and git history for sensitive info —
      came back clean (no credentials, keys, or hardcoded local paths
      anywhere). One real finding: `misc/report.txt` (an unrelated CEC
      grant deliverable used only as sample text for dev scratch scripts)
      was untracked and gitignored — it stays on disk locally since the
      scripts pick it via a file dialog, not a hardcoded path, but it's no
      longer committed. Also cleaned up leftover scratch content (a
      class-method-listing one-liner and a build command) from the bottom
      of `README.md`.

## Phase 7 — Usability `[~]`

Focus on the actual user (analyst) experience of the tool (issue #8). Started
with a structured usability interview with the project owner (the primary
analyst stakeholder) rather than guessing at friction points; other analysts'
input may come in a later round. Executing incrementally, one item at a time,
with tests added alongside each change — same rhythm as Phase 5.

- [x] Usability interview — see below for the resulting requirements
- [x] Calibration folder guidance — `filedialog.askdirectory`'s title now
      names the four expected NPZ files, and `calibration_io.py`'s
      missing-files error message restates the full requirement rather than
      just naming what's absent.
- [x] Pan when zoomed in — new middle-mouse-drag binding (`on_pan_down`/
      `on_pan_drag`/`on_pan_up` in `main.py`, bound in `video_overlay.py`);
      left-click stays point placement/drag, right-click stays the existing
      explicit-refine gesture — panning deliberately uses a separate button
      so it can't collide with either. Drives the pan offset that already
      existed per-pane (`viewL`/`viewR`'s `off_x`/`off_y`), previously only
      touched by mouse-wheel zoom. Redraws via the new
      `_redisplay_current_frames` (reuses the already-decoded current
      frame) rather than `_render_current_frames`, so continuous drag
      motion doesn't force a `cap.set()` keyframe seek on every event.
      Found and fixed an unrelated small bug along the way — see
      `FINDINGS.md` #8.
- [x] Resync control — `on_toggle_lock` already computed and stored
      `lock_offset_frames` whenever Lock was enabled while both timelines
      were manually scrubbed to the same moment; that was already the
      resync mechanism in substance. Now exposed as a directly-editable
      toolbar Spinbox (`self.offset_var`, `on_offset_changed`) that
      immediately re-aligns the right timeline when Lock is on, instead of
      only being settable implicitly via re-toggling Lock. Deliberately out
      of scope: correcting drift that changes over a video's length (a
      single offset can't fix that) — not an observed problem yet, revisit
      only if it becomes one. Also found and fixed a real bug surfaced by
      this feature: pressing Play with Lock on and a nonzero offset made
      the timelines count backward — see `FINDINGS.md` #9.
- [x] Project file — new `project_io.py` (dialog-free, testable, same
      pattern as `calibration_io.py`): saves/loads a small JSON manifest of
      left/right video paths, calibration folder path, and the resync
      offset above. New File menu items "Save Project…"/"Open Project…" so
      a session doesn't require reloading everything from scratch.
      `on_load_left_video`/`on_load_right_video`/`on_load_calibration_folder`
      were each split into a dialog-only wrapper plus a dialog-free
      `_load_*_from_path`/`_load_calibration_from_folder` helper, so
      `on_open_project` reuses the exact same loading/error-handling logic
      a manual reload would use, per stage.
- [ ] Measurement output overhaul — `measurement_window.py`'s copy block
      currently has no identifying context (video/pair name, frame,
      timestamp) and only ever shows the current measurement. Add an
      identifying header row, and an accumulating log with one new row per
      explicit "Record" action (not auto-logged on every recalculation, to
      avoid flooding it with in-progress drag states).
- [ ] Playback speed — the speed dropdown (0.25x-4x) already exists and
      "1x" is already intended to match native fps, but it's a hardcoded
      40ms tick, not read from the loaded video's actual fps — only
      coincidentally correct for ~25fps footage. Fix: read native fps from
      video metadata. Real achieved playback fps during actual use is
      reported as much lower than the ~25fps target despite Phase 5's
      rendering fixes (110fps in that benchmark, on the small `examples/`
      clip) — needs profiling against real (likely higher-resolution)
      footage to find where the gap actually is; `root.after()` only
      guarantees a *minimum* delay, so if per-tick work exceeds the
      scheduled interval that alone would explain it.
- [ ] Anaglyph 3D preview — confirmed novelty-only, not a usability issue.
      No action; deprioritized for future investment.
- [ ] Sort the above into automated-test-covered vs. manual-judgment-only,
      and extend `tests/` accordingly as each item lands
- [ ] Update `ARCHITECTURE.md`/`FINDINGS.md` to reflect any new modules
      (`project_io.py`) or structural changes

## Phase 8 — MARE API integration (future, not yet scoped) `[ ]`

Interface with the overall MARE API to record measurement data, etc. Noted
here so it isn't forgotten, but not to be planned in detail until we reach it.
