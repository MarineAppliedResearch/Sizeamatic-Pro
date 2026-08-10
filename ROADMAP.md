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

## Phase 6 — Open source readiness `[ ]`

- [ ] Choose a license
- [ ] Contributing guidelines (how contributors should work with us, PR
      expectations, code of conduct if wanted)
- [ ] Public-facing repo cleanup pass

## Phase 7 — Usability `[ ]`

Focus on the actual user (analyst) experience of the tool, informed by real
usage from Phases 1-6.

- [ ] Usability review with actual analysts using the tool
- [ ] Address friction points (the README already flags rectification as "very
      unacceptably slow", for example)

## Phase 8 — MARE API integration (future, not yet scoped) `[ ]`

Interface with the overall MARE API to record measurement data, etc. Noted
here so it isn't forgotten, but not to be planned in detail until we reach it.
