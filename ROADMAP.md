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

## Phase 5 — Architecture restructure `[ ]`

Using everything learned in Phases 3-4, break `main.py` out of the monolith
into a real module structure, with the Phase 4 test suite as a safety net
against regressions. The specific shape is intentionally undecided as of
Phase 1 — see the "Architecture" section of `AGENTS.md`.

- [ ] Decide target architecture/module layout
- [ ] Decide whether to keep Tkinter or migrate GUI frameworks
- [ ] Execute the restructure

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
