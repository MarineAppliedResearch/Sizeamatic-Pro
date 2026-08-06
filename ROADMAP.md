# Sizeamatic Pro Roadmap

This roadmap tracks the multi-phase plan for turning Sizeamatic Pro from a working
single-file prototype into a maintainable, documented, testable, agent-friendly
project. Phases are sequential by intent, but scope within a phase can shift as we
learn more.

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done.

## Phase 1 — Conventions, architecture notes, and agent instructions `[~]`

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

## Phase 2 — Documentation generation system `[ ]`

Install and configure MkDocs + mkdocstrings so developer docs generate from
docstrings, per the conventions established in Phase 1.

- [ ] Add `mkdocs`, `mkdocstrings[python]` as dev dependencies
- [ ] `mkdocs.yml` site config, nav structure
- [ ] Verify `mkdocs serve` renders a working API reference page from at least
      one real docstring

## Phase 3 — Document and analyze the existing system as-is `[ ]`

Go through `main.py` (and the other first-party scripts) and add docstrings to
everything per the Phase 1 convention (all public and private classes,
functions, methods, and notable variables/objects). While going through the
code this closely, do a deliberate pass looking for obvious bugs, flaws, and
risky patterns (measurement math, rectification, calibration loading, UI state
edge cases). Also stand up whatever tooling an agent needs to be able to
exercise and run the app itself for testing purposes — this is documentation +
analysis, not a refactor. The goal is a fully-documented, well-understood
monolith before we decide how to restructure it.

- [ ] `main.py`
- [ ] `generate_calibration_report.py`
- [ ] `create_charuco_calibration_target.py`
- [ ] Bugs/flaws findings written up somewhere durable (an issue list, or a
      findings doc)
- [ ] Tooling so an agent can run/exercise the app for testing purposes
- [ ] Pass over the rendered docs site to sanity-check the output

## Phase 4 — Regression testing system `[ ]`

Design and build the actual test suite (unit + any feasible integration
tests), now that the codebase is documented and understood well enough to know
what's testable and how. Usable by both a developer and an agent to catch
regressions before/after changes.

- [ ] Decide the testing framework/approach given the GUI+OpenCV shape of the
      code
- [ ] Real test fixtures (likely a small trimmed subset of what's in
      `examples/` today)
- [ ] Test suite covering core measurement/calibration logic
- [ ] Document how to run tests (for humans and agents) in `AGENTS.md`

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
