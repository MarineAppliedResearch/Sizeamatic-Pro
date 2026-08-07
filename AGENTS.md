# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, Cursor, etc.) working on
Sizeamatic Pro. This file is tool-agnostic; tool-specific notes live in that
tool's own file (e.g. `CLAUDE.md`), which points back here.

This file holds **durable conventions** — things that don't change week to
week. For anything about the current state of the system, look elsewhere and
keep those files honest:

- **`ARCHITECTURE.md`** — describes the code's actual current structure.
  Whenever you change the code's structure, update this file in the same
  change so it doesn't go stale.
- **`ROADMAP.md`** — the phased project plan, with per-phase checkboxes for
  what's built vs. not (docs site, test suite, tooling, etc.). Check it before
  starting work so you know what's in scope for the current phase versus
  deliberately deferred to a later one — don't start later-phase work (e.g.
  restructuring `main.py`, or building a test framework) without asking first,
  even if it seems like a natural next step. Check off items as you finish
  them.

## What this project is

A Windows desktop app (Tkinter + OpenCV) that lets analysts scrub through a
synchronized stereo video pair, click matching points in each view, and get
real-world depth/size measurements in millimeters, using a prior stereo
calibration. See `README.md` for the user-facing description.

## Coding conventions

These apply to every first-party file in the repo — `main.py` and its
supporting modules (see `ARCHITECTURE.md`), `generate_calibration_report.py`,
`create_charuco_calibration_target.py`, and `tests/`. They do not apply to
`misc/`. This includes test files: every test function gets a docstring
too, not just the fixtures and helpers around them — it's easy to assume
"the test name says what it checks" is enough and skip it, but the
convention doesn't carve out an exception for tests.

### Comments — dense and explanatory (not minimal)

This project intentionally uses a **denser comment style** than typical
Python/Claude Code defaults. Carry over the existing convention already
present in `main.py`:

- Frequent `#` comments explaining what a line or small block does, not just
  non-obvious "why" — comment even when the code is fairly readable on its
  own.
- `# ---- Section Name ----` banner comments to group related state/logic
  (e.g. `# ---- Timeline state ----`).
- This is a deliberate override of the "minimal comments, WHY only" default
  some agents default to. Follow the dense style here.

### Docstrings — required on everything, for the docs site

Every class, function, and method — public **and** private (including
underscore-prefixed) — gets a docstring. Also document notable module-level
variables/objects and non-trivial local variables/objects where their purpose
isn't obvious from the name alone. Use Google-style docstring sections
(`Args:`, `Returns:`, `Raises:`) — this is what the MkDocs + mkdocstrings docs
site (see `ROADMAP.md`) renders into the developer docs.

This is an intentionally exhaustive requirement (mirrors the project owner's
JSDoc conventions from JavaScript work) — don't skip private helpers just
because they're internal.

### Type hints — required on new/touched code

Any function or method you write new, or substantially edit, should have
parameter and return type hints. No obligation to retrofit type hints onto
existing code you're not otherwise touching.

### Errors and status output

Use `tkinter.messagebox` for user-facing errors/dialogs, `print()` for
console/debug output. Don't introduce the `logging` module as a new
convention without asking first.

### Naming, formatting, tooling

No linter, formatter, or type checker (ruff, mypy, black, etc.) is configured,
deliberately — the project owner reviews style manually rather than via
enforcement tooling. Don't add lint/format config unless asked. Follow
standard PEP 8 conventions (snake_case for functions/variables, PascalCase for
classes) and match the style already present in the file you're editing.

## Dependency management

This project uses [`uv`](https://docs.astral.sh/uv/). `pyproject.toml` lists
runtime dependencies; `uv.lock` pins exact versions.

- Add a dependency: `uv add <package>`
- Add a dev-only dependency (e.g. docs tooling): `uv add --dev <package>`
- Run a script inside the project's environment: `uv run python main.py`
- Sync the environment to match the lockfile: `uv sync`

## Docs site — build it, don't serve it

To check what the generated developer docs actually look like:

```
uv run mkdocs build
```

then open `site/index.html` (or any other file under `site/`) directly in
a browser. No server needed.

**Do not use `mkdocs serve` to verify content.** It has a confirmed,
reproducible bug where instance-attribute docstrings render as
completely empty, even though the exact same source renders correctly via
`mkdocs build` — see `FINDINGS.md` item D. Module, function, and method
docstrings weren't affected, only instance/class attributes, which made
it a very misleading way to check whether documentation content was
actually there. If you want to preview the docs, always build fresh and
open the static file — don't trust a running dev server, live-reloaded or
not.

## Testing

Run the suite: `uv run pytest`. Everything lives under `tests/`.

- **Prefer a `FakeApp` stand-in over the real GUI.** Most of the app's
  actual logic (`stereo_matching.py`, `calibration_summary.py`'s
  `map_oob_percent`, etc.) takes an `app` parameter but only reads a
  handful of specific attributes off it. Use the `make_fake_app` fixture
  (see `tests/conftest.py`) to build a minimal object with just those
  attributes set, rather than constructing a real `SizeamaticProApp`. Much
  faster, and avoids the Tk pitfalls below entirely.
- **Only one real `tk.Tk()` per test session, ever.** Confirmed
  empirically: Tcl/Tk does not reliably support creating and fully
  destroying more than one interpreter within a single process on this
  setup — it fails with `TclError: invalid command name "tcl_findLibrary"`
  or a broken `init.tcl` path lookup, intermittently, after the second or
  third such cycle. `tests/conftest.py`'s `hidden_tk_root` fixture is
  session-scoped for exactly this reason — reuse it (and the
  `sizeamatic_app` fixture built on it) rather than calling `tk.Tk()`
  directly in a new test.
- **If a test needs to exercise code that calls `self.root.destroy()`**
  (like `on_app_close`), don't let it actually destroy the shared session
  root — redirect it first. Use `root.quit` (stops `mainloop()` without
  destroying anything) if the code under test calls `mainloop()`, or a
  plain no-op lambda if it doesn't. See `test_smoke.py` and
  `test_regressions.py`'s `on_app_close` test for both patterns.
- **Prefer synthetic fixtures with known-correct expected values over real
  captured calibration data.** `tests/conftest.py`'s `synthetic_cal`/
  `known_point_pixels` build a small hand-picked rectified stereo rig
  specifically so tests can compute the exact expected triangulation
  result analytically and assert against it — real calibration files have
  no such "known correct answer" to check against.
- Every test function gets a docstring too, same as any other code — see
  the "Coding conventions" section above; it isn't scoped to exclude
  `tests/`.

## Git workflow

- **Branching model:** we loosely follow the Git branching model described in
  [nvie.com/posts/a-successful-git-branching-model](https://nvie.com/posts/a-successful-git-branching-model/)
  ("git-flow") — `develop` as the integration branch, `master` as
  release-only, feature branches for everything else. We haven't formally
  adopted its `release/*`/`hotfix/*` branches yet; revisit that if the need
  comes up.
- **Branching:** create a feature branch off `develop` for each task (e.g.
  `phase1/agents-md-setup`). Never commit directly to `develop` or `master`.
  Open a PR into `develop` when the work is ready for review. `master` is
  reserved for releases merged from `develop`.
- **Commit messages:** plain, descriptive messages (no Conventional Commits
  prefix requirement) — describe what changed and why in the body if it's not
  obvious from the summary line.
- **Issue references:** when work is tied to a GitHub issue, include the issue
  number (e.g. `#1`) in the commit message. If you don't have an issue number
  for the current task, ask before assuming one isn't needed.
- **Autonomy:** always ask before committing, and always ask before pushing.
  Don't commit or push proactively — even mid-task — without checking in
  first.

## Repo hygiene conventions

- `examples/` and `output/` are gitignored (real video/calibration fixtures
  and generated artifacts, respectively). Don't try to commit large fixture
  or generated content there without asking first.
- `.vscode/` is intentionally tracked (shared debug/run config) — don't
  gitignore it.
