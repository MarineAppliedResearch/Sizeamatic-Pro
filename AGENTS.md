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
- **`agents_history.md`** — an append-only human- and assistant-readable
  history of the work done on this project with a coding agent. Unlike
  `ARCHITECTURE.md`/`ROADMAP.md`, this file is never edited to stay
  current — each project/phase/issue gets its own roughly-four-paragraph
  summary appended once that unit of work wraps up (or reflecting
  progress so far, if it's still in progress). Read it for context on how
  and why past work happened; add a new entry to it when you finish a
  phase or a similarly-scoped piece of work, rather than only relying on
  git history or chat logs, which don't carry the reasoning behind
  decisions forward to a future session.

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

Use `QMessageBox` (`PySide6.QtWidgets`) for user-facing errors/dialogs,
`print()` for console/debug output. Don't introduce the `logging` module
as a new convention without asking first.

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

## Packaging — building the standalone .exe

Sizeamatic Pro can be built as a single-file, offline-capable Windows
`.exe` via [PyInstaller](https://pyinstaller.org/) (a dev-only
dependency — see `ROADMAP.md`'s Phase 9):

```
uv run pyinstaller sizeamatic.spec
```

The finished build lands at `dist/SizeamaticPro-v<version>.exe` (e.g.
`dist/SizeamaticPro-v0.1.0.exe`) — the version comes from the same
`pyproject.toml` read below, so the filename always matches what's
inside it, no separate bump needed. `build/`, `dist/`,
and `assets/icon.ico` are all gitignored (regenerated by every build);
`sizeamatic.spec` and `assets/default-icon.png`/`assets/splash.png`/
`assets/splash-pro.png` are tracked, since they're build *config* and
source *art*, not build *output*.

`assets/default-icon.png` is the one source-of-truth icon image the
project owner maintains — `sizeamatic.spec` runs
`create_app_icon.build_icon` on it at the top of every build (a `.spec`
file is just executed as a plain Python script by `pyinstaller`),
regenerating `assets/icon.ico` fresh each time rather than needing it
kept manually in sync. Windows expects one `.ico` file containing
several baked-in sizes (16/32/48/256 — see `create_app_icon.py`'s
`ICON_SIZES`), not separate files per size, so swapping in new source
art is just replacing `default-icon.png` and rebuilding — no code or
spec changes. `main.py`'s `main()` reads the regenerated icon at
runtime via `resource_path("assets/icon.ico")` (works both from source
and from inside a PyInstaller bundle).

The `.exe`'s own Windows version resource (visible in Explorer's file
Properties dialog — FileVersion/ProductVersion) is generated the same
way: `sizeamatic.spec` reads `pyproject.toml`'s version and calls
`build_version_info.write_version_file` to produce the gitignored
`version_info.txt` PyInstaller's `EXE(version=...)` consumes, so there's
still exactly one place the version number lives. `main.py`'s
module-level `get_app_version()` reads the same `pyproject.toml` at
runtime (bundled as a data file specifically so this resolves correctly
from inside a packaged build too) — every window's title bar
(`_app_window_title`) and the startup splash both show it, e.g.
"Sizeamatic Pro v0.1.0".

To run the icon conversion by itself (e.g. to check a candidate source
image before committing it as `default-icon.png`), without a full
build:

```
uv run python create_app_icon.py path/to/source_image.png
```

The startup splash is `main.py`'s own `_show_startup_splash` — a Qt
`QSplashScreen`, deliberately **not** PyInstaller's separate bootloader
`--splash` feature. An earlier version of this build used both: the
bootloader splash covering the onefile build's self-extraction phase,
handed off to this app's own splash once the Qt app started. In
practice that showed as two near-identical splashes back to back, which
looked like a bug rather than a smooth handoff — the bootloader
`--splash` support was removed from `sizeamatic.spec` entirely rather
than trying to tune the handoff, since this app's own
splash already covers the whole startup window on its own, identically
whether launched from source (`uv run python main.py`) or from the
packaged `.exe`. It stays up for at least
`STARTUP_SPLASH_MIN_SECONDS` (2s) even if building the rest of the app
finishes faster, and scales `assets/splash-pro.png` down to
`STARTUP_SPLASH_MAX_WIDTH_PX` (720px wide) if it's larger, so a big
source image doesn't fill most of the screen.

The splash image is currently hardcoded to `assets/splash-pro.png`
(there's also a plain `assets/splash.png`, without the "Pro" badge) —
the project owner's plan is a later build option choosing between them
(and gating functionality by which), not implemented yet. Both source
PNGs have partial transparency (a glow effect fading out at the edges),
which naive Tcl/Tk-level image loading doesn't composite correctly — it
shows corrupted magenta/pink instead of blending. `_show_startup_splash`
flattens the image onto solid black first
(`prepare_splash_image.flatten_splash_image`, in-memory) to avoid that.
The file-writing `flatten_splash` still exists for standalone use (e.g.
checking a candidate splash image via `prepare_splash_image.py` from the
command line) but nothing in the build calls it anymore.

VS Code's Run and Debug view has a "Build Sizeamatic Pro .exe
(PyInstaller)" launch config (`.vscode/launch.json`) that runs the same
`sizeamatic.spec` build with one click, alongside the existing "run
main.py" and "docs server" configs.

This is a local build only — there's no CI/GitHub Actions automation for
it yet (deliberately deferred, see `ROADMAP.md`'s Phase 9). Always
actually run the built `.exe` before trusting a packaging change — a
clean build with no errors doesn't guarantee the bundled app launches or
finds its resources correctly.

## Testing

Run the suite: `uv run pytest`. Everything lives under `tests/`.

`tests/conftest.py` also replaces pytest's default per-test "PASSED
[ x%]" stream with a custom terminal report: one aggregated line per
test file (green if it's all-passing, red if not), printed as each
file's tests finish, then a boxed totals summary at the very end.
Pytest's own tracebacks for failures/errors are untouched — only the
per-test progress line and final counts line are replaced. This is
implemented via a few `TerminalReporter` hooks/monkeypatches at the
bottom of `conftest.py`; if you're touching it, note the `trylast=True`
on `pytest_configure` (needed because `_pytest.terminal`'s own
`pytest_configure` is what registers the `TerminalReporter` plugin in
the first place) and that `pytest_report_teststatus` preserves the real
category pytest computes (only blanking the visible letter/word) so
`self.stats["failed"]`/`self.stats["error"]` — which the traceback
sections read from — don't come up empty.

- **Prefer a `FakeApp`/`FakeAppWidget` stand-in over the real GUI.** Most
  of the app's actual logic (`stereo_matching.py`,
  `calibration_summary.py`'s `map_oob_percent`, etc.) takes an `app`
  parameter but only reads a handful of specific attributes off it. Use
  the `make_fake_app` fixture (see `tests/conftest.py`) to build a
  minimal plain-object stand-in with just those attributes set, rather
  than constructing a real `SizeamaticProApp`. Much faster, and avoids
  spinning up real Qt widgets for logic that doesn't need them. Use
  `make_fake_app_widget` instead when the code under test needs `app` to
  be a real `QObject` (e.g. something that parents a `QShortcut` to it).
- **The whole suite runs with `QT_QPA_PLATFORM=offscreen`**, set at the
  very top of `tests/conftest.py` before any other import. This means
  the suite never flashes a visible window, and it must stay set before
  `PySide6.QtWidgets` is imported anywhere — don't reorder conftest's
  top-of-file imports.
- **Tests that touch real Qt widgets need the `qapp` fixture**
  (session-scoped `QApplication.instance() or QApplication([])`, in
  `tests/conftest.py`) — Qt only supports one `QApplication` per process,
  so every test that needs one shares this instance rather than
  constructing its own. `sizeamatic_app` (a real, per-test
  `SizeamaticProApp`, closed via `.close()` after each test) is built on
  top of it.
- **Qt paints immediate-mode, not retained-mode** — nothing about a
  `paintEvent`'s drawing persists as an inspectable object afterward (no
  Tkinter-style canvas item IDs to query). To assert on rendered output,
  render to an offscreen image and sample pixels:
  `pane.grab().toImage()`, then `image.pixelColor(x, y).name()`. See
  `test_video_overlay.py` for the pattern.
- **Prefer synthetic fixtures with known-correct expected values over real
  captured calibration data.** `tests/conftest.py`'s `synthetic_cal`/
  `known_point_pixels` build a small hand-picked rectified stereo rig
  specifically so tests can compute the exact expected triangulation
  result analytically and assert against it — real calibration files have
  no such "known correct answer" to check against.
- Every test function gets a docstring too, same as any other code — see
  the "Coding conventions" section above; it isn't scoped to exclude
  `tests/`.

### Manual proof test — required before committing

`pytest` passing is not the same as the project owner having verified
anything. Automated tests catch regressions in logic an agent already
wrote and already decided what "correct" means for; they don't catch a
UX decision that's wrong, a UI element that's confusing or misplaced, or
a bug that only shows up when a human actually drives the real app. This
matters most for usability-facing work, but applies to bug fixes too —
"I profiled it and the numbers look right" is not the same as the
project owner seeing the fix work.

**Before proposing a commit for any change a human could observe in the
running app** (a bug fix, a new UI control, a behavior change — not a
pure docs/internal-refactor change with no observable effect), give the
project owner an explicit manual test: numbered steps describing exactly
what to click/do in the real app, and exactly what they should see if it
worked (and, where useful, what the old broken/missing behavior looked
like for contrast). Wait for them to actually perform it and confirm
before committing — don't treat "the tests pass" or "I verified it
myself with a script" as a substitute for the project owner's own
hands-on confirmation. If several changes have piled up before this step
happened, walk through all of them, not just the most recent one.

## Git workflow

- **Branching model:** we loosely follow the Git branching model described in
  [nvie.com/posts/a-successful-git-branching-model](https://nvie.com/posts/a-successful-git-branching-model/)
  ("git-flow") — `develop` as the integration branch, `master` as
  release-only, feature branches for everything else. We haven't formally
  adopted its `release/*`/`hotfix/*` branches yet; revisit that if the need
  comes up.
- **Branching:** create a feature branch off `develop` for each task, named
  `issue-N/short-description` (e.g. `issue-7/open-source-readiness`). Never
  commit directly to `develop` or `master`. `master` is reserved for
  releases merged from `develop`; the project owner merges feature branches
  into `develop` themselves.
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
