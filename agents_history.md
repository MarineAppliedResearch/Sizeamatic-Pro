# Agents History

A running, human- **and** assistant-readable history of the work done on
Sizeamatic Pro with the help of a coding agent (Claude Code). Unlike
`ROADMAP.md` (current phase status) or `ARCHITECTURE.md` (current code
shape), this file doesn't get edited to stay "current" — it's an
append-only log. Each project/phase/issue gets one entry, roughly four
paragraphs, written after that unit of work wraps up (or, for an
in-progress entry, reflecting where things stand so far). See
`AGENTS.md` for the pointer into this file and the durable conventions
this history was built under.

---

## Phase 1 — Conventions, architecture notes, and agent instructions (issue #1)

Isaac set out to put this project — a Tkinter + OpenCV desktop tool that
lets MARE analysts measure real-world distances from synchronized stereo
video — onto a real footing for agentic development, having not worked
with Claude Code before. Rather than have the agent assume conventions,
he asked to be "interrogated" up front on every foundational choice:
comment style, docstring scope, type hints, dependency management, error
handling, testing approach, and git workflow.

That interrogation produced `AGENTS.md` as the durable-conventions
document: a deliberately dense, JSDoc-style comment convention (carried
over from Isaac's JavaScript background, overriding the agent's normal
minimal-comment default); exhaustive Google-style docstrings on every
public *and* private class/function/method/notable variable, explicitly
including test code; type hints required on new or substantially-touched
code only; `tkinter.messagebox`/`print` for errors rather than
introducing `logging`; and no linter/formatter/type-checker, since Isaac
prefers manual review over enforcement tooling.

Partway through, Isaac corrected an early structural mistake: `AGENTS.md`
had started accumulating current-state and phase-planning content
alongside the durable conventions. That got split out into
`ARCHITECTURE.md` (living description of the code's actual current
structure) and `ROADMAP.md` (the phased plan with per-phase checkboxes),
leaving `AGENTS.md` purely for things that don't change week to week.
`CLAUDE.md` was added as a short, tool-specific pointer back to
`AGENTS.md`.

This phase also established the git workflow that held for every phase
since: git-flow branching (`issue-N/short-description` off `develop`),
plain descriptive commit messages signed with `git commit -a -s`, issue
references in the commit *body* only (never the title — corrected more
than once when this slipped), never a bare `#N` except that one
deliberate reference (GitHub auto-links every bare `#N`), always ask
before every commit and push, never amend, and Isaac merges feature
branches into `develop` himself. A repo-hygiene pass (renaming `tests/` →
`misc/`, gitignoring `examples/`/`output/`) rounded out the phase.

## Phase 2 — Documentation generation system

With conventions in place, this phase stood up MkDocs + mkdocstrings so
developer docs generate automatically from the docstrings Phase 1 had
just mandated. The dependency was added via `uv add --dev`, and
`mkdocs.yml` was configured with a nav structure and a working API
reference page, verified against the `SizeamaticProApp` docstring.

Two documentation-tooling gaps were caught during this phase (fully
written up in `FINDINGS.md`'s "Documentation tooling gaps" section): the
mkdocstrings default filter was hiding all `_`-prefixed private members
(fixed via `filters: []`), and nested (function-local) functions never
render regardless of any filter setting, since griffe only collects
module- and class-level members.

The most consequential finding, though, came slightly later while
verifying the fix worked: `mkdocs serve` was confirmed to render every
instance-attribute docstring on a page as completely empty, while
`mkdocs build` — run against the exact same source — rendered all of
them correctly. This wasn't a one-off; it reproduced consistently even
on the server's very first build.

The practical fix — documented prominently in `AGENTS.md` under "Docs
site — build it, don't serve it" — is to never use `mkdocs serve` to
verify content, and always run `mkdocs build` then open the static HTML
directly in a browser. This became a standing rule for the rest of the
engagement and caused real friction later in Phase 3/4 when docs
appeared to be missing purely because of which command was used to check
them.

## Phase 3 — Document and analyze the existing system as-is

This phase went through `main.py` and every supporting module, adding
docstrings to all public and private classes, functions, methods, and
notable variables per the Phase 1 convention — not a refactor, just
documentation and analysis, deliberately, so the codebase would be fully
understood before any restructuring was decided.

While documenting this closely, three real bugs were found and fixed
(all now in `FINDINGS.md`): a nested `_on_close` closure in
`calibration_summary.py` missing its own `global` declaration (leaving
stale references to destroyed widgets after the window closed); an
uninitialized `anaglyph_after_id` module global that could raise
`NameError`; and `on_app_close` calling `stop_anaglyph_preview()` with no
`app` argument. Several other flaws were flagged but deliberately left
unfixed, noted as either currently unreachable or better addressed during
the Phase 5 restructure.

This phase is also where the `mkdocs serve` vs. `mkdocs build` discovery
actually surfaced in practice — Isaac pushed back hard mid-task
("none of these god damned things have any fucking notes on them!!!!")
when docs he expected to be present appeared completely bare, which led
directly to isolating the root cause described in Phase 2's entry above.

Beyond docs and bugs, this phase stood up `smoke_test.py` so an agent
(not just a human) could actually run and exercise the app for testing
purposes, laying groundwork for Phase 4.

## Phase 4 — Regression testing system

With the codebase now documented and understood, this phase designed and
built the actual `pytest` suite. Given the GUI+OpenCV shape of the code,
the key design decision was a `FakeApp`/`FakeVar` stand-in pattern: most
of the app's real logic only reads a handful of specific attributes off
an `app` parameter, so tests could build a minimal fake object instead of
constructing a real Tkinter GUI wherever possible.

Two other testing conventions came out of this phase and held for the
rest of the engagement: a single session-scoped `hidden_tk_root` fixture,
since Tcl/Tk doesn't reliably support creating and destroying more than
one interpreter per process; and synthetic, known-correct stereo
calibration fixtures (rather than trimmed real calibration data) so tests
could assert exact expected values analytically.

The resulting suite covered `stereo_matching.py`, calibration/report
generation, and one regression test per bug fixed in Phase 3's
`FINDINGS.md`, plus `smoke_test.py` folded in as an integration test.
Simulated GUI interaction testing (clicks/drags in `video_overlay.py`)
was deliberately deferred, since that module's module-level mutable state
made it awkward to test in isolation before the Phase 5 restructure.

This phase also cemented the "ask before assuming" pattern for scope
itself: when the testing framework question turned out to have real
architectural implications, Isaac was consulted on the approach rather
than the agent picking one and running with it.

## Phase 5 — Architecture restructure (issue #6)

Using everything learned in Phases 3-4, this phase converted `main.py`'s
module-level-global pattern into real classes: `CalibrationSummaryWindow`,
`MeasurementWindow`, `AnaglyphPreview`, and `VideoOverlay`, each owned by
the `SizeamaticProApp` instance, with the Phase 4 test suite run after
every incremental step as a safety net. This structurally eliminated
three of the bug classes found in Phase 3 — there's no module-level
global left to have a missing `global` declaration or an
uninitialized-before-first-use race.

Two real performance bottlenecks were found and fixed along the way, both
confirmed with before/after benchmarks against real footage: the old
per-frame PNG-encode + base64 + `tk.PhotoImage` round trip in
`_display_bgr_on_canvas` benchmarked at 42ms/frame, dropped to 2.36ms/frame
(17.9x) by switching to Pillow's `ImageTk.PhotoImage`; and
`cap.set(CAP_PROP_POS_FRAMES)` being called before every single frame read
(even sequential ones) forced an expensive keyframe seek every time —
checking the capture's own reported position first and only seeking when
it didn't already match dropped end-to-end rectified playback from
10.2fps to 110.7fps.

Isaac corrected the scope of this phase twice during planning: once to
clarify that the GitHub issue description was conflating this phase with
a different one ("i think you are getting two phases mixed up"), and once
near the end to deliberately close the phase with its concrete goal done
rather than let it expand into fully thinning `main.py` itself — that
open design question was pushed to a future phase rather than decided as
an afterthought.

Also decided during this phase: staying on Tkinter rather than migrating
GUI frameworks, since the "unacceptably slow" rendering that motivated
considering a migration turned out to be the self-inflicted PNG/base64
bottleneck above, not a Tkinter limitation. Visual polish remains a
separate, unaddressed consideration for later.

## Phase 6 — Open source readiness (issue #7)

This phase prepared the repo for outside contributors: an Apache License
2.0 `LICENSE` (chosen for the permissive terms and patent grant, fitting
a research nonprofit wanting wide reuse), a `CONTRIBUTING.md` describing
the git-flow branching model and pointing back to `AGENTS.md` for code
conventions, a Contributor Covenant `CODE_OF_CONDUCT.md`, `license`/
`authors` metadata in `pyproject.toml`, a License/Contributing section in
`README.md`, and GitHub issue/PR templates under `.github/`. Flipping the
GitHub repo's actual visibility to public was explicitly and deliberately
*not* part of this phase — that's a separate decision for later.

The Code of Conduct became a notable mid-task correction: the agent
initially wrote the whole document itself, including invented specifics
(a vague enforcement-contact placeholder), which Isaac flagged directly
— he wanted to be asked about every specific decision, not just told the
standard template was being adopted. The subsequent questions (who
enforces it, what contact to list, whether to keep the standard 4-tier
enforcement ladder or simplify it for a single-maintainer project) were
asked and answered properly before the document was finalized.

A sensitive-info audit (git history and the working tree) came back
mostly clean — no credentials, keys, or hardcoded local paths anywhere —
with one real finding: `misc/report.txt`, an unrelated CEC grant
deliverable that had been used only as sample input text for some dev
scratch scripts, was untracked and gitignored (kept on disk locally,
since the scripts pick their input via a file dialog rather than a
hardcoded path). Leftover scratch content at the bottom of `README.md`
was cleaned up in the same pass.

This phase also produced a lasting change to the working agreement
itself: after repeated "ask before every push" confirmations became
friction, `CLAUDE.md` was updated so a confirmed commit pushes
automatically, without a second separate confirmation — commit
confirmation stayed required, push confirmation was dropped.

## Phase 7 — Usability (issue #8)

This phase started differently from the others: rather than the agent
proposing usability fixes, Isaac asked to be quizzed first on every
usability aspect of the tool from an analyst's actual workflow, with the
resulting requirements only then turned into a test-backed implementation
plan — extending, not replacing, the existing test suite for whichever
requirements turned out to be automatable. The quiz covered navigation/
scrubbing, point-placement precision, view switching, keyboard shortcuts,
zoom/pan, calibration-folder clarity, and measurement export, and
surfaced several concrete, previously-unplanned feature requests: full
native-speed video playback, panning while zoomed in, a manual resync
control for misaligned pairs, a save/reload project file, and a
spreadsheet-ready measurement export with an accumulating log.

Landed so far: calibration-folder guidance (the folder picker and the
missing-files error message now both name the four expected NPZ files);
middle-mouse-drag panning (deliberately a separate mouse button from
point placement and point refinement, so it can't collide with either);
a directly-editable resync offset control exposing the `lock_offset_frames`
mechanism that had existed in substance since Phase 5 but was only ever
settable implicitly; and a project file (`project_io.py`, "Save
Project…"/"Open Project…") that saves/restores video paths, the
calibration folder, and the resync offset, built by splitting each
existing video/calibration loader into a dialog-only wrapper plus a
reusable dialog-free helper.

While testing the resync control, a real, pre-existing, and fairly
serious bug was found and fixed: pressing Play with Lock enabled and a
nonzero resync offset made both timelines count *backward* instead of
forward. The root cause was `_playback_tick`'s locked branch setting both
slider widgets directly without suppressing their callbacks — every other
call site that programmatically moves both sliders already guarded
against this — which chained into the offset-aware jump helper twice per
tick with contradictory targets. The fix replaced that branch's
duplicated (and wrong) logic with a direct call to the same helper the
slider and step controls already used correctly, fixing the direction bug
and, as a side effect, making Play finally respect the resync offset
during continuous playback at all (it had silently ignored the offset
even before this bug).

Partway through, Isaac established a standing rule that reshaped the rest
of the phase (now in `AGENTS.md`'s Testing section): automated tests
passing, or the agent verifying a fix with its own script, is not the
same as Isaac actually confirming it himself. Every observable change from
that point on got an explicit numbered manual walkthrough before any
commit — and those walkthroughs immediately surfaced real problems in
work already believed done. The calibration-folder guidance needed a
second pass: a file-open-dialog trick alone didn't tell the user what to
expect, so the dialog's title kept the full expected-filename list too.
The project file grew to also save the app version that created it and
the rectified-view toggle state; then, once Isaac asked "shouldn't the
log be saved too?", to save the full measurement Log and a snapshot of
the last-recorded frame/points, so reopening a project jumps straight
back to exactly where you left off with that measurement visibly back on
screen, not just a historical number in a table.

The measurement-output overhaul itself grew well past its original
formatting scope once its actual goal came up: "place multiple segments
in one frame, each measured, then the connected segments summed up."
`main.py`'s per-pane point cap, hardcoded to 2 since early on, turned out
to be the entire reason multi-segment chains had never worked, even
though the segment math already generalized to any chain length. Raising
it to 20, adding a quadrature-summed chain "Total" row, and merging
points/segments/total into one flat, spreadsheet-ready table landed
together with an explicit "Record" action — followed immediately by a
"Measurement ID" column and a fully editable (not read-only) Log, once
Isaac asked how he'd fix or remove a bad recorded measurement. A smaller
but real usability fix rode along: the automated stereo-mate guess placed
on the opposite pane's first click was, in practice, more often wrong
than helpful, so the default became placing the mate at the exact same
image pixel instead, leaving the smarter scanline match available only
via the existing explicit right-click "refine" gesture.

The most substantial bug of the phase surfaced during that same
manual-testing loop: Isaac reported the left pane's zoom not tracking
points correctly while the right pane's did. Investigation — numeric
checks against the real running app, then literal screenshots via
`PIL.ImageGrab` to visually confirm — found this wasn't a left/right
asymmetry at all (every zoom/pan code path was already correctly
symmetric) but a real bug reproducible on *either* pane specifically when
zoomed out far enough to hit the source image's edges, reachable at
`zoom_min` itself with any leftover pan offset, not some rare extreme
case. A clamped, shrunk crop was being stretched to fill the full display
rect regardless of the clamping, silently scaling the displayed video
differently from the point overlays' own transform. The fix maps the
actual clamped crop back through the identical transform the point
overlays use, so the two can no longer diverge — confirmed both
mathematically and visually.

Phase 7 closed with every quiz-derived item done except the
playback-speed fix itself, deliberately deferred: profiling ruled out the
suspected cause (live rectification, costing only ~1-2ms/tick) and found
the real cost is generic Tkinter canvas-paint overhead (~50ms/tick
against a 40ms budget) — a real finding, but fixing it is separate work
not squeezed into this pass. Two new phases were scoped immediately after
closing this one: Phase 8, a second usability round specified directly by
Isaac (a video-time-sync feature anchoring a burned-in on-screen clock to
the video's own frame timeline, plus a precise center-dot on each point
handle), and Phase 9, packaging Sizeamatic Pro as a standalone,
offline-capable distributable.

Phase 8 (issue #9) opened with the three items scoped at the end of Phase
7. Point visibility landed first and smallest: a fixed 2px solid red dot
drawn exactly at each point's center, on top of the existing hollow
handle ring, so the precise clicked/dragged pixel is visible rather than
just the ring's general vicinity. Video-time sync — anchoring a
real-world date/time to whatever frame currently shows a burned-in
on-screen camera clock, so the app can calculate real time at any other
frame — took three full design iterations, each rejected sharply and
specifically by Isaac: a single free-text date string ("I don't want to
type in a string"), then six `ttk.Spinbox`es pre-filled with "now" that
auto-applied on every change ("I don't want them to be spinners! and i
don't want them to be populated with todays time!" — the eager
FocusOut-triggered validation against leftover pre-filled values was
also throwing spurious "not a valid date/time" errors), before landing on
six empty plain `ttk.Entry` boxes with auto-advance-on-typing and a
single explicit "Set Time Sync" button as the only validation trigger.
Once that stuck, Isaac asked for the readout relocated below the scrub
bars, a green checkmark synced indicator instead of trying to recolor
the button itself (unreliable under Windows ttk theming), the six boxes
to keep live-tracking the calculated time during playback rather than
staying frozen at the anchor, and — after seeing ".840" on a timestamp —
the sub-second display changed from a fractional-seconds decimal to an
`HH:MM:SS:FF` frame-in-second count, "the actual frame number in this
particular second, not a percentage."

The third originally-scoped item, a rectified/not-rectified indicator,
followed the same request-then-refine pattern via two quick design
questions rather than a rejected first attempt: a bold toolbar label
reading "NOT RECTIFIED" in red or "RECTIFIED" in green (placed in the
toolbar rather than per-pane), wired into the existing
`_refresh_status_left` call sites so every place that already reacted to
view/calibration changes updated it for free, plus an orange overlay
color for the point ring/connecting line/index label while unrectified —
deliberately leaving the small red center-dot from the point-visibility
item unchanged either way, since it marks the exact clicked pixel rather
than measurement validity. With all three quiz-derived items done and
confirmed, Isaac added two more directly: a File > Recent Projects
submenu backed by a new `recent_projects.py` module, storing the last
five saved/opened project paths under `%APPDATA%\SizeamaticPro\` (chosen
specifically so it survives the app folder being replaced/updated once
Phase 9 packages this as a standalone .exe) and rebuilding itself fresh
on every open so a moved/deleted project quietly drops off the list
instead of showing a dead entry; and project-name-aware window titles
("Sizeamatic Pro - <project name>") applied to all three of the app's
windows (main, Measurement, Calibration Summary), deliberately replacing
those two sub-windows' fixed "Measurement"/"Calibration Summary" labels
rather than appending to them, per Isaac's explicit design choice between
two offered options.

Shortly after Recent Projects shipped, Isaac reported "my recent
projects just disappeared when i did a new build" — which turned out to
mean the list still existed but had been replaced by unfamiliar entries,
"the demo projects that were there before." Investigation found a real,
already-shipped bug: three pre-existing tests that called
`on_save_project`/`on_open_project` for unrelated reasons (from before
Recent Projects existed) had no reason to know a new call inside those
methods would now read and write the actual per-user
`%APPDATA%\SizeamaticPro\recent_projects.json`, so every test run
silently overwrote Isaac's real list with pytest's own `tmp_path`
entries — confirmed by reading the real file directly and finding it
full of `AppData\Local\Temp\pytest-of-isaac\...` paths. Rather than
patching just those three tests, the fix was an `autouse=True` pytest
fixture (`_isolate_recent_projects_file` in `conftest.py`) that redirects
every test's copy of that path automatically, closing the whole bug
class rather than the one instance of it — verified by temporarily
disabling the fixture and confirming a new regression test failed
exactly as expected before re-enabling it, then deleting Isaac's
by-then-pytest-polluted real file so it starts clean. Documented as
`FINDINGS.md` #12. Phase 8 closed with every item — the original three
plus the two added mid-phase — done and confirmed by Isaac.
