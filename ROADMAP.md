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

## Phase 7 — Usability `[x]`

Focus on the actual user (analyst) experience of the tool (issue #8). Started
with a structured usability interview with the project owner (the primary
analyst stakeholder) rather than guessing at friction points; other analysts'
input may come in a later round. Executed incrementally, one item at a time,
with tests added alongside each change — same rhythm as Phase 5, and manual
proof-test walkthroughs with the project owner before each commit (a
standing rule from partway through this phase — see `AGENTS.md`'s Testing
section).

**Scope note:** closes with every quiz-derived item done except the
playback-speed paint-path optimization, deliberately deferred (see that
item below) rather than squeezed in — the same kind of explicit,
not-silently-dropped deferral Phase 5 closed with for `main.py` splitting.
A second usability round (Phase 8) and a packaging/distribution phase
(Phase 9) are planned next. (The deferred playback-speed item was later
picked back up and fixed post-Qt-migration — see its checklist entry
below.)

- [x] Usability interview — see below for the resulting requirements
- [x] Calibration folder guidance — first pass just renamed the
      `filedialog.askdirectory` dialog's title to list the four expected
      NPZ files, but manual testing caught that this doesn't actually
      help: Windows' native folder picker only shows folder names, never
      the files inside them, so there's no way to see which candidate
      folder actually has the right files before picking. Switched to
      `filedialog.askopenfilename` (filtered to `calibration_*.npz`) and
      take the containing folder from whichever file gets picked — the
      file list itself now shows the expected files directly, *and* the
      dialog's title still spells out all four expected filenames (manual
      testing again: seeing the filter's matching files isn't the same as
      being told what the full expected set is). Also `calibration_io.py`'s
      missing-files error message restates the full requirement rather
      than just naming what's absent.
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
      left/right video paths, calibration folder path, the resync
      offset above, the rectified-view toggle state, and the app version
      that created the file (`main.py`'s `_get_app_version`, read straight
      from `pyproject.toml` — informational only, no compatibility check).
      New File menu items "Save Project…"/"Open Project…" so a session
      doesn't require reloading everything from scratch.
      `on_load_left_video`/`on_load_right_video`/`on_load_calibration_folder`
      were each split into a dialog-only wrapper plus a dialog-free
      `_load_*_from_path`/`_load_calibration_from_folder` helper, so
      `on_open_project` reuses the exact same loading/error-handling logic
      a manual reload would use, per stage. Restoring the rectified-view
      toggle goes through the real `on_toggle_view_rectified` handler
      (not just the `BooleanVar`), so its existing resolution-mismatch
      validation still applies if the saved calibration no longer matches.
      Manual testing surfaced a bigger gap than "reselect files": opening a
      project should put you back where you actually left off, not just
      which files were loaded. Added `measurement_log_text` (the
      Measurement window's Log, saved verbatim as its exact text — there's
      no separate structured store to save instead, see the measurement
      output item below) and `last_recorded_snapshot` (which frame each
      timeline was on, and the exact clicked points, at the moment the
      *last* "Record" click happened — `main.py`'s `_on_measurement_recorded`,
      called from `record_current_measurement`). `on_open_project` now
      restores the Log, then jumps both timelines back to that last-recorded
      frame and re-places those exact points, so the measurement is visibly
      back on screen, not just a historical number in the Log.
- [x] Measurement output overhaul — turned out to need more than a
      formatting change once the actual goal ("place multiple segments in
      one frame, each measured, then the connected segments summed up")
      came up: `self.max_points_per_pane` was hardcoded to 2, making a
      multi-segment chain impossible to place at all even though the
      segment math already generalized to any chain length. Raised the
      cap to 20 (a generous fixed ceiling, not a precisely-reasoned
      limit), added a chain "Total" row (sum of connected segment
      lengths, with a quadrature-summed sigma —
      `sqrt(sum(sigma_i**2))` — across the chain's independently-estimated
      segment sigmas), and merged points/segments/total into one flat
      `Type`-tagged table (`measurement_window.py`'s `RESULT_COLUMNS`)
      with leading Video/Frame/Timestamp columns on every row, so a
      recorded session pastes into a spreadsheet as one continuous,
      filterable block. Added an explicit "Record" button + accumulating
      Log (not auto-logged on every recalculation, so it doesn't fill
      with in-progress drag states) alongside the existing "current
      measurement" copy box. Manual testing surfaced a real gap: there
      was no way to fix or remove a bad recorded measurement. Added a
      "Measurement ID" column, shared across every row from one Record
      click (so a whole bad chain can be found together), and made the
      Log a plain always-editable `tk.Text` rather than disabled/
      read-only — editing the text directly *is* the fix/delete
      mechanism, by request, not a placeholder for a future one.
- [x] Stereo mate point placement — not originally in the quiz, but
      caught during manual testing of the above: clicking a point on one
      pane used to ask `stereo_matching.guess_mate_point_on_scanline` to
      guess where the matching point should go on the opposite pane. In
      practice this guess was more often unhelpful than not. Changed the
      initial placement to just reuse the exact same image pixel
      coordinates on the opposite pane instead — each pane's own draw
      pipeline already applies that pane's current zoom/pan independently,
      so this lands correctly on-screen regardless of the two panes'
      current view state, no extra transform needed. The scanline matcher
      is still there and still used by the existing right-click-drag
      "refine" gesture, for whoever wants that assist explicitly.
- [x] Zoom/pan crop-scale bug — reported as "left pane zoom doesn't line
      up, right pane does"; turned out to not be a left/right asymmetry
      at all (every zoom/pan code path is already correctly symmetric) but
      a real bug reproducible on *either* pane when zoomed out far enough
      to hit the image-bounds clamp (reachable at `zoom_min` itself with
      any leftover pan offset). `_display_bgr_on_canvas` was stretching a
      clamped (shrunk) crop to fill the full display rect regardless,
      scaling the displayed video differently from the un-clamped scale
      point overlays use — fixed by mapping the actual clamped crop bounds
      back through the same screen transform `_image_to_screen` uses, so
      video content and point overlays can no longer diverge. See
      `FINDINGS.md` #11.
- [x] Playback speed — fixed under Qt (post-migration). Two real bugs,
      not one: (1) "1x" was a hardcoded 40ms tick derived from an
      assumed ~25fps rather than the loaded video's actual fps, wrong
      for any other real fps — `_compute_playback_timing` now derives
      the tick delay from `metaL`/`metaR`'s real fps. (2) 2x/4x
      playback was reported as playing *slower* than 1x: `_read_frame_at`'s
      seek-skip optimization only covered "same index" and "exactly one
      frame ahead," so 2x/4x's small forward skips (+2/+4 frames every
      tick) fell through to a full `cap.set()` seek on *every* tick —
      profiled at ~54-105ms/frame vs. ~3-10ms/frame to decode-and-discard
      the same gap, making faster-than-1x speeds slower in wall-clock
      terms despite needing fewer ticks. Fixed by decoding-and-discarding
      small forward gaps (`MAX_SEQUENTIAL_SKIP_FRAMES`) instead of
      seeking. Also switched `playback_timer` to `Qt.TimerType.PreciseTimer`
      (Qt's default is an intentionally-imprecise "coarse" timer). Verified
      via real profiling through the actual `app.exec()` loop, not just
      unit tests: 1x/2x/4x now scale correctly relative to each other
      (previously 2x was slower than 1x); some gap from 100% of each
      speed's target fps remains at higher multipliers (decoding-and-
      discarding several frames per tick on both panes is itself real
      work), which is a normal software-decode ceiling rather than a bug.
- [x] Anaglyph 3D preview — confirmed novelty-only, not a usability issue.
      No action; deprioritized for future investment.
- [x] Sort the above into automated-test-covered vs. manual-judgment-only,
      and extend `tests/` accordingly as each item lands — done
      incrementally alongside each item rather than as a separate pass:
      `tests/test_project_io.py` (round-trip save/load), a chain-placement
      test in `tests/test_video_overlay.py`, `tests/test_main.py`'s
      resync/pan/playback-direction/measurement-chain tests, and
      `tests/test_measurement_window.py` (results table + Record log).
      The playback-perf investigation itself isn't test-covered — it's a
      one-off profiling finding, not a behavior to regress-test.
- [x] Update `ARCHITECTURE.md`/`FINDINGS.md` to reflect any new modules
      (`project_io.py`) or structural changes — done; also caught and
      fixed a pre-existing gap while at it: `calibration_io.py` had never
      been added to `mkdocs.yml`'s nav back in Phase 5, so it was never
      actually documented on the site despite being pure, tested code.

## Phase 8 — Usability, round 2 (issue #9) `[x]`

A second round of usability items — this time specified directly by the
project owner rather than via a fresh quiz (see Phase 7 for that pattern).
Recent Projects and the window-title project name were both added
mid-phase, after the first three items were already done and confirmed —
not part of the original plan, but small enough to fold in rather than
spin off their own phase.

- [x] Point visibility — `video_overlay.py`'s `draw_pane` now draws a
      small solid red dot (`CENTER_DOT_RADIUS_PX`, 2px screen radius,
      fixed regardless of zoom — matching the existing handle ring's own
      fixed-screen-size behavior) exactly at each point's center, on top
      of the existing hollow green ring. Purely visual — not tagged
      `"handle"`, so it doesn't change click hit-testing.
- [x] Video-time sync — let the user set the real-world timestamp shown
      burned into the video image itself (e.g. a camera's on-screen clock
      overlay) at whatever frame they're currently on. One shared anchor
      (`self.real_time_anchor_frame`/`self.real_time_anchor_dt`)
      referenced to the left/master timeline — not per-pane, consistent
      with measurements already treating left as the reference. Entered
      via six plain `ttk.Entry` boxes (year/month/day/hour/minute/second,
      each with a label underneath) near "Clear Points" — went through
      two rejected designs first (a single free-text date string, then
      pre-filled `ttk.Spinbox`es that auto-applied on every change) before
      landing here: the project owner wanted empty boxes you type into,
      with auto-advance-to-next-box on `<KeyRelease>`
      (`_advance_real_time_focus`) and validation happening only once,
      on an explicit "Set Time Sync" button (`on_real_time_entered`) —
      the Spinbox version's eager FocusOut validation was firing against
      leftover pre-filled values and throwing spurious "not a valid
      date/time" errors. A green "✓ Synced" label
      (`_show_time_sync_indicator`) appears next to the button once set
      (a ttk button's own background color isn't reliably themeable on
      Windows, hence the separate label). The six boxes keep live-tracking
      the calculated time as playback moves (`_refresh_real_time_entries`,
      called from `_update_frame_labels`), not just showing the anchor.
      A shared readout row spanning both panes, directly below the scrub
      bars (`self.time_readout_label`, grid row 3 — moved out from beside
      the Set Time Sync button per the project owner's request) shows
      Frame i/max, Video Time, and calculated Actual Time together rather
      than duplicated per pane. Both `_format_timestamp` and
      `_format_actual_time` render the sub-second part as an
      `HH:MM:SS:FF` frame-in-second count (`divmod(frame_index, fps)`),
      not a fractional-seconds decimal — the project owner wanted "the
      actual frame number in this particular second, not a percentage."
      `_format_actual_time` projects the anchor forward/backward using
      the left video's fps: actual time = anchor time + whole seconds of
      (frame − anchor frame) / fps, with the remaining frames as the
      `:FF` suffix. The anchor is saved in the project file
      (`project_io.py`'s `real_time_anchor_frame`/`real_time_anchor_iso`)
      and restored on reopen — including refreshing the six entry boxes
      to match — before `_update_frame_labels` runs so the readout
      reflects it immediately, same pattern as the Phase 7 resync offset.
      Ran into the FINDINGS.md #6 attribute-ordering pitfall twice
      (`_update_frame_labels` already runs once during `__init__` itself,
      before the app's later, scattered-init state is set) — first for
      the entry StringVars, then again for the anchor frame/datetime
      attributes themselves. Recorded measurements now also carry an
      `actual_time` column (empty until an anchor is set), added to
      `measurement_window.py`'s `RESULT_COLUMNS`.
- [x] Rectified/not-rectified indicator — measurements and clicked
      points are only real-world-accurate in rectified view, so raw view
      needed to look visibly different. A bold toolbar label
      (`self.rectified_indicator`, next to Clear Points) reads
      "NOT RECTIFIED" in red or "RECTIFIED" in green, kept in sync by
      `_refresh_rectified_indicator` — called from `_refresh_status_left`,
      so every existing call site (toggling the view, loading/failing
      calibration, opening a project) updates it for free without a new
      call site of its own. `video_overlay.py`'s `draw_pane` picks between
      `RECTIFIED_OVERLAY_COLOR` (green, the original color) and
      `NOT_RECTIFIED_OVERLAY_COLOR` (orange) for the ring, connecting
      line, and index label based on `app.view_rectified.get()`; the
      small red center dot from the point-visibility item above
      deliberately stays red in both modes, since it marks the exact
      clicked pixel — a concern unrelated to rectification state.
- [x] Recent Projects — a File > Recent Projects submenu lists the last
      `MAX_RECENT_PROJECTS` (5) project files saved or opened, so
      reopening one doesn't need a file dialog every time. Backed by a
      new small module, `recent_projects.py`, deliberately separate from
      `project_io.py` (which reads/writes one project's own content) —
      this one just persists a tiny cross-session list of *paths to*
      project files. Stored under `%APPDATA%\SizeamaticPro\
      recent_projects.json`, not next to the app itself, so it survives
      the app folder being replaced/updated and works the same whether
      running from source or (once Phase 9 packages it) a standalone
      .exe. The submenu is rebuilt fresh every time it's about to open
      (`self.recent_projects_menu`'s `postcommand`, wired to
      `_refresh_recent_projects_menu`) rather than once at startup, so a
      project that's since been moved/renamed/deleted just quietly
      drops off the list (`load_recent_projects` filters via
      `os.path.isfile`) instead of showing an entry that would only
      error if clicked. `on_save_project`/`on_open_project` both call
      `add_recent_project` on success; `on_open_project`'s actual load/
      restore logic was pulled out into `_open_project_from_path` so the
      submenu's click handler (`on_open_recent_project`) goes through
      identical logic, just skipping the file dialog. Shortly after this
      shipped, running the test suite was found to be silently
      clobbering the real per-user recent-projects file — see
      `FINDINGS.md` #12 for the bug and the autouse test-isolation fix.
- [x] Window titles show the loaded project name — every window's title
      bar (main window, Measurement, Calibration Summary) reads
      "Sizeamatic Pro" normally, or "Sizeamatic Pro - <project name>"
      once a project has been saved or opened this session
      (`self.current_project_name`, the file's base name without its
      directory or ".json" extension), so it's obvious at a glance
      which project a given window belongs to. `main.py`'s
      `_app_window_title` builds the string; `_refresh_window_titles`
      (called from `on_save_project`/`_open_project_from_path`) applies
      it to the main window plus the Measurement/Calibration Summary
      windows if they're already open. A window opened for the first
      time *after* a project is already loaded picks up the right title
      immediately too, since each window's own `ensure_window` reads
      `self.app._app_window_title()` directly at creation time rather
      than hardcoding "Measurement"/"Calibration Summary" — the
      project-name suffix intentionally replaces those fixed labels
      rather than appending to them, per the project owner's request.

## Phase 9 — Packaging and distribution (issue #10) `[x]`

Package Sizeamatic Pro as a standalone, distributable build that bundles
all its requirements and runs fully offline — so an analyst can get and
run it without setting up Python/`uv` themselves. Planned out with the
project owner before implementation, rather than scoped from a guess.

**Scope note:** closes with the onefile Windows `.exe` build done, run,
and confirmed working, plus everything else that ended up bundled in
along the way (real icon/splash art, embedded version metadata, a VS
Code build task). The installer and Linux packaging are deliberately
deferred rather than silently dropped — the project owner chose to skip
Linux for this pass rather than stand up a Linux build environment
(PyInstaller can't cross-compile; a Linux binary has to actually be
built on Linux), and the installer was always planned as a later
addition once the simpler onefile build was proven. Both can resume
inside this same phase whenever picked back up, without reopening it as
a new one.

- [x] Single-file `.exe` (PyInstaller `--onefile`, `sizeamatic.spec`) —
      built, run, and confirmed working by the project owner. A proper
      installer (likely Inno Setup) is still planned as a later addition
      within this same phase, not started yet. Tradeoffs the project
      owner was walked through and accepted for the onefile build: no
      install step and easy to hand someone directly, at the cost of
      slower startup (self-extracts to a temp folder every launch) and a
      higher chance of antivirus/SmartScreen false-positive flags, both
      common for unsigned onefile PyInstaller builds.
- [x] App icon — real branded art (`assets/default-icon.png`, a fish
      logo), not a placeholder. `sizeamatic.spec` regenerates
      `assets/icon.ico` from it on every build via `create_app_icon.py`
      (Windows needs one `.ico` containing several baked-in sizes —
      16/32/48/256 — not separate files per size), so dropping in future
      revised art is just replacing that one PNG and rebuilding, no code
      changes. Also embedded as the `.exe` file's own Explorer icon
      (`EXE(icon=...)`) — confirmed correct by extracting the icon
      directly from the built `.exe` (bypassing Windows' icon cache,
      which otherwise shows a stale thumbnail for a path rebuilt
      in-place — a real red herring hit during this work, not a packaging
      bug).
- [x] Splash screen shown during startup — real branded art
      (`assets/splash-pro.png`), with the version number drawn onto it
      at runtime. Ended up as `main.py`'s own Tkinter `Toplevel`
      (`_show_startup_splash`), not PyInstaller's bootloader `--splash`
      feature: an earlier version used both (bootloader splash during
      onefile self-extraction, handed off to this app's own splash once
      Tk started), which in practice showed as two near-identical
      splashes back to back — removed the bootloader `--splash` entirely
      rather than tuning the handoff, since the app's own splash already
      covers the whole startup window on its own, identically whether
      run from source or from the `.exe` (the project owner specifically
      wanted it to still show from `uv run python main.py`, which a
      PyInstaller-only splash never could). Stays up at least 4 seconds
      even on a fast machine, and scales down large source art rather
      than showing it at full native resolution. Two more real bugs
      found and fixed along the way: the splash's alpha-transparent glow
      effect rendered as corrupted magenta/pink under naive Tcl/Tk image
      loading (fixed by flattening onto solid black first,
      `prepare_splash_image.py`), and an `overrideredirect`+topmost
      Tkinter window rendered solid black on Windows until made
      borderless *after* it already had content to paint, not before.
- [x] The `.exe`'s own Windows version resource (Explorer's file
      Properties dialog) — added mid-phase, after the project owner
      asked for the version to show "on" the exe itself, not scoped
      originally. `build_version_info.py` generates the version-resource
      file `EXE(version=...)` consumes, from `pyproject.toml`'s version.
      The same version now also shows in the startup splash and every
      window's title bar (`_app_window_title`, e.g. "Sizeamatic Pro
      v0.1.0") — one source of truth in all three places.
- [ ] Linux packaging — not attempted or tested this phase; Windows was
      the actual priority. `sizeamatic.spec` isn't deliberately
      Windows-only, but nothing about it has been verified cross-platform
      either.
- [x] Local build script/command only (`uv run pyinstaller
      sizeamatic.spec`, or VS Code's "Build Sizeamatic Pro .exe
      (PyInstaller)" launch config) — no CI/GitHub Actions automation
      this phase; the project owner runs the build locally and
      distributes the result themselves.

Not yet decided: installer tooling (beyond "likely Inno Setup"), and a
later build option choosing between `assets/splash.png` and
`assets/splash-pro.png` (gating some functionality by which) — not
implemented yet. Note: `origin/feature-onlineCalibrations`, an existing
unmerged remote branch, looks like it may be prior exploration relevant
to Phase 10 (below) rather than this phase — worth checking before
Phase 10 starts, not necessarily before this one.

## Phase 10 — In-app stereo calibration workflow `[x]`

Bring the stereo camera calibration step itself into Sizeamatic Pro,
rather than treating a calibration NPZ as something produced entirely
outside the app and only ever loaded (`calibration_io.py`) or QA'd after
the fact (`generate_calibration_report.py`). Today, nothing in this repo
actually runs `cv2.calibrateCamera`/`cv2.stereoCalibrate` — the four
calibration NPZ files are produced by some external process from
checkerboard/ChArUco stills, then handed to the app as a finished folder.
Planned out with the project owner before implementation, in explicit
numbered steps to be done one after another rather than all at once:

**Scope note:** closes with Steps 0-2 done and confirmed working by the
project owner — frame-pair capture, running calibration, and fully
configurable checkerboard/ChArUco board generation (both detection
*and* printable targets, generated in-app rather than via a standalone
script). Step 3 (OAK-D 3D camera compatibility) is deliberately
cancelled rather than completed - the project owner decided not to
pursue OAK-D-specific integration for now. None of Step 3's
investigation was started, so picking OAK-D support back up later would
need its own fresh planning pass rather than resuming this checklist
item as-is.

- [x] **Step 0 — Reference the original zcam calibration code.** Before
      this project existed, camera calibration was done in a separate
      zcam project: a Node.js process loading Python and sending it
      calibration requests, before the decision was made to do
      calibration here instead. Resolved: the unmerged
      `origin/feature-onlineCalibrations` branch turned out to be that
      same ported-from-zcam calibration code, and was used as reference
      material throughout Steps 1-2 (see below) rather than a separate
      zcam source ever being located.
- [x] **Step 1 — Capture calibration frame pairs from live video.** A new
      top-level "Calibration" menu (`Load Calibration…` / `Perform
      Calibration…` / `Calibration Report…`, matching a UI precedent
      already found on the unmerged `origin/feature-onlineCalibrations`
      branch — see below) starts calibration using whatever left/right
      video pair is *already loaded* in the main window, rather than a
      separate calibration-only video load. Scrubbing uses the existing
      Lock/sync controls; a capture action saves both panes' current
      frame at once as a `left_<id>`/`right_<id>` pair (the same
      pairing/naming convention already implemented and working on that
      branch's `_find_stereo_calibration_pairs`, chosen specifically so
      the existing detection/calibration code can be adapted with
      minimal changes rather than redesigned).
- [x] **Step 2 — Create a new calibration.** A calibration is just a
      folder containing the four calibration NPZ files
      `calibration_io.py` already knows how to load — this step is
      about producing one of those folders from captured frame pairs
      (running detection + `cv2.calibrateCamera`/`stereoCalibrate`/
      `stereoRectify`, matching the math already implemented on the
      `feature-onlineCalibrations` branch, whose NPZ output already uses
      this app's exact current key names), likely alongside the
      captured calibration images themselves so a run can be reviewed or
      redone later. Detection supports both checkerboard and ChArUco
      boards (also matching that branch), with fully user-configurable
      board dimensions for both (columns/rows, square size, marker
      size) rather than a single fixed size for either. A new Generate
      Calibration Target window (`generate_calibration_target.py`) prints
      either board type from inside the app, with a live preview and the
      actual board settings printed on the page as small text.
- [ ] **Step 3 — Investigate OAK-D 3D camera compatibility.** Cancelled -
      the project owner decided not to pursue OAK-D-specific integration
      for now (see this phase's scope note). The project owner supplied
      `assets/oakd_camera_info.json`, the camera's own factory
      calibration data. Three angles were identified but never
      investigated:
      1. Whether the OAK-D is already effectively pre-calibrated for
         this app's purposes — the JSON has per-camera intrinsics,
         extrinsics/baseline between sockets, and even precomputed
         stereo rectification rotations, but its distortion coefficients
         are a 14-value array in DepthAI's own extended model, not
         OpenCV's standard Brown-Conrady one, so it likely can't be used
         as-is without conversion.
      2. Whether the `depthai` Python SDK can do that conversion (or more)
         directly — it ships a `CalibrationHandler` API built for
         exactly this kind of calibration-data access, worth using
         directly rather than hand-parsing the JSON.
      3. How to support the OAK-D's three camera views (a center color
         autofocus camera plus a left/right mono stereo pair — matching
         `oakd_camera_info.json`'s `camera_features`) and its `.mcap`
         recording format, which is a general-purpose multiplexed
         sensor-log container (Foxglove/ROS-style), not a plain video
         file — loading one will likely need a demuxing step before this
         app's existing `cv2.VideoCapture`-based pipeline can use it at
         all.

`origin/feature-onlineCalibrations`, an existing unmerged remote branch
flagged during Phase 9 planning, turned out to have substantial prior
work directly relevant to Steps 1-2: a 3290-line `perform_calibration.py`
with working checkerboard/ChArUco detection and the full stereo
calibration/rectification math, producing NPZ files in the exact format
`calibration_io.py` already expects. The branch itself is too stale to
merge (it predates almost this project's entire current structure —
`AGENTS.md`, `ROADMAP.md`, the docs site, the current test suite, all of
it) — treated as reference material to adapt algorithms and conventions
from, not something to merge wholesale.

## Phase 11 — Look and feel polish `[x]`

A dedicated pass on making the app's look and feel as cohesive and
professional as possible, rather than squeezing visual polish into
whatever feature happened to touch a given screen. Mid-phase, the UI
framework itself was switched from Tkinter to PySide6/Qt (Tkinter's
native menu bar can't be dark-themed on Windows) — `main.py` and
`video_overlay.py` were ported first; `measurement_window.py`,
`calibration_summary.py`, `perform_calibration.py`, and
`generate_calibration_target.py` are now ported too, along with
restoring the measurement pipeline/status bar/real-time sync that had
been dropped mid-port, and wiring `anaglyph_preview.py` back in. The
whole test suite (previously left with `--ignore` flags on four
Tkinter-era files) has also been fully rewritten for Qt — see
`AGENTS.md`'s Testing section.

**Scope note:** closes with the Qt migration and test-suite rewrite
done and confirmed working — the general spacing/layout consistency
pass below is deliberately deferred rather than done now, the same
kind of explicit, not-silently-dropped deferral prior phases have
closed with (see Phase 5's `main.py`-splitting note, Phase 7's
playback-speed fix). It needs more hands-on time with the ported app
to even know what still looks off, which hasn't happened yet; revisit
as its own pass whenever that time exists, rather than guessing at
spacing issues sight-unseen.

- [x] Port `main.py`/`video_overlay.py` to PySide6/Qt, dark title bar,
      multi-monitor-correct window placement, `qt_helpers.py` shared
      infrastructure (`ClosableDialog`, `pil_image_to_qpixmap`,
      `move_to_same_screen_as`)
- [x] Port `measurement_window.py`, `calibration_summary.py`,
      `perform_calibration.py`, `generate_calibration_target.py` to
      `QDialog`-based windows; restore the measurement pipeline/3-part
      status bar/real-time sync in `main.py`; wire `anaglyph_preview.py`
      back in via `QTimer`
- [x] Toolbar iconography — the transport controls and Clear Points
      button use real Font Awesome icons via the `qtawesome` package
      (`qta.icon("fa5s.play", color=ICON_COLOR)`, etc.) instead of plain
      Unicode glyphs, which rendered inconsistently (font-dependent,
      fell back to emoji/missing-glyph boxes for a couple of the
      transport symbols specifically). Play/pause now also swaps icon
      to reflect actual playback state. Any *other* icons this app adds
      later (menu items, additional dialogs) should reuse the same
      `qta.icon(name, color=ICON_COLOR)` pattern rather than introducing
      a second source.
- [x] Rewrite the whole test suite for Qt — `test_main.py`,
      `test_video_overlay.py`, `test_rendering_performance.py`,
      `test_smoke.py`/`smoke_test.py`, plus the four ported sub-window
      modules' own test files, plus a handful of Qt-specific regression
      tests (`test_regressions.py`) for bugs found along the way
      (dark-title-bar/multi-monitor placement, measurement-window
      focus/positioning, anaglyph-preview resizing the main window)
- [ ] General spacing/layout consistency pass beyond what's already been
      fixed (menu/toolbar padding, font size, video pane borders,
      default window sizing) — not yet scoped in detail; revisit once
      there's been more hands-on time with the ported app to see what
      still looks off.

## Phase 12 — Object-space stereo ray residual (`StereoRayResidual(mm)`) `[x]`

Add an EventMeasure-comparable object-space diagnostic alongside the existing
pixel-space `ReprojRMS(px)`, so match/calibration quality can be judged in
physical units too. Motivated by a comparison of Sizeamatic Pro's stereo
measurement math against SeaGIS EventMeasure/CAL: EventMeasure's published
"RMS" is the shortest distance between the two original observation rays in
object-space millimeters, a genuinely different quantity from Sizeamatic's
pixel-space reprojection RMS, even though both are residual/quality metrics.
Purely additive — doesn't change triangulation or any existing output.

- [x] Derive each pane's camera center and ray direction directly from the
      calibration's existing rectified `PL`/`PR` (no new calibration fields)
      — `stereo_matching.py`'s `camera_center_and_ray_direction`. First
      implementation used P's homogeneous null space (SVD) for the camera
      center and a pseudoinverse solution for a ray point; caught by its
      own test before merge that this degenerates to a point at infinity
      whenever a camera sits exactly at the world origin — the normal
      case for a rectified left/reference camera, including this
      project's own synthetic test rig. Replaced with the standard
      `M`/`p4` split of `P = [M | p4]` (`center = -M^-1 @ p4`, `direction
      = M^-1 @ [x, y, 1]`), which has no such degeneracy for any real
      finite camera.
- [x] Implement closest-approach-distance-between-two-skew-rays as a new
      `stereo_matching.py` function (`ray_residual_mm`, plus an
      app-state wrapper `stereo_ray_residual_mm` matching
      `reprojection_rms_px`'s validation pattern)
- [x] Unit tests against `synthetic_cal`/`known_point_pixels` with a
      hand-computed expected residual (7 new tests in
      `tests/test_stereo_matching.py`)
- [x] Surface the new metric in the Measurement window's Log/results table
      next to `ReprojRMS(px)`, clearly labeled in mm and visually distinct
      from it — new `ray_residual_mm` column in `measurement_window.py`'s
      `RESULT_COLUMNS`/`RESULT_HEADERS`, populated in `main.py`'s
      `_update_measurement_status_stub`
- [x] Update `ARCHITECTURE.md`
- [x] Manual proof test with project owner — confirmed working

## Phase 13 — Jacobian/covariance uncertainty propagation `[x]`

Add a statistically more formal alternative to the current 8-perturbation
sample-standard-deviation uncertainty estimate: a finite-difference
Jacobian / propagated-variance estimate, reusing the same clicked-point
perturbations already computed — putting `SigmaZ`/`SigmaRange`/segment
`sigma_L` on formal statistical footing instead of a sensitivity indicator.
Also motivated by the EventMeasure/CAL comparison: SeaGIS derives precision
via analytical variance propagation, and while Sizeamatic's numerical
approach can retain its advantage of using the *complete* calibrated
stereo geometry rather than a simplified analytical model, it should
compute a real propagated variance rather than just the spread of ad hoc
perturbations.

- [x] **Decision needed up front:** this changes the numeric values shown
      for every existing measurement (including ones already recorded in
      saved project files/exported logs) — confirm with project owner
      whether that's acceptable as a straight replacement, or whether
      old/new should be shown side-by-side for a transition period —
      **decided: side-by-side.** The old sample-standard-deviation
      estimators stay as-is; new Jacobian estimators are added alongside
      rather than replacing them, so nothing about an existing saved
      measurement's numbers changes.
- [x] Implement Jacobian-based sigma calculation reusing `endpoint_perturbs`
      — `stereo_matching.py`'s `estimate_point_sigma_mm_jacobian`/
      `estimate_segment_sigma_len_mm_jacobian`. Reuses `endpoint_perturbs`'s
      existing four `(+sigma_px, -sigma_px)` coordinate pairs as central-
      difference partial derivatives, combined as an explicit propagated
      variance (`sigma_g^2 = sum_i ((g_plus_i - g_minus_i) / 2) ** 2`)
      rather than the old sample standard deviation of all perturbations
      together.
- [x] ~~Update `estimate_point_sigma_mm`/`estimate_segment_sigma_len_mm`~~ —
      superseded by the side-by-side decision above: left unchanged, new
      `_jacobian`-suffixed functions added instead.
- [x] Add new tests for the Jacobian estimators (9 new tests in
      `tests/test_stereo_matching.py`, including one that independently
      re-derives the propagated-variance formula rather than just
      checking sign/growth), and update existing row-shape tests/fixtures
      for the two new `sigma1_jac`/`sigma2_jac` columns
- [x] Manual proof test comparing old vs. new numbers on a real measurement
      — confirmed working

## Phase 14 — Investigate 3D bundle-adjustment calibration (research spike) `[x]`

Investigate — without committing to implement — a photogrammetric
bundle-adjustment calibration mode using a 3-D calibration target, to see
how much of the SeaGIS CAL calibration-philosophy gap is realistically
closeable. This is the biggest and least software-only item to come out of
the EventMeasure/CAL comparison: CAL uses an internally constrained bundle
adjustment across a whole photogrammetric network (camera parameters,
poses, and 3-D target coordinates solved simultaneously), typically driven
by a 3-D calibration cube rather than a planar checkerboard/ChArUco board.
Sizeamatic's current pipeline (independent per-camera `cv2.calibrateCamera`
→ `cv2.stereoCalibrate` with `CALIB_FIX_INTRINSIC` → `cv2.stereoRectify`) is
a normal, defensible CV stereo calibration approach, but a materially
different (and less rigorous) one. A genuine bundle adjustment needs a
custom per-view initial pose plus a hand-built sparse least-squares solve —
`cv2.calibrateCamera`'s Zhang-method initial guess assumes a planar target
per view, so it can't just be handed 3-D object points as-is.

**Important framing, confirmed with the project owner:** this would be an
**additional, optional** calibration mode alongside the existing
checkerboard/ChArUco workflow (matching `perform_calibration.py`'s
existing board-type-choice pattern) — not a replacement for it. Nothing
here proposes removing or changing today's default calibration path.

- [x] Investigate 3-D target options (ArUco-faced cube vs.
      precisely-surveyed discrete targets) and what's fabricable/measurable
      in-house — project owner confirmed the realistic fabrication path is
      **3D printing + manual measurement** (calipers or similar), not
      professional machining/surveying. Given that, an **ArUco-faced cube**
      is the clear choice over discrete surveyed targets: it reuses this
      app's existing ChArUco/ArUco detection code
      (`perform_calibration.py`'s `build_charuco_detector`/
      `detect_charuco_points`) almost entirely unchanged (just detecting
      each face as its own small ChArUco board, tagged by which cube face
      it is), and per-marker ID-based correspondence removes the need for
      manual point-matching across views. A discrete-surveyed-point target
      (e.g. a frame with a few widely-spaced precisely-known points) would
      need either a different detection method entirely or manual point
      clicking per calibration view, and gains little given 3D-printing
      accuracy is already the tolerance ceiling, not detection accuracy.
- [x] Investigate `scipy` (not currently a dependency) as a new dependency
      vs. a hand-rolled least-squares solve — confirmed `scipy` (1.18.0)
      installs cleanly against this project's actual Python version
      (3.14.0) in an isolated check venv, and `scipy.optimize.least_squares`
      with `method="lm"` solved the prototype below (20 views × 8 points ×
      2 cameras = 320 point observations, ~140 free parameters) without
      needing a custom sparse Jacobian or bounds. **Recommendation: use
      `scipy`, don't hand-roll.** A hand-rolled Levenberg-Marquardt/Gauss-
      Newton solver would just be reimplementing well-tested, actively
      maintained code for no real benefit at this problem size; `scipy` is
      a mainstream, permissively-licensed (BSD) dependency already common
      in the scientific Python ecosystem this project's other dependencies
      (`numpy`, `opencv-python`, `matplotlib`) come from. Added as a
      **dev-only** dependency for now (`uv add --dev scipy`) since it's
      backing an investigation prototype, not shipped app code yet; would
      need promoting to a real runtime dependency if a future phase
      actually builds this into the app.
- [x] Prototype a minimal bundle-adjustment solve against **synthetic**
      data only, to validate the math before investing in physical target
      fabrication — `misc/bundle_adjustment_prototype.py` (not wired into
      the app; synthetic data only, per this phase's scope). Simulates a
      fixed stereo rig observing an 8-corner cube (with small simulated
      3D-printing error vs. its nominal/as-designed geometry) across 20
      random views with realistic pixel noise, then jointly refines the
      stereo extrinsics, every view's pose, and the cube's own point
      geometry via `scipy.optimize.least_squares`.
      **Result: reprojection RMS dropped from 3.5px to ~0.27px and the
      recovered stereo baseline error dropped from ~3.9mm to ~0.2-0.4mm**
      — the core approach works.
      **Important finding along the way, not a bug:** an unconstrained
      "release object" bundle adjustment (letting the solver freely adjust
      every object point's XYZ) has a **gauge/datum ambiguity** — nothing
      in the reprojection residuals alone pins down the object-space
      point cloud's absolute scale/position/orientation jointly with the
      stereo extrinsics, so a naive first attempt's recovered cube
      geometry drifted **55mm** from the true manufactured geometry
      despite near-perfect reprojection error and a well-recovered stereo
      baseline. **Fix:** add a soft "anchor" residual pulling solved
      object points back toward the nominal/as-designed geometry, weighted
      by the expected manufacturing tolerance (`1 / tolerance_mm`) rather
      than treating the nominal geometry as exact. With that anchor added,
      the same run recovered the cube geometry to ~0.24mm RMS (an actual
      improvement over the 0.3mm nominal-vs-true error, not just a
      no-op) while reprojection RMS and baseline recovery stayed just as
      good. This directly validates the earlier EventMeasure/CAL write-up's
      mention of CAL using "network/datum/stereo constraints" — a real
      implementation would need this same kind of explicit constraint,
      not just a naive joint least-squares solve.
- [x] Investigate known-distance scale-bar validation as a standing
      calibration QA step (useful regardless of which calibration mode
      produced the calibration) — this needs no new math or dependency:
      it's the app's *existing* triangulation/length pipeline
      (`stereo_matching.triangulate_from_pixels` + Euclidean distance),
      just pointed at an object of precisely known real-world length
      instead of an animal. **Recommendation:** add this as a lightweight
      calibration-report step (`generate_calibration_report.py`/the
      Calibration Report window) — after any calibration (checkerboard,
      ChArUco, or a future bundle-adjustment mode), let the user click a
      known-length object's two ends in both rectified views and report
      measured-vs-known length as a pass/fail-style QA line. This is cheap,
      useful today with zero dependency on the rest of this phase, and
      isn't gated on the bundle-adjustment work below — worth scoping as
      its own small future phase rather than only-if-bundle-adjustment
      ships.
- [x] Close with a decision: pursue full implementation as its own future
      phase, adopt partial pieces only (e.g. scale-bar validation without
      full bundle adjustment), or defer entirely — **decision: adopt
      partial pieces now, defer the rest.** Known-distance scale-bar
      validation is worth its own near-term phase regardless of what
      happens with bundle adjustment. Full bundle-adjustment calibration
      (as an **additional, optional** mode, never a replacement for
      checkerboard/ChArUco — see framing note above) is real and
      achievable given this phase's findings, but is a substantially
      bigger lift than Phases 12-13: physical cube fabrication +
      measurement, a new in-app calibration-mode UI path (ArUco-cube
      capture, multi-view management), the anchor-weight/tolerance
      tuning this phase's prototype surfaced as necessary, and real
      validation against physically captured footage (not just synthetic
      data) before it could be trusted. Recommend scoping that as its own
      dedicated future phase, planned in detail the way Phase 5/9/10 were,
      once there's appetite to commit the physical-fabrication effort -
      not opened as open-ended work off the back of this research spike.
- [x] No changes to the existing checkerboard/ChArUco calibration path in
      this phase — confirmed: `perform_calibration.py` untouched;
      `misc/bundle_adjustment_prototype.py` is a new, standalone,
      synthetic-data-only file with no import/call path from the app.

## Phase 15 — Tutorial mode `[ ]`

Guided, in-app, game-style tutorial: a tutorial window with a step
checklist, each step showing a description (plus an expandable "more
info" section), with in-app visual guidance (arrows/highlighted or
"waiting for click" glowing controls) pointing at the real UI element to
interact with next. Covers the full operational workflow a video
processor needs before they can start working — loading left/right
video, loading a calibration, toggling rectified view, syncing/locking
the two timelines, placing/recording measurements, saving/loading
projects, and what the Measurement window's columns and row types
(`Point`/`Segment`/`Total`) and quality/error metrics (`ReprojRMS(px)`,
`RayResidual(mm)`, `sigma1`/`sigma2`, `sigma1_jac`/`sigma2_jac`) actually
mean. **This is the next phase to be worked on** (confirmed with the
project owner ahead of Phases 16-18, which are either hardware-blocked
or explicitly unscoped).

**Explicitly built to extend later:** this phase covers only the
*operational workflow* tutorial. A future phase will add a *calculations*
tutorial track (explaining the measurement math/uncertainty itself, not
just how to click through the app) — this phase's architecture must
support more than one track without a rewrite, even though only the
operational track ships now.

**Step 0 — Mandatory clarifying-question session with the project owner.**
The project owner explicitly wants to be interrogated extensively before
any design or implementation work starts, not have decisions assumed on
their behalf — treat this as a hard gate, not a formality, the same way
Phase 5/9/10 got dedicated planning rounds before execution. At minimum,
resolve:

- **Checklist scope.** Confirm/adjust the exact v1 step list. Candidate,
  drawn from what already exists in the app: load left video → load
  right video → load calibration → toggle rectified view → the
  RECTIFIED/NOT RECTIFIED indicator → Lock/resync the two timelines →
  pan/zoom a pane → place a measurement point pair → Disp/dY/ReprojRMS/
  RayResidual → place a second point to form a Segment → the Total row
  and chain-sigma → Record vs. just viewing "current measurement" →
  editing/deleting a bad recorded row in the Log → Save Project → Open
  Project → Recent Projects → the real-time sync anchor → the
  copy-to-clipboard block. What's explicitly out of scope for v1 (e.g.
  Perform Calibration's *creation* flow, anaglyph preview, playback
  speed controls)? Strictly linear (step N unlocks step N+1), or freely
  navigable?
- **The "glowing button" mechanism.** What should highlighting actually
  look like, mechanically, in Qt (an overlay border on the real widget, a
  pulsing animation, an arrow, some combination)? Does advancing to the
  next step require detecting the real action actually happening (hooking
  into the real handler - e.g. `on_save_project`, `record_current_
  measurement`, `on_toggle_view_rectified`), or is a manual "I did this,
  Next" button acceptable for steps that are hard to detect
  programmatically? This needs an explicit decision, not an assumption -
  the two approaches have very different implementation cost.
- **The tutorial "project."** The project owner's own framing: "a
  tutorial mode, maybe even a tutorial project, that the user can load."
  Confirm which of: (a) a small bundled sample stereo video pair +
  calibration folder shipped with the app (raises size/licensing
  questions - real footage is currently gitignored under `examples/`,
  and Phase 9's PyInstaller onefile build already has size/startup
  tradeoffs to weigh against), (b) synthetic/generated video+calibration
  produced at tutorial-start time, or (c) no real video needed at all -
  the tutorial mocks/simulates enough app state to demonstrate each step
  without real footage.
- **Window placement.** Docked panel, floating non-modal window (like
  `MeasurementWindow`/`CalibrationSummaryWindow`), or a full overlay on
  top of the main window? Reachable any time via a menu (e.g. "Help >
  Tutorial"), first-launch-only, or both? Does tutorial progress persist
  across app restarts / get saved in the project file, or does it always
  start fresh?
- **Content architecture.** Confirm a content-authoring approach that
  keeps this phase's operational-workflow tutorial and the future
  calculations-tutorial track as separate modules within one tutorial
  engine, not hardcoded together. Plain Python/JSON step definitions
  (matching this project's existing "no CMS, everything in git"
  convention) are the default assumption - confirm or adjust.
- **Scientific-accuracy sign-off.** Does the explanation text for
  `ReprojRMS`/`RayResidual`/the sigma columns need the project owner's
  review before shipping, given those are easy to describe incorrectly
  (see Phase 12/13's own write-ups for how subtle the distinctions are)?

**Later steps** (deliberately not detailed yet - refine once Step 0's
answers land, the same way Phase 10's steps were refined as work
progressed):

- [ ] Design the tutorial's step data model (step list, per-step
      description/more-info text, target-widget reference,
      completion-detection hook) - keep it decoupled from any specific
      step's content so the future calculations-tutorial track can reuse
      the same engine.
- [ ] Design and build the actual overlay/highlighting mechanism in Qt,
      per Step 0's decision.
- [ ] Build the Tutorial window itself (checklist + description +
      expandable "more info" area), matching this project's existing
      `QDialog`-based window patterns.
- [ ] Build/acquire whatever tutorial "project"/sample data Step 0
      decided on.
- [ ] Wire real completion-detection hooks into the relevant existing
      handlers, wherever Step 0 decided that's needed.
- [ ] Automated tests: `FakeApp`-based tests for the tutorial engine's
      step-tracking logic, plus real-`QApplication` tests for the
      overlay/highlight rendering (following the `qapp`/`sizeamatic_app`
      fixture patterns already in `tests/conftest.py`).
- [ ] Manual proof-test walkthroughs with the project owner, per
      `AGENTS.md`'s standing rule - likely more than once given this
      feature's size, not just once at the end.
- [ ] **Update `CLAUDE.md`** with a new standing instruction: whenever a
      future change alters any workflow step, button, menu item, or
      Measurement-window column/output that the tutorial covers or
      explains, the tutorial's content/target-widget references must be
      updated in the same change - treat the tutorial like
      `ARCHITECTURE.md`: a living doc that goes stale the moment behavior
      changes out from under it. Draft wording now, finalize once the
      tutorial's actual module/file structure is known from the steps
      above:

      > Whenever you change a workflow step, button, menu item, or
      > Measurement-window column/output that the in-app Tutorial mode
      > (see `<tutorial module(s), TBD>`) walks a user through or
      > explains, update the tutorial's step content and target-widget
      > references in the same change.
- [ ] Update `ROADMAP.md`/`ARCHITECTURE.md` to describe the new tutorial
      module(s) once built.

**Non-negotiable process requirements, from the project owner:**

- New branch off `develop`, named `issue-N/short-description` per
  `AGENTS.md`'s git workflow - ask the project owner for the issue
  number before naming it if one hasn't been provided.
- Real automated tests are required, not optional - follow this repo's
  existing testing conventions.
- Manual testing/proof-test walkthroughs with the project owner are
  required before any commit that changes observable app behavior, per
  `AGENTS.md`'s existing "Manual proof test" section - throughout this
  phase, not just at the end.
- The `CLAUDE.md` update above is required, not optional.

## Phase 16 — 3D bundle-adjustment calibration (build) `[ ]`

Build the photogrammetric bundle-adjustment calibration mode investigated
in Phase 14, now that its math and dependency choice (`scipy`) are
validated against synthetic data. Driven by scientific rigor / closing the
methodology gap with SeaGIS EventMeasure/CAL — **not** a response to an
observed field accuracy problem with the existing pipeline (confirmed with
the project owner when this phase was planned), so this is built and
validated carefully rather than rushed, and treated as genuinely optional
the whole way through.

**Framing, carried over from Phase 14 and non-negotiable for this phase:**
this is an **additional, optional** calibration mode alongside the
existing checkerboard/ChArUco workflow — never a replacement for it. The
existing pipeline (`perform_calibration.py`, `calibration_io.py`,
`stereo_matching.py`) stays the default and stays untouched in its
behavior; this phase only adds a new, separately-chosen path that ends in
the same `PL`/`PR` NPZ format everything downstream already expects.

Numbered steps (Phase-10-style, since this has a real physical-fabrication
dependency, not just software):

- [ ] **Step 0 — Design and fabricate the 3D calibration target.** Decide
      cube size (should span the real working range this app is used at),
      design distinct ArUco marker IDs per face so cross-face
      correspondence is unambiguous, 3D print it, then manually measure
      it with calipers to build the "nominal" object-point geometry the
      solver anchors to (Phase 14's finding: this anchor is load-bearing,
      not optional). Open question to resolve during this step: how many
      cube faces are typically visible per view, and whether a
      single-/two-face view (fewer detected points than the full 8-corner
      set the Phase 14 prototype assumed) still gives a robust per-view
      pose.
- [ ] **Step 1 — Per-face detection + multi-face correspondence.** Extend
      `perform_calibration.py`'s existing `build_charuco_detector` pattern
      into a "cube board" detector: detect each visible face
      independently, tag points by `(face_id, corner_id)`, and combine
      multiple simultaneously-visible faces into one set of
      image-point/object-point correspondences per captured image.
- [ ] **Step 2 — Multi-view capture flow.** New capture UI built on top of
      the existing frame-pair capture pattern, but designed for many
      diverse cube poses/rolls (not the flat sweeps a planar checkerboard
      capture session uses) — likely with on-screen "rotate the cube and
      capture again" guidance, since view diversity is what actually
      decorrelates the solved parameters. Open question: how many views
      is "enough" for a given real rig - Phase 14's synthetic prototype
      used 20; validate a real target count against reprojection RMS and
      parameter-recovery stability rather than assuming that number
      transfers directly to real data.
- [ ] **Step 3 — Per-view initial pose estimation.** `cv2.solvePnP` per
      view against the nominal cube geometry and that view's detected
      left-image points, same approach as the Phase 14 prototype. Needs a
      minimum-detected-points-per-view threshold for views with poor cube
      visibility, mirroring `run_stereo_calibration`'s existing "at least
      3 valid pairs" pattern.
- [ ] **Step 4 — Production bundle-adjustment solver.** Port the Phase 14
      prototype's math (`misc/bundle_adjustment_prototype.py`) into a real
      module; promote `scipy` from a dev-only dependency to a full runtime
      one. Implement the soft anchor-to-nominal-geometry constraint with a
      configurable tolerance (matching whatever manufacturing tolerance
      Step 0's calipers actually measured, not a hardcoded guess). Open
      decision, worth a deliberate ablation rather than guessing: refine
      camera intrinsics jointly within the bundle adjustment, or keep them
      fixed from a preliminary per-camera `cv2.calibrateCamera` pass (the
      prototype fixed them). Output must exactly match the existing
      `PL`/`PR` NPZ shape `calibration_io.py` already loads, so nothing
      downstream needs to change.
- [ ] **Step 5 — New calibration-mode UI.** Add a new option alongside
      Checkerboard/ChArUco in `perform_calibration.py`'s board-type choice
      (e.g. "3D Cube (Advanced)"). Open UX decision to make during this
      step: how the user enters/manages the cube's as-measured geometry
      from Step 0 (a small config file, manual per-corner entry, or
      something else).
- [ ] **Step 6 — Validate against real footage.** Requires Step 0's
      physical cube to exist. Capture real stereo footage of it across
      many views, run both the existing pipeline and this new one against
      comparable data, and validate using the known-distance scale-bar
      check (a separately-scoped, near-term phase per Phase 14's
      findings — build/borrow it before this step, not from scratch here)
      plus Phase 12's `ReprojRMS`/`RayResidual` diagnostics as consistency
      checks. **This step is a hard gate, not a formality:** if real-world
      results don't show a meaningful improvement at the ranges this app
      is actually used at, that is a legitimate, fully documented
      "investigated, not worth the added complexity for our use case"
      outcome — not a reason to push the new mode into default use anyway,
      given this phase's own driving motivation was rigor, not a known
      problem to fix.
- [ ] **Step 7 — Decide presentation.** Given the existing pipeline stays
      the default calibration path regardless of this step's outcome,
      decide how the new mode is labeled/surfaced in the UI (clearly
      marked advanced/experimental rather than presented as a plain
      alternative), and how Step 6's real-footage validation results feed
      the Phase 18 white paper update below.

## Phase 17 — MARE API integration (future, not yet scoped) `[ ]`

Interface with the overall MARE API to record measurement data, etc. Noted
here so it isn't forgotten, but not to be planned in detail until we reach it.

## Phase 18 — Update the measurement white paper (future, not yet scoped) `[ ]`

Update `docs/Sizeamatic_Pro_Stereo_Length_Measurement_Method.docx (1).pdf`
(the scientific write-up of Sizeamatic Pro's stereo length measurement
method) to document the new calculations added in Phases 12-13 — the
object-space `StereoRayResidual(mm)` metric and the Jacobian/covariance
uncertainty propagation — plus, if Phase 16 is completed, the 3D
bundle-adjustment calibration mode and its Step 6 real-footage validation
results (including an honest account if that step concluded it wasn't
worth adopting for this project's actual use case). Noted here so it
isn't forgotten, but not to be planned in detail (including how a
`.docx`/PDF source gets edited, given the rest of this project's tooling
is plain-text/git-based) until we reach it.
