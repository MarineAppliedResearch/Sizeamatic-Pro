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
(Phase 9) are planned next.

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
- [~] Playback speed — investigated, not yet fixed. The speed dropdown
      (0.25x-4x) already exists and "1x" is already intended to match
      native fps, but it's a hardcoded 40ms tick, not read from the
      loaded video's actual fps (only coincidentally correct for ~25fps
      footage) — that specific fix is still open. The suspected cause of
      the reported slowness (live per-frame rectification) was
      profiled and ruled out: simulating a real 4-second Play click
      (the actual self-scheduling `after()` loop, with real screen
      painting) measured 19.7fps raw vs. 19.0fps rectified against a
      25fps target — rectification itself only costs ~1-2ms/tick.
      Forcing real `root.update()` calls (vs. a synthetic tight loop with
      no actual screen paint, which measured ~85-98 ticks/sec) revealed
      the real cost is generic Tkinter canvas-paint/event-loop overhead,
      roughly 50ms/tick against the 40ms budget, regardless of
      rectification. Deliberately not pursued further right now — a
      persistent `PhotoImage` + `.paste()` instead of recreating one
      every frame is a plausible next step, but is real additional
      investigation/work of its own.
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
Recent Projects was added mid-phase, after the other three items were
already done and confirmed — not part of the original plan, but small
enough to fold in rather than spin off its own phase.

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
      identical logic, just skipping the file dialog.

## Phase 9 — Packaging and distribution `[ ]`

Package Sizeamatic Pro as a standalone, distributable (possibly
installable) build that bundles all its requirements and runs fully
offline — so an analyst can get and run it without setting up Python/`uv`
themselves. Not yet scoped in detail (needs its own planning pass —
likely built on `pyinstaller`, given the existing build note already
sitting in git history from before this file existed, or an alternative
bundler if that turns out not to fit); noted here so it isn't forgotten.

## Phase 10 — MARE API integration (future, not yet scoped) `[ ]`

Interface with the overall MARE API to record measurement data, etc. Noted
here so it isn't forgotten, but not to be planned in detail until we reach it.
