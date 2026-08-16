# Phase 3 Findings — Bugs and Flaws

Findings from the documentation + analysis pass across the codebase
(Phase 3 in `ROADMAP.md`). Bugs marked **Fixed** were corrected as part of
this pass; items marked **Flagged, not fixed** are noted for awareness but
left as-is, either because they're currently unreachable given existing
invariants, or because fixing them well overlaps with the Phase 5
restructure.

## Documentation tooling gaps found and fixed

While auditing whether the docs site actually showed what the source
files' docstrings promised, found two separate ways content was silently
invisible on the rendered site despite being present and correct in
source:

### A. Variables/attributes documented only with `#` comments never rendered

A `#` comment directly above a variable assignment (module-level or
`self.x = ...` inside `__init__`) is invisible to `griffe` (the static
analyzer `mkdocstrings-python` uses) — only a bare string literal placed
**immediately after** the assignment is picked up as that variable's
"attribute docstring" and rendered. Every notable module-level global
(`anaglyph_preview.py`, `calibration_summary.py`, `video_overlay.py`) and
every notable `self.*` instance attribute in `main.py`'s
`SizeamaticProApp.__init__` had good `#` comments that were nonetheless
rendering as completely empty entries on the docs site.

**Fix:** converted every notable module-level and instance attribute to
the `value = ...` / `"""docstring"""` (immediately following) pattern.

### B. Underscore-prefixed ("private") members were filtered out entirely

`mkdocstrings-python` filters out members whose name starts with `_` by
default. Most of `SizeamaticProApp`'s actual implementation lives in
`_`-prefixed helper methods (`_build_menu`, `_render_current_frames`,
`_jump_frames_locked_with_offset`, etc.) — only 21 of 57 methods were
rendering on the `main.py` reference page before this was caught. This is
almost certainly the biggest single reason the docs read as "bare."

**Fix:** set `filters: []` under the mkdocstrings python handler options
in `mkdocs.yml`, matching the Phase 1 convention that private members get
full docstrings too — the default filter was fighting that convention.

### C. Nested (function-local) functions never render at all, regardless of filters

A `def` nested inside another function's body — `_on_close` in both
`measurement_window.py` and `calibration_summary.py` — is invisible to
`mkdocstrings` no matter how good its own docstring is; griffe only
collects module- and class-level members. There's no filter or config
option that changes this; the fix is structural.

**Fix:** promoted both to top-level module functions
(`_on_measurement_window_close(app, win)` /
`_on_calibration_window_close(win)`), passing in what they used to get
from the closure. This also happens to remove the nested-function +
`global` fragility that caused bug #1 below in the first place — the
whole bug class no longer exists once the function isn't nested.

### D. `mkdocs serve` renders instance-attribute docstrings as empty; `mkdocs build` doesn't

Confirmed, reproducible: running `mkdocs build` and checking the output
HTML directly shows every instance-attribute docstring on
`main.SizeamaticProApp`'s reference page rendering correctly (0 of 44
empty). Running `mkdocs serve` and fetching the exact same page from the
live server shows every one of those same attributes with a completely
empty content block (40 of 40 empty) — even on the server's first build,
not just after a hot-reload. Function/method and module docstrings were
unaffected; this only hit instance attributes specifically.

Root cause not fully tracked down (likely a griffe/mkdocstrings-python
caching or code-path difference between the two `mkdocs` subcommands), and
not worth spending more time on right now.

**Practical fix:** don't use `mkdocs serve` to preview or verify this
site. Use `mkdocs build`, then open the generated static HTML directly
(e.g. `site/index.html`) in a browser — no server needed. This is now the
documented workflow in `AGENTS.md`.

## Bugs fixed

### 1. `calibration_summary.py` — `_on_close` closure scoping bug

`ensure_calibration_window`'s nested `_on_close` function assigned to the
module-level `cal_win`, `cal_tree`, and `cal_copy_text` globals without its
own `global` declaration. A `global` statement in an enclosing function
does not extend into a nested function — so those assignments silently
created local variables in `_on_close` instead, leaving the module-level
references pointing at destroyed Tkinter widgets after the window closed.
The next call to `update_calibration_window` would then operate on a
destroyed `Treeview`/`Text` widget and crash with a `TclError`.

**Repro:** Open the calibration summary window, close it, then trigger
`update_calibration_window` again (e.g. reload calibration).

**Fix:** Added the missing `global cal_win`, `global cal_tree`,
`global cal_copy_text` inside `_on_close`.

### 2. `anaglyph_preview.py` — uninitialized `anaglyph_after_id` global

`anaglyph_after_id` was referenced via `global` in `stop_anaglyph_preview`
and `anaglyph_tick`, but unlike the other preview state variables
(`anaglyph_active`, `anaglyph_playing`, `anaglyph_index`), it was never
initialized at module load. It only came into existence once
`anaglyph_tick` ran far enough to reach its own assignment to it. If an
exception fired before that point on the very first tick (the code already
anticipates `cv2.getWindowProperty` raising right after window creation,
via its own `try/except`), `stop_anaglyph_preview` would hit a `NameError`
referencing an undefined name.

**Fix:** Initialized `anaglyph_after_id = None` alongside the other
module-level state variables.

### 3. `main.py` — `stop_anaglyph_preview()` called with no `app` argument

`on_app_close` called `anaglyph_preview.stop_anaglyph_preview()` with zero
arguments, but the function requires `app` as its only parameter (it uses
`app.root.after_cancel` and `app._set_status_mid`). Closing the app while
the anaglyph preview window was open would crash with a `TypeError`
instead of shutting down cleanly.

**Repro:** Open the anaglyph preview (View → Anaglyph 3D Preview…), then
close the main window.

**Fix:** Changed the call to `anaglyph_preview.stop_anaglyph_preview(self)`.

### 8. `main.py` — `current_frameL`/`current_frameR` never initialized in `__init__`

Found while building the Phase 7 pan feature. `self.current_frameL`/
`self.current_frameR` (the cached, already-decoded current frame per pane)
only came into existence once `_render_current_frames` ran for a pane with
a loaded capture — there was no `self.current_frameL = None` /
`self.current_frameR = None` in `__init__`. Any code reading either
attribute before the first successful render of that pane (or when that
pane's video was never loaded at all) would hit `AttributeError` instead
of a clean `None`. Not reachable before Phase 7 since nothing read these
outside `_render_current_frames` itself; became reachable once
`_redisplay_current_frames` (used by panning) needed to read both
attributes regardless of which panes have ever rendered.

**Fix:** Initialized both to `None` in `__init__`, alongside `metaL`/
`metaR`.

### 9. `main.py` — `_playback_tick`'s locked branch could run playback backward

Reported by the project owner while testing Phase 7's resync control:
pressing Play with Lock on and a nonzero resync offset set made both
timelines count *down* instead of up.

Root cause: the locked branch of `_playback_tick` computed the next
index (`nxt`) itself and set both `left_frame_index`/`right_frame_index`
*and* both slider widgets directly — critically, always setting the
right side to the same value as the left (ignoring
`self.lock_offset_frames` entirely), and without wrapping the slider
`.set()` calls in `self._suppress_slider_callbacks` the way every other
call site that programmatically moves both sliders does. Setting a
`ttk.Scale` widget's value directly fires its bound `command` callback,
so each tick fired `on_left_slider_changed` then `on_right_slider_changed`
unsuppressed. Both are wired, when Lock is on, to call
`_jump_frames_locked_with_offset` — first with `"L"` driving (correctly
recomputing the right index using the offset), then immediately after
with `"R"` driving the *same* `nxt` value (since `_playback_tick` had
just set the right slider to `nxt`, not `nxt + offset`), which recomputed
the *left* index as `nxt - offset`. With a positive offset, that's less
than the just-advanced value — so every tick ended by silently pulling
the left index backward by (offset × 2) net of the forward step,
compounding on each subsequent tick.

**Repro:** Load both videos, scrub them apart, enable Lock (capturing a
nonzero offset), then press Play.

**Fix:** Replaced the locked branch's manual index/slider-setting with a
call to `_jump_frames_locked_with_offset("L", nxt)` — the same helper the
slider-drag and step-forward/back controls already used correctly. This
fixes both problems at once: the offset is now preserved during
continuous playback (previously it was silently dropped even without the
callback-cascade bug), and the helper's own slider updates are already
wrapped in `_suppress_slider_callbacks`.

### 11. `main.py` — `_display_bgr_on_canvas` stretched a clamped crop to fill the full display rect

Reported by the project owner as "when i zoom the right video everything
zooms correctly, but when i try to zoom the left video, the points don't
zoom and move correctly with the video" — investigation (numeric checks
against the real app, then a visual repro via `PIL.ImageGrab` screenshots)
found this wasn't actually a left/right asymmetry: every zoom/pan code
path (`on_mouse_wheel`, `_get_view`, `_image_to_screen`, `_get_display_rect`)
is already correctly parametrized by `which` and symmetric between panes.

The real bug reproduced on *either* pane, triggered by "zoom out all the
way" specifically: whenever the view's intended region (computed from the
current zoom/pan) extended past the source image's edges — reachable even
right at `zoom_min` itself with any nonzero leftover pan offset from an
earlier off-center zoom, not just some extreme out-of-bounds case — the
ROI gets clamped to the image's actual bounds before cropping (correct),
but the crop was then unconditionally resized to fill the *entire*
`(dw, dh)` display rect and drawn at the display rect's origin `(dx, dy)`
regardless of whether clamping had actually shrunk it. That silently
stretched a smaller-than-intended crop to fill the same on-screen space,
scaling the displayed video differently from the un-clamped scale
`_image_to_screen` uses to place point overlays — so points appeared to
"jump" relative to the video content whenever this triggered.

**Repro:** Zoom in several notches near one corner of a pane, then zoom
back out (many notches, past where it visibly stops) with the cursor at a
different position — this leaves a large residual pan offset even once
zoom clamps back to `zoom_min`, forcing the ROI clamp.

**Fix:** Map the *actual* clamped crop bounds `(rx0, ry0, rx1, ry1)` back
through the same screen transform `_image_to_screen` uses
(`screen = display_origin + pixel * scale + pan_offset`) to compute where
this exact crop belongs and how large it should be on screen, instead of
always assuming the full, unclamped display rect. Verified both
mathematically (the fix uses the identical formula `_image_to_screen`
uses, so a video pixel and a point overlay at that same pixel can no
longer diverge) and empirically (a point placed inside a heavily
clamped/panned view now lands within ~2px of the actual displayed crop's
edge, matching sub-pixel rounding, instead of the two being scaled
differently). No behavior change in the common, unclamped case — the fix
reduces to exactly the old `(dw, dh)` at `(dx, dy)` whenever nothing was
actually clamped.

### 12. `tests/test_main.py` — running the test suite clobbered the real per-user Recent Projects file

Reported by the project owner as "my recent projects just disappeared
when i did a new build" — turned out to mean the list wasn't gone, but
had been replaced by unfamiliar entries ("the demo projects that were
there before"). The Recent Projects feature (ROADMAP.md Phase 8) added
a call to `recent_projects.add_recent_project(path)` inside
`on_save_project`/`_open_project_from_path`, which by default reads and
writes the real per-user `%APPDATA%\SizeamaticPro\recent_projects.json`
unless a test explicitly overrides
`recent_projects.get_recent_projects_path`. Most new tests for the
feature did override it — but three pre-existing tests
(`test_save_project_writes_current_app_state`,
`test_open_project_restores_video_calibration_and_offset`,
`test_open_project_restores_last_recorded_frame_points_and_log`) already
called `on_save_project`/`on_open_project` for unrelated reasons, from
before the Recent Projects feature existed, and had no reason to
override a path they didn't know would matter. Every test run silently
wrote pytest's own `tmp_path` project paths into the real file,
overwriting whatever the project owner had actually saved/opened —
confirmed by inspecting the real file directly and finding it full of
`AppData\Local\Temp\pytest-of-isaac\...` entries instead of real
projects.

**Fix:** added an `autouse=True` fixture,
`conftest.py`'s `_isolate_recent_projects_file`, that redirects
`recent_projects.get_recent_projects_path` to a `tmp_path` location for
*every* test automatically, regardless of whether that test's author
knew to ask for it. This closes the whole class of bug rather than
just patching the three tests caught this time — a test written next
month that calls `on_save_project` for some unrelated reason is
automatically protected too. Verified by temporarily setting the
fixture's `autouse` back to `False` and confirming a new regression
test (`test_saving_a_project_never_touches_the_real_appdata_recent_projects_file`,
which deliberately adds no monkeypatch of its own) failed exactly as
expected, then restoring the fix and deleting the project owner's real
recent-projects file (by then containing only pytest debris, no real
data) so it starts clean.

## Flaws / risky patterns flagged, not fixed

### 4. `main.py` — `_display_bgr_on_canvas` dead fallback branch

When `_get_image_size(which)` returns `None` (pane metadata not yet
known), the function computes a width-fit resized copy of the frame, then
`return`s immediately without ever drawing it — the resize result is
discarded and nothing is shown.

In the current code, this branch should be unreachable in practice:
`capL`/`metaL` (and `capR`/`metaR`) are always set together when a video
loads, and `_display_bgr_on_canvas` is only called with an already-decoded
frame after that pane's capture exists — so image size is always known by
the time this runs. Worth fixing if that invariant ever changes (e.g.
during the Phase 5 restructure), since right now it would fail silently
rather than showing an error.

### 5. `main.py` — partial-triangulation-failure status inconsistency

In `_update_measurement_status_stub`, if a point in the *middle* of the
list fails to triangulate, the loop `break`s but the function still
continues on to report a summary for whatever points succeeded
beforehand. The partial-failure message (`err_msg`) is passed through to
`measurement_window.update_measurement_window`, which *does* display it
in the measurement popup's error line — but the main window's own status
bar (`_set_status_right`) gets overwritten by the "Measured N pts, M segs"
summary further down, so the reason for the missing point(s) is only
visible in the popup, not the main status bar. Cosmetic/UX inconsistency,
not a crash.

### 6. `main.py` — scattered attribute initialization order in `__init__`

State consumed by `video_overlay.py`/`stereo_matching.py` (`self.ptsL`,
`self.max_points_per_pane`, `self.handle_radius_px`, drag state, zoom
state, `self.cal`, etc.) is initialized *after* `_build_menu`/
`_build_toolbar`/`_build_viewers`/`_build_statusbar` and the first
placeholder render already ran. This doesn't currently cause a bug —
nothing during widget construction reads that later-initialized state, and
no user interaction can fire before `__init__` returns and `mainloop()`
starts — but it's a fragile ordering that a future edit could break
silently (e.g. adding a call inside `_build_viewers` that triggers an
overlay redraw synchronously). Worth tidying up during the Phase 5
restructure rather than patching now.

### 7. `main.py` — vestigial drag-state attributes disconnected from `video_overlay.py`

`SizeamaticProApp.__init__` declares `self.drag_active`/`drag_which`/
`drag_index` and `self.refine_drag_active`/`refine_drag_which`/
`refine_drag_index`, and `on_clear_points` resets them. But the actual
point-drag handling lives in `video_overlay.py`, which uses its own
identically-named module-level globals exclusively and never reads the
copies on `self`. Left over from before `video_overlay.py` was split out
of `main.py` — at that point `self.drag_active` etc. stopped being read by
anything.

Practical effect: clicking "Clear Points" while a drag is in progress
doesn't actually stop that drag in `video_overlay.py`. Harmless in
practice — `on_overlay_left_drag`/`on_overlay_right_drag` bounds-check the
point index against the (now-empty) point list and just no-op — but it's
dead state that could confuse a future reader into thinking `on_clear_points`
does more than it does. Worth removing during the Phase 5 restructure.

### 10. `main.py` — real-world playback fps still below the native-fps target

Reported by the project owner as "playback is significantly better than
before, but still playing very slow." The suspected cause (live
per-frame rectification, `cv2.remap` on every tick) was profiled and
ruled out: simulating a real 4-second Play click through the actual
self-scheduling `after()` loop, with real screen painting, measured
19.7fps raw vs. 19.0fps rectified against a 25fps target — essentially
no difference, meaning rectification isn't the bottleneck.

Isolating compute from paint made the real cost obvious: the same
per-tick work measured ~85-98 ticks/sec (10-12ms/tick) when the window
was hidden and no real `root.update()` was forced, but dropped to
~20fps (~50ms/tick) once actual screen compositing was included. So the
gap is generic Tkinter canvas-paint/event-loop overhead for two large
panes, not the stereo math — `root.after()` only guarantees a *minimum*
delay, so a callback that runs longer than its scheduled interval simply
runs slower than requested, with no error or warning.

**Not fixed at the time.** A persistent `PhotoImage` updated in place via
`.paste()` (instead of constructing a new one every frame, the current
approach in `_display_bgr_on_canvas`) is a plausible next step, but
investigating and verifying that is real additional work, deliberately
deferred rather than squeezed into the same pass that ruled out
rectification. See `ROADMAP.md` Phase 7's playback speed item.

**Update (post-Qt-migration): fixed, and a second, more severe bug
found alongside it.** Re-profiling under the Qt port (Phase 11) found
this specific Tkinter-canvas-paint theory no longer applied - full
per-tick cost (decode + rectify-if-needed + paint, both panes) measured
~14ms, comfortably under the 40ms budget. The actual playback-speed
report ("1x doesn't play at full rate, and 2x plays even slower than
1x") traced to two different bugs: (1) "1x" was still a hardcoded 40ms
tick assuming ~25fps, wrong for any other real fps - fixed by deriving
the tick delay from the loaded video's actual fps
(`_compute_playback_timing`). (2) `_read_frame_at`'s seek-skip
optimization only covered the exact-index and exactly-one-frame-ahead
cases; 2x/4x's small forward skips (+2/+4 frames/tick) fell through to
a full `cap.set()` seek on *every* tick, profiled at ~54-105ms/frame
vs. ~3-10ms/frame to decode-and-discard the same gap - explaining why
faster-than-1x speeds played slower in wall-clock terms despite needing
fewer ticks. Fixed by decoding-and-discarding small forward gaps
instead of seeking (`MAX_SEQUENTIAL_SKIP_FRAMES`). Also switched
`playback_timer` to `Qt.TimerType.PreciseTimer` (Qt's default timer
type is intentionally imprecise). Verified via real profiling through
the actual `app.exec()` event loop: 1x/2x/4x now scale correctly
relative to each other. See `tests/test_main.py`'s
`test_read_frame_at_skips_seeking_for_small_forward_gaps_only` and
`test_compute_playback_timing_uses_the_videos_real_fps`.

## Files reviewed with no bugs found

- `stereo_matching.py` — triangulation/reprojection/uncertainty math holds
  up; edge cases (degenerate homogeneous coordinates, insufficient
  perturbation samples) are already guarded.
- `measurement_window.py` — display-only logic, no state-management bugs
  (uses `app.*` attributes rather than module globals, so it doesn't share
  the closure bug found in `calibration_summary.py`).
- `video_overlay.py` — drag/refine state machine and hit-testing look
  sound; confirmed the "polyline" segment drawing matches
  `main.py`'s `_update_measurement_status_stub` segment computation, so
  that's intentional, not a mismatch.
- `generate_calibration_report.py` — defensively coded (guards against
  divide-by-zero, missing optional map files, degenerate histograms).
- `create_charuco_calibration_target.py` — small and straightforward;
  fails loudly (via `RuntimeError`) rather than silently on the two
  failure modes that matter (board doesn't fit the page, PNG encode
  fails).
