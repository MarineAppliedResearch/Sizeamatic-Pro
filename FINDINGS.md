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
