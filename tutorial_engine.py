"""Tutorial engine for Sizeamatic Pro - pure step-tracking data model.

This module is deliberately framework-independent, matching the
existing pure-logic modules (`stereo_matching.py`, `calibration_io.py`,
`project_io.py`) - no Qt import anywhere in this file. `tutorial_window.py`
is the Qt-facing module that displays a `Tutorial`'s steps and calls into
it; this module only tracks which step is current and which steps are
done.

Kept decoupled from any specific track's *content* on purpose (ROADMAP.md
Phase 15's "explicitly built to extend later" requirement) - a
`Tutorial` is built from a plain list of `TutorialStep`s handed to it, so
the future calculations-tutorial track can reuse this exact engine with
its own step list, unmodified.

Contents:
    - `TutorialStep` - one step's static content (title/description/
      details/target-widget reference/completion-action key).
    - `Tutorial` - an ordered list of `TutorialStep`s plus the runtime
      state (current step, which steps are done) and the navigation
      rules decided in ROADMAP.md Phase 15's Step 0: linear by default,
      with a free jump-to-any-step escape hatch, and real
      completion-detection hooks (`mark_action_done`) rather than a
      manual "I did this" button.

Author:
    Isaac Travers

Created:
    2026-08-17
"""


class TutorialStep:
    """One step's static content - never mutated once built.

    A step's "target" tells the Qt-facing `tutorial_window.py` which
    real widget to highlight, and against which window: `(host, kind,
    ref)`. `host` is `"main"` (the main `SizeamaticProApp` window) or
    `"measurement_window"` (the separate `MeasurementWindow` top-level
    dialog - a few steps, like Record and the quality-metric columns,
    live there instead) - each host gets its own `HighlightOverlay`
    instance in `tutorial_window.py`, since an overlay is a child
    widget of one specific top-level window and can't span two. `kind`
    is `"menu"` (a closed top-level menu-bar entry - Step 0 decided
    against drilling into an open submenu item, see ROADMAP.md) or
    `"widget"` (a plain attribute to read off that host and highlight
    directly, e.g. `"btn_play_pause"` on the main window or
    `"results_table"` on the measurement window).
    """

    def __init__(self, step_id, title, description, details, target, completion_action=None):
        """Store one step's content.

        Args:
            step_id (str): Stable, unique identifier for this step
                (e.g. `"load_left_video"`) - used to look this step up
                by name (`Tutorial.index_of`) independent of its
                position in the list, so reordering steps later doesn't
                silently break anything referencing them by index.
            title (str): Short step title shown in both the checklist
                panel and the current-step bubble's heading.
            description (str): The step's main instruction text, shown
                in the current-step bubble (e.g. "Click File -> Load
                Left Video... to open the left camera's footage.").
            details (str): The expandable "Details" section's text -
                background/context that doesn't need to be read before
                acting, but is there for a curious or confused user.
            target (tuple[str, str, str]): `(host, kind, ref)` - see
                this class's own docstring for what each part means.
            completion_action (str | None): The action-name key a real
                app handler reports (via `Tutorial.mark_action_done`)
                that completes this step, or None if this step is
                purely informational (nothing new to *do*, e.g.
                explaining what the RECTIFIED/NOT RECTIFIED indicator
                means) - such a step is only ever completed by the
                manual "Next" fallback Step 0 named as the exception to
                real-hook detection, not by assumption that every step
                must have a hook.

        Returns:
            None
        """
        # Plain 1:1 assignments from the constructor args - nothing here
        # is derived or validated, since every `TutorialStep` is
        # currently hand-written in `tutorial_content_operational.py`
        # rather than parsed from an untrusted external source.
        self.step_id = step_id
        """Stable identifier for this step, independent of list position."""

        self.title = title
        """Short title shown in the checklist and the current-step bubble."""

        self.description = description
        """Main instruction text shown in the current-step bubble."""

        self.details = details
        """Expandable "Details" section text."""

        self.target = target
        """`(kind, ref)` pair identifying the real widget to highlight."""

        self.completion_action = completion_action
        """Action-name key that marks this step done, or None."""


class Tutorial:
    """Tracks progress through an ordered list of `TutorialStep`s.

    Navigation matches ROADMAP.md Phase 15's Step 0 decision exactly:
    linear by default (`advance` moves forward one step at a time,
    and a step's own real handler firing auto-advances past it via
    `mark_action_done`), with a free skip/jump escape hatch
    (`jump_to` accepts any index, no locking) - not a hard
    step-to-step gate. There is no persistence: a fresh `Tutorial`
    instance (built fresh every "Start Tutorial…" click, per
    `tutorial_window.py`) always starts at step 0 with nothing marked
    done.
    """

    def __init__(self, steps):
        """Store the step list and initialize progress at step 0.

        Args:
            steps (list[TutorialStep]): The ordered steps to track,
                e.g. `tutorial_content_operational.STEPS`.

        Returns:
            None
        """
        # ---- Step content (static, handed in by the caller) ----
        self.steps = steps
        """The ordered list of `TutorialStep`s this tutorial covers."""

        # ---- Runtime progress state ----
        self.current_index = 0
        """Index into `steps` of the step currently being shown. Always
        starts at 0 - there is no persistence between tutorial runs."""

        self.done = [False] * len(steps)
        """Parallel list to `steps` - `done[i]` is True once step `i`'s
        `completion_action` has fired at least once, or the user has
        advanced past it. Independent of `current_index`, so jumping
        around via the checklist doesn't lose track of what's actually
        been completed."""

    # -------------------------------------------------------------------------
    # Read-only queries
    # -------------------------------------------------------------------------

    def current_step(self):
        """Get the step currently being shown.

        Returns:
            TutorialStep: `self.steps[self.current_index]`.
        """
        return self.steps[self.current_index]

    def is_finished(self):
        """Check whether every step has been marked done.

        Returns:
            bool: True if `self.done` has no remaining False entries.
        """
        # `all([])` is True, but `self.done` can never be empty in
        # practice - `Tutorial` is always built from a non-empty step
        # list (`tutorial_content_operational.STEPS`).
        return all(self.done)

    def index_of(self, step_id):
        """Find a step's position by its stable `step_id`.

        Args:
            step_id (str): The `TutorialStep.step_id` to look up.

        Returns:
            int | None: The matching step's index, or None if no step
            has that ID.
        """
        # Linear scan - step lists are small (a few dozen entries at
        # most), so there's no need for a dict-based lookup here.
        for index, step in enumerate(self.steps):
            if step.step_id == step_id:
                return index
        return None

    # -------------------------------------------------------------------------
    # Navigation / progress mutation
    # -------------------------------------------------------------------------

    def advance(self):
        """Move to the next step, marking the current one done first.

        The last step's own `advance()` call is a no-op past the end -
        `current_index` simply stays pinned at the final index, since
        there's nothing after it to move to.

        Returns:
            None
        """
        # Mark the step we're leaving as done regardless of whether it
        # had a real completion hook - this is what lets a manual
        # "Next" fallback (Step 0's named exception, not the default)
        # still record progress the same way a real hook would.
        self.done[self.current_index] = True
        if self.current_index < len(self.steps) - 1:
            self.current_index += 1

    def jump_to(self, index):
        """Jump directly to any step, by position.

        This is the decided skip/jump escape hatch - deliberately does
        NOT check whether earlier steps are done first; every step is
        always reachable, matching the checklist panel's own
        click-any-step behavior validated in this phase's prototype.

        Args:
            index (int): The step index to jump to. Out-of-range values
                are silently clamped rather than raising, since this is
                driven by UI clicks on a list whose length is already
                known to the caller.

        Returns:
            None
        """
        # Clamp rather than raise - a caller passing an out-of-range
        # index is a programming slip, not a user-facing error worth
        # surfacing, and clamping keeps the tutorial in a valid state
        # regardless.
        self.current_index = max(0, min(index, len(self.steps) - 1))

    def mark_action_done(self, action_name):
        """Report that a real app action just happened.

        Called from `tutorial_window.TutorialController.notify_action`,
        which every wired-up handler in `main.py`/`video_overlay.py`/
        `measurement_window.py` calls into - this is the real
        completion-detection mechanism decided in Step 0 (hooks by
        default, not a manual "I did this" button). Every step whose
        `completion_action` matches gets marked done, not just the
        current one - so performing an already-passed or not-yet-
        reached step's action (e.g. via the jump escape hatch) still
        records it correctly rather than only ever trusting linear
        order.

        If the *current* step is among those completed, this also
        auto-advances to the next step - the whole point of real-hook
        detection is that finishing a step should visibly move the
        tutorial forward without the user needing to separately click
        "Next".

        Args:
            action_name (str): The action-name key that just fired
                (e.g. `"load_left_video"`).

        Returns:
            list[int]: Indices of every step marked done by this call
            (usually zero or one, but not guaranteed unique - nothing
            stops two steps from sharing a `completion_action` if a
            future step's content ever calls for that).
        """
        # Scan every step, not just the current one - see the
        # docstring above for why (the jump escape hatch means "current"
        # and "next to complete" can disagree).
        newly_done = []
        for index, step in enumerate(self.steps):
            if step.completion_action == action_name and not self.done[index]:
                self.done[index] = True
                newly_done.append(index)

        # Only auto-advance if the step we were actually looking at is
        # one of the ones that just completed - completing some other,
        # not-currently-shown step (via the jump escape hatch) shouldn't
        # yank the view away from whatever the user is currently reading.
        if self.current_index in newly_done and self.current_index < len(self.steps) - 1:
            self.current_index += 1

        return newly_done
