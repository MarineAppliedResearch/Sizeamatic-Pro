"""Tests for tutorial_engine.py's pure step-tracking data model.

No Qt/fixtures needed anywhere here - `TutorialStep`/`Tutorial` are
deliberately framework-independent (see tutorial_engine.py's module
docstring), so these tests just build small step lists directly.
"""

import tutorial_engine


def _make_steps(n):
    """Build `n` simple `TutorialStep`s, IDs "step0".."step{n-1}", each
    with its own distinct `completion_action` of the same name - a
    shared helper so each test doesn't repeat this boilerplate."""

    return [
        tutorial_engine.TutorialStep(
            step_id=f"step{i}",
            title=f"Title {i}",
            description=f"Description {i}",
            details=f"Details {i}",
            target=("main", "widget", f"widget{i}"),
            completion_action=f"step{i}",
        )
        for i in range(n)
    ]


def test_tutorial_step_stores_every_field_verbatim():
    """A `TutorialStep` should just store exactly what it's given, with
    no transformation - `Tutorial`/`tutorial_window.py` read these back
    directly."""

    step = tutorial_engine.TutorialStep(
        step_id="load_left_video",
        title="Load the left video",
        description="Click File...",
        details="The left video is the reference timeline.",
        target=("main", "menu", "File"),
        completion_action="load_left_video",
    )

    assert step.step_id == "load_left_video"
    assert step.title == "Load the left video"
    assert step.description == "Click File..."
    assert step.details == "The left video is the reference timeline."
    assert step.target == ("main", "menu", "File")
    assert step.completion_action == "load_left_video"


def test_tutorial_step_completion_action_defaults_to_none():
    """A step with no explicit `completion_action` should default to
    None, not raise or require a placeholder string."""

    step = tutorial_engine.TutorialStep(
        step_id="x", title="X", description="X", details="X", target=("main", "widget", "x")
    )

    assert step.completion_action is None


def test_tutorial_starts_at_step_zero_with_nothing_done():
    """A freshly built `Tutorial` always starts fresh - no persistence,
    per ROADMAP.md Phase 15's Step 0 decision."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))

    assert tutorial.current_index == 0
    assert tutorial.current_step().step_id == "step0"
    assert tutorial.done == [False, False, False]
    assert tutorial.is_finished() is False


def test_advance_marks_current_step_done_and_moves_to_the_next_one():
    """Advancing should mark the step being left as done and move the
    current index forward by one."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))

    tutorial.advance()

    assert tutorial.done == [True, False, False]
    assert tutorial.current_index == 1
    assert tutorial.current_step().step_id == "step1"


def test_advance_past_the_last_step_stays_pinned_at_the_end():
    """Calling advance() once already on the last step should mark it
    done but not move `current_index` out of range."""

    tutorial = tutorial_engine.Tutorial(_make_steps(2))

    tutorial.advance()  # step0 -> step1
    tutorial.advance()  # step1 done, but no step2 to move to

    assert tutorial.current_index == 1
    assert tutorial.done == [True, True]


def test_jump_to_moves_to_any_step_regardless_of_completion_state():
    """The skip/jump escape hatch (ROADMAP.md Phase 15's Step 0
    decision) should allow jumping to any step, done or not, without
    needing earlier steps completed first."""

    tutorial = tutorial_engine.Tutorial(_make_steps(5))

    tutorial.jump_to(3)

    assert tutorial.current_index == 3
    assert tutorial.done == [False] * 5  # jumping alone doesn't complete anything


def test_jump_to_clamps_out_of_range_indices_instead_of_raising():
    """An out-of-range jump target should clamp to the nearest valid
    index rather than raising - see jump_to's docstring for why."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))

    tutorial.jump_to(999)
    assert tutorial.current_index == 2

    tutorial.jump_to(-5)
    assert tutorial.current_index == 0


def test_index_of_finds_a_step_by_its_stable_id():
    """A step should be findable by its `step_id`, independent of
    where it sits in the list."""

    tutorial = tutorial_engine.Tutorial(_make_steps(4))

    assert tutorial.index_of("step2") == 2


def test_index_of_returns_none_for_an_unknown_step_id():
    """Looking up a step_id that doesn't exist should return None, not
    raise."""

    tutorial = tutorial_engine.Tutorial(_make_steps(2))

    assert tutorial.index_of("no_such_step") is None


def test_mark_action_done_completes_the_current_step_and_auto_advances():
    """Firing the current step's real completion action should mark it
    done AND auto-advance - the whole point of real-hook detection
    (Step 0's decision) is that the tutorial visibly moves forward
    without a separate "Next" click."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))

    newly_done = tutorial.mark_action_done("step0")

    assert newly_done == [0]
    assert tutorial.done[0] is True
    assert tutorial.current_index == 1


def test_mark_action_done_for_a_non_current_step_does_not_move_the_view():
    """Completing some other (not currently-shown) step's action - e.g.
    via the jump escape hatch putting the user somewhere else first -
    should record that step as done without yanking the current view
    away from what the user is actually looking at."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))
    tutorial.jump_to(2)  # now looking at step2

    newly_done = tutorial.mark_action_done("step0")

    assert newly_done == [0]
    assert tutorial.done == [True, False, False]
    assert tutorial.current_index == 2  # unchanged - step0 isn't what's shown


def test_mark_action_done_with_no_matching_step_returns_empty_list():
    """An action name that doesn't match any step's `completion_action`
    should be a no-op, not raise."""

    tutorial = tutorial_engine.Tutorial(_make_steps(2))

    assert tutorial.mark_action_done("no_such_action") == []
    assert tutorial.done == [False, False]


def test_mark_action_done_is_idempotent_for_an_already_done_step():
    """Firing the same completion action twice (e.g. the user redoes an
    action already credited) shouldn't re-report it as newly done or
    advance a second time."""

    tutorial = tutorial_engine.Tutorial(_make_steps(3))

    tutorial.mark_action_done("step0")  # step0 done, now on step1
    tutorial.jump_to(0)  # user jumps back to look at step0 again

    newly_done = tutorial.mark_action_done("step0")

    assert newly_done == []
    assert tutorial.current_index == 0  # no further auto-advance triggered


def test_is_finished_is_true_only_once_every_step_is_done():
    """`is_finished` should stay False until every single step has been
    completed, not just the current one."""

    tutorial = tutorial_engine.Tutorial(_make_steps(2))

    assert tutorial.is_finished() is False

    tutorial.mark_action_done("step0")
    assert tutorial.is_finished() is False

    tutorial.mark_action_done("step1")
    assert tutorial.is_finished() is True
