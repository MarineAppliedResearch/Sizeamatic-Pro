"""Tests for tutorial_content_operational.py's v1 step content.

These aren't testing the tutorial *engine* (see test_tutorial_engine.py)
- they're guarding the content itself against regressions as it's
edited over time: duplicate step IDs, a malformed target tuple, or an
accidentally-blank field would otherwise only surface as a confusing
runtime bug deep inside tutorial_window.py.
"""

import tutorial_content_operational
import tutorial_engine

VALID_HOSTS = ("main", "measurement_window")
VALID_KINDS = ("menu", "widget")


def test_steps_is_a_non_empty_list_of_tutorial_steps():
    """STEPS should be a real, non-empty list of TutorialStep instances -
    the whole tutorial is built from this one list."""

    assert len(tutorial_content_operational.STEPS) > 0
    assert all(isinstance(s, tutorial_engine.TutorialStep) for s in tutorial_content_operational.STEPS)


def test_every_step_id_is_unique():
    """Duplicate step_ids would break Tutorial.index_of's by-name lookup
    (it returns the first match, silently hiding the second)."""

    ids = [s.step_id for s in tutorial_content_operational.STEPS]
    assert len(ids) == len(set(ids))


def test_every_step_has_non_empty_title_description_and_details():
    """A step missing any of these would render as a blank section in
    the real tutorial window - catch that here instead of visually."""

    for step in tutorial_content_operational.STEPS:
        assert step.title.strip(), step.step_id
        assert step.description.strip(), step.step_id
        assert step.details.strip(), step.step_id


def test_every_step_target_is_a_well_formed_three_tuple():
    """Every target must be a (host, kind, ref) tuple with a known host
    and kind - see tutorial_engine.TutorialStep's docstring for what
    each part means."""

    for step in tutorial_content_operational.STEPS:
        assert isinstance(step.target, tuple) and len(step.target) == 3, step.step_id
        host, kind, ref = step.target
        assert host in VALID_HOSTS, step.step_id
        assert kind in VALID_KINDS, step.step_id
        assert ref, step.step_id


def test_the_step_list_builds_a_working_tutorial_starting_at_the_first_step():
    """A basic end-to-end sanity check: the real content list should
    build a working `Tutorial` and start at its first step, same as any
    other step list would."""

    tutorial = tutorial_engine.Tutorial(tutorial_content_operational.STEPS)

    assert tutorial.current_step().step_id == tutorial_content_operational.STEPS[0].step_id
    assert tutorial.current_step().step_id == "load_left_video"
