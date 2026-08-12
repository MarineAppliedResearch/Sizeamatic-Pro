"""Tests for recent_projects.py's persistent recent-projects list.

All tests pass an explicit `path=` override pointing at a tmp_path file,
never touching the real `%APPDATA%\\SizeamaticPro\\recent_projects.json`
this module would otherwise read/write on the machine running the tests.
"""

import os

import recent_projects


def test_load_recent_projects_returns_empty_list_when_file_does_not_exist(tmp_path):
    """A first run (no recent-projects file written yet) should read back
    as an empty list, not raise."""

    path = str(tmp_path / "recent_projects.json")

    assert recent_projects.load_recent_projects(path) == []


def test_add_recent_project_persists_and_loads_back(tmp_path):
    """Recording a project path should make it show up on the next load,
    most-recent first."""

    path = str(tmp_path / "recent_projects.json")
    project_a = str(tmp_path / "a.json")
    project_b = str(tmp_path / "b.json")
    open(project_a, "w").close()
    open(project_b, "w").close()

    recent_projects.add_recent_project(project_a, path)
    recent_projects.add_recent_project(project_b, path)

    assert recent_projects.load_recent_projects(path) == [project_b, project_a]


def test_add_recent_project_moves_existing_entry_to_front_instead_of_duplicating(tmp_path):
    """Re-saving/reopening a project already in the list should bump it to
    the top rather than showing it twice."""

    path = str(tmp_path / "recent_projects.json")
    project_a = str(tmp_path / "a.json")
    project_b = str(tmp_path / "b.json")
    open(project_a, "w").close()
    open(project_b, "w").close()

    recent_projects.add_recent_project(project_a, path)
    recent_projects.add_recent_project(project_b, path)
    recent_projects.add_recent_project(project_a, path)

    assert recent_projects.load_recent_projects(path) == [project_a, project_b]


def test_add_recent_project_trims_to_max_recent_projects(tmp_path):
    """The list should never grow past MAX_RECENT_PROJECTS entries, even
    if more than that many distinct projects have been saved/opened."""

    path = str(tmp_path / "recent_projects.json")
    projects = []
    for i in range(recent_projects.MAX_RECENT_PROJECTS + 3):
        p = str(tmp_path / f"project_{i}.json")
        open(p, "w").close()
        projects.append(p)
        recent_projects.add_recent_project(p, path)

    loaded = recent_projects.load_recent_projects(path)
    assert len(loaded) == recent_projects.MAX_RECENT_PROJECTS
    # Most recently added should be first, oldest-of-the-kept should be the
    # (MAX_RECENT_PROJECTS)th-from-last added, not the very first ever added.
    assert loaded[0] == projects[-1]
    assert loaded == list(reversed(projects[-recent_projects.MAX_RECENT_PROJECTS:]))


def test_load_recent_projects_silently_drops_files_that_no_longer_exist(tmp_path):
    """A project that's since been moved/renamed/deleted should just
    quietly disappear from the list on the next load, rather than
    showing a stale entry that would only error if clicked."""

    path = str(tmp_path / "recent_projects.json")
    project_a = str(tmp_path / "a.json")
    project_b = str(tmp_path / "b.json")
    open(project_a, "w").close()
    open(project_b, "w").close()

    recent_projects.add_recent_project(project_a, path)
    recent_projects.add_recent_project(project_b, path)

    os.remove(project_a)

    assert recent_projects.load_recent_projects(path) == [project_b]
