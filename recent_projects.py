"""Persistent "recent projects" list for Sizeamatic Pro's File menu.

Tracks the last few project files saved or opened, so the File >
Recent Projects submenu (ROADMAP.md Phase 8) can offer a quick reopen
without a file dialog. Separate from `project_io.py`: that module reads/
writes one project's own JSON manifest, while this one persists a tiny
cross-session list of *paths to* project manifests, independent of any
single project's content.

Contents:
    - `get_recent_projects_path` — where the list itself is stored.
    - `load_recent_projects` — read the list, dropping stale entries.
    - `add_recent_project` — record a path as most-recently-used.

Assumptions:
    - Lives under the per-user Windows app-data folder rather than next
      to the app itself, so it survives the app folder being replaced or
      updated - relevant once Phase 9 packages this as a standalone
      .exe - and behaves the same whether running from source or
      packaged.
"""

import json
import os

MAX_RECENT_PROJECTS = 5


def get_recent_projects_path():
    """Return the path to the persistent recent-projects file.

    Returns:
        str: The absolute path to the recent-projects JSON file, under
        `%APPDATA%\\SizeamaticPro\\` (or the user's home directory if
        `APPDATA` isn't set).
    """
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "SizeamaticPro", "recent_projects.json")


def load_recent_projects(path=None):
    """Load the list of recently saved/opened project file paths.

    Silently drops any path that no longer points at an existing file,
    rather than surfacing a stale entry that would just error if
    clicked - the caller doesn't need to check for staleness itself.

    Args:
        path (str | None): Override for the recent-projects file path,
            for testing. Defaults to `get_recent_projects_path()`.

    Returns:
        list[str]: Up to `MAX_RECENT_PROJECTS` project file paths,
        most-recently-used first. Empty if the file doesn't exist yet,
        isn't valid JSON, or every entry has gone stale.
    """
    path = path or get_recent_projects_path()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    raw_paths = data.get("recent_projects", []) if isinstance(data, dict) else []
    existing = [p for p in raw_paths if isinstance(p, str) and os.path.isfile(p)]
    return existing[:MAX_RECENT_PROJECTS]


def add_recent_project(project_path, path=None):
    """Record a project path as the most recently used, persisting it.

    Moves `project_path` to the front of the list (removing any earlier
    occurrence rather than duplicating it), then trims to
    `MAX_RECENT_PROJECTS`.

    Args:
        project_path (str): The project file path just saved or opened.
        path (str | None): Override for the recent-projects file path,
            for testing. Defaults to `get_recent_projects_path()`.

    Returns:
        None
    """
    path = path or get_recent_projects_path()

    existing = load_recent_projects(path)
    updated = [project_path] + [p for p in existing if p != project_path]
    updated = updated[:MAX_RECENT_PROJECTS]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"recent_projects": updated}, f, indent=2)
