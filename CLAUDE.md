# CLAUDE.md

This project's conventions, architecture notes, git workflow, and roadmap live
in [`AGENTS.md`](./AGENTS.md) — read that first. It's written to be
tool-agnostic so it stays useful if this project is ever worked on with a
different coding agent.

See also [`ROADMAP.md`](./ROADMAP.md) for the current phase and what's
explicitly out of scope until a later phase.

## Claude Code specific notes

- Ask before every `git commit` — no exceptions, even mid-task. This project
  owner wants to review each commit before it happens.
- Once a commit has been confirmed and made, push it without asking again
  separately.
- **Never** add a `Co-Authored-By: Claude` (or any other AI co-author)
  trailer to a commit message — no exceptions, ever, for any commit in
  this repo.
- Commit with `git commit -a -s -m "..."` — `-a` stages already-tracked
  modifications (never use this as a substitute for reviewing untracked
  files before adding them), `-s` adds the project owner's own
  `Signed-off-by` trailer (distinct from, and not a substitute for,
  avoiding the `Co-Authored-By: Claude` trailer above).
- Don't add ruff/mypy/black or similar lint/format tooling unless explicitly
  asked — see the "Tooling" section in `AGENTS.md` for why.
- Don't start Phase 4 (testing system) or Phase 5 (restructure) work from
  `ROADMAP.md` unless explicitly asked, even if it seems like a natural next
  step while doing earlier-phase work.
- Whenever you change a workflow step, button, menu item, or Measurement-
  window column/output that the in-app Tutorial mode (`tutorial_engine.py`,
  `tutorial_content_operational.py`, `tutorial_window.py`,
  `tutorial_fixtures.py`) walks a user through or explains, update the
  tutorial's step content (`tutorial_content_operational.py`'s
  `description`/`details` text) and target-widget references (`target`
  tuples) in the same change — treat the tutorial like `ARCHITECTURE.md`: a
  living doc that goes stale the moment the behavior it describes changes
  out from under it.
