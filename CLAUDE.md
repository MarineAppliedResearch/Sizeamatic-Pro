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
- Don't add ruff/mypy/black or similar lint/format tooling unless explicitly
  asked — see the "Tooling" section in `AGENTS.md` for why.
- Don't start Phase 4 (testing system) or Phase 5 (restructure) work from
  `ROADMAP.md` unless explicitly asked, even if it seems like a natural next
  step while doing earlier-phase work.
