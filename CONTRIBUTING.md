# Contributing to Sizeamatic Pro

Thanks for your interest in contributing. Sizeamatic Pro is developed by
[Marine Applied Research & Exploration](https://mareresearch.org) (MARE); this
document covers how outside contributions fit into that workflow.

## Before you start

- **Open an issue first.** For anything beyond a trivial typo fix, open a
  GitHub issue describing the bug or proposed change before writing code.
  This avoids duplicated effort and lets a maintainer weigh in on approach
  before you've sunk time into an implementation. Pull requests that don't
  reference an issue may be asked to open one before review continues.
- **Check `ROADMAP.md`.** It tracks what's currently in scope, what's
  deliberately deferred, and what phase the project is in. A change that
  looks like a natural next step may already be planned for a later phase —
  check there (or ask in your issue) before starting substantial work.

## Branching and commits

This project follows the branching model described in
[nvie.com/posts/a-successful-git-branching-model](https://nvie.com/posts/a-successful-git-branching-model/)
("git-flow"):

- `develop` is the integration branch. `master` is release-only.
- Create your feature branch off `develop`, named `issue-N/short-description`
  (e.g. `issue-12/fix-rectification-crash`), where `N` is the issue number
  you opened or are addressing.
- Never commit directly to `develop` or `master`.
- Write plain, descriptive commit messages — no Conventional Commits prefix
  required. Explain what changed and why in the body if it isn't obvious
  from the summary line.
- If your change addresses a specific issue, reference it in the commit
  body (not the title) — e.g. a line like `Part of issue #12`. Be careful
  with bare `#N` elsewhere in a commit message: GitHub auto-links any bare
  `#N` to that issue or PR number, so an unrelated `#N` reference will
  create a misleading link.
- Open a pull request into `develop` when your change is ready for review.
  Maintainers merge approved PRs; please don't merge your own.

## Code conventions

Full conventions for this codebase — comment style, docstring requirements,
type hints, testing patterns, and dependency management — are documented in
[`AGENTS.md`](AGENTS.md). Please read it before submitting a PR; in short:

- Every class, function, and method (public and private) needs a
  Google-style docstring — this project's documentation site is generated
  from them.
- New or substantially edited code should have type hints.
- No linter/formatter/type-checker is enforced; match the style already
  present in the file you're editing.
- Add or update tests under `tests/` for behavior you change. See
  `AGENTS.md`'s Testing section for the patterns this project uses (the
  `FakeApp` stand-in, the shared `hidden_tk_root` fixture, synthetic
  calibration fixtures).

## Reporting bugs

Use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`. Include
enough detail to reproduce: OS/Python version, the video/calibration file
properties involved (not the files themselves unless asked), and steps to
reproduce.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you're expected to uphold it.

## License

By contributing, you agree that your contributions will be licensed under
the project's [Apache License 2.0](LICENSE).
