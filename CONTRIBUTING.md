# Contributing

## Branches

- `main`: reviewed and runnable code only
- `dev/fengyiovo`: Fengyiovo's persistent development branch
- `dev/colleague`: the colleague's persistent development branch
- `feat/<owner>/<name>`: optional short-lived feature branch
- `fix/<owner>/<name>`: optional short-lived bug-fix branch

Both developers can read every branch. Branch names describe responsibility,
not access restrictions.

## Daily Workflow

1. Update `main` from `origin/main`.
2. Merge the updated `main` into your personal development branch.
3. Commit and push work only to your personal or task branch.
4. Open a pull request from the personal branch into `main`.
5. The other developer reviews and merges the pull request.

Do not push unfinished work directly to `main`.

## Pull Requests

- Explain behavior and validation, not only changed files.
- Keep data/model artifacts out of the diff.
- Require one review from the other developer.
- Run `pytest` and `ruff check .` before merge.

The complete two-developer workflow is documented in
`docs/collaboration.md`.

## Commit Messages

Use concise imperative messages, for example:

- `Add detector adapter interface`
- `Fix Label Studio percentage coordinates`
- `Validate class mapping during export`
