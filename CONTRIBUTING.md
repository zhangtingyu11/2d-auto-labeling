# Contributing

## Branches

- `main`: reviewed and runnable code only
- `feat/<name>`: new behavior
- `fix/<name>`: bug fixes
- `chore/<name>`: tooling and maintenance

## Pull Requests

- Explain behavior and validation, not only changed files.
- Keep data/model artifacts out of the diff.
- Require one review from the other developer.
- Run `pytest` and `ruff check .` before merge.

## Commit Messages

Use concise imperative messages, for example:

- `Add detector adapter interface`
- `Fix Label Studio percentage coordinates`
- `Validate class mapping during export`
