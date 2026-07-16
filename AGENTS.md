# Agent Working Agreement

This file applies to the entire repository.

## Repository Identity

- Remote: `https://github.com/fengyiovo/car5-2d-auto-labeling.git`
- Shared requirements branch: `main`
- Fengyiovo implementation branch: `dev/fengyiovo`
- Colleague implementation branch: `dev/colleague`

Use `git rev-parse --show-toplevel` and `git remote -v` before changing files.
Do not assume the local clone path.

## Source of Truth

Read these files before implementation:

1. `docs/technical-route.md`
2. `docs/work-packages.md`
3. `docs/research/2026-07-2d-autolabeling-landscape.md`
4. `docs/collaboration.md`

The `main` branch defines requirements and work packages. Do not implement
directly on `main`. Merge the latest `origin/main` into the assigned personal
branch, implement there, and open a pull request.

## Initial Colleague Task

The first task on `dev/colleague` is an independent review of the technical
route, not model implementation. Write the review to:

`docs/reviews/technical-route-review-colleague.md`

Use the checklist in `docs/agent-prompts/colleague-route-review.md`.

## Repository Boundaries

- Application code: `src/car5_autolabel/`
- Model and policy configuration: `configs/`
- Tests: `tests/`
- Shared architecture and decisions: `docs/`
- Agent reviews: `docs/reviews/`

Do not commit datasets, company images, Label Studio databases, raw annotation
exports, credentials, model weights, checkpoints, engines, or generated output.
The ignore rules in `.gitignore` are mandatory.

## Engineering Rules

- Keep internal boxes in pixel-space `xyxy`.
- Convert to Label Studio percentage coordinates only at the integration edge.
- Preserve deterministic `frame_id`, `timestamp`, and `camera_id` ordering.
- Treat each camera as an independent sequence in tracking v1.
- Never convert a skipped task into a verified annotation. A reviewed empty
  image must use the explicit `verified_empty` state.
- Every prediction must record model version, config hash, data manifest ID,
  source, score, and review state.
- Add focused tests for coordinate conversion, ordering, class mapping, and any
  acceptance-policy change.

## Before Pushing

```powershell
git status
pytest
ruff check .
git diff --check
```

If a command cannot run, record the reason and residual risk in the pull
request. Do not claim validation that was not performed.
