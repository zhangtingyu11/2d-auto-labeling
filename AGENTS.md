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
3. `docs/decisions/`
4. `docs/evaluation-policy.md`
5. `docs/label-studio-contract.md`
6. `docs/research/2026-07-2d-autolabeling-landscape.md`
7. `docs/collaboration.md`

The `main` branch defines requirements and work packages. Do not implement
directly on `main`. Merge the latest `origin/main` into the assigned personal
branch, implement there, and open a pull request.

## Current Work

The independent route review is complete at:

`docs/reviews/technical-route-review-colleague.md`

Current parallel work:

- `dev/fengyiovo`: WP-01 data contract and validator;
- `dev/colleague`: WP-02 annotation policy and gold-set protocol.

The colleague uses
`docs/agent-prompts/colleague-wp02-annotation-policy.md`. Neither developer
starts detector training until the detector-start gate in
`docs/work-packages.md` passes.

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
- Treat only the 5,568 reviewed keyframe-camera tasks as v1 supervised data;
  high-rate intermediate frames are temporal context.
- Use the six active taxonomy classes from ADR 0002. Do not train zero-positive
  `Pedestrian` or `BoxTruck` heads in v1.
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
