# car5 2D Auto Labeling

Collaborative 2D image auto-labeling service for the car5 six-camera dataset.

## Goals

- Run detector inference over single images or ordered camera sequences.
- Send pre-annotations to Label Studio for human correction.
- Keep a stable six-class v1 schema and version candidate-class promotion.
- Export reviewed annotations for training and evaluation.
- Add temporal propagation and tracking without coupling them to one detector.

## Safety Boundary

This repository contains source code and small synthetic test fixtures only.
Company images, annotations, model weights, databases, tokens, and generated
outputs must stay outside Git. See `.gitignore` and `SECURITY.md`.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn car5_autolabel.api.main:app --reload --port 8090
```

Then open `http://127.0.0.1:8090/health`.

## Repository Layout

```text
configs/                 Label and pipeline configuration
docs/                    Architecture and engineering decisions
src/car5_autolabel/      Application source
  api/                   FastAPI endpoints
  detectors/             Pluggable detector interface
  integrations/          Label Studio adapter boundary
  tracking/              Temporal association boundary
tests/                   Automated tests
```

## Development Workflow

1. Use `dev/fengyiovo` or `dev/colleague` for daily work.
2. Synchronize the latest `main` into the personal branch before coding.
3. Keep commits small and run `pytest` and `ruff check .`.
4. Push the personal branch and open a pull request into `main`.
5. The other developer reviews before merge.

See `docs/collaboration.md` for the complete two-developer workflow. The first
implementation milestone is documented in `docs/roadmap.md`.

## Technical Direction

- Detailed route: `docs/technical-route.md`
- Accepted decisions: `docs/decisions/`
- Verified data audit: `docs/audits/2026-07-16-car5-v1-data-audit.md`
- Evaluation policy: `docs/evaluation-policy.md`
- Label Studio contract: `docs/label-studio-contract.md`
- Review resolution: `docs/reviews/technical-route-review-resolution.md`
- Mainstream solution research: `docs/research/2026-07-2d-autolabeling-landscape.md`
- Shared work packages: `docs/work-packages.md`
- Agent working agreement: `AGENTS.md`
- Colleague WP-02 prompt: `docs/agent-prompts/colleague-wp02-annotation-policy.md`

## Local 2D Auto-Label V1

The first review-assistance CLI is documented in `docs/v1-user-guide.md`.
On the prepared workstation, run:

```powershell
.\run_v1_autolabel.bat "C:\path\to\car5_dataset_root"
```

The command produces resumable prediction JSONL and a Label Studio task JSON.
V1 is human-review assistance only; it does not automatically accept labels.
