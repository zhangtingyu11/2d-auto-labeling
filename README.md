# car5 2D Auto Labeling

Collaborative 2D image auto-labeling service for the car5 six-camera dataset.

## Goals

- Run detector inference over single images or ordered camera sequences.
- Send pre-annotations to Label Studio for human correction.
- Keep a stable eight-class annotation schema.
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
