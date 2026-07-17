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

## Dataset Validation

Build the 928-keyframe manifest without committing company data:

```powershell
car5-autolabel build-manifest `
  --data-root "D:\data\car5_20260611_day_02" `
  --output "D:\private-output\car5_v1_manifest.jsonl" `
  --report "D:\private-output\car5_v1_manifest_report.json"
```

Validate a reviewed Label Studio export against that manifest:

```powershell
car5-autolabel validate-label-studio `
  --export "D:\private-output\label-studio-export.json" `
  --manifest "D:\private-output\car5_v1_manifest.jsonl" `
  --output "D:\private-output\car5_v1_annotations.json" `
  --report "D:\private-output\car5_v1_annotations_report.json"
```

Both commands exit with code `2` when the data contract is violated. Use
`--include-context` to include all 27,840 images in the manifest and
`--verify-dimensions` for release validation.

Build the frozen 720-image gold set as 120 complete six-camera timestamp
groups, then export one of the five leave-one-sequence-out COCO folds:

```powershell
car5-autolabel build-gold-set `
  --manifest "D:\private-output\car5_v1_manifest.jsonl" `
  --annotations "D:\private-output\car5_v1_annotations.json" `
  --output "D:\private-output\gold_v1.json" `
  --report "D:\private-output\gold_v1_report.json" `
  --dataset-release-id car5_20260611_day_02_labels_v1

car5-autolabel export-coco `
  --manifest "D:\private-output\car5_v1_manifest.jsonl" `
  --annotations "D:\private-output\car5_v1_annotations.json" `
  --output-dir "D:\private-output\coco_s00" `
  --held-out-fold s00
```

Training is pilot-first and compares two pure 2D detectors on the same held-out
fold: RTMDet-S is the fast baseline and RF-DETR-S is the accuracy challenger.
Each model runs a short smoke test and a 1 to 2 hour pilot. Long training starts
only after metrics and prediction images pass `configs/training/pilot_gate.yaml`.

Stage the RF-DETR COCO layout without duplicating the cached images, then run a
one-epoch smoke test:

```bash
python tools/stage_rfdetr_dataset.py \
  --coco-root /opt/car5-workspace/Docker/annotations/private/phase0/coco_s00 \
  --images-root /opt/car5-data/car5_v1/images \
  --output-dir /opt/car5-data/rfdetr_s00_smoke \
  --max-train-images 128 \
  --max-valid-images 120

RF_HOME=/opt/car5-model-cache/rfdetr \
python tools/run_rfdetr.py \
  --stage smoke \
  --dataset-dir /opt/car5-data/rfdetr_s00_smoke \
  --output-dir /opt/car5-runs/rfdetr_s_smoke_s00
```

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
