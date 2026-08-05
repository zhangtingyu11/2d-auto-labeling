# car5 2D Auto Labeling

Collaborative 2D image auto-labeling service for the car5 six-camera dataset.

## Reviewer Start Here / 评审先看这里

The current deliverable is **RF-DETR-M V4.4 (70 px policy)**. The files below
are the important implementation, not generated examples:

| Priority | File | Purpose |
| --- | --- | --- |
| **1 - Core inference** | [`tools/infer_rfdetr_image_folder.py`](tools/infer_rfdetr_image_folder.py) | Loads the V4.4 checkpoint, runs an image folder, applies ROI/small-object handling, deduplication, class policy, and the 70 px export rule. |
| **2 - Docker entry** | [`Dockerfile.rfdetr`](Dockerfile.rfdetr) | Reproducible Linux/GPU inference container. Build this file; mount weights and images at runtime. |
| **3 - Model release** | [`configs/models/rfdetr/mining8_v44_70px.json`](configs/models/rfdetr/mining8_v44_70px.json) | Model identity, seven classes, checkpoint filename/SHA-256, environment, and 70 px operating point. |
| **4 - Thresholds** | [`configs/policies/mining8_v44_class_thresholds_v1.json`](configs/policies/mining8_v44_class_thresholds_v1.json) | Per-class confidence policy. |
| **5 - Annotation rule** | [`configs/policies/annotation_v2_70px.json`](configs/policies/annotation_v2_70px.json) | Human-GT and evaluation size boundary. |
| **6 - Core algorithms** | [`src/car5_autolabel/postprocessing.py`](src/car5_autolabel/postprocessing.py), [`src/car5_autolabel/roi_verification.py`](src/car5_autolabel/roi_verification.py), [`src/car5_autolabel/tiling.py`](src/car5_autolabel/tiling.py), [`src/car5_autolabel/policies/size.py`](src/car5_autolabel/policies/size.py) | Duplicate suppression, native-resolution ROI verification, tiled inference, and original-image size filtering. |
| **7 - Training entry** | [`tools/train_rfdetr.py`](tools/train_rfdetr.py) | Cross-platform RF-DETR training CLI. |
| **8 - Full instructions** | [`docs/rfdetr-v44-70px.md`](docs/rfdetr-v44-70px.md), [`docs/docker-deployment.md`](docs/docker-deployment.md) | Exact inference, evaluation, Docker, and handoff commands. |

To inspect or run all code from GitHub, clone the implementation branch rather
than downloading individual files:

```bash
git clone --branch dev/colleague \
  https://github.com/fengyiovo/car5-2d-auto-labeling.git
cd car5-2d-auto-labeling
```

The checkpoint is intentionally not stored in Git. Obtain the verified
`checkpoint_best_total.pth` through the approved internal transfer channel and
verify its SHA-256 against the model-release JSON above. No training or test
images are required merely to review the source; a new image folder can be
mounted when running inference.

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

## Current RF-DETR Release

The current mining operating point is RF-DETR-M V4.4 with a formal 70-pixel
long-side policy. Source, model metadata, inference, evaluation, and Docker
adaptation instructions are documented in `docs/rfdetr-v44-70px.md`. Model
weights, company data, and generated results remain outside Git.

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

## Local 2D Auto-Label V1.1

V1.1 adds a full-view deployment checkpoint, cross-class duplicate removal,
and the mine-specific rule that ambiguous WaterTruck candidates fall back to
Truck. See `docs/v1.1-user-guide.md`. Run:

```powershell
.\run_v1_1_autolabel.bat "C:\path\to\car5_dataset_root"
```

V1.1 still requires human review. Its all-data fit score must not be reported
as independent validation accuracy.
