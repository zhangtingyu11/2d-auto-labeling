# Work Packages

The `main` branch owns this task list. Implementation happens on personal
branches and reaches `main` only through a reviewed pull request.

## Branch Responsibilities

| Branch | Responsibility |
| --- | --- |
| `main` | Requirements, accepted architecture, task definitions, released code |
| `dev/fengyiovo` | Fengyiovo implementation and experiments |
| `dev/colleague` | Colleague implementation, experiments, and independent review |

## WP-00: Independent Route Review

Owner: colleague

Branch: `dev/colleague`

Status: complete; review commit `52d1561`, incorporated into `main`

Read:

- `AGENTS.md`
- `docs/technical-route.md`
- `docs/research/2026-07-2d-autolabeling-landscape.md`
- `docs/work-packages.md`

Deliver:

- `docs/reviews/technical-route-review-colleague.md`

The review must evaluate:

- model choice and missing alternatives;
- RTX 4060 feasibility;
- licenses and external service dependencies;
- six-camera ordering and split leakage;
- annotation states and Label Studio assumptions;
- tracking failure modes;
- scorecard realism;
- task boundaries and ownership;
- concrete changes required before implementation.

Completed evidence:

- review committed and pushed to `dev/colleague`;
- documentation-only change;
- review preserved at `docs/reviews/technical-route-review-colleague.md`;
- accepted decisions recorded under `docs/decisions/`.

## WP-01: Data Contract and Validator

Owner: Fengyiovo

Branch: `dev/fengyiovo`

Create:

- canonical manifest and annotation schemas;
- Label Studio export parser;
- deterministic image/frame/camera reconciliation;
- class mapping validation;
- duplicate, missing-file, invalid-box, and ordering reports;
- synthetic fixtures and unit tests.

Primary folders:

- `src/car5_autolabel/datasets/`
- `src/car5_autolabel/integrations/label_studio.py`
- `configs/datasets/`
- `tests/datasets/`

Gate:

- validates the exact 928-by-six supervised layout or reports every exception;
- identifies all 4,640 frames per camera and marks non-keyframes as temporal
  context rather than ground truth;
- emits the five natural sequence IDs and development folds from ADR 0004;
- same input produces byte-stable manifest and split output;
- no dependence on Label Studio task IDs for ordering.

## WP-02: Annotation Policy and Gold Set

Owner: colleague, reviewed by Fengyiovo

Branch: `dev/colleague`

Create:

- class definitions and confusion rules;
- host-vehicle exclusion policy;
- truncation, occlusion, minimum visibility, and empty-image rules;
- six camera configs and ignore polygons;
- double-review protocol and disagreement log;
- frozen gold-set manifest.

Primary folders:

- `configs/cameras/`
- `configs/policies/`
- `docs/annotation-policy.md`
- `tests/fixtures/`

Gate:

- two developers independently agree on at least 98% of sampled boxes after
  policy clarification;
- every disagreement category has a documented resolution.

Use `docs/agent-prompts/colleague-wp02-annotation-policy.md` for the assigned
Agent task. Fengyiovo supplies and reviews camera ignore polygons against the
external images; the colleague owns the policy text and synthetic schemas.

## Detector-Start Gate

WP-03 and WP-04 may start only after all of the following are reviewed on
`main`:

- WP-01 manifest, validator, sequence-fold report, and Label Studio parser;
- WP-02 taxonomy policy, camera policy, gold-set protocol, and disagreement
  schema;
- the synthetic Label Studio round-trip fixture from
  `docs/label-studio-contract.md`;
- evaluator matching tests implementing `docs/evaluation-policy.md`.

## WP-03: RTMDet Baseline

Owner: Fengyiovo

Depends on: WP-01 and WP-02

Create:

- pinned MMDetection GPU environment;
- RTMDet-S/M configs;
- training and inference adapter;
- native-resolution and optional tile benchmark;
- reproducible run manifest;
- Label Studio prediction export.

Primary folders:

- `src/car5_autolabel/detectors/rtmdet.py`
- `configs/models/rtmdet/`
- `tools/train_rtmdet.py`
- `tests/detectors/`

Gate:

- clean-environment train/evaluate smoke test;
- five-fold sequence metrics and RTX 4060 profile recorded;
- predictions round-trip through Label Studio.

## WP-04: RF-DETR Challenger

Owner: colleague

Depends on: WP-01, WP-02, and the detector-start gate

Create:

- pinned RF-DETR release and weights;
- RF-DETR-S/M training and inference adapter;
- same-fold benchmark output compatible with WP-03;
- export and reproducibility report;
- recommendation to promote or reject.

Primary folders:

- `src/car5_autolabel/detectors/rfdetr.py`
- `configs/models/rfdetr/`
- `tools/train_rfdetr.py`
- `tests/detectors/`

Gate:

- no data, weights, or credentials committed;
- comparison uses identical test images and metric code;
- promotion decision references correction time, not public AP alone.

## WP-05: Evaluation and Selection Harness

Owner: colleague, reviewed by Fengyiovo

Create:

- COCO metrics;
- per-class, camera, size, occlusion, and truncation slices;
- confidence calibration plots/data;
- correction-action and review-time importer;
- hardware benchmark command;
- detector selection scorecard.

Primary folders:

- `src/car5_autolabel/evaluation/`
- `src/car5_autolabel/calibration/`
- `configs/policies/acceptance.yaml`
- `tests/evaluation/`

## WP-06: Temporal Refinement

Owner: colleague, reviewed by Fengyiovo

Depends on: selected detector and WP-05

Create:

- ByteTrack adapter;
- sequence reset and scene-cut guards;
- forward/backward consensus;
- track confidence and provenance;
- synthetic sequence tests;
- optional SAM 2.1 spike report.

Primary folders:

- `src/car5_autolabel/tracking/`
- `configs/models/tracking/`
- `tests/tracking/`

## WP-07: Human-in-the-Loop Integration

Owner: Fengyiovo, reviewed by colleague

Create:

- idempotent Label Studio task/prediction sync;
- explicit verified-empty workflow;
- model provenance in every prediction;
- review-priority queue;
- correction audit export;
- active-learning batch selector.

Primary folders:

- `src/car5_autolabel/integrations/`
- `src/car5_autolabel/policies/`
- `src/car5_autolabel/active_learning/`
- `tests/integrations/`

## WP-08: Durable Service

Owner: Fengyiovo, reviewed by colleague after the CLI pipeline passes

Create:

- FastAPI job endpoints;
- single-GPU worker and lock;
- progress, cancellation, retry, and restart recovery;
- model registry and rollback;
- Docker/WSL deployment files;
- operational logs and backup procedure.

Primary folders:

- `src/car5_autolabel/api/`
- `src/car5_autolabel/inference/`
- `deploy/`
- `tests/integration/`

## Pull Request Rule

Every implementation pull request includes:

- work-package ID;
- behavior and assumptions;
- commands actually run;
- test and benchmark results;
- data-safety confirmation;
- known limitations and rollback method.
