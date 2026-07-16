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

Definition of done:

- review committed and pushed to `dev/colleague`;
- no implementation code mixed into the review commit;
- a pull request targets `main`;
- disagreements are written as explicit decision questions.

## WP-01: Data Contract and Validator

Proposed owner: Fengyiovo

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

- validates the expected 928-by-six layout or reports every exception;
- same input produces byte-stable manifest and split output;
- no dependence on Label Studio task IDs for ordering.

## WP-02: Annotation Policy and Gold Set

Proposed owner: both developers

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

## WP-03: RTMDet Baseline

Proposed owner: Fengyiovo

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
- frozen-split metrics and RTX 4060 profile recorded;
- predictions round-trip through Label Studio.

## WP-04: RF-DETR Challenger

Proposed owner: colleague

Depends on: WP-00, WP-01, and WP-02

Create:

- pinned RF-DETR release and weights;
- RF-DETR-S/M training and inference adapter;
- same-split benchmark output compatible with WP-03;
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

Proposed owner: colleague, reviewed by Fengyiovo

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

Proposed owner: assigned after detector selection

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

Proposed owner: Fengyiovo, reviewed by colleague

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

Proposed owner: assigned after the CLI pipeline passes

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
