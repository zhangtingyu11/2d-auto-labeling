# Roadmap

The detailed algorithm, architecture, acceptance gates, and task ownership are
defined in `technical-route.md` and `work-packages.md`. This page remains the
milestone summary.

## Milestone 0 - Collaboration Foundation

- Private GitHub repository
- Branch and pull-request workflow
- Data leakage protections
- Runnable API skeleton and tests

## Milestone 1 - Detector Baseline

- Independent technical-route review
- Freeze annotation policy and double-reviewed gold set
- Convert reviewed car5 boxes into a training format
- Create leakage-safe train/validation/test splits by time sequence
- Benchmark RTMDet and RF-DETR on one frozen split
- Export confidence-calibrated Label Studio predictions

## Milestone 2 - Human-in-the-Loop

- Batch inference jobs
- Label Studio project import/export
- Difficult-sample queue and quality checks
- Reproducible model/version metadata on every prediction

## Milestone 3 - Temporal Assistance

- Per-camera tracking
- Forward/backward box propagation
- Scene-cut and large-motion guards
- Human confirmation for propagated boxes

## Milestone 4 - Production Tooling

- Job queue, progress, cancellation, and resumability
- GPU batching and mixed precision
- Model registry and rollback
- Audit logs and dataset versioning
