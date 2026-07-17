# Evaluation and Acceptance Policy

Status: accepted for v1 implementation

Date: 2026-07-16

## Evaluation Purposes

The project separates three questions:

1. **Development comparison:** which detector produces better reviewed
   proposals on the current single-day dataset?
2. **Human-efficiency validation:** does assisted review reduce correction time
   without increasing misses?
3. **Automatic acceptance:** is a class-camera slice proven safe enough to
   bypass human review?

The current data can answer the first two with documented limitations. It
cannot answer the third because there is no independent collection run.

## Detection Matching

- Clip predictions and ground truth to image bounds before matching.
- Exclude camera ignore polygons from both prediction and ground-truth scoring.
- Match only identical active classes.
- Use one-to-one maximum-IoU assignment within each image and class.
- A true positive at the review operating point requires IoU at least 0.50.
- Report COCO AP50 and AP50:95 in addition to the review operating point.
- Report localization error and correction actions for matched boxes.
- A prediction overlapping an ignored host-vehicle region is ignored only when
  the configured overlap rule is met; the evaluator must report this count.

## Required Slices

Every benchmark reports:

- active class;
- camera;
- box-size bucket;
- truncation and occlusion when available;
- natural sequence;
- confidence band;
- native versus tiled inference;
- detector-only versus temporally refined output.

An unsupported slice is labeled `insufficient_evidence`; it is never merged
silently into an aggregate score.

## Development Model Gate

Use the five development folds from ADR 0004. Compare RTMDet and RF-DETR with
identical manifests and evaluator code.

Targets for supported slices:

- review-threshold recall at least 95% overall;
- review-threshold recall at least 95% for `Car` and `Truck` when a fold has at
  least 50 ground-truth boxes;
- review-threshold recall at least 90% for heavy-equipment and `Sign` slices
  when a fold has at least 50 ground-truth boxes;
- no supported active-class recall regression greater than one percentage
  point when promoting a model;
- all absent or low-support classes remain review-only.

Model selection also includes correction time, peak VRAM, throughput, failure
rate, reproducibility, and license status. Public COCO AP is not a selection
gate.

## Gold-Set Protocol

Create a frozen, double-reviewed gold set of at least 720 image tasks.

- Sample complete six-camera timestamp groups for the representative portion.
- Oversample rare classes, truncation, occlusion, glare, dust, small boxes, and
  camera-specific edge cases.
- Include explicit verified-empty images from every camera.
- Both developers independently review the initial boxes.
- Resolve disagreements against `docs/annotation-policy.md` and record the
  policy category, not personal preference.
- Agreement target after policy clarification is at least 98% at box/class
  decision level.

The gold manifest contains no images. It stores stable image IDs, checksums,
sampling reasons, review state, and dataset-release ID.

## Human-Time Experiment

Use a randomized crossover pilot with at least 200 representative image tasks
per annotator and at least two annotators:

- half manual-first, half assisted-first;
- exclude warm-up tasks from timing;
- record drawing, resizing, relabeling, deletion, creation, and submission;
- compare median seconds per image and missed objects;
- target at least 60% median time reduction without a recall regression.

## Automatic-Acceptance Gate

Automatic acceptance is disabled in the first release.

Later, a class-camera-size slice becomes eligible only when all conditions are
true on an independent frozen collection:

- at least 200 high-confidence prediction opportunities in the slice;
- measured precision at least 98%;
- 95% Wilson lower confidence bound at least 98%;
- class recall floor passes at the separate review threshold;
- no unresolved model/teacher/temporal disagreement;
- no truncation, ignore-region ambiguity, or unsupported domain shift;
- written approval recorded in a versioned acceptance configuration.

Slices that do not meet the sample minimum or confidence bound remain human
review tasks regardless of point estimate.

## Efficiency Profile

Targets on the recorded RTX 4060 Laptop GPU are project goals, not assumptions:

- native batch inference: at least 8 images/second;
- tiled inference when enabled: at least 2 images/second;
- peak VRAM below the measured safe limit with at least 512 MiB reserved;
- no duplicate outputs after interrupted-job resume.
