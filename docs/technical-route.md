# Technical Route for the car5 2D Auto-Labeling Tool

Status: accepted with v1 controls after independent review `52d1561`

Binding decisions and gates:

- `docs/decisions/0001-v1-data-scope.md`
- `docs/decisions/0002-v1-taxonomy.md`
- `docs/decisions/0003-camera-order.md`
- `docs/decisions/0004-v1-split-and-release-policy.md`
- `docs/evaluation-policy.md`
- `docs/label-studio-contract.md`

## 1. Product Goal

Build a self-hosted system that converts ordered images from six cameras into
high-quality 2D bounding-box proposals, uses temporal evidence to improve those
proposals, sends them to Label Studio for efficient review, and learns from the
corrected results.

The system optimizes total labeling cost, not detector FPS alone.

## 2. Classes and Domain Rules

The active v1 class order is:

1. `Car`
2. `Truck`
3. `Bulldozer`
4. `Excavator`
5. `WaterTruck`
6. `Sign`

`Pedestrian` and `BoxTruck` are inactive candidates because the reviewed v1
export has no positive boxes for them. They require a taxonomy-version
migration and reviewed positive data before promotion. Class IDs are immutable
inside a released taxonomy. See ADR 0002 for visual decision rules.

Each camera has a versioned configuration containing:

- camera ID and deterministic display order;
- image dimensions and expected orientation;
- fixed ignore polygons for the host vehicle/bodywork;
- optional valid road region;
- per-class review and acceptance thresholds;
- calibration version.

The visible host vehicle at the image edge is not a target. External objects
that are truncated at the image edge remain targets when their class is
identifiable. This distinction is represented by camera ignore polygons and a
`truncated` annotation attribute, not by deleting all border boxes.

## 3. Non-Goals for Version 1

- No 3D projection or LiDAR dependency.
- No cross-camera identity association.
- No custom annotation canvas.
- No unreviewed foundation-model output treated as ground truth.
- No model weights, raw images, private exports, or credentials in Git.
- No automatic relabeling of historical data without a versioned migration.
- No automatic acceptance of boxes in the first product release.

## 4. Canonical Data Contract

### 4.1 Dataset Manifest

One manifest row represents one image:

```text
dataset_id
image_id
relative_path
sha256
width
height
sequence_id
frame_id
capture_timestamp
raw_sequence_index
keyframe_id (nullable)
is_keyframe
sync_delta_ms
camera_id
camera_order
fold_id
annotation_status
```

Requirements:

- `image_id` is stable and never derived from a Label Studio task ID.
- All six images at one timestamp belong to the same split.
- Ordering is `(sequence_id, frame_id, camera_order)`.
- Duplicate paths, hashes, IDs, frame-camera pairs, and missing images fail
  validation.
- A report states expected and actual camera coverage for every timestamp.
- The supervised v1 population is 928 exact-match keyframes by six cameras,
  for 5,568 reviewed images.
- The remaining high-rate camera images are `temporal_context`, never ground
  truth or implicit negative examples.

### 4.2 Annotation Format

COCO detection JSON is the portable dataset format. The internal box is pixel
`xyxy`, while COCO `xywh` and Label Studio percentages exist only at adapters.

Every box carries project metadata:

```text
annotation_id
image_id
class_id
xyxy
track_id (optional)
occluded
truncated
source = human | detector | tracker | foundation_teacher
model_version (optional)
score (optional)
review_status
created_at
```

Allowed review states:

- `unreviewed_prediction`
- `human_verified`
- `human_modified`
- `human_created`
- `verified_empty`
- `rejected_prediction`

`skip` means unreviewed. It must never be converted to `human_verified` or
`verified_empty`.

### 4.3 Versioning

Each dataset release has:

- manifest ID and SHA-256;
- annotation file SHA-256;
- taxonomy version;
- camera policy version;
- split version;
- creation command and Git commit;
- validation report.

Raw assets and annotation exports remain in approved external storage. Git
stores schemas, converters, configs, checksums, and sanitized reports.

## 5. Leakage-Safe Development Evaluation

Do not randomly split images. Adjacent frames, six simultaneous views, and
long-lived source objects are highly correlated.

The verified timestamp gaps define five natural sequences. Use five
leave-one-sequence-out development folds:

1. keep all six cameras from a timestamp in the same fold;
2. keep non-keyframe context with its enclosing sequence;
3. reset tracking at every natural sequence boundary;
4. report near duplicates and persistent source object IDs across folds;
5. mark a class `not_evaluable` when either fold side lacks support;
6. compare every detector on identical folds and evaluator code.

This one-day collection has no valid independent production test. A second
day, route, or site is required before automatic acceptance. See ADR 0004.

## 6. End-to-End Architecture

```text
external image storage
        |
        v
manifest builder -> deterministic validator -> canonical COCO release
        |                                      |
        |                                      v
        |                               train/evaluate models
        |                                      |
        v                                      v
batch inference -> calibration -> temporal refinement -> policy engine
                                                        |
                                                        v
                                            Label Studio predictions
                                                        |
                                                        v
                                               human review/correction
                                                        |
                                                        v
                                      reviewed export -> QA -> next release
```

The detector, tracker, policy engine, and Label Studio integration are separate
interfaces. A detector replacement must not change data or review semantics.

## 7. Detector Strategy

### 7.1 Baseline A: RTMDet

Implement RTMDet through MMDetection first.

- Start with RTMDet-S pretrained on COCO at long side 640 with AMP.
- Increase to 768 after the first stable memory and throughput profile.
- Run RTMDet-M only after RTMDet-S establishes the safe micro-batch.
- Preserve aspect ratio and pad; do not stretch six-camera images.
- Treat long side 960 and tiling as measured options, not defaults.
- Use automatic mixed precision.
- Probe maximum safe micro-batch, then use gradient accumulation for an
  effective batch near 16.
- Begin with a maximum of 200 epochs and early stopping on validation mAP plus
  review-oriented metrics.

Augmentation is conservative:

- brightness, contrast, gamma, haze, mild blur, sensor noise, and compression;
- small scale and translation changes;
- horizontal flip only after camera and sign semantics are approved;
- no vertical flip;
- avoid aggressive random crops that remove the evidence needed to distinguish
  truck subclasses;
- measure Mosaic/MixUp rather than assuming they help this domain.

### 7.2 Challenger B: RF-DETR

Benchmark RF-DETR-S first and RF-DETR-M if memory permits.

- Pin an exact released package and checkpoint hash.
- Use the same manifest, split, augmentations, and evaluation harness.
- Record peak VRAM, training time, inference latency, and export behavior.
- Do not use Plus-only model components without license approval.
- Promote RF-DETR only if it materially improves the acceptance scorecard and
  passes reproducibility tests on a clean environment.

### 7.3 Optional Benchmark C: YOLO26

YOLO26-S may be measured as a speed reference only. No production dependency or
code copy is allowed until the company approves AGPL obligations or obtains an
Enterprise license.

### 7.4 Detector Selection Gate

The development winner is selected over the five identical sequence folds and
human-correction pilot. Public COCO AP is not a selection gate. Promotion to
automatic acceptance additionally requires an independent frozen collection.

Required comparison:

- COCO mAP50 and mAP50:95;
- per-class AP and recall;
- AP by small/medium/large box;
- metrics per camera;
- precision at the proposed auto-accept threshold;
- recall at the low review threshold;
- mean human correction time;
- RTX 4060 throughput, peak VRAM, and failure rate;
- clean-environment reproducibility;
- license and deployment review.

## 8. Small and Distant Objects

First increase detector input resolution while preserving aspect ratio. If a
camera still has unacceptable small-object recall, enable overlapping tiled
inference for that camera only.

Tiled results use class-aware NMS or weighted fusion in full-image coordinates.
Tests cover objects crossing tile boundaries. Latency and duplicate rate are
reported separately from native inference.

## 9. Temporal Refinement

### 9.1 Sequence Rules

- Process each camera independently.
- Sort by manifest frame order, never filename lexical order alone.
- Reset on sequence change, timestamp gap, camera change, or detected scene cut.
- Do not share track IDs across cameras in v1.

### 9.2 ByteTrack Pass

Run ByteTrack on low-threshold detector outputs.

- Keep high and low detector thresholds configurable per class.
- Require class-compatible association.
- Use camera-specific motion and IoU gates.
- Record track age, hit count, gaps, mean confidence, and box jitter.
- Keep detector boxes and refined boxes so every change is auditable.

### 9.3 Bidirectional Consensus

Run forward and backward association for offline labeling. A propagated box is
trusted only when forward/backward tracks agree within configured IoU and class
constraints. Disagreement creates a high-priority review item.

Temporal processing may:

- recover a low-score detection inside a stable track;
- suppress an isolated detection unsupported by adjacent frames;
- smooth small box jitter;
- propose a box during a short occlusion gap.

It may not silently change class labels or bridge scene cuts.

### 9.4 SAM 2.1 Challenger

On a small gold sequence subset, compare SAM 2.1 box-prompt propagation against
ByteTrack for occlusion, fast motion, and distant targets. Convert masks to
boxes only after clipping and minimum-area checks. Keep this optional until it
improves correction time enough to justify GPU cost.

## 10. Foundation-Model Teacher

offline proposal set. Grounded-SAM-2 may refine proposals or propagate masks.
Grounding DINO uses prompts and synonyms for the six active classes to produce
an offline proposal set. Grounded-SAM-2 may refine proposals or propagate
masks.
offline proposal set. Grounded-SAM-2 may refine proposals or propagate masks.

Teacher outputs are used for disagreement mining:

- teacher finds a box and production detector does not;
- production detector finds a high-confidence box absent from human labels;
- detector and teacher disagree on class or localization;
- adjacent frames disagree strongly.

All such samples enter review. They are not auto-accepted during the first two
dataset cycles.

## 11. Confidence Calibration and Policy Engine

Raw detector scores are not probabilities. Fit calibration on validation data,
initially using per-class temperature or isotonic calibration when sample size
is sufficient. Measure expected calibration error and reliability curves.

Use three policy bands:

1. `auto_accept_candidate`: very high calibrated precision, stable temporal
   support, outside ignore regions, no model disagreement.
2. `review`: medium confidence, isolated box, disagreement, truncation, or rare
   condition.
3. `manual_search`: low detector confidence but teacher/track evidence, or a
   safety-critical slice with uncertain absence.

In the first release, `auto_accept_candidate` is an audit label only and still
requires human submission. Later eligibility requires an independent frozen
collection, at least 98% precision, and a 95% Wilson lower confidence bound of
at least 98% for the exact class-camera-size slice. Rare and unsupported slices
remain review-only.

Never infer that an image is empty solely because the detector returned no
boxes until the absence policy is separately validated.

## 12. Active Learning

Create a ranked review queue using a weighted combination of:

- class uncertainty and low margin;
- RTMDet versus RF-DETR or teacher disagreement;
- temporal inconsistency and track fragmentation;
- camera and time-of-day underrepresentation;
- rare class and small-object coverage;
- embedding diversity to avoid near-duplicate frames;
- possible annotation mistake score.

Select batches with class-camera quotas. Review a bounded batch, create a new
dataset release, retrain, and compare against the same frozen test set.

## 13. Label Studio Integration

Keep Label Studio as the review interface in v1.

The integration service must:

- import deterministic tasks from the manifest;
- attach predictions using stable `image_id` and `model_version`;
- convert pixel `xyxy` to Label Studio percentages with round-trip tests;
- preserve prediction IDs and provenance;
- export annotations without relying on Label Studio numeric task order;
- distinguish submitted empty annotations from skipped tasks;
- support idempotent retry without duplicate tasks or predictions;
- produce a reconciliation report for missing and duplicate images.

Run a CVAT spike only after measuring that Label Studio sequence navigation or
track editing is the dominant remaining human cost.

## 14. Service and Runtime Design

### Phase-one runtime

Use deterministic CLI jobs first:

```text
validate -> convert -> train -> evaluate -> infer -> track -> export -> audit
```

Every command writes a run manifest and can resume without overwriting previous
artifacts.

### Service runtime

FastAPI becomes the control plane after the CLI contracts are stable. A single
GPU worker owns CUDA execution and consumes durable jobs. The first deployment
supports one GPU, cancellation, progress, retry, and restart recovery.

Separate environments are allowed:

- API/control environment: current project Python runtime;
- GPU worker image: pinned Python, PyTorch, CUDA, MMDetection/RF-DETR versions.

This avoids forcing incompatible model dependencies into the API process.

### Model registry

Each model artifact directory contains:

```text
model.yaml
weights file
training config
dataset manifest ID
taxonomy and camera-policy versions
metrics.json
calibration.json
environment lock
Git commit
SHA-256 checksums
```

Weights remain outside Git. Git stores registry schemas and sanitized metadata.

## 15. Target Repository Layout

```text
configs/
  cameras/                 camera order and ignore polygons
  datasets/                manifest and split templates
  models/                  detector and tracker configs
  policies/                acceptance and review thresholds
docs/
  research/                evidence and alternatives
  reviews/                 independent design reviews
src/car5_autolabel/
  active_learning/
  api/
  calibration/
  datasets/
  detectors/
  evaluation/
  inference/
  integrations/
  policies/
  tracking/
tests/
  fixtures/                synthetic or sanitized fixtures only
tools/                      reproducible CLI entry points
```

## 16. Validation Scorecard

### Data integrity gates

- 100% manifest-to-file resolution.
- Zero duplicate `(frame_id, camera_id)` pairs.
- Zero unknown class IDs.
- Deterministic export order and hashes.
- Round-trip coordinate error at most one pixel after clipping.

### Model gates

- Apply the matching, support, confidence, and regression rules in
  `docs/evaluation-policy.md`.
- Review-threshold recall target at least 95% overall on supported gold slices.
- No class-camera slice hidden by aggregate metrics.
- No automatic acceptance in the first release.

### Efficiency gates

- At least 60% reduction in median human time per 1,000 images versus manual
  drawing, measured on a representative blind batch.
- Native inference target at least 8 images/second on the available RTX 4060.
- Tiled inference target at least 2 images/second when enabled.
- Peak GPU memory stays below the detected safe limit with a reserved margin.
- Batch jobs resume after interruption without duplicate output.

### Quality audit

Measure:

- boxes accepted unchanged;
- boxes resized, relabeled, deleted, or created;
- missed objects per 1,000 images;
- false boxes per 1,000 images;
- time per reviewed image;
- results by class, camera, box size, occlusion, and truncation.

## 17. Delivery Phases

### Phase 0: contracts and gold set

Freeze taxonomy, annotation policy, camera order, ignore regions, canonical
schema, manifest builder, split, and a double-reviewed gold subset.

### Phase 1: single-frame baseline

Complete RTMDet training, evaluation, inference, Label Studio import/export,
and correction-time measurement.

### Phase 2: challenger and detector selection

Run RF-DETR on the same split and select a production detector through the
scorecard.

### Phase 3: temporal assistance

Add ByteTrack, scene-cut guards, bidirectional consensus, and track-oriented
evaluation.

### Phase 4: human-in-the-loop cycle

Add calibration, triage, active learning, and audit reports. Collect independent
acceptance data; keep automatic acceptance disabled until its separate gate
passes.

### Phase 5: operational service

Add durable jobs, GPU worker, model registry, deployment image, monitoring,
backup, and rollback.

## 18. Stop and Pivot Rules

- If RTMDet and RF-DETR both fail the review-recall target, revisit annotation
  policy, class ambiguity, resolution, and data coverage before adding models.
- If small objects are the dominant failure, test camera-selective tiling before
  a larger backbone.
- If tracking increases false positives or identity errors, keep it as a review
  prioritizer rather than a box generator.
- If Label Studio correction time remains high because of sequence navigation,
  run the defined CVAT migration spike.
- If a dependency license is not approved, remove it from the production path
  even if its benchmark is better.

## 19. Decision Summary

The approved starting route is:

```text
MMDetection RTMDet baseline
        + RF-DETR same-fold challenger
        + ByteTrack per-camera temporal refinement
        + Grounding DINO/SAM 2.1 offline disagreement teacher
        + calibrated policy engine
        + Label Studio human review
        + active-learning feedback loop
```

The independent review is merged. WP-01 and WP-02 now proceed in parallel.
Detector training remains blocked until the data validator, annotation policy,
gold-set protocol, evaluator contract, and Label Studio round-trip fixture pass.
