# Independent Review of the car5 2D Auto-Labeling Technical Route

Review date: 2026-07-16

Reviewer branch: `dev/colleague`

Reviewed baseline: `origin/main` at `8c2417d`

## Overall conclusion

**Decision: approve with changes.**

The proposed route is appropriate for a self-hosted, pure-2D, human-in-the-loop
auto-labeling product. The separation between data contract, detector,
temporal refinement, policy, evaluation, and Label Studio integration is sound.
RTMDet as the first end-to-end baseline and RF-DETR as a same-split challenger
is a reasonable two-model strategy.

Implementation should not begin with detector training, however. Four issues
are blocking: the exact image population is unresolved; the canonical taxonomy
conflicts with the known seven-class baseline; the leakage-safe split cannot be
specified until sequence boundaries are known; and the acceptance gates lack
the sample-size and IoU definitions needed to make them executable. WP-01 and
WP-02 must resolve these issues before WP-03 or WP-04 starts.

This review is a design assessment only. No detector training, inference,
latency benchmark, VRAM benchmark, or Label Studio round-trip test was run.

Submission validation: `git diff --check` passed. `pytest` and `ruff check .`
were attempted but could not run because neither executable is installed in
the current shell environment. This commit contains documentation only, so the
remaining risk is limited to the absence of repository-tooling validation; no
code behavior is claimed to have been tested.

## Strong parts of the proposal

### Product and architecture

- The product optimizes human correction cost rather than detector FPS or COCO
  AP alone. This is the correct objective for an annotation system.
- Pure 2D is kept independent from LiDAR and 3D projection. That controls scope
  and allows the team to deliver a useful first product.
- Pixel-space `xyxy` is the internal representation, with COCO and Label Studio
  conversion only at adapters. This reduces repeated coordinate drift.
- Predictions, human annotations, tracker output, and foundation-model output
  have distinct provenance and review states. In particular, `skip` is not
  treated as an empty verified image.
- CLI contracts precede FastAPI service work. This is appropriate for two
  developers and makes failures reproducible before durable-job complexity is
  added.

### Detector choice

- MMDetection RTMDet-S/M is a defensible primary baseline: the stack is mature,
  supports RTMDet, and is Apache 2.0. The main risk is environment pinning, not
  the suitability of the detector family.
- RF-DETR-S/M is a useful independent challenger rather than a replacement
  chosen from public results. Its current open-source package and the
  Apache-designated Nano/Small/Medium/Large detection weights are Apache 2.0;
  the XL/2XL Plus detection components use PML 1.0 and must remain excluded
  without legal approval.
- YOLO26 is correctly limited to an optional benchmark because the official
  Ultralytics distribution is offered under AGPL-3.0 or an Enterprise license.
- PaddleDetection RT-DETR is a credible Apache-2.0 fallback, but keeping Paddle
  out of phase one avoids maintaining a third training ecosystem before it is
  needed.

### Data and evaluation principles

- Keeping six simultaneous camera views in one split and grouping adjacent
  frames is essential.
- Per-class, per-camera, size, occlusion, and truncation slices are more useful
  than a single aggregate mAP.
- Camera-specific ignore polygons, immutable IDs after release, deterministic
  manifests, checksums, and round-trip coordinate tests are all appropriate
  production controls.
- The stop-and-pivot rules are valuable: tiling is conditional, tracking can be
  downgraded to prioritization, and Label Studio can be replaced only after a
  measured workflow bottleneck.

## Blocking risks and incorrect assumptions

### 1. “928 timestamps = 5,568 images” is not yet an established data contract

The locally available car5 source contains 928 LiDAR/label keyframes but 4,640
images per camera, or 27,840 camera images in total. The exact 5x ratio strongly
suggests a higher-rate camera stream around 928 keyframes. Therefore 5,568 is
only the size of a six-camera keyframe subset, not automatically the complete
2D dataset.

Before creating a manifest, the team must decide whether v1 labels:

1. only the six images exactly matched to each of 928 keyframes;
2. all 4,640 timestamps from each of six cameras; or
3. a versioned subset of the higher-rate stream.

The manifest needs at least `capture_timestamp`, `keyframe_id` (nullable),
`sync_delta_ms`, `raw_sequence_index`, and `is_keyframe`. Matching must define
timestamp units, tolerance, tie-breaking, missing-camera behavior, and whether
camera files with equal stems are truly simultaneous. `frame_id` alone is too
ambiguous for this dataset.

This also changes tracking and splitting. If the intermediate camera frames are
used for propagation, they must inherit the split of their enclosing sequence;
they may not leak into validation or test because only keyframes carry labels.

### 2. The taxonomy is unresolved

The route declares eight immutable classes including `Sign`, while the existing
MMDetection3D baseline discussed for this dataset uses seven classes without
`Sign`. The route cannot freeze IDs until the authoritative 2D annotation policy
and the actual Label Studio export are reconciled.

`Truck`, `WaterTruck`, and `BoxTruck` also need visual decision rules. If a
subclass cannot be identified from a single 2D view, the policy must state
whether to use `Truck`, mark uncertainty, or require temporal/context evidence.
Otherwise apparent model error will actually be annotator-policy disagreement.

### 3. Sequence construction and split leakage are underspecified

“Build `sequence_id` from timestamp gaps and route/scene boundaries” is a goal,
not an algorithm. Required inputs are missing: run/day identifiers, clock
reset handling, timestamp frequency, duplicate timestamps, scene-cut policy,
and minimum exclusion gap.

If all 928 keyframes come from one continuous drive, a 70/15/15 contiguous
split can still share nearly identical geography and objects across boundaries.
Rare-class balancing must not move individual frames between splits. Prefer
grouping by independent collection run/day/route. If only one run exists, use
contiguous blocks with a documented temporal buffer, freeze the test block, and
state explicitly that external-scene generalization has not been measured.

Near-duplicate detection by perceptual hash or embeddings should be part of the
split audit, including cross-camera and cross-split duplicate reports.

### 4. Acceptance gates are not yet executable

“98% precision with an acceptable lower confidence bound” must specify:

- IoU and class-match rules for a correct box;
- clipping and ignore-region treatment;
- one-to-one matching method;
- confidence-interval method and confidence level;
- minimum positive samples per class-camera-size slice;
- maximum missed pedestrians or other safety-critical objects per 1,000 images;
- allowed regression tolerance per class;
- who times manual-only and assisted blind batches, and how timing is captured.

With few rare-class examples, a point estimate of 98% is misleading. A slice
must remain review-only until its predeclared lower confidence bound passes.
The 95% overall review-threshold recall target should be supplemented by
class-specific recall floors, especially for `Pedestrian` and rare heavy
equipment.

The “60% median time reduction per 1,000 images” gate is directionally useful
but expensive. Start with a powered pilot on representative blind batches,
report annotator count and warm-up, then confirm on a larger acceptance batch.

### 5. RTX 4060 feasibility is conditional, not established

The route is feasible on an RTX 4060 if the product is an offline batch tool and
the implementation probes actual VRAM. It is not yet proven that RTMDet-M at a
960-pixel long side or RF-DETR-M training will fit with useful micro-batches on
the available card. Desktop/laptop power limits and VRAM must be recorded.

Required order:

1. record GPU name, VRAM, driver, CUDA, PyTorch, precision, image size, batch,
   and peak allocated/reserved memory;
2. smoke-test RTMDet-S at 640 or 768 with AMP;
3. increase resolution, then model size, one variable at a time;
4. use gradient accumulation only after a stable micro-batch is known;
5. treat 960, tiling, RTMDet-M, and RF-DETR-M as measured options rather than
   mandatory defaults.

SAM 2.1 should initially use Tiny or Small checkpoints for an offline spike.
The Large checkpoint, concurrent detector execution, and video-state memory may
be unsuitable for an 8 GB-class card. CPU RAM and WSL2/Docker shared-memory use
must also be logged.

### 6. Label Studio integration needs an end-to-end contract test

Label Studio officially supports predictions/pre-annotations and preserves
prediction result IDs when a prediction becomes an annotation, but that alone
does not guarantee the proposed state machine.

Pin the Label Studio version and test the full workflow against the Community
edition actually deployed:

`manifest -> task -> prediction -> human unchanged/modified/created/deleted ->`
`submitted empty/skipped -> export -> canonical annotation`

The adapter must use stable external IDs, not numeric task order; verify
`from_name`, `to_name`, label spelling/case, original image dimensions,
percentage conversion, model version, duplicate retries, prediction deletion,
and project configuration hash. Local files additionally require a mounted
document root and a project local-storage registration. API tokens and the
Label Studio database remain outside Git.

Do not enable Label Studio “auto accept annotation suggestions” during the
first dataset cycles. A displayed prediction is not a human verification.

### 7. ByteTrack can amplify detector mistakes on moving cameras

ByteTrack is MIT-licensed and simple, but its low-score second association can
extend false detections or assign an incorrect track through crossings. The
base method has no appearance ReID or camera-motion compensation. A vehicle-
mounted six-camera rig creates strong ego-motion, so IoU/Kalman assumptions may
break during turns, bumps, close passes, and scene cuts.

ByteTrack output must therefore remain a proposal with provenance. Measure box
recovery precision, false propagation, fragmentation, ID switches, and class
changes separately. Require maximum gap, minimum hits, forward/backward
agreement, and detector support around every inserted box. Do not smooth large
boxes across abrupt scale changes.

Add a small permissively licensed BoT-SORT or global-motion-compensated tracker
spike if ByteTrack fails specifically on ego-motion. Do not obtain that code by
copying the AGPL Ultralytics package into production. Independently review the
chosen implementation and transitive licenses.

### 8. SAM 2.1 mask propagation is not a generic box tracker

SAM 2.1 code and checkpoints are Apache 2.0, but masks can drift to adjacent
look-alike trucks, merge objects, lose tiny targets, or retain an occluded
object after it disappears. Mask-to-box conversion can also enlarge boxes due
to shadows, dust, or partial masks. Use short bounded clips, per-camera resets,
prompt refreshes, and backward consistency. It should remain review-only until
it beats ByteTrack on correction time and propagation precision.

## Recommended model and algorithm changes

1. Keep RTMDet-S/M and RF-DETR-S/M as the only phase-one trained detectors.
   Pin MMDetection/MMCV/MMEngine and one released RF-DETR version plus weight
   hashes. Do not install RF-DETR from its moving `develop` branch.
2. Add PaddleDetection RT-DETR-R18 as a documented fallback, not a required
   phase-one benchmark. Trigger it only if OpenMMLab compatibility or both
   selected detectors block delivery.
3. Keep YOLO26 outside the production dependency graph until the company makes
   a written AGPL/Enterprise decision. A speed number may be collected in an
   isolated environment without copying code into the product.
4. Use camera-specific higher-resolution inference first. Add SAHI-style tiles
   only for cameras/slices with measured small-object recall failure, with
   boundary de-duplication tests.
5. Add duplicate/near-duplicate auditing before split freeze.
6. Compare ByteTrack against a camera-motion-compensated association method on
   a small sequence only if ego-motion is a measured failure mode.
7. Use Grounding DINO and SAM 2.1 strictly as offline disagreement/recovery
   teachers during early cycles. Store their proposals separately from human
   annotations.

## Required changes before detector implementation

The following must land on `main` before WP-03 or WP-04 begins:

1. A decision record defining 928 keyframes versus 4,640 images per camera,
   timestamp matching, synchronization tolerance, and whether intermediate
   frames are labeled, tracking-only, or excluded.
2. An authoritative eight-versus-seven-class taxonomy decision, class ID map,
   and rules for `Sign` and the three truck categories.
3. A canonical camera ID/order table derived from source data, not UI order,
   with orientation, expected resolution, and missing-camera policy.
4. A deterministic sequence/split specification with run boundaries, exclusion
   gap, near-duplicate audit, and frozen test manifest.
5. A double-reviewed gold-set protocol including explicit verified-empty
   images and difficult/rare/class-camera slices.
6. An executable evaluation policy defining matching IoU, confidence intervals,
   sample minima, per-class safety floors, timing protocol, and regression
   tolerance.
7. A pinned Label Studio version and a synthetic/sanitized round-trip fixture
   covering prediction, modification, deletion, creation, skip, and verified
   empty.
8. An RTX 4060 capability record and smoke-test matrix. Results are generated
   artifacts and must not be committed if they expose machine or company data;
   sanitized hardware metadata may be committed.
9. A dependency/license register covering code, model weights, and transitive
   runtime components. Explicitly exclude RF-DETR Plus and Ultralytics
   production use until approved.

WP-01 and WP-02 may proceed in parallel only after the decisions in items 1-3
are written. Model training remains blocked until items 1-7 are complete.

## Recommended ownership for WP-01 through WP-08

| Work package | Primary owner | Reviewer / supporting owner | Boundary |
| --- | --- | --- | --- |
| WP-01 Data Contract and Validator | Fengyiovo | Colleague | Fengyiovo owns schemas, parsers, manifest and split generator; colleague writes review cases for 5x camera cadence, leakage and ordering, then reviews. |
| WP-02 Annotation Policy and Gold Set | Colleague | Fengyiovo | Colleague owns taxonomy decision draft, ambiguity matrix, gold-set sampling and QA protocol; Fengyiovo owns camera ignore polygons and validates policy against source data. Both independently review the same sample, but only one edits the policy document at a time. |
| WP-03 RTMDet Baseline | Fengyiovo | Colleague | Fengyiovo owns the OpenMMLab environment, configs and adapter; colleague reviews reproducibility, metrics contract and Label Studio output. |
| WP-04 RF-DETR Challenger | Colleague | Fengyiovo | Colleague owns the pinned RF-DETR environment, adapter and same-split report; Fengyiovo verifies parity with WP-03 inputs and evaluation. |
| WP-05 Evaluation and Selection Harness | Colleague | Fengyiovo | Colleague owns matching, slicing, calibration and scorecard code; Fengyiovo validates expected outputs against RTMDet and reviews acceptance policy changes. |
| WP-06 Temporal Refinement | Colleague | Fengyiovo | Colleague owns ByteTrack, resets, consensus and optional SAM spike; Fengyiovo reviews detector/tracker boundaries and runs independent sequence QA. |
| WP-07 Human-in-the-Loop Integration | Fengyiovo | Colleague | Fengyiovo owns Label Studio sync and active-learning orchestration; colleague owns round-trip contract tests and audits review-state semantics. |
| WP-08 Durable Service | Fengyiovo | Colleague | Fengyiovo owns FastAPI/control-plane and deployment; colleague owns GPU-worker locking, restart/idempotency tests and model-registry validation. Reconfirm this split only after the CLI pipeline passes. |

This allocation gives each detector and its environment a separate owner,
places the common evaluator with the challenger owner to reduce baseline bias,
and keeps the existing Label Studio work with Fengyiovo. Shared decisions use
review, not simultaneous edits to the same file.

## Questions requiring a joint decision

1. Is the v1 product labeling 5,568 synchronized keyframe images or all 27,840
   camera images? If only keyframes, what exact mapping selects them?
2. Is `Sign` an authoritative eighth class? If yes, where are its reviewed
   examples and how is it represented in existing seven-class exports/models?
3. What are the authoritative six camera IDs and display order, and are any
   stored images rotated or mirrored?
4. How many independent collection runs/days/routes exist? Can one be reserved
   as a genuinely independent test set?
5. Which classes are safety-critical, and what per-class missed-object ceiling
   applies instead of only overall recall?
6. What IoU threshold(s), confidence level, and minimum sample count define a
   passing 98% auto-accept slice?
7. Is automatic acceptance required in the first month, or is the first release
   allowed to provide reviewed predictions only?
8. Which exact RTX 4060 variant and VRAM are available, and is overnight
   training acceptable?
9. Has the company approved Apache-2.0 dependencies and model weights, and who
   owns the written decision for AGPL-3.0 and PML 1.0 components?
10. Will Label Studio Community remain the supported review interface, and what
    correction-time threshold triggers the CVAT spike?
11. May intermediate high-rate camera frames be used for tracking when they do
    not have human labels, and how will propagation quality be audited?

## Primary evidence checked independently

- [MMDetection repository and Apache-2.0 license](https://github.com/open-mmlab/mmdetection)
- [RF-DETR repository, model table, release guidance, and split licensing](https://github.com/roboflow/rf-detr)
- [RF-DETR releases](https://github.com/roboflow/rf-detr/releases)
- [Ultralytics YOLO26 licensing](https://docs.ultralytics.com/models/yolo26)
- [PaddleDetection repository and Apache-2.0 license](https://github.com/PaddlePaddle/PaddleDetection)
- [ByteTrack repository and MIT license](https://github.com/FoundationVision/ByteTrack)
- [SAM 2 / SAM 2.1 repository and Apache-2.0 license](https://github.com/facebookresearch/sam2)
- [Label Studio prediction integration](https://labelstud.io/guide/ml)
- [Label Studio task and prediction import format](https://labelstud.io/guide/tasks)
- [Label Studio annotation export and percentage coordinates](https://labelstud.io/guide/export)
- [Label Studio local storage](https://labelstud.io/guide/storage_local)
