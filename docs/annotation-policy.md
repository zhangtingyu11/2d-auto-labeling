# car5 v1 2D Annotation Policy

Status: implementation draft for double review

Binding versions: `taxonomy_v1`, `annotation_v1`, `camera_policy_v1`

## 1. Scope and Ground-Truth Principle

The supervised v1 set contains the 5,568 keyframe-camera images defined by ADR
0001. Each annotation describes what a reviewer can justify from the pixels in
that image. LiDAR projection, a detector, a tracker, another camera, or an
adjacent frame may suggest where to inspect, but none may make an invisible
object ground truth in the current image.

An object behind a wall or completely occluded is not labeled. An object with a
small visible part is labeled only when its visible box meets the minimum-size
rule and its active class is visually stable. This prevents 3D projection from
creating the apparent "foreknowledge" observed in earlier experiments.

## 2. Active Classes

| ID | Class | Include when | Main confusion rule |
| ---: | --- | --- | --- |
| 0 | `Car` | A passenger or light road vehicle is visually identifiable. | Do not use for heavy site trucks. |
| 1 | `Truck` | A heavy road truck is visible, or its subtype is uncertain. | This is the fallback for an uncertain `WaterTruck`. |
| 2 | `Bulldozer` | The blade and body identify a bulldozer. | Do not infer it from tracks or wheels alone. |
| 3 | `Excavator` | Boom, dipper, bucket, or sufficient body geometry identifies it. | A generic construction silhouette without class evidence is unresolved. |
| 4 | `WaterTruck` | A tank or spraying system is visible and distinguishes it from `Truck`. | If the tank/sprayer is not visible, label `Truck`. |
| 5 | `Sign` | An external traffic or worksite sign is visible. | Host-vehicle text, logos, lamps, and scene text are not signs. |

`Pedestrian` and `BoxTruck` are inactive candidates. They have no v1 class ID,
are not detector heads, and must not be turned into positive training examples.
If a reviewer observes one, record a candidate observation outside the released
box set and escalate it for the promotion process in section 10.

When two active classes remain genuinely ambiguous, do not guess. Mark the
object `needs_adjudication`; unresolved boxes are excluded from a training
release.

## 3. Box Geometry

- Draw a tight axis-aligned box around visible object pixels only.
- Do not hallucinate the hidden extent behind another object, dust, a wall, or
  the image boundary.
- Clip every box to the 1920 by 1080 image bounds.
- A target continuing beyond the image boundary is retained and marked
  `truncated=true`.
- The v1 consistency floor is 8 pixels in both width and height. A smaller
  object is `too_small_for_consistent_label` and is not added to ground truth.
- Shadows, dust clouds, reflections, and projected 3D extents are not part of
  the box.

The 8-pixel floor is a versioned consistency rule, not a claim that smaller
objects are unimportant. Candidate models may still surface them for policy
review; changing the floor requires a new annotation-policy version and a
re-audit of existing labels.

## 4. Occlusion and Truncation

Occlusion describes pixels hidden by another foreground object or dense dust:

- `none`: at least 90% of the estimated object is visible;
- `partial`: at least 50% but less than 90% is visible;
- `heavy`: less than 50% is visible, but the class is still visually stable;
- completely occluded: no box.

Truncation is independent of occlusion. Set `truncated=true` when the object
continues outside the image. A box may be both occluded and truncated.

## 5. Host Vehicle and Ignore Regions

The recording vehicle/bodywork visible at image edges is never a target. Each
camera has a versioned polygon in `configs/cameras/camera_policy_v1.yaml`.
External objects that overlap or cross this region remain valid targets.

A prediction is ignored as host vehicle only when at least 60% of its box area
overlaps a reviewed ignore polygon. Reviewers may override this for an external
object. Empty polygon lists mean "not reviewed" rather than "no host vehicle";
the dataset release is blocked until Fengyiovo supplies polygons from real
images and both developers approve them. Coordinates must never be guessed.

## 6. Empty, Skipped, and Deleted Predictions

- `verified_empty`: a human inspected the complete image and explicitly
  submitted it with zero active-class boxes.
- skipped or unopened task: unreviewed; never empty and never ground truth.
- displayed prediction only: `unreviewed_prediction`.
- submitted unchanged prediction: `human_verified`.
- resized or relabeled prediction: `human_modified`.
- new human box: `human_created`.
- deleted prediction: preserve its provenance as `rejected_prediction`; do not
  silently erase the audit event.

An empty detector result is never evidence that an image is empty.

## 7. Prediction and Temporal Assistance

Detector, Grounding DINO, 3D projection, and tracker outputs are proposals. They
may focus attention or recover a candidate, but a reviewer must confirm its
current-image pixels. Propagated boxes cannot cross camera, natural-sequence,
missing-frame, or scene-cut boundaries. The first release has no automatic
acceptance.

## 8. Gold-Set Selection

The frozen gold set contains at least 720 image tasks and always samples whole
six-camera timestamp groups; therefore it contains at least 120 timestamps.

Selection procedure:

1. Start with deterministic, sequence-stratified sampling across `s00` to
   `s04` and all six cameras.
2. Add targeted groups for rare classes, small/distant boxes, truncation,
   occlusion, glare, dust, host-vehicle overlap, and verified-empty cases.
3. Store only stable image IDs, SHA-256 values, sequence/camera metadata,
   sampling reasons, and release IDs in Git-safe manifests. Store no image,
   absolute path, personal name, or raw export in Git.
4. Freeze the sorted manifest and its SHA-256 before independent review.
5. Replacing an image creates a new gold-set version; do not edit a frozen
   manifest in place.

Every camera must contribute reviewed empty images and difficult examples. If
a rare class lacks enough examples, record `insufficient_evidence`; do not fill
the quota with adjacent near-duplicates and claim broader coverage.

## 9. Double Review and Disagreement Resolution

Two developers independently annotate the frozen tasks. During the first pass,
neither sees the other's labels, detector source, confidence, or proposed
resolution. Matching uses identical class and IoU at least 0.50.

Log every unmatched box, class mismatch, geometry mismatch, attribute mismatch,
empty/skip mismatch, host-ignore disagreement, and policy ambiguity. Each log
entry records both decisions, a policy category, resolution, policy reference,
resolver alias, and timestamp. Do not record personal names.

After policy clarification, repeat the affected review. Agreement must be at
least 98% at box/class decision level. Report box presence, class, geometry,
and attributes separately so high aggregate agreement cannot hide systematic
class errors. Every disagreement category must be resolved or explicitly
marked `release_blocking`.

## 10. Candidate-Class Promotion

Promoting `Pedestrian`, `BoxTruck`, or a new class requires all of:

1. written visual rules with positive and confusion examples;
2. a reviewed positive sample covering relevant cameras and sizes;
3. a new taxonomy version and explicit class-ID migration;
4. revalidation of Label Studio config, exports, model heads, and metrics;
5. review and merge through the shared repository.

Teacher-model predictions alone never satisfy this process.

## 11. Release Checklist

- taxonomy and annotation-policy versions match the manifest;
- all six camera ignore polygons are double reviewed;
- no inactive or unknown class appears in released boxes;
- no unresolved or `release_blocking` disagreement remains;
- skipped tasks are absent from ground truth;
- verified-empty tasks carry explicit human submission evidence;
- boxes meet geometry, bounds, visibility, and provenance rules;
- gold manifest order and checksum are deterministic;
- two-reviewer agreement report reaches the 98% gate.
