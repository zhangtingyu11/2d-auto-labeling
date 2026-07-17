# ADR 0001: v1 Data Scope and Timebase

Status: accepted

Date: 2026-07-16

## Context

The source has 4,640 images per camera but only 928 human-labeled keyframes.
The final reviewed 2D export contains all six cameras for those 928 keyframes.

## Decision

The v1 supervised dataset is exactly the 5,568 reviewed keyframe images:

`928 keyframes x 6 cameras = 5,568 images`

The full 27,840-image camera stream is retained as external source data. Its
22,272 non-keyframe images have the role `temporal_context` and must not be
treated as ground truth, negative images, or evaluation examples.

Each manifest row includes:

```text
dataset_id
image_id
relative_path
sha256
width
height
capture_timestamp
raw_sequence_index
keyframe_id (nullable)
is_keyframe
sync_delta_ms
camera_id
camera_order
sequence_id
fold_id
annotation_status
```

The filename stem is parsed as a capture timestamp. It is not used as the sole
identity; `dataset_id`, timestamp, and camera ID form the source identity, and
the manifest assigns a stable `image_id`.

## Matching Rules

- A keyframe is defined by the authoritative 3D label stem.
- The v1 camera match is an exact same-stem JPEG in each camera directory.
- All 928 keyframes currently have six exact matches.
- A missing camera image fails supervised release validation.
- Context matching uses source order and timestamp, records `sync_delta_ms`,
  and never silently substitutes a context frame for a keyframe.
- Duplicate paths, hashes, source identities, or image IDs fail validation.

## Temporal Context Rules

- Context frames may be read only by temporal modules.
- They inherit the sequence and fold of neighboring keyframes.
- Propagated boxes remain predictions until human review.
- A later fully reviewed 27,840-image release requires a new dataset version;
  it is not an in-place expansion of v1.

## Evidence

See `docs/audits/2026-07-16-car5-v1-data-audit.md`.
