# Label Studio Integration Contract

Status: required before detector training output is imported

Pinned deployed version: Label Studio Community 1.23.0

## Identity Rules

- The external `image_id` is the integration key.
- Label Studio project, task, annotation, and result IDs are adapter metadata;
  they never define dataset order or identity.
- Import retries are idempotent by dataset release, image ID, model version,
  and prediction version.
- The project configuration hash and Label Studio version are recorded with an
  import/export run.

## Coordinate Rules

- Internal boxes are clipped pixel `xyxy`.
- Label Studio receives percentage `x`, `y`, `width`, and `height` using the
  original 1920 x 1080 dimensions unless the manifest says otherwise.
- `from_name`, `to_name`, label spelling, case, and active class IDs are
  validated before import.
- Round-trip pixel error after clipping must be at most one pixel per edge.
- Image rotation is zero for the v1 camera policy.

## Review-State Mapping

| Label Studio event | Canonical state |
| --- | --- |
| prediction displayed only | `unreviewed_prediction` |
| submitted unchanged prediction | `human_verified` |
| submitted resized or relabeled box | `human_modified` |
| submitted new box | `human_created` |
| submitted prediction deletion | `rejected_prediction` |
| submitted task with zero boxes | `verified_empty` |
| skipped task | unreviewed; never empty or verified |

Automatic suggestion acceptance is disabled for initial dataset cycles.

## Required Contract Fixture

Create a synthetic or sanitized fixture that exercises:

1. task creation with stable external ID;
2. prediction creation with model and dataset provenance;
3. unchanged submission;
4. resize and relabel;
5. human-created box;
6. deleted prediction;
7. submitted empty task;
8. skipped task;
9. duplicate import retry;
10. export back to canonical annotations.

The test asserts task count, class mapping, coordinates, provenance, review
states, and byte-stable canonical ordering. It must run against the pinned
Community edition before WP-03 or WP-04 predictions enter a real project.

## Storage and Secrets

Local images are exposed through the approved Label Studio document root and
registered local storage. Repository code receives paths through configuration.
API tokens, databases, local absolute data paths, exports, and images stay out
of Git.
