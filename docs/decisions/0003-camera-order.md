# ADR 0003: Canonical Camera Order
stored images. Polygons are versioned configuration, not annotations.
stored images. Polygons are versioned configuration, not annotations.
Status: accepted

Date: 2026-07-16

## Decision

The canonical v1 camera IDs and display order are:

| Order | Camera ID | Expected size | Stored rotation |
| ---: | --- | ---: | ---: |
| 0 | `front` | 1920 x 1080 | 0 degrees |
| 1 | `front_left` | 1920 x 1080 | 0 degrees |
| 2 | `front_right` | 1920 x 1080 | 0 degrees |
| 3 | `back` | 1920 x 1080 | 0 degrees |
| 4 | `back_left` | 1920 x 1080 | 0 degrees |
| 5 | `back_right` | 1920 x 1080 | 0 degrees |

This order matches all 928 six-camera groups in the final reviewed export. It
is independent of filesystem alphabetical order and Label Studio task IDs.

No software rotation or mirroring is applied in v1. A future orientation
change requires a camera-policy version bump and box-coordinate migration.

## Missing-Camera Policy

- A missing keyframe camera fails a supervised dataset release.
- A missing temporal-context image is recorded and breaks temporal continuity;
  it is never replaced silently.
- Tracking state resets on camera changes, natural sequence gaps, or missing
  context.

## Pending Camera Work

Host-vehicle ignore polygons are not inferred from calibration. WP-02 must
draw and independently review one polygon per camera using representative
stored images. Polygons are versioned configuration, not annotations.
