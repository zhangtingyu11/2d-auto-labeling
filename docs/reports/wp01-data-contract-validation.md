# WP-01 Data Contract Validation

Status: passed on 2026-07-17

Branch: `dev/fengyiovo`

## Delivered

- Pydantic contracts for manifest rows, canonical boxes, review states, and
  validation reports.
- Deterministic car5 scanner with the accepted six-camera order.
- Stable image IDs based on dataset, timestamp, and camera.
- Natural sequence assignment from timestamp gaps and `s00` through `s04`
  development fold IDs.
- Optional SHA-256 hashing and actual image decoding/dimension checks.
- Unicode-safe image decoding for Windows paths.
- Label Studio export conversion with strict identity, class, coordinate,
  dimension, rotation, duplicate, skipped-task, and explicit-empty checks.
- CLI entry points for manifest creation and reviewed-export validation.

## Real Data Result

The source-tree structural audit passed:

| Check | Result |
| --- | ---: |
| Cameras | 6 |
| Source images per camera | 4,640 |
| Source images total | 27,840 |
| Keyframe timestamps | 928 |
| Reviewed keyframe images | 5,568 |
| Temporal-context images | 22,272 |
| Natural sequences | 5 |
| Structural errors | 0 |

The strict reviewed-keyframe release audit decoded all 5,568 images, verified
1920 x 1080 dimensions, and produced SHA-256 hashes with zero errors and zero
warnings.

The final Label Studio export passed against the strict manifest:

| Check | Result |
| --- | ---: |
| Export tasks | 5,568 |
| Reviewed tasks | 5,568 |
| Unreviewed or skipped tasks | 0 |
| Verified-empty images | 2,790 |
| Boxes | 4,249 |
| Export errors | 0 |
| Export warnings | 0 |

Box counts by active class:

| Class | Boxes |
| --- | ---: |
| Car | 62 |
| Truck | 3,143 |
| Bulldozer | 35 |
| Excavator | 846 |
| WaterTruck | 84 |
| Sign | 79 |

## Reproduction

Strict reviewed-keyframe manifest:

```powershell
car5-autolabel build-manifest `
  --data-root "<car5-source-root>" `
  --output "<private-output>/car5_v1_manifest.jsonl" `
  --report "<private-output>/car5_v1_manifest_report.json" `
  --verify-dimensions
```

Reviewed Label Studio export:

```powershell
car5-autolabel validate-label-studio `
  --export "<private-output>/label-studio-final.json" `
  --manifest "<private-output>/car5_v1_manifest.jsonl" `
  --output "<private-output>/car5_v1_annotations.json" `
  --report "<private-output>/car5_v1_annotations_report.json"
```

Full temporal-context manifests use `--include-context`. Hashes and image
decoding may be skipped for a fast local audit, but a release artifact must use
the strict command above.

## Automated Verification

- `ruff check .`: passed.
- `pytest`: 11 passed.
- Company images, annotations, hashes, and local paths remain outside Git.

## Release Boundary

This validates dataset identity and reviewed annotations. It does not authorize
automatic acceptance. The v1 product remains review-assistance only until the
independent test release required by ADR 0004 exists.
