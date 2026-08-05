# ADR 0005: Formal 70-Pixel Long-Side Policy

Status: accepted for the current mining 2D auto-labeling operating point

Date: 2026-08-04

## Decision

Formal ground truth, formal model output, and reported operating-point metrics
use the same rule on the original image coordinates:

```text
max(round(box_width_px), round(box_height_px)) >= 70
```

Boxes below the boundary may be retained in a separate human-review candidate
queue, but they are not formal ground truth or formal predictions. Model
confidence, temporal support, truncation, and occlusion do not silently override
the rule.

Metrics are reported both with and without `Sign` because the current `Sign`
ground truth has known consistency limitations.

## Rationale

The 50/60/70-pixel grid on the 296-image strong-deduplication calibration set
showed similar aggregate F1 across these boundaries. The team selected 70
pixels to improve annotation consistency and suppress tiny ambiguous boxes.
This is an operating policy, not evidence that sub-70-pixel objects are
irrelevant to safety.

## Consequences

- Existing training and evaluation releases must be re-audited before claiming
  conformance to `annotation_v2_70px`.
- The 296-image set has influenced this choice and is now calibration data, not
  an untouched final acceptance test.
- A new independent collection is required for a final unbiased claim.
