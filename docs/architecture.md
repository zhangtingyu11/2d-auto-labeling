# Architecture

## Components

1. **API service** accepts image or batch inference requests.
2. **Detector adapter** isolates model-specific preprocessing and inference.
3. **Tracking adapter** associates detections across ordered frames.
4. **Label Studio integration** converts detections to and from percentage
   rectangle coordinates.
5. **Dataset tools** validate image identity, class mapping, ordering, and
   train/validation splits.

## Core Rule

Detection, tracking, annotation storage, and UI integration are separate
boundaries. Replacing a detector must not change the annotation schema or API.

## Initial Data Flow

```text
camera images
    -> detector
    -> optional temporal association
    -> confidence/class filters
    -> Label Studio predictions
    -> human review
    -> validated training export
```

## Coordinate Convention

Internally, boxes use pixel-space `xyxy`. Label Studio conversion happens only
at the integration boundary. API responses include image width and height so
conversions are deterministic.
