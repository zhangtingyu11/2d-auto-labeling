# RTMDet-Tiny Fold-0 Speed Probe

Date: 2026-07-17

Status: rejected as the formal v1 detector; retained as a lightweight speed
probe only

## Scope

- Model: MMDetection RTMDet-Tiny, COCO-pretrained
- Input: aspect-ratio-preserving 640 x 640 canvas
- Development fold: leave sequence `s00` out
- Train: 4,182 reviewed images, 3,305 boxes
- Validation: 1,386 reviewed images, 944 boxes
- Hardware: RTX 4060 Laptop GPU, 8,188 MiB
- Precision: automatic mixed precision
- Epochs: 40; best checkpoint: epoch 38
- Peak training memory reported by MMEngine: 2,781 MiB

This run used RTMDet's aggressive reference Mosaic, MixUp, crop, and flip
pipeline. Those choices were not approved as the formal mining-domain
augmentation policy, so the result must not be promoted directly.

## Best Validation Result

| Metric | Value |
| --- | ---: |
| COCO mAP50:95 | 0.126 |
| COCO mAP50 | 0.246 |
| COCO mAP75 | 0.099 |
| Small-object mAP | 0.000 |
| Medium-object mAP | 0.019 |
| Large-object mAP | 0.172 |

Supported class results on fold 0:

| Class | mAP50:95 | mAP50 | Evidence |
| --- | ---: | ---: | --- |
| Truck | 0.358 | 0.651 | 581 validation boxes |
| Excavator | 0.001 | 0.003 | 293 validation boxes |
| Sign | 0.019 | 0.085 | 70 validation boxes |
| Car | not evaluable | not evaluable | no fold-0 validation boxes |
| Bulldozer | not evaluable | not evaluable | no fold-0 validation boxes |
| WaterTruck | not evaluable | not evaluable | no fold-0 validation boxes |

## Decision

Reject RTMDet-Tiny as the formal baseline because heavy-equipment, sign, and
small-object performance is inadequate. Continue with the approved RTMDet-S
conservative baseline and five-fold evaluation. Keep Tiny only as a later
throughput reference or emergency low-resource deployment candidate.

