# RF-DETR-S Pilot on s00

Date: 2026-07-17

Status: completed; challenger remains active, long training is blocked on a
better class-support plan.

## Reproducible Setup

- Model: official RF-DETR-S, 32.1M parameters, 512 base resolution.
- Package: `rfdetr[train]==1.8.3`.
- Runtime: isolated `car5-rfdetr` environment with PyTorch 2.5.1+cu118.
- Fold: the same natural-sequence `s00` split used by RTMDet-S.
- Train: 4,182 images and 3,305 boxes.
- Validation: 1,386 images and 944 boxes.
- Micro-batch: 2; gradient accumulation: 8; effective batch: 16.
- AMP: FP16; EMA enabled; eight persistent workers.
- Pilot duration: 4,817 seconds for six epochs, including validation.

The run configuration is `configs/models/rfdetr/car5_rfdetr_s.yaml`. Generated
weights, logs, images, and annotations remain outside Git.

## Measured Result

| Epoch | mAP50:95 | mAP50 | EMA mAP50:95 | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1458 | 0.3399 | **0.1490** | 0.3338 |
| 2 | 0.1216 | 0.3014 | 0.1310 | 0.3207 |
| 3 | 0.1230 | 0.2861 | 0.1252 | 0.3227 |
| 4 | 0.1190 | 0.2750 | 0.1263 | 0.3217 |
| 5 | 0.1238 | 0.2730 | 0.1324 | 0.2953 |
| 6 | 0.1293 | 0.2955 | 0.1313 | 0.3268 |

The deployable pilot winner is the epoch-1 EMA checkpoint, selected as
`checkpoint_best_total.pth` by the trainer.

At epoch 1, per-class AP50:95 was 0.3414 for Truck, 0.0959 for Excavator, and
0.0001 for Sign. Warm optimized inference on one 512-resolution image measured
40.24 ms median, or 24.85 FPS, over 20 serial calls. This is a local pipeline
measurement, not an end-to-end six-camera service benchmark.

## RTMDet-S Comparison

The RTMDet-S early baseline on the identical `s00` validation images reached
mAP50:95 0.1290 and mAP50 0.2790 at epoch 2. RF-DETR-S therefore improved the
early overall metrics to 0.1458 and 0.3399. RTMDet-S remained slightly stronger
on Truck AP50:95 (0.359 versus 0.341), while RF-DETR-S was substantially better
on Excavator (0.096 versus 0.023).

## Findings and Gate Decision

1. RF-DETR-S is a viable challenger and currently has the better overall early
   metric on the controlled split.
2. Batch 4 reached about 7.7 GiB but reduced image throughput. Batch 2 reached
   roughly 6.5 GiB during the full run and sustained high GPU utilization, so
   batch 2 is the retained RTX 4060 profile.
3. The best score at epoch 1 followed by degradation indicates that the pilot
   learning rate is too aggressive for this small domain dataset.
4. `s00` train contains only 9 Sign boxes while validation contains 70. It is a
   useful sequence stress test but cannot independently qualify a final
   six-class model.
5. Long training is not authorized from this run. The next controlled test is
   `pilot_tuned`, with lower detector and encoder learning rates and early
   stopping. Model selection must aggregate all five sequence folds; the final
   release model is then retrained on all reviewed data.
