# Mainstream 2D Auto-Labeling Landscape

Research date: 2026-07-16

## Scope

The target is a self-hosted, pure 2D bounding-box auto-labeling system for the
car5 dataset. The expected complete set is 928 timestamps and six cameras,
which should produce 5,568 images when every camera is present. The ingestion
validator must verify that count rather than assume it.

The production workstation is an NVIDIA RTX 4060. VRAM must be detected at
runtime because desktop and laptop variants differ. The solution must work in
WSL2/Docker, preserve company data locally, and integrate with the team's
existing Label Studio workflow.

## Evaluation Principles

A model is not selected from public COCO results alone. Each candidate must be
trained and evaluated on one immutable car5 split using the same annotation
policy, image set, hardware, precision mode, and timing method.

The decision order is:

1. Annotation correctness and missed-object rate.
2. Human correction time per image.
3. Stability, reproducibility, and license suitability.
4. Throughput and GPU memory.
5. Public benchmark score.

## Detector Candidates

| Candidate | Strengths | Risks | Project decision |
| --- | --- | --- | --- |
| MMDetection RTMDet-S/M | Mature OpenMMLab stack, Apache 2.0, strong real-time trade-off, official MMDetection plus Label Studio example, familiar to the team | Older dependency stack and MMCV/CUDA compatibility need a pinned image | Primary baseline and first complete pipeline |
| RF-DETR-S/M | Modern DETR family, strong published accuracy/latency trade-off, simple fine-tuning API, active 2026 releases, Apache-designated package and weights | Faster-moving package, recent breaking changes, custom-domain and RTX 4060 behavior must be proven locally | Accuracy challenger; promote only after the same-split gate |
| Ultralytics YOLO26-S/M | Excellent developer experience, export tooling, built-in BoT-SORT/ByteTrack, strong speed | AGPL-3.0 or Enterprise licensing; company distribution/service use needs an approved license | Optional speed benchmark only; not a default dependency |
| PaddleDetection RT-DETR-R18/R50 | Official end-to-end real-time DETR, TensorRT path, mature published benchmark | Adds Paddle as a second training/deployment ecosystem and increases maintenance | Reference fallback, not phase-one implementation |
| DINO/Co-DETR teacher | High offline accuracy and useful disagreement signal | Slow and memory-heavy on an RTX 4060 | Optional offline teacher after baseline |

## Open-Vocabulary and Segmentation Assistance

Grounding DINO accepts image and text prompts and is useful for discovering
objects missed by a closed-set detector. SAM 2.1 supports prompted image masks
and propagation through video frames. Grounded-SAM-2 combines open-vocabulary
detection and temporal mask propagation.

These models are not the production source of truth for the eight car5
classes. They are used for:

- cold-start proposals for rare classes;
- detector-disagreement review queues;
- finding likely missing annotations;
- optional mask-to-box propagation through short, continuous sequences.

Open-vocabulary class ambiguity is expected for `Truck`, `WaterTruck`, and
`BoxTruck`. Their outputs must remain review candidates until measured on the
car5 gold set.

## Tracking Candidates

ByteTrack is selected for the first tracking implementation. It associates
high- and low-score detection boxes, is detector-agnostic, MIT licensed, and is
simple enough to tune and audit. Tracking is per camera only in v1.

SAM 2.1 propagation is a later challenger for difficult occlusion and box
jitter. It is heavier and should not block the detector plus ByteTrack MVP.

## Small-Object Strategy

The system first uses native aspect-ratio inference at a larger long side. If
small or distant targets still dominate false negatives, SAHI-style overlapping
tile inference is enabled only for affected cameras or frames. Tiling is not
the default because it multiplies latency and can duplicate boxes near tile
boundaries.

## Annotation Platform Options

| Platform | Evidence | Decision |
| --- | --- | --- |
| Label Studio | Existing team data, API and prediction format, ML backend support, Apache 2.0 | Keep for MVP and production review unless measured workflow limits remain |
| CVAT Community | Strong image/video workflow, track mode, keyframe interpolation, automatic annotation, MIT core | Run a small migration spike only if sequential review time remains the main bottleneck |
| New custom annotation canvas | Full control | Rejected for MVP; high cost and avoidable QA risk |

The project builds its own orchestration, model, tracking, policy, evaluation,
and Label Studio adapters. It does not rebuild a bounding-box editor in phase
one.

## Dataset Curation

FiftyOne is a useful optional inspection tool for duplicates, uniqueness,
hardness, and possible annotation mistakes. It is not part of the production
runtime. The core pipeline must still provide deterministic validators and
machine-readable reports without requiring FiftyOne.

## Final Recommendation

1. Build a complete RTMDet-M baseline through MMDetection and Label Studio.
2. In parallel, benchmark RF-DETR-S/M on the identical split.
3. Select the production detector using the project acceptance scorecard.
4. Add ByteTrack and temporal consensus after the single-frame baseline is
   trustworthy.
5. Use Grounding DINO and SAM 2.1 only as offline proposal/recovery teachers.
6. Add camera-specific ignore regions and calibration before any automatic
   acceptance.
7. Keep Label Studio for review; evaluate CVAT only against measured review
   time, not preference.

## Primary Sources

- MMDetection repository and license:
  https://github.com/open-mmlab/mmdetection
- MMDetection RTMDet plus Label Studio guide:
  https://mmdetection.readthedocs.io/en/main/user_guides/label_studio.html
- MMDetection inference guide:
  https://mmdetection.readthedocs.io/en/v3.3.0/user_guides/inference.html
- RF-DETR repository, model table, and license split:
  https://github.com/roboflow/rf-detr
- RF-DETR releases:
  https://github.com/roboflow/rf-detr/releases
- Ultralytics YOLO26 model and license options:
  https://docs.ultralytics.com/models/yolo26
- Ultralytics tracking mode:
  https://docs.ultralytics.com/modes/track
- PaddleDetection RT-DETR implementation:
  https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/rtdetr
- Grounding DINO:
  https://github.com/IDEA-Research/GroundingDINO
- SAM 2.1:
  https://github.com/facebookresearch/sam2
- Grounded-SAM-2:
  https://github.com/IDEA-Research/Grounded-SAM-2
- ByteTrack:
  https://github.com/FoundationVision/ByteTrack
- SAHI:
  https://github.com/obss/sahi
- Label Studio predictions and ML integration:
  https://labelstud.io/guide/predictions.html
  https://labelstud.io/guide/ml
- CVAT automatic annotation and track mode:
  https://docs.cvat.ai/docs/manual/advanced/automatic-annotation/
  https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/
- FiftyOne Brain dataset curation:
  https://docs.voxel51.com/brain/index.html
