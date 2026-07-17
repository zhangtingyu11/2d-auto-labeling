# Pure 2D Auto-Label V1 User Guide

Status: local review-assistance release candidate

## What It Does

V1 reads the six camera JPG folders and produces six-class 2D bounding-box
proposals for `Car`, `Truck`, `Bulldozer`, `Excavator`, `WaterTruck`, and
`Sign`. It does not read point clouds, calibration files, 3D boxes, or any
MMDetection3D project code.

Every prediction records the model version, training-config SHA-256,
checkpoint SHA-256, dataset-manifest SHA-256, source, score, and review state.

## Run the Full Reviewed Keyframe Set

From the repository root:

```powershell
.\run_v1_autolabel.bat "C:\path\to\car5_20260611_day_02"
```

Custom output, threshold, and batch size are available through PowerShell:

```powershell
.\run_v1_autolabel.ps1 `
  -DatasetRoot "C:\path\to\car5_20260611_day_02" `
  -Output ".\artifacts\v1_predictions\predictions.jsonl" `
  -ScoreThreshold 0.2 `
  -BatchSize 32
```

Outputs:

- `predictions.jsonl`: canonical, deterministic pixel-space predictions;
- `predictions.jsonl.run.json`: model hashes, counts, failures, and throughput;
- `predictions.jsonl.labelstudio.json`: Label Studio task import with
  pre-annotations and provenance.

Re-running the same command resumes the job and skips existing `image_id`
rows. It never appends a duplicate for a completed image.

## Label Studio

The exported task JSON assumes Label Studio local files are exposed with the
standard `/data/local-files/?d=` prefix and that the document root contains
the dataset's `camera/` directory. Import the generated
`*.labelstudio.json` as tasks into a six-class rectangle-label project.

All V1 suggestions remain `unreviewed_prediction`. A human must submit every
task. Do not interpret a task with no predicted boxes as a verified empty
image.

## Measured Local Behavior

- True batch inference tested at batch sizes 8 and 32.
- Batch 32 processed all 5,568 keyframe-camera images at 28.42 images/second
  after model initialization.
- Resume test skipped 64 of 64 existing rows with zero duplicates.
- End-to-end launcher test produced predictions and Label Studio tasks with
  zero failures.
- Full fold-0 model evaluation: mAP50:95 0.126 and mAP50 0.246.

The full 5,568-image run completed in 195.9 seconds after model initialization,
or about 3 minutes 16 seconds. Hardware load and disk cache can change this
estimate; short smoke runs understate steady-state throughput.

## Quality Boundary

The current RTMDet-Tiny checkpoint is an initial usable proposal model, not an
automatic-label acceptance model. Fold 0 measured good relative performance
for `Truck` but weak `Excavator`, `Sign`, and small-object performance. Three
classes are absent from that validation sequence and therefore not evaluable
on the fold.

At the default 0.2 score threshold, fold-0 class-aware matching measured 29.0%
precision and 36.5% recall overall. `Truck` measured 84.7% precision and 58.2%
recall, while `Excavator` and `Sign` remained inadequate. Lowering the threshold
increases candidate recall and deletion work; raising it reduces false boxes
but increases manual drawing.

Every V1 box requires human review. RTMDet-S, five-fold evaluation, threshold
calibration, RF-DETR, and ByteTrack remain V1.1 work and must not be implied by
the V1 name.

## Rollback

Generated output and weights are outside Git. To roll back code, switch to the
local V1 commit recorded at handoff. To roll back predictions, select an older
output directory; never overwrite a reviewed Label Studio export in place.
