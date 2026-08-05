# RF-DETR V4.4 70-Pixel Release

This release packages reproducible source and policy metadata for the current
mining 2D auto-labeling operating point. Company images, annotations, Label
Studio state, generated evaluations, and model weights are deliberately not in
Git.

## Environment

Create Python 3.11 or newer environment, install a CUDA-compatible PyTorch
build approved for the target machine, and then run:

```bash
python -m pip install -e .
python -m pip install -r requirements-rfdetr.txt
```

The verified workstation used `rfdetr==1.8.3`. The model registry metadata and
checkpoint SHA-256 are in
`configs/models/rfdetr/mining8_v44_70px.json`.

## Build an Internal Handoff Archive

The checkpoint must not be committed to Git. Build a company-internal transfer
archive from the exact committed source plus the verified checkpoint:

```bash
python tools/package_rfdetr_handoff.py \
  --checkpoint /models/checkpoint_best_total.pth \
  --output /handoff/car5-rfdetr-v44-70px.zip
```

The command refuses a checkpoint whose SHA-256 differs from the model registry.
The resulting ZIP includes the committed source, policies, documentation,
weight, and `MODEL_RELEASE.json`; it includes no images or annotations. Transfer
this ZIP only through a company-approved internal channel.

## Inference

```bash
python tools/infer_rfdetr_image_folder.py \
  --checkpoint /models/checkpoint_best_total.pth \
  --image-root /data/images \
  --output-dir /output/run-001 \
  --rf-home /cache/rf-home \
  --dataset-name mine-day-001 \
  --label-studio-root /data/images \
  --threshold 0.20 \
  --minimum-long-side-px 70 \
  --model-version mining8-v44-70px
```

The inference summary records the checkpoint hash, dataset-manifest hash,
thresholds, and the number of boxes rejected by the 70-pixel policy.

## Evaluation

The formal default is 70 pixels. Both ground truth and predictions are filtered
with the same original-image rule.

```bash
python tools/evaluate_size_policy_grid.py \
  --gt /evaluation/ground_truth.coco.json \
  --raw-predictions /evaluation/predictions.coco.json \
  --model-name mining8-v44-70px \
  --size-thresholds 70 \
  --output-dir /output/evaluation-70px
```

Report metrics with and without `Sign`. Do not publish the current 296-image
calibration result as an independent final-test score.

## Deployment

The Python entry points use CLI paths and run on Windows or Linux. Windows
PowerShell launchers are optional workstation conveniences. For company Docker
deployment, build `Dockerfile.rfdetr`, follow `docs/docker-deployment.md`, and
mount data and weights at runtime. The Docker image deliberately contains
neither the checkpoint nor any company images.
