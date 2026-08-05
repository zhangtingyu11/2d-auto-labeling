# Docker and Linux Deployment

The Python source is operating-system independent. Windows `.bat` and
PowerShell files are workstation launchers only; they are not required inside
the company container.

## Build the RF-DETR image

The repository includes `Dockerfile.rfdetr`. Its default base is the official
PyTorch 2.12 / CUDA 13.0 runtime image. A company-approved compatible PyTorch
image can be substituted with `--build-arg BASE_IMAGE=...`.

```bash
docker build -f Dockerfile.rfdetr -t car5-rfdetr:v4.4-70px .
```

The `.dockerignore` explicitly excludes checkpoints, images, annotations,
databases, generated outputs, and common data directories. The model weight is
mounted at runtime rather than baked into the image.

## Runtime contract

The company GPU host must provide:

- NVIDIA Container Toolkit and a driver compatible with the selected base;
- the Dockerfile's pinned `rfdetr==1.8.3` runtime;
- read-only mounts for images and the model checkpoint;
- a writable mount for outputs and RF-DETR cache files.

Do not copy images, annotations, credentials, or model weights into the Docker
image or Git repository. Mount them at runtime.

Example container paths:

```text
/workspace/car5-2d-auto-labeling  source checkout
/data/images                      read-only input images
/models/checkpoint_best_total.pth read-only V4.4 checkpoint
/output                           generated predictions
/cache/rf-home                    RF-DETR cache
```

Run the image with the checkpoint and a different image set mounted read-only:

```bash
docker run --rm --gpus all --shm-size=2g \
  -v /host/images:/data/images:ro \
  -v /host/models:/models:ro \
  -v /host/output:/output \
  -v /host/rf-cache:/cache/rf-home \
  car5-rfdetr:v4.4-70px \
  --checkpoint /models/checkpoint_best_total.pth \
  --image-root /data/images \
  --output-dir /output/run-001 \
  --rf-home /cache/rf-home \
  --dataset-name mine-day-001 \
  --label-studio-root /data/images \
  --minimum-long-side-px 70 \
  --model-version mining8-v44-70px
```

The input images do not need to be the original test set. They only need to be
ordinary supported image files arranged below `/data/images`; camera
subdirectories are recommended because the output records the parent folder as
the camera name.

## Company-image adaptation

Before production promotion, confirm the company NVIDIA driver, internal image
mirror, proxy policy, and outbound package-download policy. Mirror and pin the
approved base image by digest in the company's deployment repository.

Before promotion, run a clean-container smoke test and compare the checkpoint
SHA-256 and a small sanitized prediction fixture against the Windows reference.
Numerical output may differ slightly across CUDA kernels, but class mapping,
70-pixel filtering, ordering, and box counts at the chosen operating point must
remain within the documented tolerance.
