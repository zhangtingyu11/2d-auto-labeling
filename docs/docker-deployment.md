# Docker and Linux Deployment

The Python source is operating-system independent. Windows `.bat` and
PowerShell files are workstation launchers only; they are not required inside
the company container.

## Runtime contract

The company GPU image must provide:

- Python 3.11 or newer;
- an NVIDIA driver/CUDA/PyTorch combination approved by the company;
- `rfdetr==1.8.3` and this repository installed as a Python package;
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

The inference CLI accepts paths as arguments and therefore does not depend on
Windows drive letters:

```bash
python tools/infer_rfdetr_image_folder.py \
  --checkpoint /models/checkpoint_best_total.pth \
  --image-root /data/images \
  --output-dir /output/run-001 \
  --rf-home /cache/rf-home \
  --dataset-name mine-day-001 \
  --label-studio-root /data/images \
  --minimum-long-side-px 70 \
  --model-version mining8-v44-70px
```

## Company-image adaptation

Do not hard-code a public CUDA base image until the company confirms its NVIDIA
driver, CUDA runtime, PyTorch version, internal package mirror, proxy policy,
and whether outbound downloads are allowed. Build from the company's approved
GPU base image and pin its digest in the deployment repository.

Before promotion, run a clean-container smoke test and compare the checkpoint
SHA-256 and a small sanitized prediction fixture against the Windows reference.
Numerical output may differ slightly across CUDA kernels, but class mapping,
70-pixel filtering, ordering, and box counts at the chosen operating point must
remain within the documented tolerance.
