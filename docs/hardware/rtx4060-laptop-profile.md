# Available GPU Profile

Recorded: 2026-07-16

## Sanitized Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Reported memory: 8,188 MiB
- Windows NVIDIA driver: 566.07
- Runtime target: WSL2 distribution `mmdet3d-env`
- Review UI: Label Studio Community 1.23.0

This record establishes capacity only. It does not prove that a particular
model, resolution, batch size, or tracker fits.

## Required Probe Order

1. Record CUDA, PyTorch, MMEngine, MMCV, MMDetection, precision, and image size.
2. Smoke-test RTMDet-S with AMP at long side 640.
3. Increase to 768 while keeping the model and augmentation fixed.
4. Probe the largest safe micro-batch and record allocated and reserved VRAM.
5. Add gradient accumulation only after the micro-batch is stable.
6. Test RTMDet-M, 960 input, tiling, and RF-DETR-M separately; change one
   capacity variable at a time.
7. Reserve at least 512 MiB and never run detector training and SAM video state
   concurrently on this GPU during the first spike.

Generated logs may not contain company paths or images. Commit only sanitized
benchmark summaries and reproducible configuration IDs.
## Verified Environment

- WSL distribution: `mmdet3d-env` on Ubuntu 22.04
- isolated Conda environment: `/opt/conda/envs/car5-2d`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- PyTorch: 2.3.1 with CUDA 11.8
- MMDetection: 3.3.0
- MMCV: 2.1.0
- MMEngine: 0.10.7

The original base environment is not modified. `car5-2d` is a clone dedicated
to the pure 2D tool.

## Throughput Strategy

Training images are staged under `/opt/car5-data/car5_v1/images` so JPEG reads
come from the WSL ext4 filesystem instead of `/mnt/d`. The initial RTMDet-S
profile uses 640 x 640 augmentation, AMP, batch size 16, eight persistent data
workers, and pinned memory. Smoke testing records
peak VRAM and iteration time before the 1 to 2 hour pilot starts.

GPU utilization is an observation, not a correctness target. Batch size is
raised only while retaining at least 512 MiB VRAM reserve and stable data
loading. Long training may compare 640 native-resize and 960/tiled inference
after the pilot confirms the data path and learning direction.

`/opt/car5-workspace` is an untracked WSL symlink to the Windows workspace. It
keeps runtime configuration ASCII-only while source data and generated COCO
files remain in their original Windows locations.

## Measured Smoke Result

The batch-16 run used about 5.8 GiB total GPU memory and left more than 2 GiB
available. Batch 8 was rejected as unnecessarily conservative. Gradient
accumulation and clipping were removed after AMP smoke logs showed non-finite
diagnostic gradient norms despite finite losses. The held-out metric after
epoch 2 is the next learning gate; a zero-learning run is stopped immediately.

## RF-DETR-S Pilot Profile

RF-DETR uses a separate `/opt/conda/envs/car5-rfdetr` environment with
PyTorch 2.5.1+cu118 and `rfdetr[train]==1.8.3`. The official 32.1M-parameter
RF-DETR-S checkpoint was MD5-validated before training.

Batch 4 reached about 7.7 GiB but processed the smoke set more slowly than batch
2. The retained profile is batch 2, gradient accumulation 8, eight persistent
workers, FP16 AMP, EMA, and a 512 base resolution with bounded multiscale
training. During the full pilot, sampled GPU utilization was typically 68% to
97% and memory use reached about 6.5 GiB. Six train-and-validation epochs took
4,817 seconds.

The optimized inference path measured 40.24 ms median per serial image over 20
warm calls, about 24.85 FPS. See
`docs/experiments/2026-07-17-rfdetr-s-pilot-s00.md` for accuracy and gate
results.
