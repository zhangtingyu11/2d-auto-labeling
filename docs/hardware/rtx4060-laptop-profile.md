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
