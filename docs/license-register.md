# Dependency and Model License Register

Status: implementation allowlist for phase one

This file is an engineering register, not legal advice. Exact package versions,
source commits, transitive dependencies, checkpoint URLs, and hashes must be
recorded in each reproducible run manifest.

| Component | Intended use | Current disposition |
| --- | --- | --- |
| MMDetection / RTMDet | Primary detector | Allowed for phase-one spike; Apache-2.0 project license |
| MMEngine / MMCV | RTMDet runtime | Allowed after compatible versions are pinned and transitive licenses captured |
| RF-DETR open package and Apache-designated S/M weights | Detector challenger | Allowed after exact release and weight hashes are pinned |
| RF-DETR Plus XL/2XL components | Detector challenger | Excluded pending written PML review |
| Ultralytics YOLO26 package | Speed reference | Excluded from production dependency graph pending AGPL/Enterprise decision |
| PaddleDetection RT-DETR | Fallback detector | Allowed fallback; do not add a third runtime unless trigger criteria are met |
| ByteTrack | Temporal proposal | Allowed spike; MIT project license, transitive implementation still reviewed |
| SAM 2.1 Tiny/Small | Optional propagation teacher | Allowed offline spike after checkpoint hash and VRAM profile |
| Grounding DINO / Grounded-SAM-2 | Offline disagreement teacher | Review-only output; pin exact implementations and transitive licenses |
| SAHI | Conditional tiled inference | Add only after measured small-object failure and dependency review |
| Label Studio Community 1.23.0 | Human review UI | Pinned deployed integration target |

No benchmark result overrides a license exclusion. Do not copy tracker or model
code from a differently licensed distribution merely because the algorithm
name is the same.
