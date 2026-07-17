from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

CONDA_PYTHON = Path("/opt/conda/envs/car5-2d/bin/python")
MMDET_TRAIN = Path(
    "/opt/conda/envs/car5-2d/lib/python3.10/site-packages/"
    "mmdet/.mim/tools/train.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pinned car5 RTMDet training stage.")
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg-options", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "configs" / "models" / "rtmdet" / f"car5_rtmdet_s_{args.stage}.py"
    command = [
        str(CONDA_PYTHON),
        str(MMDET_TRAIN),
        str(config),
        "--work-dir",
        str(args.work_dir),
    ]
    if args.amp:
        command.append("--amp")
    if args.resume:
        command.append("--resume")
    if args.cfg_options:
        command.extend(["--cfg-options", *args.cfg_options])

    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    print(" ".join(command), flush=True)
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
