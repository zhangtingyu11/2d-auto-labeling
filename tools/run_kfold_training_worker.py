"""Run queued K-fold web jobs without exposing Docker to the web service."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--docker-image", default="car5-rfdetr:v4.4-70px-train-user")
    parser.add_argument("--gpu", type=int, action="append", dest="gpus")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--once", action="store_true", help="Process at most one job, then exit")
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"job metadata is not an object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_job_directory(jobs_root: Path, metadata_path: Path, metadata: dict[str, Any]) -> Path:
    job_id = str(metadata.get("job_id", ""))
    job_directory = metadata_path.parent.resolve()
    if (
        not job_id
        or job_directory.parent != jobs_root.resolve()
        or job_directory.name != job_id
    ):
        raise ValueError(f"invalid queued job identity: {metadata_path}")
    return job_directory


def _command(
    *,
    repository_root: Path,
    image_root: Path,
    docker_image: str,
    gpu_indices: tuple[int, ...],
    workspace_root: Path,
    artifact_root: Path,
    job_directory: Path,
    metadata: dict[str, Any],
) -> list[str]:
    job_id = str(metadata["job_id"])
    workspace = workspace_root / job_id
    artifacts = artifact_root / "runs" / job_id
    if Path(metadata["workspace_directory"]).resolve() != workspace:
        raise ValueError("queued job workspace is outside the fixed workspace root")
    if Path(metadata["artifact_directory"]).resolve() != artifacts:
        raise ValueError("queued job artifacts are outside the fixed artifact root")
    command = [
        sys.executable,
        str(repository_root / "tools/run_rfdetr_kfold_label_audit.py"),
        "--source",
        f"reviewed1085={job_directory / 'human.coco.json'}::{image_root}",
        "--assignments",
        str(job_directory / "assignments.csv"),
        "--workspace",
        str(workspace),
        "--artifact-root",
        str(artifacts),
        "--docker-image",
        docker_image,
        "--batch-size",
        "32",
        "--grad-accum-steps",
        "1",
        "--epochs",
        "80",
        "--early-stopping-patience",
        "15",
        "--idle-confirmations",
        "1",
        "--poll-seconds",
        "5",
    ]
    for gpu in gpu_indices:
        command.extend(["--gpu", str(gpu)])
    return command


def queued_jobs(jobs_root: Path) -> list[Path]:
    result = []
    for metadata_path in sorted(jobs_root.glob("*/job.json")):
        try:
            if _read(metadata_path).get("state") == "queued":
                result.append(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return result


def process_job(args: argparse.Namespace, metadata_path: Path) -> None:
    metadata = _read(metadata_path)
    job_directory = _safe_job_directory(args.jobs_root, metadata_path, metadata)
    command = _command(
        repository_root=args.repository_root,
        image_root=args.image_root,
        docker_image=args.docker_image,
        gpu_indices=tuple(args.gpus or (3, 4)),
        workspace_root=args.workspace_root,
        artifact_root=args.artifact_root,
        job_directory=job_directory,
        metadata=metadata,
    )
    log_path = job_directory / "runner.log"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.repository_root / "src")
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.repository_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        metadata.update(
            {
                "state": "running",
                "started_at": datetime.now().astimezone().isoformat(),
                "pid": process.pid,
            }
        )
        _write(metadata_path, metadata)
        return_code = process.wait()
    metadata = _read(metadata_path)
    metadata.update(
        {
            "state": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    _write(metadata_path, metadata)


def main() -> None:
    args = parse_args()
    args.jobs_root = args.jobs_root.resolve()
    args.image_root = args.image_root.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.repository_root = args.repository_root.resolve()
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    args.jobs_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.jobs_root / "worker.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another K-fold training worker is already running") from error
        while True:
            jobs = queued_jobs(args.jobs_root)
            if jobs:
                process_job(args, jobs[0])
                if args.once:
                    return
            elif args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
