"""Run queued K-fold web jobs without exposing Docker to the web service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from car5_autolabel.integrations.label_studio_project_api import LabelStudioProjectApi
from car5_autolabel.integrations.label_studio_training_upload import (
    convert_training_export,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--assignment-template", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--docker-image", default="car5-rfdetr:v4.4-70px-train-user")
    parser.add_argument("--gpu", type=int, action="append", dest="gpus")
    parser.add_argument("--label-studio-url", default="http://127.0.0.1:8090")
    parser.add_argument("--label-studio-project-id", type=int, default=3)
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


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _input_digest(coco_path: Path, assignments_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (coco_path, assignments_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _log_message(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().astimezone().isoformat()}] {message}\n")


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


def _label_studio_api(args: argparse.Namespace) -> LabelStudioProjectApi:
    token = os.environ.get("CAR5_LABEL_STUDIO_API_TOKEN", "")
    if not token:
        raise ValueError("CAR5_LABEL_STUDIO_API_TOKEN is required by the K-fold worker")
    return LabelStudioProjectApi(base_url=args.label_studio_url, token=token)


def _prepare_label_studio_source(
    args: argparse.Namespace,
    metadata_path: Path,
    metadata: dict[str, Any],
    job_directory: Path,
) -> dict[str, Any]:
    source_mode = metadata.get("source_mode", "legacy_upload")
    if source_mode != "label_studio_api":
        for required in ("human.coco.json", "assignments.csv"):
            if not (job_directory / required).is_file():
                raise FileNotFoundError(job_directory / required)
        return metadata
    if (
        metadata.get("input_digest")
        and (job_directory / "human.coco.json").is_file()
        and (job_directory / "assignments.csv").is_file()
    ):
        return metadata
    project_id = int(metadata.get("label_studio_project_id", 0))
    if project_id != args.label_studio_project_id:
        raise ValueError("queued job Label Studio project differs from the fixed worker project")
    log_path = job_directory / "runner.log"
    metadata.update(
        {
            "state": "exporting",
            "export_started_at": datetime.now().astimezone().isoformat(),
        }
    )
    _write(metadata_path, metadata)
    _log_message(log_path, f"读取 Label Studio 项目 {project_id} 的已提交标注")
    snapshot = _label_studio_api(args).snapshot(project_id)
    export_path = job_directory / "label_studio_export.json"
    _write_json(export_path, snapshot.tasks)
    metadata["state"] = "preparing"
    _write(metadata_path, metadata)
    summary = convert_training_export(
        label_studio_export=export_path,
        assignment_template=args.assignment_template,
        coco_output=job_directory / "human.coco.json",
        assignments_output=job_directory / "assignments.csv",
        summary_output=job_directory / "conversion_summary.json",
    )
    metadata.update(
        {
            "conversion": asdict(summary),
            "input_digest": _input_digest(
                job_directory / "human.coco.json",
                job_directory / "assignments.csv",
            ),
            "export_finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    _write(metadata_path, metadata)
    _log_message(
        log_path,
        (
            f"标注快照完成：任务 {summary.task_count}，人工框 {summary.human_box_count}，"
            f"删除提示 {summary.removed_non_human_result_count}"
        ),
    )
    return metadata


def _current_label_studio_input_digest(
    args: argparse.Namespace,
    job_directory: Path,
) -> str:
    validation = job_directory / "pre_sync_validation"
    validation.mkdir(parents=True, exist_ok=True)
    snapshot = _label_studio_api(args).snapshot(args.label_studio_project_id)
    export_path = validation / "label_studio_export.json"
    _write_json(export_path, snapshot.tasks)
    convert_training_export(
        label_studio_export=export_path,
        assignment_template=args.assignment_template,
        coco_output=validation / "human.coco.json",
        assignments_output=validation / "assignments.csv",
        summary_output=validation / "conversion_summary.json",
    )
    return _input_digest(
        validation / "human.coco.json",
        validation / "assignments.csv",
    )


def _sync_command(
    *,
    repository_root: Path,
    artifact_directory: Path,
    job_directory: Path,
    job_id: str,
    label_studio_url: str,
    label_studio_project_id: int,
) -> list[str]:
    return [
        sys.executable,
        str(repository_root / "tools/sync_kfold_audit_to_label_studio.py"),
        "--project-id",
        str(label_studio_project_id),
        "--artifact-root",
        str(artifact_directory),
        "--model-version",
        f"rfdetr-{job_id}",
        "--url",
        label_studio_url,
        "--backup",
        str(job_directory / "label_studio_before_sync.json"),
        "--report",
        str(job_directory / "sync_report.json"),
        "--workers",
        "1",
        "--reuse-backup",
        "--apply",
    ]


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
            if _read(metadata_path).get("state") in {
                "queued",
                "exporting",
                "preparing",
                "syncing",
                "interrupted",
            }:
                result.append(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return result


def process_job(args: argparse.Namespace, metadata_path: Path) -> None:
    metadata = _read(metadata_path)
    job_directory = _safe_job_directory(args.jobs_root, metadata_path, metadata)
    log_path = job_directory / "runner.log"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.repository_root / "src")
    phase = "preparing"
    try:
        metadata = _prepare_label_studio_source(
            args, metadata_path, metadata, job_directory
        )
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
        phase = "training"
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
        metadata.pop("pid", None)
        metadata["return_code"] = return_code
        if return_code != 0:
            metadata.update(
                {
                    "state": "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                }
            )
            _write(metadata_path, metadata)
            return

        phase = "syncing"
        metadata.update(
            {
                "state": "syncing",
                "sync_started_at": datetime.now().astimezone().isoformat(),
            }
        )
        _write(metadata_path, metadata)
        _log_message(log_path, "训练完成；重新读取 Label Studio，检查人工标注是否变化")
        current_digest = _current_label_studio_input_digest(args, job_directory)
        if current_digest != metadata.get("input_digest"):
            metadata.update(
                {
                    "state": "stale",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": "训练期间人工标注发生变化；为避免旧审计提示覆盖新标注，未回写",
                    "current_input_digest": current_digest,
                }
            )
            _write(metadata_path, metadata)
            _log_message(log_path, metadata["error"])
            return

        sync_command = _sync_command(
            repository_root=args.repository_root,
            artifact_directory=Path(metadata["artifact_directory"]),
            job_directory=job_directory,
            job_id=str(metadata["job_id"]),
            label_studio_url=args.label_studio_url,
            label_studio_project_id=args.label_studio_project_id,
        )
        with log_path.open("ab", buffering=0) as log_handle:
            sync_process = subprocess.Popen(
                sync_command,
                cwd=args.repository_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            sync_return_code = sync_process.wait()
        metadata = _read(metadata_path)
        metadata["sync_return_code"] = sync_return_code
        if sync_return_code != 0:
            metadata.update(
                {
                    "state": "sync_failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": "五折训练成功，但 Label Studio 自动回写失败；人工标注未被删除",
                }
            )
            _write(metadata_path, metadata)
            return
        sync_report_path = job_directory / "sync_report.json"
        sync_report = _read(sync_report_path) if sync_report_path.is_file() else {}
        metadata.update(
            {
                "state": "completed",
                "sync": sync_report,
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        _write(metadata_path, metadata)
        _log_message(log_path, "Label Studio 审计提示已自动更新")
    except Exception as error:
        try:
            metadata = _read(metadata_path)
            metadata.pop("pid", None)
            metadata.update(
                {
                    "state": "sync_failed" if phase == "syncing" else "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": str(error),
                }
            )
            _write(metadata_path, metadata)
            _log_message(log_path, f"任务失败：{error}")
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    args.jobs_root = args.jobs_root.resolve()
    args.assignment_template = args.assignment_template.resolve()
    args.image_root = args.image_root.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.repository_root = args.repository_root.resolve()
    if args.poll_seconds <= 0 or args.label_studio_project_id <= 0:
        raise ValueError("poll seconds must be positive")
    if not args.assignment_template.is_file():
        raise FileNotFoundError(args.assignment_template)
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
