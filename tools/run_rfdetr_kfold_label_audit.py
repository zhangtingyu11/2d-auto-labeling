"""Prepare and serially run leakage-safe RF-DETR K-fold annotation audits."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from car5_autolabel.kfold_audit import (
    clip_ground_truth_boxes,
    enforce_v1_active_taxonomy,
    filter_training_frames,
    parse_nvidia_smi_csv,
    select_idle_gpu,
)

SAFE_SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Source:
    name: str
    coco: Path
    image_root: Path


@dataclass(frozen=True)
class Assignment:
    package: str
    source_image_id: int
    source_dataset: str
    camera: str
    timestamp: datetime
    validation_fold: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=COCO_JSON::IMAGE_ROOT",
        help=("input package in NAME=COCO_JSON::IMAGE_ROOT form; repeat once per package"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="CSV assigning each source image to validation_fold",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="prepared fold datasets and runner lock; must share a filesystem with source images",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="write checkpoints and reports here; defaults to workspace",
    )
    parser.add_argument("--folds", type=int, default=5, help="number of folds (default: 5)")
    parser.add_argument(
        "--only-fold",
        type=int,
        help="zero-based fold to run; defaults to all folds and a combined OOF report",
    )
    parser.add_argument(
        "--minimum-long-side-px",
        type=int,
        default=70,
        help="drop a whole frame if any GT long side is below this many pixels (default: 70)",
    )
    parser.add_argument(
        "--purge-seconds",
        type=float,
        default=3.0,
        help="exclude training frames within this many seconds of validation (default: 3.0)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare and validate fold datasets, then exit without using Docker or a GPU",
    )
    parser.add_argument(
        "--docker-image",
        default="car5-rfdetr:v4.4-70px-eval",
        help="RF-DETR training image name or digest (default: car5-rfdetr:v4.4-70px-eval)",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        help="initial checkpoint unseen by audited images; defaults to model initialization",
    )
    parser.add_argument(
        "--model", default="medium", help="RF-DETR model size passed to training (default: medium)"
    )
    parser.add_argument(
        "--taxonomy",
        default="car5-v1-six-class",
        help="taxonomy version recorded in provenance (default: car5-v1-six-class)",
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="maximum epochs per fold (default: 80)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="force one epoch and isolate outputs under runs_smoke",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="epochs without improvement before stopping (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="images per device step (default: 4)"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="device steps accumulated before an optimizer step (default: 4)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="training data-loader workers (default: 0)"
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.001,
        help="low inference score cutoff retained for COCO mAP curves (default: 0.001)",
    )
    parser.add_argument(
        "--operating-score-threshold",
        type=float,
        default=0.20,
        help="score cutoff used only for TP/FP/FN and error candidates (default: 0.20)",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.70,
        help="same-class NMS IoU used only at the audit operating point (default: 0.70)",
    )
    parser.add_argument(
        "--cross-class-nms-iou-threshold",
        type=float,
        default=0.95,
        help="merge nearly identical cross-class audit boxes and report conflict (default: 0.95)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        action="append",
        dest="gpus",
        help="allowed physical GPU index; repeat for more, or omit to allow all visible GPUs",
    )
    parser.add_argument(
        "--minimum-free-memory-mib",
        type=int,
        default=22000,
        help="minimum free memory required before acquiring a GPU (default: 22000 MiB)",
    )
    parser.add_argument(
        "--maximum-gpu-utilization",
        type=int,
        default=5,
        help="maximum utilization considered idle (default: 5 percent)",
    )
    parser.add_argument(
        "--idle-confirmations",
        type=int,
        default=3,
        help="consecutive idle checks required before a fold starts (default: 3)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="seconds between GPU idle checks; allowed range 1-60 (default: 60)",
    )
    return parser.parse_args()


def parse_source(value: str) -> Source:
    try:
        name, paths = value.split("=", 1)
        coco, image_root = paths.split("::", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid --source {value!r}; expected NAME=COCO_JSON::IMAGE_ROOT"
        ) from error
    source = Source(
        name=name.strip(), coco=Path(coco).resolve(), image_root=Path(image_root).resolve()
    )
    if (
        not SAFE_SOURCE_NAME.fullmatch(source.name)
        or source.name in {".", ".."}
        or not source.coco.is_file()
        or not source.image_root.is_dir()
    ):
        raise argparse.ArgumentTypeError(f"invalid or missing source path in {value!r}")
    return source


def parse_timestamp(value: str) -> datetime:
    for pattern in ("%Y%m%d_%H%M%S_%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass
    raise ValueError(f"unsupported assignment timestamp: {value!r}")


def load_assignments(path: Path, folds: int) -> dict[tuple[str, int], Assignment]:
    result: dict[tuple[str, int], Assignment] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = Assignment(
                package=row["package"],
                source_image_id=int(row["source_image_id"]),
                source_dataset=row["source_dataset"],
                camera=row["camera"],
                timestamp=parse_timestamp(row["timestamp"]),
                validation_fold=int(row["validation_fold"]),
            )
            if not 0 <= item.validation_fold < folds:
                raise ValueError(f"fold out of range for {item}")
            key = (item.package, item.source_image_id)
            if key in result:
                raise ValueError(f"duplicate assignment: {key}")
            result[key] = item
    return result


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise RuntimeError(f"existing split image does not match source: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        raise RuntimeError(
            f"cannot hard-link {source} to {destination}; place --workspace on the same filesystem"
        ) from error


def build_split(
    destination: Path,
    records: list[tuple[Source, dict[str, Any], list[dict[str, Any]]]],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    for next_image_id, (source, image, source_annotations) in enumerate(records, start=1):
        source_relative = Path(str(image["file_name"]))
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError(f"unsafe COCO file_name: {image['file_name']!r}")
        source_image = (source.image_root / source_relative).resolve()
        if not source_image.is_relative_to(source.image_root):
            raise ValueError(f"COCO image escapes source root: {image['file_name']!r}")
        relative = Path("images") / source.name / source_relative
        hardlink(source_image, destination / relative)
        images.append(
            {
                **image,
                "id": next_image_id,
                "file_name": relative.as_posix(),
                "source_package": source.name,
                "source_image_id": int(image["id"]),
            }
        )
        for annotation in source_annotations:
            annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                }
            )
            next_annotation_id += 1
    json_write(
        destination / "_annotations.coco.json",
        {"info": {}, "images": images, "annotations": annotations, "categories": categories},
    )
    category_name = {int(category["id"]): str(category["name"]) for category in categories}
    class_box_counts: dict[str, int] = defaultdict(int)
    for annotation in annotations:
        class_box_counts[category_name[int(annotation["category_id"])]] += 1
    return {
        "images": len(images),
        "boxes": len(annotations),
        "class_box_counts": dict(sorted(class_box_counts.items())),
    }


def prepare(args: argparse.Namespace, sources: list[Source]) -> dict[str, Any]:
    preparation_config = {
        "schema_version": 4,
        "filter_policy_version": "strict-float-70-v2-clipped-gt",
        "folds": args.folds,
        "minimum_long_side_px": args.minimum_long_side_px,
        "purge_seconds": args.purge_seconds,
        "assignments": {
            "path": str(args.assignments.resolve()),
            "sha256": file_sha256(args.assignments.resolve()),
        },
        "sources": [
            {
                "name": source.name,
                "coco": str(source.coco),
                "image_root": str(source.image_root),
                "coco_sha256": file_sha256(source.coco),
            }
            for source in sources
        ],
    }
    ready = args.workspace / "plan" / "_READY.json"
    if ready.is_file():
        existing = json.loads(ready.read_text(encoding="utf-8"))
        if existing.get("preparation_config") != preparation_config:
            raise RuntimeError(
                "workspace was prepared with different inputs or policy; use a new --workspace"
            )
        expected_annotations = [
            args.workspace / "folds" / f"fold_{fold}" / "dataset" / split / "_annotations.coco.json"
            for fold in range(args.folds)
            for split in ("train", "valid", "test")
        ]
        missing = [str(path) for path in expected_annotations if not path.is_file()]
        if missing:
            raise RuntimeError(f"prepared workspace is incomplete; missing: {missing}")
        print(f"reuse prepared folds: {ready}", flush=True)
        return existing

    assignments = load_assignments(args.assignments.resolve(), args.folds)
    categories: list[dict[str, Any]] | None = None
    records: list[tuple[Source, dict[str, Any], list[dict[str, Any]], Assignment]] = []
    source_reports: dict[str, Any] = {}
    for source in sources:
        coco = json.loads(source.coco.read_text(encoding="utf-8-sig"))
        coco, clipped_boxes = clip_ground_truth_boxes(coco)
        filtered, report = filter_training_frames(
            coco, minimum_long_side_px=args.minimum_long_side_px
        )
        filtered, taxonomy_report = enforce_v1_active_taxonomy(filtered)
        current_categories = filtered.get("categories", [])
        if categories is None:
            categories = current_categories
        elif current_categories != categories:
            raise ValueError(f"category definitions differ in source {source.name!r}")
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in filtered["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image in filtered["images"]:
            image_id = int(image["id"])
            assignment = assignments.get((source.name, image_id))
            if assignment is None:
                raise ValueError(f"missing assignment for {(source.name, image_id)}")
            image_path = source.image_root / str(image["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append((source, image, annotations_by_image[image_id], assignment))
        source_reports[source.name] = {
            "size_and_empty_filter": asdict(report),
            "v1_taxonomy_filter": asdict(taxonomy_report),
            "clipped_gt_boxes": clipped_boxes,
        }

    fold_reports: dict[str, Any] = {}
    for fold in range(args.folds):
        valid = [record for record in records if record[3].validation_fold == fold]
        valid_times: dict[str, list[datetime]] = defaultdict(list)
        for record in valid:
            valid_times[record[3].source_dataset].append(record[3].timestamp)
        train = []
        purged = 0
        for record in records:
            assignment = record[3]
            if assignment.validation_fold == fold:
                continue
            nearby = valid_times[assignment.source_dataset]
            if any(
                abs((assignment.timestamp - valid_time).total_seconds()) <= args.purge_seconds
                for valid_time in nearby
            ):
                purged += 1
                continue
            train.append(record)
        fold_root = args.workspace / "folds" / f"fold_{fold}" / "dataset"
        train_report = build_split(
            fold_root / "train",
            [(source, image, annotations) for source, image, annotations, _ in train],
            categories or [],
        )
        valid_records = [(source, image, annotations) for source, image, annotations, _ in valid]
        valid_report = build_split(fold_root / "valid", valid_records, categories or [])
        test_report = build_split(fold_root / "test", valid_records, categories or [])
        fold_reports[str(fold)] = {
            "train": train_report,
            "valid": valid_report,
            "test": test_report,
            "purged_near_validation": purged,
        }

    report = {
        "folds": args.folds,
        "minimum_long_side_px": args.minimum_long_side_px,
        "purge_seconds": args.purge_seconds,
        "eligible_images": len(records),
        "source_filter_reports": source_reports,
        "fold_reports": fold_reports,
        "assignments": str(args.assignments.resolve()),
        "preparation_config": preparation_config,
        "data_manifest_id": stable_hash(preparation_config),
    }
    json_write(ready, report)
    return report


def query_gpus() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def wait_for_idle_gpu(args: argparse.Namespace) -> tuple[int, TextIO]:
    confirmations = 0
    last_index: int | None = None
    while True:
        selected = select_idle_gpu(
            parse_nvidia_smi_csv(query_gpus()),
            minimum_free_memory_mib=args.minimum_free_memory_mib,
            maximum_utilization_percent=args.maximum_gpu_utilization,
            allowed_indices=set(args.gpus) if args.gpus else None,
        )
        if selected is None:
            confirmations, last_index = 0, None
            print("no idle GPU; waiting", flush=True)
        elif selected.index == last_index:
            confirmations += 1
        else:
            confirmations, last_index = 1, selected.index
        if selected is not None:
            print(
                f"GPU {selected.index}: free={selected.memory_free_mib} MiB, "
                f"util={selected.utilization_percent}%, idle check {confirmations}/"
                f"{args.idle_confirmations}",
                flush=True,
            )
            if confirmations >= args.idle_confirmations:
                candidates = parse_nvidia_smi_csv(query_gpus())
                remaining = set(args.gpus) if args.gpus else {status.index for status in candidates}
                while remaining:
                    candidate = select_idle_gpu(
                        candidates,
                        minimum_free_memory_mib=args.minimum_free_memory_mib,
                        maximum_utilization_percent=args.maximum_gpu_utilization,
                        allowed_indices=remaining,
                    )
                    if candidate is None:
                        break
                    lock = Path(f"/tmp/car5-kfold-gpu-{candidate.index}.lock").open(
                        "a", encoding="utf-8"
                    )
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        remaining.remove(candidate.index)
                        continue
                    current = select_idle_gpu(
                        parse_nvidia_smi_csv(query_gpus()),
                        minimum_free_memory_mib=args.minimum_free_memory_mib,
                        maximum_utilization_percent=args.maximum_gpu_utilization,
                        allowed_indices={candidate.index},
                    )
                    if current is not None:
                        return candidate.index, lock
                    lock.close()
                    remaining.remove(candidate.index)
                confirmations, last_index = 0, None
        time.sleep(args.poll_seconds)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def docker_base(
    args: argparse.Namespace,
    gpu: int | None,
    mounts: list[tuple[Path, str, bool]],
    *,
    environment: dict[str, str] | None = None,
) -> list[str]:
    container_user = os.environ.get("USER") or f"uid-{os.getuid()}"
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--ipc=host",
        "--entrypoint",
        "python",
    ]
    if gpu is not None:
        command.extend(["--gpus", f"device={gpu}"])
    container_environment = {
        "USER": container_user,
        "LOGNAME": container_user,
        **(environment or {}),
    }
    for name, value in container_environment.items():
        command.extend(["-e", f"{name}={value}"])
    for host, container, read_only in mounts:
        command.extend(["-v", f"{host.resolve()}:{container}{':ro' if read_only else ''}"])
    return [*command, args.docker_image]


def find_checkpoint(run_dir: Path) -> Path:
    preferred = [run_dir / "checkpoint_best_total.pth", run_dir / "checkpoint_best_ema.pth"]
    for checkpoint in preferred:
        if checkpoint.is_file():
            return checkpoint
    candidates = sorted(run_dir.rglob("checkpoint*.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint produced under {run_dir}")
    return candidates[-1]


def run_fold(args: argparse.Namespace, fold: int, repo_root: Path) -> None:
    fold_dataset = args.workspace / "folds" / f"fold_{fold}" / "dataset"
    runs_name = "runs_smoke" if args.smoke else "runs"
    run_dir = args.artifact_root / runs_name / f"fold_{fold}"
    success = run_dir / "_SUCCESS.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = args.artifact_root / "cache" / "rf-home"
    cache.mkdir(parents=True, exist_ok=True)
    mounts = [
        (fold_dataset, "/dataset", True),
        (run_dir, "/output", False),
        (cache, "/cache/rf-home", False),
    ]
    train_success = run_dir / "_TRAIN_SUCCESS.json"
    training_complete = train_success.is_file()
    training_config = {
        "docker_image": args.docker_image,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "early_stopping_patience": args.early_stopping_patience,
        "smoke": args.smoke,
        "taxonomy": args.taxonomy,
        "data_manifest_id": args.data_manifest_id,
        "pretrain_sha256": (
            file_sha256(args.pretrain_weights.resolve()) if args.pretrain_weights else None
        ),
    }
    training_config_hash = stable_hash(training_config)
    if training_complete:
        train_state = json.loads(train_success.read_text(encoding="utf-8"))
        if train_state.get("config_hash") != training_config_hash:
            raise RuntimeError(f"fold {fold} training config differs from its success sentinel")
        checkpoint = run_dir / train_state["checkpoint_relative"]
        if not checkpoint.is_file():
            raise RuntimeError(f"training sentinel points to missing checkpoint: {checkpoint}")
        if train_state.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise RuntimeError(f"fold {fold} checkpoint differs from its success sentinel")
        print(f"fold {fold} training already complete; reuse {checkpoint}", flush=True)
    inference_success = run_dir / "_INFERENCE_SUCCESS.json"
    inference_complete = inference_success.is_file()
    gpu_lock: TextIO | None = None
    gpu: int | None = None
    if not training_complete or not inference_complete:
        gpu, gpu_lock = wait_for_idle_gpu(args)
    if not training_complete:
        runtime_environment = {
            "HOME": "/cache/rf-home",
            "RF_HOME": "/cache/rf-home",
            "XDG_CACHE_HOME": "/cache/rf-home/.cache",
        }
        train_base = docker_base(args, gpu, mounts, environment=runtime_environment)
        train = list(train_base)
        train.extend(
            [
                "/workspace/car5-2d-auto-labeling/tools/train_rfdetr.py",
                "--model",
                args.model,
                "--dataset-dir",
                "/dataset",
                "--output-dir",
                "/output",
                "--rf-home",
                "/cache/rf-home",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum-steps",
                str(args.grad_accum_steps),
                "--num-workers",
                str(args.num_workers),
                "--checkpoint-interval",
                "1",
                "--eval-interval",
                "1",
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--release",
                f"kfold-label-audit-fold-{fold}",
                "--taxonomy",
                args.taxonomy,
                "--frozen-test",
                f"fold-{fold}",
            ]
        )
        if args.pretrain_weights:
            train_mount = args.pretrain_weights.resolve()
            pretrain_base = docker_base(
                args,
                gpu,
                [*mounts, (train_mount.parent, "/pretrain", True)],
                environment=runtime_environment,
            )
            train = pretrain_base + train[len(train_base) :]
            train.extend(["--pretrain-weights", f"/pretrain/{train_mount.name}"])
        if args.smoke:
            train.append("--smoke")
        resumable = sorted(run_dir.rglob("checkpoint*.pth"), key=lambda path: path.stat().st_mtime)
        if resumable:
            resume = resumable[-1]
            train.extend(["--resume", f"/output/{resume.relative_to(run_dir)}"])
    try:
        if not training_complete:
            run_logged(train, run_dir / "train.log")
            checkpoint = find_checkpoint(run_dir)
            checkpoint_sha256 = file_sha256(checkpoint)
            json_write(
                train_success,
                {
                    "checkpoint_relative": str(checkpoint.relative_to(run_dir)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "config_hash": training_config_hash,
                    "config": training_config,
                    "gpu": gpu,
                    "finished_at": datetime.now().isoformat(),
                },
            )
        else:
            checkpoint_sha256 = train_state["checkpoint_sha256"]
        inference_config = {
            "training_config_hash": training_config_hash,
            "checkpoint_sha256": checkpoint_sha256,
            "prediction_threshold": args.prediction_threshold,
            "minimum_long_side_px": args.minimum_long_side_px,
            "data_manifest_id": args.data_manifest_id,
        }
        inference_config_hash = stable_hash(inference_config)
        if inference_complete:
            inference_state = json.loads(inference_success.read_text(encoding="utf-8"))
            if inference_state.get("config_hash") != inference_config_hash:
                raise RuntimeError(
                    f"fold {fold} inference config differs from its success sentinel"
                )
            for output_name in (
                "oof_predictions_70px.coco.json",
                "oof_inference_summary.json",
            ):
                if not (run_dir / output_name).is_file():
                    raise RuntimeError(
                        f"inference sentinel exists but output is missing: {output_name}"
                    )
        if not inference_complete:
            inference = docker_base(
                args,
                gpu,
                [*mounts, (repo_root, "/audit-code", True)],
                environment={
                    "HOME": "/cache/rf-home",
                    "RF_HOME": "/cache/rf-home",
                    "XDG_CACHE_HOME": "/cache/rf-home/.cache",
                },
            )
            inference.extend(
                [
                    "/audit-code/tools/infer_rfdetr_coco.py",
                    "--checkpoint",
                    f"/output/{checkpoint.relative_to(run_dir)}",
                    "--split-dir",
                    "/dataset/valid",
                    "--output",
                    "/output/oof_predictions_70px.coco.json",
                    "--summary",
                    "/output/oof_inference_summary.json",
                    "--rf-home",
                    "/cache/rf-home",
                    "--threshold",
                    str(args.prediction_threshold),
                    "--minimum-long-side-px",
                    str(args.minimum_long_side_px),
                    "--batch-size",
                    str(args.batch_size),
                    "--config-hash",
                    inference_config_hash,
                    "--data-manifest-id",
                    args.data_manifest_id,
                ]
            )
            run_logged(inference, run_dir / "inference.log")
            json_write(
                inference_success,
                {
                    "checkpoint_relative": str(checkpoint.relative_to(run_dir)),
                    "config_hash": inference_config_hash,
                    "config": inference_config,
                    "finished_at": datetime.now().isoformat(),
                },
            )
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            gpu_lock.close()
            gpu_lock = None
        audit = docker_base(
            args,
            None,
            [*mounts, (repo_root, "/audit-code", True)],
            environment={"PYTHONPATH": "/audit-code/src"},
        )
        audit.extend(
            [
                "/audit-code/tools/audit_coco_predictions.py",
                "--ground-truth",
                "/dataset/valid/_annotations.coco.json",
                "--predictions",
                "/output/oof_predictions_70px.coco.json",
                "--metrics",
                "/output/oof_metrics_iou50.json",
                "--candidates",
                "/output/annotation_error_candidates.csv",
                "--iou-threshold",
                "0.5",
                "--operating-score-threshold",
                str(args.operating_score_threshold),
                "--nms-iou-threshold",
                str(args.nms_iou_threshold),
                "--cross-class-nms-iou-threshold",
                str(args.cross_class_nms_iou_threshold),
            ]
        )
        run_logged(audit, run_dir / "audit.log")
    except Exception as error:
        json_write(run_dir / "_FAILED.json", {"fold": fold, "gpu": gpu, "error": repr(error)})
        raise
    finally:
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            gpu_lock.close()
    json_write(
        success,
        {
            "fold": fold,
            "gpu": gpu,
            "training_config_hash": training_config_hash,
            "inference_config_hash": inference_config_hash,
            "checkpoint": str(checkpoint),
            "finished_at": datetime.now().isoformat(),
        },
    )


def combine_oof_results(args: argparse.Namespace, repo_root: Path) -> None:
    """Merge all fold validation GT/predictions and compute one overall OOF report."""

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] | None = None
    next_image_id = 1
    next_annotation_id = 1
    for fold in range(args.folds):
        ground_truth_path = (
            args.workspace
            / "folds"
            / f"fold_{fold}"
            / "dataset"
            / "valid"
            / "_annotations.coco.json"
        )
        predictions_path = (
            args.artifact_root / "runs" / f"fold_{fold}" / "oof_predictions_70px.coco.json"
        )
        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        fold_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        if categories is None:
            categories = ground_truth["categories"]
        elif categories != ground_truth["categories"]:
            raise RuntimeError(f"fold {fold} category definitions differ")
        image_id_map: dict[int, int] = {}
        for image in ground_truth["images"]:
            old_id = int(image["id"])
            image_id_map[old_id] = next_image_id
            images.append(
                {
                    **image,
                    "id": next_image_id,
                    "file_name": f"fold_{fold}/{image['file_name']}",
                }
            )
            next_image_id += 1
        for annotation in ground_truth["annotations"]:
            annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": image_id_map[int(annotation["image_id"])],
                }
            )
            next_annotation_id += 1
        for prediction in fold_predictions:
            predictions.append(
                {
                    **prediction,
                    "id": len(predictions) + 1,
                    "image_id": image_id_map[int(prediction["image_id"])],
                }
            )
    combined = args.artifact_root / "oof_combined"
    json_write(
        combined / "ground_truth.coco.json",
        {
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": categories or [],
        },
    )
    json_write(combined / "predictions_70px.coco.json", predictions)
    audit = docker_base(
        args,
        None,
        [(args.artifact_root, "/audit-output", False), (repo_root, "/audit-code", True)],
        environment={"PYTHONPATH": "/audit-code/src"},
    )
    audit.extend(
        [
            "/audit-code/tools/audit_coco_predictions.py",
            "--ground-truth",
            "/audit-output/oof_combined/ground_truth.coco.json",
            "--predictions",
            "/audit-output/oof_combined/predictions_70px.coco.json",
            "--metrics",
            "/audit-output/oof_combined/metrics.json",
            "--candidates",
            "/audit-output/oof_combined/annotation_error_candidates.csv",
            "--iou-threshold",
            "0.5",
            "--operating-score-threshold",
            str(args.operating_score_threshold),
            "--nms-iou-threshold",
            str(args.nms_iou_threshold),
            "--cross-class-nms-iou-threshold",
            str(args.cross_class_nms_iou_threshold),
        ]
    )
    run_logged(audit, combined / "audit.log")


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2 or args.minimum_long_side_px <= 0 or args.purge_seconds < 0:
        raise ValueError("invalid fold, size, or purge configuration")
    if args.idle_confirmations <= 0 or not 1 <= args.poll_seconds <= 60:
        raise ValueError("idle-confirmations must be positive and poll-seconds must be 1..60")
    if args.only_fold is not None and not 0 <= args.only_fold < args.folds:
        raise ValueError("only-fold must be within the configured fold range")
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.grad_accum_steps <= 0
        or args.early_stopping_patience <= 0
    ):
        raise ValueError("epochs, batch-size, and grad-accum-steps must be positive")
    if (
        args.num_workers < 0
        or not 0 <= args.prediction_threshold <= 1
        or not 0 <= args.operating_score_threshold <= 1
        or not 0 <= args.nms_iou_threshold <= 1
        or not 0 <= args.cross_class_nms_iou_threshold <= 1
        or args.cross_class_nms_iou_threshold < args.nms_iou_threshold
    ):
        raise ValueError("num-workers or prediction threshold is out of range")
    if args.pretrain_weights and not args.pretrain_weights.is_file():
        raise FileNotFoundError(args.pretrain_weights)


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    """Keep smoke runs bounded even though the trainer's flag does not cap epochs."""

    if args.smoke:
        args.epochs = 1


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    validate_args(args)
    sources = [parse_source(value) for value in args.source]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    args.workspace = args.workspace.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.artifact_root = (args.artifact_root or args.workspace).resolve()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.workspace / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another K-fold runner owns {args.workspace}") from error
    report = prepare(args, sources)
    args.data_manifest_id = report["data_manifest_id"]
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    repo_root = Path(__file__).resolve().parents[1]
    selected_folds = range(args.folds) if args.only_fold is None else (args.only_fold,)
    for fold in selected_folds:
        run_fold(args, fold, repo_root)
    if args.only_fold is None:
        combine_oof_results(args, repo_root)
        json_write(
            args.artifact_root / "_ALL_SUCCESS.json",
            {"folds": args.folds, "finished_at": datetime.now().isoformat()},
        )
    else:
        marker = "SMOKE" if args.smoke else "FOLD"
        json_write(
            args.artifact_root / f"_{marker}_{args.only_fold}_SUCCESS.json",
            {
                "fold": args.only_fold,
                "smoke": args.smoke,
                "finished_at": datetime.now().isoformat(),
            },
        )


if __name__ == "__main__":
    main()
