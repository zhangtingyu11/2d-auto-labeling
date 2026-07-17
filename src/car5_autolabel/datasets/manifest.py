from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import cv2

from car5_autolabel.datasets.label_studio_export import (
    LabelStudioExportError,
    parse_label_studio_export,
)
from car5_autolabel.datasets.models import CAMERA_ORDER, ManifestRow, ValidationReport


HashScope = Literal["none", "keyframes", "all"]
DimensionScope = Literal["none", "sample", "keyframes", "all"]


def _timestamp(stem: str) -> datetime:
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S_%f")
    except ValueError as exc:
        raise ValueError(f"invalid timestamp filename stem: {stem}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_ids(stems: list[str], gap_seconds: float) -> list[str]:
    sequence = 0
    result: list[str] = []
    previous: datetime | None = None
    for stem in stems:
        current = _timestamp(stem)
        if previous is not None and (current - previous).total_seconds() > gap_seconds:
            sequence += 1
        result.append(f"s{sequence:02d}")
        previous = current
    return result


def _write_jsonl(rows: Iterable[ManifestRow], path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = (line + "\n").encode()
            digest.update(encoded)
            stream.write(line + "\n")
    return digest.hexdigest()


def build_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str | None = None,
    label_studio_export: str | Path | None = None,
    hash_scope: HashScope = "keyframes",
    dimension_scope: DimensionScope = "sample",
    expected_width: int = 1920,
    expected_height: int = 1080,
    sequence_gap_seconds: float = 1.0,
    strict: bool = True,
) -> tuple[list[ManifestRow], ValidationReport]:
    root = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera_root = root / "camera"
    label_root = root / "label"
    resolved_dataset_id = dataset_id or root.name
    report = ValidationReport(dataset_id=resolved_dataset_id, source_root=str(root))

    camera_stems: dict[str, list[str]] = {}
    camera_paths: dict[tuple[str, str], Path] = {}
    for camera in CAMERA_ORDER:
        directory = camera_root / camera
        files = sorted(directory.glob("*.jpg")) if directory.is_dir() else []
        stems = [path.stem for path in files]
        camera_stems[camera] = stems
        report.camera_counts[camera] = len(stems)
        camera_paths.update({(camera, path.stem): path for path in files})
        if not files:
            report.errors.append(f"camera directory is missing or empty: {camera}")

    canonical_stems = camera_stems["front"]
    canonical_set = set(canonical_stems)
    report.source_timestamps = len(canonical_stems)
    if len(canonical_set) != len(canonical_stems):
        report.errors.append("front camera contains duplicate filename stems")

    for camera, stems in camera_stems.items():
        stem_set = set(stems)
        missing = sorted(canonical_set - stem_set)
        extra = sorted(stem_set - canonical_set)
        if missing:
            report.missing_camera_stems[camera] = missing
            report.errors.append(f"{camera} is missing {len(missing)} source timestamps")
        if extra:
            report.extra_camera_stems[camera] = extra
            report.errors.append(f"{camera} has {len(extra)} extra source timestamps")

    label_paths = sorted(label_root.glob("*.json")) if label_root.is_dir() else []
    keyframe_stems = [path.stem for path in label_paths if not path.name.startswith("._")]
    keyframe_set = set(keyframe_stems)
    report.keyframe_timestamps = len(keyframe_stems)
    if len(keyframe_set) != len(keyframe_stems):
        report.errors.append("3D label directory contains duplicate keyframe stems")

    for camera in CAMERA_ORDER:
        missing = sorted(keyframe_set - set(camera_stems[camera]))
        if missing:
            report.keyframes_missing_images[camera] = missing
            report.errors.append(f"{camera} is missing {len(missing)} keyframe images")

    unknown_keyframes = sorted(keyframe_set - canonical_set)
    if unknown_keyframes:
        report.errors.append(f"{len(unknown_keyframes)} label stems are absent from the camera stream")

    sequence_ids = _sequence_ids(canonical_stems, sequence_gap_seconds)
    sequence_counts = Counter(
        sequence_id for stem, sequence_id in zip(canonical_stems, sequence_ids, strict=True) if stem in keyframe_set
    )
    report.sequence_counts = dict(sorted(sequence_counts.items()))

    reviewed_tasks = {}
    if label_studio_export:
        try:
            reviewed_tasks, export_summary = parse_label_studio_export(label_studio_export)
            report.export_duplicate_keys = export_summary["duplicate_keys"]
            report.reviewed_box_counts = export_summary["box_counts"]
            if report.export_duplicate_keys:
                report.errors.append(
                    f"Label Studio export has {len(report.export_duplicate_keys)} duplicate task keys"
                )
        except LabelStudioExportError as exc:
            report.errors.append(f"Label Studio export is invalid: {exc}")

    expected_review_keys = {(camera, stem) for stem in keyframe_stems for camera in CAMERA_ORDER}
    export_keys = set(reviewed_tasks)
    if label_studio_export:
        report.export_missing_keys = sorted(
            f"{camera}/{stem}" for camera, stem in expected_review_keys - export_keys
        )
        report.export_extra_keys = sorted(
            f"{camera}/{stem}" for camera, stem in export_keys - expected_review_keys
        )
        if report.export_missing_keys:
            report.errors.append(
                f"Label Studio export is missing {len(report.export_missing_keys)} keyframe-camera tasks"
            )
        if report.export_extra_keys:
            report.errors.append(
                f"Label Studio export has {len(report.export_extra_keys)} tasks outside the supervised set"
            )

    should_hash = {
        "none": lambda _is_keyframe: False,
        "keyframes": lambda is_keyframe: is_keyframe,
        "all": lambda _is_keyframe: True,
    }[hash_scope]
    should_check_dimension = {
        "none": lambda _camera, _index, _is_keyframe: False,
        "sample": lambda _camera, index, _is_keyframe: index == 0,
        "keyframes": lambda _camera, _index, is_keyframe: is_keyframe,
        "all": lambda _camera, _index, _is_keyframe: True,
    }[dimension_scope]

    rows: list[ManifestRow] = []
    status_counts: Counter[str] = Counter()
    for raw_index, (stem, sequence_id) in enumerate(
        zip(canonical_stems, sequence_ids, strict=True)
    ):
        is_keyframe = stem in keyframe_set
        for camera, camera_order in CAMERA_ORDER.items():
            path = camera_paths.get((camera, stem))
            if path is None:
                continue
            if should_check_dimension(camera, raw_index, is_keyframe):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                actual = None if image is None else (image.shape[1], image.shape[0])
                if actual != (expected_width, expected_height):
                    report.image_dimension_errors.append(
                        {
                            "camera": camera,
                            "timestamp": stem,
                            "expected": [expected_width, expected_height],
                            "actual": list(actual) if actual else None,
                        }
                    )

            if is_keyframe:
                reviewed = reviewed_tasks.get((camera, stem))
                if label_studio_export:
                    annotation_status = reviewed.review_status if reviewed else "missing_review"
                else:
                    annotation_status = "expected_review"
            else:
                annotation_status = "temporal_context"
            status_counts[annotation_status] += 1

            rows.append(
                ManifestRow(
                    dataset_id=resolved_dataset_id,
                    image_id=f"{resolved_dataset_id}:{stem}:{camera}",
                    relative_path=path.relative_to(root).as_posix(),
                    sha256=_sha256(path) if should_hash(is_keyframe) else None,
                    width=expected_width,
                    height=expected_height,
                    sequence_id=sequence_id,
                    frame_id=raw_index,
                    capture_timestamp=stem,
                    raw_sequence_index=raw_index,
                    keyframe_id=stem if is_keyframe else None,
                    is_keyframe=is_keyframe,
                    sync_delta_ms=0.0,
                    camera_id=camera,
                    camera_order=camera_order,
                    fold_id=sequence_id,
                    annotation_status=annotation_status,
                )
            )

    if report.image_dimension_errors:
        report.errors.append(f"{len(report.image_dimension_errors)} images have invalid dimensions")
    report.manifest_rows = len(rows)
    report.supervised_rows = sum(row.is_keyframe for row in rows)
    report.temporal_context_rows = len(rows) - report.supervised_rows
    report.annotation_status_counts = dict(sorted(status_counts.items()))

    manifest_path = output / "manifest.jsonl"
    report.manifest_sha256 = _write_jsonl(rows, manifest_path)
    (output / "validation_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.sha256").write_text(
        f"{report.manifest_sha256}  manifest.jsonl\n", encoding="ascii"
    )

    if strict and report.errors:
        raise ValueError("dataset validation failed: " + "; ".join(report.errors))
    return rows, report
