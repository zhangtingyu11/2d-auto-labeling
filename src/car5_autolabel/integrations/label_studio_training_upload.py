"""Convert a native Label Studio JSON export into K-fold training inputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from car5_autolabel.integrations.label_studio_human_export import (
    _coco_box,
    _image_file_name,
    _task_dimensions,
    clean_human_results,
    clean_task_data,
)
from car5_autolabel.kfold_audit import KFOLD_TRAINING_CLASSES


@dataclass(frozen=True)
class TrainingUploadSummary:
    task_count: int
    submitted_task_count: int
    empty_task_count: int
    human_box_count: int
    removed_non_human_result_count: int
    assignment_count: int
    fold_counts: dict[str, int]


def _object_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{context} must be a JSON array of objects")
    return value


def _latest_annotation(task: dict[str, Any], task_id: int) -> dict[str, Any]:
    annotations = _object_list(task.get("annotations", []), context=f"task {task_id} annotations")
    submitted = [item for item in annotations if not item.get("was_cancelled", False)]
    if not submitted:
        raise ValueError(
            f"task {task_id} has no submitted annotation; submit every task before training"
        )
    return max(
        submitted,
        key=lambda item: (str(item.get("updated_at", "")), int(item.get("id", 0))),
    )


def _load_assignment_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "package",
        "source_image_id",
        "source_dataset",
        "camera",
        "timestamp",
        "file_name",
        "block_key",
        "validation_fold",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"assignment template is missing columns: {sorted(missing)}")
    return fieldnames, rows


def _assignment_by_file_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        file_name = row["file_name"].replace("\\", "/").lstrip("/")
        if file_name in result:
            raise ValueError(f"duplicate assignment file_name: {file_name}")
        result[file_name] = row
    return result


def _assignment_for_image(
    file_name: str, assignment_index: dict[str, dict[str, str]]
) -> dict[str, str] | None:
    normalized = file_name.replace("\\", "/").lstrip("/")
    candidates = [normalized]
    marker = "images/"
    if marker in normalized:
        candidates.append(marker + normalized.split(marker, 1)[1])
    matches = {
        assignment_index[candidate]["package"]
        + ":"
        + assignment_index[candidate]["source_image_id"]: assignment_index[candidate]
        for candidate in candidates
        if candidate in assignment_index
    }
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def convert_training_export(
    *,
    label_studio_export: Path,
    assignment_template: Path,
    coco_output: Path,
    assignments_output: Path,
    summary_output: Path,
    package_name: str = "reviewed1085",
    default_width: int = 1920,
    default_height: int = 1080,
) -> TrainingUploadSummary:
    """Keep submitted human boxes and inherit fold assignments by source filename."""
    if default_width <= 0 or default_height <= 0:
        raise ValueError("default image dimensions must be positive")
    tasks = _object_list(
        json.loads(label_studio_export.read_text(encoding="utf-8-sig")),
        context="Label Studio export",
    )
    if not tasks:
        raise ValueError("Label Studio export is empty")
    _, assignment_rows = _load_assignment_rows(assignment_template)
    assignment_index = _assignment_by_file_name(assignment_rows)
    expected_labels = set(KFOLD_TRAINING_CLASSES)

    prepared: list[dict[str, Any]] = []
    seen_task_ids: set[int] = set()
    for position, task in enumerate(tasks, start=1):
        try:
            task_id = int(task["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"task at position {position} has no integer id") from error
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate Label Studio task id: {task_id}")
        seen_task_ids.add(task_id)
        data = task.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"task {task_id} data must be an object")
        annotation = _latest_annotation(task, task_id)
        raw_results = _object_list(
            annotation.get("result", []), context=f"task {task_id} annotation results"
        )
        human_results = clean_human_results(raw_results)
        invalid_labels = sorted(
            {
                result["value"]["rectanglelabels"][0]
                for result in human_results
                if result["value"]["rectanglelabels"][0] not in expected_labels
            }
        )
        if invalid_labels:
            raise ValueError(f"task {task_id} has unsupported human labels: {invalid_labels}")
        cleaned_data = clean_task_data(data)
        file_name = _image_file_name(cleaned_data, task_id)
        assignment = _assignment_for_image(file_name, assignment_index)
        if assignment is None:
            raise ValueError(
                f"task {task_id} image {file_name!r} has no unique prior fold assignment"
            )
        width, height = _task_dimensions(
            human_results, default_width, default_height, task_id
        )
        prepared.append(
            {
                "task_id": task_id,
                "data": cleaned_data,
                "file_name": file_name,
                "width": width,
                "height": height,
                "human_results": human_results,
                "removed_results": len(raw_results) - len(human_results),
                "assignment": assignment,
            }
        )

    prepared.sort(key=lambda item: (item["file_name"], item["task_id"]))
    category_ids = {
        name: index for index, name in enumerate(KFOLD_TRAINING_CLASSES, start=1)
    }
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    assignment_output_rows: list[dict[str, str]] = []
    annotation_id = 1
    fold_counts: dict[str, int] = {}
    for image_id, item in enumerate(prepared, start=1):
        data = item["data"]
        image = {
            "id": image_id,
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
            "label_studio_task_id": item["task_id"],
        }
        for output_key, candidates in {
            "external_image_id": ("image_id", "external_id"),
            "camera_id": ("camera_id", "camera"),
            "timestamp": ("timestamp", "capture_timestamp"),
            "sequence_id": ("sequence_id",),
        }.items():
            value = next((data[key] for key in candidates if data.get(key) is not None), None)
            if value is not None:
                image[output_key] = value
        images.append(image)
        for result in item["human_results"]:
            label = result["value"]["rectanglelabels"][0]
            bbox = _coco_box(result, item["width"], item["height"])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[label],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "label_studio_task_id": item["task_id"],
                    "label_studio_result_id": result["id"],
                }
            )
            annotation_id += 1
        inherited = item["assignment"]
        fold = inherited["validation_fold"]
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        assignment_output_rows.append(
            {
                "package": package_name,
                "source_image_id": str(image_id),
                "source_dataset": inherited["source_dataset"],
                "camera": inherited["camera"],
                "timestamp": inherited["timestamp"],
                "file_name": item["file_name"],
                "block_key": inherited["block_key"],
                "validation_fold": fold,
            }
        )

    coco = {
        "info": {
            "description": "Human-only Label Studio upload for K-fold training",
            "human_results_only": True,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "object"}
            for name in KFOLD_TRAINING_CLASSES
        ],
    }
    summary = TrainingUploadSummary(
        task_count=len(prepared),
        submitted_task_count=len(prepared),
        empty_task_count=sum(not item["human_results"] for item in prepared),
        human_box_count=len(annotations),
        removed_non_human_result_count=sum(item["removed_results"] for item in prepared),
        assignment_count=len(assignment_output_rows),
        fold_counts=dict(sorted(fold_counts.items())),
    )
    _write_json(coco_output, coco)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_assignments = assignments_output.with_suffix(assignments_output.suffix + ".tmp")
    with temporary_assignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "package",
                "source_image_id",
                "source_dataset",
                "camera",
                "timestamp",
                "file_name",
                "block_key",
                "validation_fold",
            ],
        )
        writer.writeheader()
        writer.writerows(assignment_output_rows)
    temporary_assignments.replace(assignments_output)
    _write_json(summary_output, asdict(summary))
    return summary
