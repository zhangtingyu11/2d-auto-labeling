"""Read-only export of human rectangle annotations from a Label Studio database."""

from __future__ import annotations

import copy
import json
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


AUDIT_DATA_KEYS = {
    "candidate_count",
    "candidate_summary",
    "review_tag",
    "review_tag_label",
    "review_summary",
}


@dataclass(frozen=True)
class ExportSummary:
    project_id: int
    task_count: int
    completed_task_count: int
    empty_task_count: int
    human_box_count: int
    removed_non_human_result_count: int
    category_count: int


def _json_object(raw: str | None, *, context: str) -> dict[str, Any]:
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _json_list(raw: str | None, *, context: str) -> list[dict[str, Any]]:
    value = json.loads(raw or "[]")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{context} must be a JSON array of objects")
    return value


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def human_label_names(label_config: str) -> list[str]:
    """Read labels from the editable RectangleLabels control named ``label``."""
    root = ET.fromstring(label_config)
    for element in root.iter():
        if _local_name(element.tag) != "RectangleLabels" or element.attrib.get("name") != "label":
            continue
        labels = [
            child.attrib["value"]
            for child in element
            if _local_name(child.tag) == "Label" and child.attrib.get("value")
        ]
        if not labels:
            raise ValueError('RectangleLabels control "label" contains no labels')
        if len(labels) != len(set(labels)):
            raise ValueError('RectangleLabels control "label" contains duplicate labels')
        return labels
    raise ValueError('label config has no RectangleLabels control named "label"')


def clean_task_data(data: dict[str, Any]) -> dict[str, Any]:
    """Remove K-fold review descriptions while retaining source identity fields."""
    return {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key not in AUDIT_DATA_KEYS and not key.startswith("kfold_")
    }


def clean_human_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only editable rectangles from the human ``label`` control."""
    cleaned: list[dict[str, Any]] = []
    for result in results:
        if (
            result.get("type") != "rectanglelabels"
            or result.get("from_name") != "label"
            or result.get("readonly") is True
        ):
            continue
        value = result.get("value")
        if not isinstance(value, dict):
            raise ValueError(f'human result {result.get("id", "<unknown>")} has no value object')
        labels = value.get("rectanglelabels")
        if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
            raise ValueError(
                f'human result {result.get("id", "<unknown>")} must have exactly one label'
            )
        width = int(result.get("original_width", 0))
        height = int(result.get("original_height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(
                f'human result {result.get("id", "<unknown>")} has invalid image dimensions'
            )
        if float(value.get("rotation", 0)) != 0 or int(result.get("image_rotation", 0)) != 0:
            raise ValueError("rotated rectangles are not supported by the COCO exporter")
        coordinates = {}
        for key in ("x", "y", "width", "height"):
            try:
                coordinates[key] = float(value[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'human result {result.get("id", "<unknown>")} has invalid {key}'
                ) from error
        if coordinates["width"] <= 0 or coordinates["height"] <= 0:
            raise ValueError(f'human result {result.get("id", "<unknown>")} has an empty box')
        cleaned.append(
            {
                "id": str(result.get("id", f"human-{len(cleaned) + 1}")),
                "from_name": "label",
                "to_name": str(result.get("to_name", "image")),
                "type": "rectanglelabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    **coordinates,
                    "rotation": 0,
                    "rectanglelabels": [labels[0]],
                },
            }
        )
    return cleaned


def _image_file_name(data: dict[str, Any], task_id: int) -> str:
    for key in ("file_name", "relative_path"):
        if data.get(key):
            return str(data[key])
    image = str(data.get("image", ""))
    parsed = urlparse(image)
    query_path = parse_qs(parsed.query).get("d", [])
    if query_path:
        return unquote(query_path[0])
    if parsed.path:
        return unquote(parsed.path.lstrip("/"))
    raise ValueError(f"task {task_id} has no usable image path")


def _task_dimensions(
    human_results: list[dict[str, Any]], default_width: int, default_height: int, task_id: int
) -> tuple[int, int]:
    dimensions = {
        (int(result["original_width"]), int(result["original_height"]))
        for result in human_results
    }
    if len(dimensions) > 1:
        raise ValueError(f"task {task_id} has inconsistent image dimensions: {sorted(dimensions)}")
    if dimensions:
        return next(iter(dimensions))
    return default_width, default_height


def _coco_box(result: dict[str, Any], image_width: int, image_height: int) -> list[float]:
    value = result["value"]
    x1 = max(0.0, min(image_width, value["x"] * image_width / 100.0))
    y1 = max(0.0, min(image_height, value["y"] * image_height / 100.0))
    x2 = max(0.0, min(image_width, (value["x"] + value["width"]) * image_width / 100.0))
    y2 = max(0.0, min(image_height, (value["y"] + value["height"]) * image_height / 100.0))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'human result {result["id"]} is empty after clipping')
    return [x1, y1, x2 - x1, y2 - y1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_human_annotations(
    *,
    database: Path,
    project_id: int,
    label_studio_json: Path,
    coco_json: Path,
    default_width: int = 1920,
    default_height: int = 1080,
) -> ExportSummary:
    """Export the latest non-cancelled annotation for every project task."""
    if default_width <= 0 or default_height <= 0:
        raise ValueError("default image dimensions must be positive")
    database_uri = database.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        project = connection.execute(
            "SELECT title, label_config FROM project WHERE id = ? AND deleted_at IS NULL",
            (project_id,),
        ).fetchone()
        if project is None:
            raise ValueError(f"Label Studio project {project_id} does not exist")
        project_title, label_config = project
        labels = human_label_names(label_config)
        rows = connection.execute(
            """
            SELECT t.id, t.data, t.is_labeled,
                   a.id, a.result, a.created_at, a.updated_at, a.ground_truth
            FROM task AS t
            LEFT JOIN task_completion AS a ON a.id = (
                SELECT candidate.id
                FROM task_completion AS candidate
                WHERE candidate.task_id = t.id AND candidate.was_cancelled = 0
                ORDER BY candidate.updated_at DESC, candidate.id DESC
                LIMIT 1
            )
            WHERE t.project_id = ?
            ORDER BY t.id
            """,
            (project_id,),
        ).fetchall()
    finally:
        connection.close()

    prepared: list[dict[str, Any]] = []
    removed_results = 0
    for row in rows:
        (
            task_id,
            raw_data,
            is_labeled,
            annotation_id,
            raw_results,
            created_at,
            updated_at,
            ground_truth,
        ) = row
        data = _json_object(raw_data, context=f"task {task_id} data")
        all_results = _json_list(raw_results, context=f"task {task_id} annotation result")
        human_results = clean_human_results(all_results)
        removed_results += len(all_results) - len(human_results)
        stable_key = str(
            data.get("image_id") or data.get("external_id") or data.get("image") or task_id
        )
        prepared.append(
            {
                "task_id": int(task_id),
                "data": clean_task_data(data),
                "is_labeled": bool(is_labeled),
                "annotation_id": annotation_id,
                "created_at": created_at,
                "updated_at": updated_at,
                "ground_truth": bool(ground_truth) if annotation_id is not None else False,
                "results": human_results,
                "stable_key": stable_key,
            }
        )
    prepared.sort(key=lambda item: (item["stable_key"], item["task_id"]))

    label_studio_tasks: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    category_ids = {name: index for index, name in enumerate(labels, start=1)}
    annotation_index = 1
    completed_tasks = 0
    empty_tasks = 0
    for image_index, item in enumerate(prepared, start=1):
        annotations = []
        if item["annotation_id"] is not None:
            completed_tasks += 1
            annotations.append(
                {
                    "id": int(item["annotation_id"]),
                    "result": item["results"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "ground_truth": item["ground_truth"],
                    "was_cancelled": False,
                }
            )
        if not item["results"]:
            empty_tasks += 1
        label_studio_tasks.append(
            {"id": item["task_id"], "data": item["data"], "annotations": annotations}
        )

        width, height = _task_dimensions(
            item["results"], default_width, default_height, item["task_id"]
        )
        data = item["data"]
        image = {
            "id": image_index,
            "file_name": _image_file_name(data, item["task_id"]),
            "width": width,
            "height": height,
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
        coco_images.append(image)
        for result in item["results"]:
            label = result["value"]["rectanglelabels"][0]
            if label not in category_ids:
                raise ValueError(
                    f"task {item['task_id']} uses label not in project config: {label}"
                )
            bbox = _coco_box(result, width, height)
            coco_annotations.append(
                {
                    "id": annotation_index,
                    "image_id": image_index,
                    "category_id": category_ids[label],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "label_studio_task_id": item["task_id"],
                    "label_studio_result_id": result["id"],
                }
            )
            annotation_index += 1

    coco = {
        "info": {
            "description": (
                f"Human annotations exported from Label Studio project "
                f"{project_id}: {project_title}"
            ),
            "label_studio_project_id": project_id,
            "human_results_only": True,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "object"}
            for name in labels
        ],
    }
    _write_json(label_studio_json, label_studio_tasks)
    _write_json(coco_json, coco)
    return ExportSummary(
        project_id=project_id,
        task_count=len(prepared),
        completed_task_count=completed_tasks,
        empty_task_count=empty_tasks,
        human_box_count=len(coco_annotations),
        removed_non_human_result_count=removed_results,
        category_count=len(labels),
    )


def summary_as_dict(summary: ExportSummary) -> dict[str, int]:
    return asdict(summary)
