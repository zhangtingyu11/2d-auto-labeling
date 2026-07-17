from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from car5_autolabel.datasets.models import ACTIVE_CLASSES, ReviewedBox, ReviewedTask


class LabelStudioExportError(ValueError):
    """Raised when an export cannot be converted without losing review semantics."""


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise LabelStudioExportError("Label Studio export must be a JSON task list")
    return payload


def _task_identity(task: dict[str, Any]) -> tuple[str, str]:
    data = task.get("data") or {}
    camera = data.get("camera")
    frame_id = data.get("frame_id")
    if not camera or not frame_id:
        image_url = str(data.get("image", ""))
        document = parse_qs(urlparse(image_url).query).get("d", [""])[0]
        parts = Path(document).parts
        if len(parts) >= 2:
            camera = camera or parts[-2]
            frame_id = frame_id or Path(parts[-1]).stem
    if not camera or not frame_id:
        raise LabelStudioExportError(f"task {task.get('id')} has no stable camera/frame identity")
    return str(camera), str(frame_id)


def _latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations") or []
    if not annotations:
        return None
    return max(
        annotations,
        key=lambda item: (str(item.get("updated_at", "")), int(item.get("id") or 0)),
    )


def _parse_box(result: dict[str, Any]) -> ReviewedBox | None:
    if result.get("type") != "rectanglelabels":
        return None
    value = result.get("value") or {}
    labels = value.get("rectanglelabels") or []
    if len(labels) != 1:
        raise LabelStudioExportError(f"result {result.get('id')} must contain exactly one label")
    class_name = str(labels[0]).strip().lower()
    if class_name not in ACTIVE_CLASSES:
        raise LabelStudioExportError(f"unknown active class: {labels[0]}")

    width = float(result.get("original_width") or 0)
    height = float(result.get("original_height") or 0)
    if width <= 0 or height <= 0:
        raise LabelStudioExportError(f"result {result.get('id')} has invalid image dimensions")

    x = float(value.get("x", 0))
    y = float(value.get("y", 0))
    box_width = float(value.get("width", 0))
    box_height = float(value.get("height", 0))
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise LabelStudioExportError(f"result {result.get('id')} has invalid percentage box")
    if x + box_width > 100.000001 or y + box_height > 100.000001:
        raise LabelStudioExportError(f"result {result.get('id')} exceeds image bounds")

    x1 = width * x / 100
    y1 = height * y / 100
    x2 = width * (x + box_width) / 100
    y2 = height * (y + box_height) / 100
    meta = result.get("meta") or {}
    return ReviewedBox(
        class_id=ACTIVE_CLASSES[class_name],
        class_name=class_name,
        xyxy=(x1, y1, x2, y2),
        source=str(meta.get("source") or "human"),
        result_id=str(result.get("id")) if result.get("id") is not None else None,
    )


def parse_label_studio_export(
    path: str | Path,
) -> tuple[dict[tuple[str, str], ReviewedTask], dict[str, Any]]:
    """Parse submitted tasks while preserving missing, skipped, and empty states."""
    export_path = Path(path)
    tasks: dict[tuple[str, str], ReviewedTask] = {}
    duplicate_keys: list[str] = []
    status_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()

    for raw_task in _load_json(export_path):
        camera, frame_id = _task_identity(raw_task)
        key = (camera, frame_id)
        if key in tasks:
            duplicate_keys.append(f"{camera}/{frame_id}")
            continue

        annotation = _latest_annotation(raw_task)
        boxes: tuple[ReviewedBox, ...] = ()
        if annotation is None:
            status = "unreviewed"
        elif annotation.get("was_cancelled"):
            status = "skipped"
        else:
            boxes = tuple(
                box
                for result in annotation.get("result") or []
                if (box := _parse_box(result)) is not None
            )
            status = "human_verified" if boxes else "verified_empty"
            box_counts.update(box.class_name for box in boxes)

        status_counts[status] += 1
        image_id = f"{camera}:{frame_id}"
        tasks[key] = ReviewedTask(
            image_id=image_id,
            frame_id=frame_id,
            camera_id=camera,
            review_status=status,
            boxes=boxes,
            task_id=raw_task.get("id"),
        )

    summary = {
        "path": str(export_path),
        "tasks": len(tasks),
        "duplicate_keys": sorted(duplicate_keys),
        "status_counts": dict(sorted(status_counts.items())),
        "box_counts": dict(sorted(box_counts.items())),
    }
    return tasks, summary
