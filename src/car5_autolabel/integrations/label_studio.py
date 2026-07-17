import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from car5_autolabel.datasets.constants import ACTIVE_CLASSES
from car5_autolabel.datasets.models import (
    CanonicalBox,
    CanonicalImageAnnotation,
    LabelStudioParseResult,
    ManifestRow,
    ReviewStatus,
    ValidationReport,
)
from car5_autolabel.schemas import BoundingBox, ImagePrediction


def box_to_label_studio(box: BoundingBox, width: int, height: int) -> dict:
    """Convert an internal pixel-space box to a Label Studio rectangle result."""
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "prediction",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": box.x1 * 100.0 / width,
            "y": box.y1 * 100.0 / height,
            "width": (box.x2 - box.x1) * 100.0 / width,
            "height": (box.y2 - box.y1) * 100.0 / height,
            "rotation": 0,
            "rectanglelabels": [box.label],
        },
        "score": box.score,
    }


def prediction_to_label_studio(prediction: ImagePrediction) -> list[dict]:
    return [
        box_to_label_studio(box, prediction.width, prediction.height)
        for box in prediction.boxes
    ]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_zero_rotation(value: Any) -> bool:
    try:
        return abs(float(value)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _task_identity(task: dict[str, Any]) -> tuple[str | None, str | None]:
    data = task.get("data") or {}
    camera_id = data.get("camera") or data.get("camera_id")
    frame_id = data.get("frame_id")
    if isinstance(camera_id, str) and isinstance(frame_id, str):
        return camera_id, frame_id
    return None, None


def _unreviewed_annotation(
    image_id: str,
    camera_id: str,
    frame_id: str,
    task: dict[str, Any],
) -> CanonicalImageAnnotation:
    return CanonicalImageAnnotation(
        image_id=image_id,
        camera_id=camera_id,
        frame_id=frame_id,
        review_status=ReviewStatus.UNREVIEWED_PREDICTION,
        source_task_id=_optional_int(task.get("id")),
    )


def _result_to_box(
    result: dict[str, Any],
    *,
    image_id: str,
    width: int,
    height: int,
    result_index: int,
    active_classes: tuple[str, ...],
    report: ValidationReport,
) -> CanonicalBox | None:
    if result.get("type") != "rectanglelabels":
        report.add_warning(
            "ignored_result_type",
            "non-rectangle Label Studio result was ignored",
            image_id=image_id,
            result_type=result.get("type"),
        )
        return None

    if result.get("from_name") != "label" or result.get("to_name") != "image":
        report.add_error(
            "invalid_label_studio_binding",
            "rectangle must bind label to image",
            image_id=image_id,
            from_name=result.get("from_name"),
            to_name=result.get("to_name"),
        )
        return None
    if not _is_zero_rotation(result.get("image_rotation", 0.0)):
        report.add_error(
            "unsupported_image_rotation",
            "v1 accepts only unrotated images",
            image_id=image_id,
            image_rotation=result.get("image_rotation"),
        )
        return None
    try:
        original_width = int(result.get("original_width", width))
        original_height = int(result.get("original_height", height))
    except (TypeError, ValueError):
        original_width = original_height = -1
    if original_width != width or original_height != height:
        report.add_error(
            "image_dimension_mismatch",
            "rectangle source dimensions do not match the manifest",
            image_id=image_id,
            manifest_width=width,
            manifest_height=height,
            result_width=result.get("original_width"),
            result_height=result.get("original_height"),
        )
        return None

    value = result.get("value") or {}
    labels = value.get("rectanglelabels") or []
    if len(labels) != 1 or labels[0] not in active_classes:
        report.add_error(
            "invalid_rectangle_label",
            "rectangle must contain exactly one active v1 class",
            image_id=image_id,
            labels=labels,
        )
        return None
    if not _is_zero_rotation(value.get("rotation", 0.0)):
        report.add_error(
            "unsupported_rotation",
            "v1 accepts only axis-aligned rectangles",
            image_id=image_id,
            rotation=value.get("rotation"),
        )
        return None

    try:
        x1_raw = float(value["x"]) * width / 100.0
        y1_raw = float(value["y"]) * height / 100.0
        x2_raw = x1_raw + float(value["width"]) * width / 100.0
        y2_raw = y1_raw + float(value["height"]) * height / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        report.add_error(
            "invalid_rectangle_coordinates",
            "rectangle percentage coordinates are missing or invalid",
            image_id=image_id,
            detail=str(exc),
        )
        return None

    x1 = min(max(x1_raw, 0.0), float(width))
    y1 = min(max(y1_raw, 0.0), float(height))
    x2 = min(max(x2_raw, 0.0), float(width))
    y2 = min(max(y2_raw, 0.0), float(height))
    if any(
        abs(clipped - raw) > 1e-6
        for clipped, raw in zip(
            (x1, y1, x2, y2),
            (x1_raw, y1_raw, x2_raw, y2_raw),
            strict=True,
        )
    ):
        report.add_warning(
            "box_clipped",
            "rectangle was clipped to image boundaries",
            image_id=image_id,
            result_id=result.get("id"),
        )

    raw_result_id = str(result.get("id") or f"result-{result_index:04d}")
    label = labels[0]
    try:
        return CanonicalBox(
            annotation_id=f"{image_id}::{raw_result_id}",
            image_id=image_id,
            class_id=active_classes.index(label),
            label=label,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            source="human",
            review_status=ReviewStatus.HUMAN_VERIFIED,
            score=result.get("score"),
            metadata={
                "label_studio_result_id": raw_result_id,
                "origin": result.get("origin"),
                "meta": result.get("meta") or {},
            },
        )
    except ValidationError as exc:
        report.add_error(
            "invalid_pixel_box",
            "converted rectangle is empty or invalid",
            image_id=image_id,
            result_id=raw_result_id,
            detail=str(exc),
        )
        return None


def parse_label_studio_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    manifest_rows: Iterable[ManifestRow] | None = None,
    dataset_id: str = "car5_20260611_day_02",
    active_classes: tuple[str, ...] = ACTIVE_CLASSES,
    require_reviewed: bool = True,
) -> LabelStudioParseResult:
    """Convert a Label Studio JSON export into reviewed canonical annotations."""
    report = ValidationReport(dataset_id=dataset_id)
    manifest = list(manifest_rows or [])
    manifest_by_key = {(row.camera_id, row.frame_id): row for row in manifest}
    manifest_position = {
        (row.camera_id, row.frame_id): index for index, row in enumerate(manifest)
    }
    expected_keys = {
        (row.camera_id, row.frame_id) for row in manifest if row.is_keyframe
    }

    annotations: list[CanonicalImageAnnotation] = []
    seen_keys: set[tuple[str, str]] = set()
    for task_index, task in enumerate(tasks):
        if not isinstance(task, dict):
            report.add_error(
                "invalid_task",
                "Label Studio export entries must be JSON objects",
                task_index=task_index,
            )
            continue
        camera_id, frame_id = _task_identity(task)
        if camera_id is None or frame_id is None:
            report.add_error(
                "missing_task_identity",
                "task data must contain camera and frame_id",
                task_index=task_index,
                task_id=task.get("id"),
            )
            continue

        key = (camera_id, frame_id)
        if key in seen_keys:
            report.add_error(
                "duplicate_task_identity",
                "multiple tasks refer to the same camera and frame",
                camera_id=camera_id,
                frame_id=frame_id,
            )
            continue
        seen_keys.add(key)

        manifest_row = manifest_by_key.get(key)
        if manifest and manifest_row is None:
            report.add_error(
                "task_not_in_manifest",
                "Label Studio task is not present in the manifest",
                camera_id=camera_id,
                frame_id=frame_id,
            )
            continue

        image_id = (
            manifest_row.image_id
            if manifest_row is not None
            else f"{dataset_id}__{frame_id}__{camera_id}"
        )
        width = manifest_row.width if manifest_row is not None else 1920
        height = manifest_row.height if manifest_row is not None else 1080
        raw_annotations = task.get("annotations") or []
        if not isinstance(raw_annotations, list):
            report.add_error(
                "invalid_annotations",
                "task annotations must be a JSON array",
                image_id=image_id,
            )
            continue
        if not raw_annotations:
            if require_reviewed:
                report.add_error(
                    "unreviewed_task",
                    "task has no submitted annotation; predictions are not ground truth",
                    image_id=image_id,
                )
            annotations.append(
                _unreviewed_annotation(image_id, camera_id, frame_id, task)
            )
            continue
        if len(raw_annotations) != 1:
            report.add_error(
                "ambiguous_annotations",
                "task must have exactly one selected submitted annotation",
                image_id=image_id,
                annotation_count=len(raw_annotations),
            )
            continue

        raw_annotation = raw_annotations[0]
        if not isinstance(raw_annotation, dict):
            report.add_error(
                "invalid_annotation",
                "submitted annotation must be a JSON object",
                image_id=image_id,
            )
            continue
        if raw_annotation.get("was_cancelled"):
            if require_reviewed:
                report.add_error(
                    "skipped_task",
                    "cancelled or skipped task is unreviewed, not verified empty",
                    image_id=image_id,
                )
            annotations.append(
                _unreviewed_annotation(image_id, camera_id, frame_id, task)
            )
            continue

        raw_results = raw_annotation.get("result") or []
        if not isinstance(raw_results, list):
            report.add_error(
                "invalid_results",
                "annotation result must be a JSON array",
                image_id=image_id,
            )
            continue
        boxes: list[CanonicalBox] = []
        seen_result_ids: set[str] = set()
        for result_index, result in enumerate(raw_results):
            if not isinstance(result, dict):
                report.add_error(
                    "invalid_result",
                    "annotation result entries must be JSON objects",
                    image_id=image_id,
                    result_index=result_index,
                )
                continue
            result_id = result.get("id")
            if result_id is not None:
                result_id = str(result_id)
                if result_id in seen_result_ids:
                    report.add_error(
                        "duplicate_result_id",
                        "result IDs must be unique within a task",
                        image_id=image_id,
                        result_id=result_id,
                    )
                    continue
                seen_result_ids.add(result_id)
            box = _result_to_box(
                result,
                image_id=image_id,
                width=width,
                height=height,
                result_index=result_index,
                active_classes=active_classes,
                report=report,
            )
            if box is not None:
                boxes.append(box)
        boxes.sort(
            key=lambda box: (
                box.class_id,
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                box.annotation_id,
            )
        )
        if raw_results and not boxes:
            report.add_error(
                "no_valid_rectangles",
                "submitted non-empty annotation has no valid v1 rectangles",
                image_id=image_id,
            )
            review_status = ReviewStatus.UNREVIEWED_PREDICTION
        elif boxes:
            review_status = ReviewStatus.HUMAN_VERIFIED
        else:
            review_status = ReviewStatus.VERIFIED_EMPTY
        annotations.append(
            CanonicalImageAnnotation(
                image_id=image_id,
                camera_id=camera_id,
                frame_id=frame_id,
                review_status=review_status,
                boxes=boxes,
                source_project_id=_optional_int(task.get("project")),
                source_task_id=_optional_int(task.get("id")),
                source_annotation_id=_optional_int(raw_annotation.get("id")),
            )
        )

    if expected_keys:
        missing = sorted(expected_keys - seen_keys)
        if missing:
            report.add_error(
                "missing_manifest_tasks",
                "review export does not cover every manifest keyframe image",
                missing_count=len(missing),
                missing_sample=[{"camera_id": key[0], "frame_id": key[1]} for key in missing[:10]],
            )

    annotations.sort(
        key=lambda annotation: manifest_position.get(
            (annotation.camera_id, annotation.frame_id),
            len(manifest_position),
        )
    )
    class_counts = Counter(
        box.label for annotation in annotations for box in annotation.boxes
    )
    report.stats = {
        "task_count": len(seen_keys),
        "canonical_image_count": len(annotations),
        "box_count": sum(len(annotation.boxes) for annotation in annotations),
        "reviewed_image_count": sum(
            annotation.review_status is not ReviewStatus.UNREVIEWED_PREDICTION
            for annotation in annotations
        ),
        "verified_empty_image_count": sum(
            annotation.review_status is ReviewStatus.VERIFIED_EMPTY
            for annotation in annotations
        ),
        "unreviewed_image_count": sum(
            annotation.review_status is ReviewStatus.UNREVIEWED_PREDICTION
            for annotation in annotations
        ),
        "box_counts_by_class": {
            label: class_counts.get(label, 0) for label in active_classes
        },
    }
    return LabelStudioParseResult(annotations=annotations, report=report)


def parse_label_studio_export(
    path: str | Path,
    **kwargs: Any,
) -> LabelStudioParseResult:
    export_path = Path(path)
    payload = json.loads(export_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Label Studio export must be a JSON array of tasks")
    return parse_label_studio_tasks(payload, **kwargs)
