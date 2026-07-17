from __future__ import annotations

from typing import Any

from car5_autolabel.datasets.constants import ACTIVE_CLASSES, CLASS_INDEX
from car5_autolabel.datasets.models import (
    CanonicalImageAnnotation,
    ManifestRow,
    ReviewStatus,
)


def build_coco_dataset(
    manifest_rows: list[ManifestRow],
    annotations: list[CanonicalImageAnnotation],
    *,
    held_out_fold: str,
    split: str,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    annotation_by_image = {annotation.image_id: annotation for annotation in annotations}
    if len(annotation_by_image) != len(annotations):
        raise ValueError("canonical annotation image IDs must be unique")

    selected = [
        row
        for row in manifest_rows
        if row.is_keyframe
        and ((row.fold_id != held_out_fold) if split == "train" else (row.fold_id == held_out_fold))
    ]
    selected.sort(key=lambda row: (row.raw_sequence_index, row.camera_order, row.image_id))
    if not selected:
        raise ValueError(
            f"no rows selected for held-out fold {held_out_fold!r} and split {split!r}"
        )

    images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    for image_number, row in enumerate(selected, start=1):
        canonical = annotation_by_image.get(row.image_id)
        if canonical is None:
            raise ValueError(f"missing canonical annotation for {row.image_id}")
        if canonical.review_status not in {
            ReviewStatus.HUMAN_VERIFIED,
            ReviewStatus.VERIFIED_EMPTY,
        }:
            raise ValueError(f"unreviewed canonical annotation for {row.image_id}")
        images.append(
            {
                "id": image_number,
                "file_name": row.relative_path,
                "width": row.width,
                "height": row.height,
                "car5_image_id": row.image_id,
                "frame_id": row.frame_id,
                "camera_id": row.camera_id,
                "sequence_id": row.sequence_id,
                "fold_id": row.fold_id,
                "verified_empty": not canonical.boxes,
            }
        )
        for box in sorted(canonical.boxes, key=lambda item: item.annotation_id):
            x1 = min(max(box.x1, 0.0), float(row.width))
            y1 = min(max(box.y1, 0.0), float(row.height))
            x2 = min(max(box.x2, 0.0), float(row.width))
            y2 = min(max(box.y2, 0.0), float(row.height))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                raise ValueError(f"box clips to zero area: {box.annotation_id}")
            if box.label not in CLASS_INDEX:
                raise ValueError(f"inactive class in canonical annotations: {box.label}")
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_number,
                    "category_id": CLASS_INDEX[box.label] + 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "car5_annotation_id": box.annotation_id,
                    "attributes": {
                        "occluded": box.occluded,
                        "truncated": box.truncated,
                        "review_status": box.review_status.value,
                    },
                }
            )
            next_annotation_id += 1

    return {
        "info": {
            "description": "car5 six-camera reviewed keyframes",
            "held_out_fold": held_out_fold,
            "split": split,
        },
        "licenses": [],
        "categories": [
            {"id": CLASS_INDEX[label] + 1, "name": label, "supercategory": "object"}
            for label in ACTIVE_CLASSES
        ],
        "images": images,
        "annotations": coco_annotations,
    }
