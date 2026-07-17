from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from car5_autolabel.datasets.models import ACTIVE_CLASSES, ManifestRow, ReviewedTask


CATEGORY_NAMES = {
    0: "Car",
    1: "Truck",
    2: "Bulldozer",
    3: "Excavator",
    4: "WaterTruck",
    5: "Sign",
}


def _write_json(payload: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _coco_payload(
    rows: list[ManifestRow],
    reviewed_tasks: dict[tuple[str, str], ReviewedTask],
    *,
    dataset_id: str,
    split_name: str,
) -> dict[str, Any]:
    supervised = sorted(
        (row for row in rows if row.is_keyframe),
        key=lambda row: (row.sequence_id, row.frame_id, row.camera_order),
    )
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, row in enumerate(supervised, start=1):
        reviewed = reviewed_tasks.get((row.camera_id, row.capture_timestamp))
        if reviewed is None or reviewed.review_status not in {"human_verified", "verified_empty"}:
            raise ValueError(
                f"cannot release unreviewed task: {row.camera_id}/{row.capture_timestamp}"
            )
        images.append(
            {
                "id": image_id,
                "stable_image_id": row.image_id,
                "file_name": row.relative_path,
                "width": row.width,
                "height": row.height,
                "camera_id": row.camera_id,
                "camera_order": row.camera_order,
                "capture_timestamp": row.capture_timestamp,
                "sequence_id": row.sequence_id,
                "frame_id": row.frame_id,
                "review_status": reviewed.review_status,
                "sha256": row.sha256,
            }
        )
        for box in reviewed.boxes:
            x1, y1, x2, y2 = box.xyxy
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box.class_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "source": "human",
                    "review_status": reviewed.review_status,
                    "source_hint": box.source,
                    "source_result_id": box.result_id,
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "car5 reviewed pure-2D bounding boxes",
            "dataset_id": dataset_id,
            "split": split_name,
            "annotation_policy": "annotation_v1",
            "taxonomy": "taxonomy_v1",
        },
        "licenses": [],
        "categories": [
            {"id": category_id, "name": CATEGORY_NAMES[category_id], "supercategory": "object"}
            for category_id in sorted(CATEGORY_NAMES)
        ],
        "images": images,
        "annotations": annotations,
    }


def build_coco_release(
    rows: list[ManifestRow],
    reviewed_tasks: dict[tuple[str, str], ReviewedTask],
    output_dir: str | Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Write the complete reviewed set and five leave-one-sequence-out folds."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = sorted({row.sequence_id for row in rows if row.is_keyframe})
    if not sequences:
        raise ValueError("no supervised sequences found")

    files: dict[str, str] = {}
    full_payload = _coco_payload(
        rows, reviewed_tasks, dataset_id=dataset_id, split_name="development_all"
    )
    files["annotations_all.json"] = _write_json(full_payload, output / "annotations_all.json")

    for fold_index, validation_sequence in enumerate(sequences):
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(exist_ok=True)
        train_rows = [row for row in rows if row.sequence_id != validation_sequence]
        validation_rows = [row for row in rows if row.sequence_id == validation_sequence]
        train_payload = _coco_payload(
            train_rows,
            reviewed_tasks,
            dataset_id=dataset_id,
            split_name=f"fold_{fold_index}_train",
        )
        validation_payload = _coco_payload(
            validation_rows,
            reviewed_tasks,
            dataset_id=dataset_id,
            split_name=f"fold_{fold_index}_validation_{validation_sequence}",
        )
        files[f"fold_{fold_index}/train.json"] = _write_json(
            train_payload, fold_dir / "train.json"
        )
        files[f"fold_{fold_index}/validation.json"] = _write_json(
            validation_payload, fold_dir / "validation.json"
        )

    class_counts = Counter(
        CATEGORY_NAMES[annotation["category_id"]] for annotation in full_payload["annotations"]
    )
    summary = {
        "dataset_id": dataset_id,
        "images": len(full_payload["images"]),
        "annotations": len(full_payload["annotations"]),
        "verified_empty_images": sum(
            image["review_status"] == "verified_empty" for image in full_payload["images"]
        ),
        "class_counts": dict(sorted(class_counts.items())),
        "sequences": sequences,
        "folds": len(sequences),
        "files": dict(sorted(files.items())),
    }
    summary["release_sha256"] = _write_json(summary, output / "release_manifest.json")
    return summary
