import json
from pathlib import Path

from car5_autolabel.datasets.models import ManifestRow, ReviewedBox, ReviewedTask
from car5_autolabel.datasets.release import build_coco_release


def _row(timestamp: str, sequence: str, camera: str, order: int) -> ManifestRow:
    return ManifestRow(
        dataset_id="synthetic-v1",
        image_id=f"synthetic-v1:{timestamp}:{camera}",
        relative_path=f"camera/{camera}/{timestamp}.jpg",
        sha256=None,
        width=1920,
        height=1080,
        sequence_id=sequence,
        frame_id=order,
        capture_timestamp=timestamp,
        raw_sequence_index=order,
        keyframe_id=timestamp,
        is_keyframe=True,
        sync_delta_ms=0.0,
        camera_id=camera,
        camera_order=order,
        fold_id=sequence,
        annotation_status="human_verified",
    )


def test_release_keeps_verified_empty_images_and_sequence_folds(tmp_path: Path) -> None:
    rows = [
        _row("20260101_000000_000000", "s00", "front", 0),
        _row("20260101_000002_000000", "s01", "front", 0),
    ]
    reviewed = {
        ("front", "20260101_000000_000000"): ReviewedTask(
            image_id=rows[0].image_id,
            frame_id=rows[0].capture_timestamp,
            camera_id="front",
            review_status="human_verified",
            boxes=(
                ReviewedBox(
                    class_id=1,
                    class_name="truck",
                    xyxy=(10, 20, 110, 220),
                    source="human",
                ),
            ),
        ),
        ("front", "20260101_000002_000000"): ReviewedTask(
            image_id=rows[1].image_id,
            frame_id=rows[1].capture_timestamp,
            camera_id="front",
            review_status="verified_empty",
        ),
    }

    summary = build_coco_release(rows, reviewed, tmp_path, dataset_id="synthetic-v1")

    assert summary["images"] == 2
    assert summary["annotations"] == 1
    assert summary["verified_empty_images"] == 1
    assert summary["class_counts"] == {"Truck": 1}
    assert summary["folds"] == 2
    fold_zero_validation = json.loads((tmp_path / "fold_0" / "validation.json").read_text())
    assert {image["sequence_id"] for image in fold_zero_validation["images"]} == {"s00"}
    assert fold_zero_validation["annotations"][0]["bbox"] == [10, 20, 100, 200]
