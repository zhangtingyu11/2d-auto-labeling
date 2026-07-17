import pytest

from car5_autolabel.datasets.models import (
    AnnotationStatus,
    ManifestRow,
    ReviewStatus,
)
from car5_autolabel.integrations.label_studio import parse_label_studio_tasks


def _manifest_row(camera_id: str, frame_id: str, camera_order: int) -> ManifestRow:
    return ManifestRow(
        dataset_id="synthetic",
        image_id=f"synthetic__{frame_id}__{camera_id}",
        relative_path=f"camera/{camera_id}/{frame_id}.jpg",
        width=1920,
        height=1080,
        sequence_id="synthetic_s00",
        raw_sequence_index=0,
        frame_id=frame_id,
        capture_timestamp="2026-06-11T16:51:30.000000",
        keyframe_id=frame_id,
        is_keyframe=True,
        sync_delta_ms=0,
        camera_id=camera_id,
        camera_order=camera_order,
        fold_id="s00",
        annotation_status=AnnotationStatus.UNREVIEWED,
    )


def _task(camera_id: str, frame_id: str, result: list[dict]) -> dict:
    return {
        "data": {"camera": camera_id, "frame_id": frame_id},
        "annotations": [
            {"result": result, "was_cancelled": False, "ground_truth": True}
        ],
        "predictions": [],
    }


def _rectangle(label: str = "Truck") -> dict:
    return {
        "id": "box-1",
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": 1920,
        "original_height": 1080,
        "value": {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def test_parses_reviewed_box_and_explicit_empty_task() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [
        _manifest_row("front", frame_id, 0),
        _manifest_row("front_left", frame_id, 1),
    ]
    tasks = [
        _task("front", frame_id, [_rectangle()]),
        _task("front_left", frame_id, []),
    ]

    result = parse_label_studio_tasks(
        tasks,
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert result.report.valid
    assert result.report.stats["box_count"] == 1
    assert result.report.stats["verified_empty_image_count"] == 1
    box = result.annotations[0].boxes[0]
    assert box.label == "Truck"
    assert box.class_id == 1
    assert box.x1 == pytest.approx(192)
    assert box.y1 == pytest.approx(216)
    assert box.x2 == pytest.approx(768)
    assert box.y2 == pytest.approx(648)
    assert result.annotations[1].review_status is ReviewStatus.VERIFIED_EMPTY


def test_skipped_task_is_unreviewed_and_invalid() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [_manifest_row("front", frame_id, 0)]
    task = _task("front", frame_id, [])
    task["annotations"][0]["was_cancelled"] = True

    result = parse_label_studio_tasks(
        [task],
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert not result.report.valid
    assert {issue.code for issue in result.report.errors} == {"skipped_task"}
    assert result.annotations[0].review_status is ReviewStatus.UNREVIEWED_PREDICTION


def test_unknown_class_cannot_become_verified_empty() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [_manifest_row("front", frame_id, 0)]

    result = parse_label_studio_tasks(
        [_task("front", frame_id, [_rectangle("Pedestrian")])],
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert not result.report.valid
    codes = {issue.code for issue in result.report.errors}
    assert codes == {"invalid_rectangle_label", "no_valid_rectangles"}
    assert result.annotations[0].review_status is ReviewStatus.UNREVIEWED_PREDICTION


def test_result_dimensions_must_match_manifest() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [_manifest_row("front", frame_id, 0)]
    rectangle = _rectangle()
    rectangle["original_width"] = 1280

    result = parse_label_studio_tasks(
        [_task("front", frame_id, [rectangle])],
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert not result.report.valid
    codes = {issue.code for issue in result.report.errors}
    assert codes == {"image_dimension_mismatch", "no_valid_rectangles"}


def test_missing_keyframe_task_is_reported() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [
        _manifest_row("front", frame_id, 0),
        _manifest_row("front_left", frame_id, 1),
    ]

    result = parse_label_studio_tasks(
        [_task("front", frame_id, [])],
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert not result.report.valid
    assert {issue.code for issue in result.report.errors} == {"missing_manifest_tasks"}


def test_malformed_result_is_reported_instead_of_crashing() -> None:
    frame_id = "20260611_165130_000000"
    manifest = [_manifest_row("front", frame_id, 0)]

    result = parse_label_studio_tasks(
        [_task("front", frame_id, ["not-an-object"])],
        manifest_rows=manifest,
        dataset_id="synthetic",
    )

    assert not result.report.valid
    codes = {issue.code for issue in result.report.errors}
    assert codes == {"invalid_result", "no_valid_rectangles"}
