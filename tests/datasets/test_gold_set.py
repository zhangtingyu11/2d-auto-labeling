from car5_autolabel.datasets.constants import CAMERA_ORDER
from car5_autolabel.datasets.gold import build_gold_set
from car5_autolabel.datasets.models import (
    AnnotationStatus,
    CanonicalBox,
    CanonicalImageAnnotation,
    ManifestRow,
    ReviewStatus,
)


def _release_fixture() -> tuple[list[ManifestRow], list[CanonicalImageAnnotation]]:
    rows: list[ManifestRow] = []
    annotations: list[CanonicalImageAnnotation] = []
    for sequence_index in range(5):
        for local_index in range(4):
            raw_index = sequence_index * 4 + local_index
            frame_id = f"20260611_1651{raw_index:02d}_000000"
            for camera_order, camera_id in enumerate(CAMERA_ORDER):
                image_id = f"synthetic__{frame_id}__{camera_id}"
                rows.append(
                    ManifestRow(
                        dataset_id="synthetic",
                        image_id=image_id,
                        relative_path=f"camera/{camera_id}/{frame_id}.jpg",
                        sha256=f"{raw_index:02X}{camera_order:02X}".ljust(64, "A"),
                        width=1920,
                        height=1080,
                        sequence_id=f"synthetic_s{sequence_index:02d}",
                        raw_sequence_index=raw_index,
                        frame_id=frame_id,
                        capture_timestamp="2026-06-11T16:51:30.000000",
                        keyframe_id=frame_id,
                        is_keyframe=True,
                        camera_id=camera_id,
                        camera_order=camera_order,
                        fold_id=f"s{sequence_index:02d}",
                        annotation_status=AnnotationStatus.REVIEWED,
                    )
                )
                boxes = []
                status = ReviewStatus.VERIFIED_EMPTY
                if local_index == 0 and camera_order == 0:
                    boxes = [
                        CanonicalBox(
                            annotation_id=f"box-{raw_index}",
                            image_id=image_id,
                            class_id=5,
                            label="Sign",
                            x1=0,
                            y1=10,
                            x2=8,
                            y2=70,
                            review_status=ReviewStatus.HUMAN_VERIFIED,
                        )
                    ]
                    status = ReviewStatus.HUMAN_VERIFIED
                annotations.append(
                    CanonicalImageAnnotation(
                        image_id=image_id,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        review_status=status,
                        boxes=boxes,
                    )
                )
    return rows, annotations


def test_gold_set_is_deterministic_and_keeps_complete_camera_groups() -> None:
    rows, annotations = _release_fixture()
    first = build_gold_set(
        rows,
        annotations,
        dataset_release_id="synthetic-v1",
        timestamp_groups=10,
    )
    second = build_gold_set(
        rows,
        annotations,
        dataset_release_id="synthetic-v1",
        timestamp_groups=10,
    )

    assert first.report.valid
    assert first.rows == second.rows
    assert len(first.rows) == 60
    assert first.report.stats["timestamp_group_count"] == 10
    assert first.report.stats["sequence_group_counts"] == {
        f"synthetic_s{index:02d}": 2 for index in range(5)
    }
    for offset in range(0, len(first.rows), 6):
        group = first.rows[offset : offset + 6]
        assert [row.camera_id for row in group] == list(CAMERA_ORDER)
        assert len({row.frame_id for row in group}) == 1


def test_gold_set_rejects_missing_release_hashes() -> None:
    rows, annotations = _release_fixture()
    rows[0].sha256 = None

    result = build_gold_set(
        rows,
        annotations,
        dataset_release_id="synthetic-v1",
        timestamp_groups=10,
    )

    assert not result.report.valid
    assert {issue.code for issue in result.report.errors} == {"missing_hashes"}
