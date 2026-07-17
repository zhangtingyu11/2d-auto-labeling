from car5_autolabel.datasets.coco import build_coco_dataset
from car5_autolabel.datasets.models import (
    AnnotationStatus,
    CanonicalBox,
    CanonicalImageAnnotation,
    ManifestRow,
    ReviewStatus,
)


def _row(image_id: str, frame_id: str, fold_id: str, raw_index: int) -> ManifestRow:
    return ManifestRow(
        dataset_id="synthetic",
        image_id=image_id,
        relative_path=f"camera/front/{frame_id}.jpg",
        width=1920,
        height=1080,
        sequence_id=f"synthetic_{fold_id}",
        raw_sequence_index=raw_index,
        frame_id=frame_id,
        capture_timestamp="2026-06-11T16:51:30.000000",
        keyframe_id=frame_id,
        is_keyframe=True,
        camera_id="front",
        camera_order=0,
        fold_id=fold_id,
        annotation_status=AnnotationStatus.REVIEWED,
    )


def test_coco_export_uses_sequence_holdout_and_preserves_empty_images() -> None:
    train_row = _row("train-image", "20260611_165130_000000", "s00", 0)
    val_row = _row("val-image", "20260611_165132_000000", "s01", 1)
    box = CanonicalBox(
        annotation_id="box-1",
        image_id="train-image",
        class_id=1,
        label="Truck",
        x1=100,
        y1=200,
        x2=400,
        y2=500,
        review_status=ReviewStatus.HUMAN_VERIFIED,
    )
    annotations = [
        CanonicalImageAnnotation(
            image_id="train-image",
            camera_id="front",
            frame_id=train_row.frame_id,
            review_status=ReviewStatus.HUMAN_VERIFIED,
            boxes=[box],
        ),
        CanonicalImageAnnotation(
            image_id="val-image",
            camera_id="front",
            frame_id=val_row.frame_id,
            review_status=ReviewStatus.VERIFIED_EMPTY,
            boxes=[],
        ),
    ]

    train = build_coco_dataset(
        [train_row, val_row], annotations, held_out_fold="s01", split="train"
    )
    val = build_coco_dataset(
        [train_row, val_row], annotations, held_out_fold="s01", split="val"
    )

    assert [image["car5_image_id"] for image in train["images"]] == ["train-image"]
    assert train["annotations"][0]["bbox"] == [100.0, 200.0, 300.0, 300.0]
    assert train["annotations"][0]["category_id"] == 2
    assert [image["car5_image_id"] for image in val["images"]] == ["val-image"]
    assert val["images"][0]["verified_empty"] is True
    assert val["annotations"] == []
