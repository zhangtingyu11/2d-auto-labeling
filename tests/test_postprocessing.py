from car5_autolabel.postprocessing import (
    apply_mining_label_policy,
    suppress_cross_class_overlaps,
)
from car5_autolabel.schemas import BoundingBox


def make_box(label: str, score: float, xyxy: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(
        label=label,
        score=score,
        x1=xyxy[0],
        y1=xyxy[1],
        x2=xyxy[2],
        y2=xyxy[3],
    )


def test_cross_class_nms_keeps_strongest_label() -> None:
    boxes = [
        make_box("Car", 0.55, (1, 1, 101, 101)),
        make_box("Truck", 0.90, (0, 0, 100, 100)),
    ]

    kept = suppress_cross_class_overlaps(boxes, iou_threshold=0.5)

    assert [(box.label, box.score) for box in kept] == [("Truck", 0.9)]


def test_cross_class_nms_does_not_merge_two_same_class_objects() -> None:
    boxes = [
        make_box("Truck", 0.90, (0, 0, 100, 100)),
        make_box("Truck", 0.80, (5, 5, 105, 105)),
    ]

    kept = suppress_cross_class_overlaps(boxes, iou_threshold=0.5)

    assert len(kept) == 2


def test_cross_class_nms_preserves_non_overlapping_classes() -> None:
    boxes = [
        make_box("Truck", 0.90, (0, 0, 100, 100)),
        make_box("Excavator", 0.80, (200, 200, 300, 300)),
    ]

    kept = suppress_cross_class_overlaps(boxes, iou_threshold=0.5)

    assert [box.label for box in kept] == ["Truck", "Excavator"]


def test_cross_class_nms_can_be_disabled() -> None:
    boxes = [
        make_box("Car", 0.55, (1, 1, 101, 101)),
        make_box("Truck", 0.90, (0, 0, 100, 100)),
    ]

    kept = suppress_cross_class_overlaps(boxes, iou_threshold=None)

    assert len(kept) == 2


def test_truck_wins_ambiguous_watertruck_overlap() -> None:
    boxes = [
        make_box("Truck", 0.60, (0, 0, 100, 100)),
        make_box("WaterTruck", 0.70, (1, 1, 101, 101)),
    ]

    kept = suppress_cross_class_overlaps(
        boxes,
        iou_threshold=0.5,
        watertruck_ambiguity_margin=0.15,
    )

    assert [box.label for box in kept] == ["Truck"]


def test_clearly_stronger_watertruck_wins_overlap() -> None:
    boxes = [
        make_box("Truck", 0.60, (0, 0, 100, 100)),
        make_box("WaterTruck", 0.80, (1, 1, 101, 101)),
    ]

    kept = suppress_cross_class_overlaps(
        boxes,
        iou_threshold=0.5,
        watertruck_ambiguity_margin=0.15,
    )

    assert [box.label for box in kept] == ["WaterTruck"]


def test_ambiguous_watertruck_falls_back_to_truck_with_audit_fields() -> None:
    boxes = [make_box("WaterTruck", 0.59, (0, 0, 100, 100))]

    resolved = apply_mining_label_policy(boxes, watertruck_min_score=0.6)

    assert resolved[0].label == "Truck"
    assert resolved[0].raw_label == "WaterTruck"
    assert resolved[0].postprocess_reason == "ambiguous_watertruck_fallback_to_truck"
