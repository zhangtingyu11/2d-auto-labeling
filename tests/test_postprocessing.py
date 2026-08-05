from car5_autolabel.postprocessing import (
    apply_mining_label_policy,
    suppress_cross_class_overlaps,
    suppress_same_class_duplicates,
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


def test_cross_class_containment_removes_vehicle_part_box() -> None:
    boxes = [
        make_box("Truck", 0.90, (0, 0, 300, 300)),
        make_box("Car", 0.40, (150, 100, 260, 220)),
    ]

    kept = suppress_cross_class_overlaps(
        boxes,
        iou_threshold=0.55,
        containment_threshold=0.80,
        comparable_labels={"Car", "Truck", "WaterTruck"},
    )

    assert [box.label for box in kept] == ["Truck"]


def test_truck_watertruck_pair_can_be_excluded_from_cross_class_nms() -> None:
    boxes = [
        make_box("Truck", 0.80, (0, 0, 100, 100)),
        make_box("WaterTruck", 0.50, (1, 1, 101, 101)),
    ]

    kept = suppress_cross_class_overlaps(
        boxes,
        iou_threshold=0.55,
        containment_threshold=0.80,
        ignored_label_pairs={frozenset(("Truck", "WaterTruck"))},
        comparable_labels={"Car", "Truck", "WaterTruck"},
    )

    assert [box.label for box in kept] == ["Truck", "WaterTruck"]


def test_ambiguous_watertruck_falls_back_to_truck_with_audit_fields() -> None:
    boxes = [make_box("WaterTruck", 0.59, (0, 0, 100, 100))]

    resolved = apply_mining_label_policy(boxes, watertruck_min_score=0.6)

    assert resolved[0].label == "Truck"
    assert resolved[0].raw_label == "WaterTruck"
    assert resolved[0].postprocess_reason == "ambiguous_watertruck_fallback_to_truck"


def test_same_class_nms_removes_repeated_whole_object_box() -> None:
    boxes = [
        make_box("Excavator", 0.80, (0, 0, 200, 200)),
        make_box("Excavator", 0.40, (10, 10, 205, 205)),
    ]

    kept = suppress_same_class_duplicates(boxes)

    assert [(box.label, box.score) for box in kept] == [("Excavator", 0.8)]


def test_same_class_nms_removes_contained_part_box() -> None:
    boxes = [
        make_box("Excavator", 0.80, (0, 0, 300, 300)),
        make_box("Excavator", 0.35, (180, 30, 280, 160)),
    ]

    kept = suppress_same_class_duplicates(boxes)

    assert [(box.label, box.score) for box in kept] == [("Excavator", 0.8)]


def test_same_class_nms_prefers_lower_confidence_whole_excavator_box() -> None:
    boxes = [
        make_box("Excavator", 0.262, (1088, 260, 1689, 585)),
        make_box("Excavator", 0.483, (1320, 367, 1688, 586)),
        make_box("Excavator", 0.405, (1094, 264, 1364, 558)),
    ]

    kept = suppress_same_class_duplicates(
        boxes,
        containment_threshold=0.80,
    )

    assert [(box.label, box.score) for box in kept] == [("Excavator", 0.262)]


def test_cross_class_part_box_is_removed_by_larger_excavator() -> None:
    boxes = [
        make_box("Excavator", 0.278, (185, 452, 368, 610)),
        make_box("Truck", 0.230, (234, 528, 360, 607)),
    ]

    kept = suppress_cross_class_overlaps(
        boxes,
        iou_threshold=0.55,
        containment_threshold=0.80,
        comparable_labels={"Truck", "Excavator"},
    )

    assert [(box.label, box.score) for box in kept] == [("Excavator", 0.278)]


def test_same_class_nms_preserves_separate_objects() -> None:
    boxes = [
        make_box("Truck", 0.80, (0, 0, 100, 100)),
        make_box("Truck", 0.70, (120, 0, 220, 100)),
    ]

    kept = suppress_same_class_duplicates(boxes)

    assert len(kept) == 2
