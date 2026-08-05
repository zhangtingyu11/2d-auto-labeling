from __future__ import annotations

from collections.abc import Sequence

from car5_autolabel.schemas import BoundingBox


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def box_intersection_over_smaller(left: BoundingBox, right: BoundingBox) -> float:
    """Return how much of the smaller box is covered by the other box."""
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    smaller = min(left_area, right_area)
    return intersection / smaller if smaller > 0 else 0.0


def suppress_same_class_duplicates(
    boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float | None = 0.55,
    containment_threshold: float | None = 0.90,
    whole_box_score_ratio: float = 0.45,
    min_whole_area_ratio: float = 1.25,
) -> list[BoundingBox]:
    """Remove repeated whole-object and part-only boxes of the same class.

    RF-DETR is end-to-end, but low deployment thresholds can still expose
    multiple queries for one object. IoU catches repeated whole-object boxes;
    intersection-over-smaller catches lower-score arm/base/body fragments that
    are almost entirely contained by the stronger full-object box.
    """
    if iou_threshold is not None and not 0 < iou_threshold <= 1:
        raise ValueError("same-class NMS IoU threshold must be in (0, 1]")
    if containment_threshold is not None and not 0 < containment_threshold <= 1:
        raise ValueError("containment threshold must be in (0, 1]")
    if not 0 <= whole_box_score_ratio <= 1:
        raise ValueError("whole-box score ratio must be between zero and one")
    if min_whole_area_ratio < 1:
        raise ValueError("minimum whole-box area ratio must be at least one")

    # Phase 1: repeated whole-object queries are confidence-led.
    ordered = sorted(
        boxes,
        key=lambda box: (-box.score, box.label, box.x1, box.y1, box.x2, box.y2),
    )
    iou_kept: list[BoundingBox] = []
    for candidate in ordered:
        duplicate = any(
            candidate.label == accepted.label
            and iou_threshold is not None
            and box_iou(candidate, accepted) >= iou_threshold
            for accepted in iou_kept
        )
        if not duplicate:
            iou_kept.append(candidate)

    # Phase 2: a plausible whole-object box can outrank higher-confidence
    # fragments such as an excavator arm or truck body section.
    def area(box: BoundingBox) -> float:
        return (box.x2 - box.x1) * (box.y2 - box.y1)

    area_ordered = sorted(
        iou_kept,
        key=lambda box: (-area(box), -box.score, box.label),
    )
    kept: list[BoundingBox] = []
    for candidate in area_ordered:
        candidate_area = area(candidate)
        contained_part = any(
            candidate.label == accepted.label
            and containment_threshold is not None
            and area(accepted) >= candidate_area * min_whole_area_ratio
            and box_intersection_over_smaller(candidate, accepted)
            >= containment_threshold
            and accepted.score >= candidate.score * whole_box_score_ratio
            for accepted in kept
        )
        if not contained_part:
            kept.append(candidate)
    return sorted(
        kept,
        key=lambda box: (-box.score, box.label, box.x1, box.y1, box.x2, box.y2),
    )


def suppress_cross_class_overlaps(
    boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float | None,
    containment_threshold: float | None = None,
    watertruck_ambiguity_margin: float = 0.0,
    ignored_label_pairs: set[frozenset[str]] | None = None,
    comparable_labels: set[str] | None = None,
    whole_box_score_ratio: float = 0.45,
    min_whole_area_ratio: float = 1.25,
) -> list[BoundingBox]:
    """Keep the strongest label when different classes describe the same object.

    RTMDet already performs per-class NMS. This second pass only compares boxes
    with different labels, so two nearby trucks are not accidentally merged.
    """
    if not 0 <= watertruck_ambiguity_margin <= 1:
        raise ValueError("water-truck ambiguity margin must be between zero and one")

    def selection_score(box: BoundingBox) -> float:
        if box.label == "WaterTruck":
            return box.score - watertruck_ambiguity_margin
        return box.score

    ordered = sorted(
        boxes,
        key=lambda box: (
            -selection_score(box),
            -box.score,
            box.label,
            box.x1,
            box.y1,
            box.x2,
            box.y2,
        ),
    )
    if iou_threshold is None:
        if containment_threshold is None:
            return ordered
    elif not 0 < iou_threshold <= 1:
        raise ValueError("cross-class NMS IoU threshold must be in (0, 1]")
    if containment_threshold is not None and not 0 < containment_threshold <= 1:
        raise ValueError("cross-class containment threshold must be in (0, 1]")
    if not 0 <= whole_box_score_ratio <= 1:
        raise ValueError("whole-box score ratio must be between zero and one")
    if min_whole_area_ratio < 1:
        raise ValueError("minimum whole-box area ratio must be at least one")

    def comparable(candidate: BoundingBox, accepted: BoundingBox) -> bool:
        return (
            candidate.label != accepted.label
            and (
                comparable_labels is None
                or (
                    candidate.label in comparable_labels
                    and accepted.label in comparable_labels
                )
            )
            and (
                ignored_label_pairs is None
                or frozenset((candidate.label, accepted.label))
                not in ignored_label_pairs
            )
        )

    # Equal-scale cross-class duplicates remain confidence-led.
    iou_kept: list[BoundingBox] = []
    for candidate in ordered:
        duplicate = any(
            comparable(candidate, accepted)
            and iou_threshold is not None
            and box_iou(candidate, accepted) >= iou_threshold
            for accepted in iou_kept
        )
        if not duplicate:
            iou_kept.append(candidate)

    # For contained cross-class fragments, prefer a plausible larger whole box.
    def area(box: BoundingBox) -> float:
        return (box.x2 - box.x1) * (box.y2 - box.y1)

    area_ordered = sorted(
        iou_kept,
        key=lambda box: (-area(box), -selection_score(box), -box.score),
    )
    kept: list[BoundingBox] = []
    for candidate in area_ordered:
        candidate_area = area(candidate)
        duplicate = any(
            comparable(candidate, accepted)
            and containment_threshold is not None
            and area(accepted) >= candidate_area * min_whole_area_ratio
            and box_intersection_over_smaller(candidate, accepted)
            >= containment_threshold
            and accepted.score >= candidate.score * whole_box_score_ratio
            for accepted in kept
        )
        if not duplicate:
            kept.append(candidate)
    return sorted(
        kept,
        key=lambda box: (-box.score, box.label, box.x1, box.y1, box.x2, box.y2),
    )


def apply_mining_label_policy(
    boxes: Sequence[BoundingBox],
    *,
    watertruck_min_score: float | None,
) -> list[BoundingBox]:
    """Use Truck as the safe fallback unless WaterTruck evidence is clear."""
    if watertruck_min_score is None:
        return list(boxes)
    if not 0 <= watertruck_min_score <= 1:
        raise ValueError("WaterTruck minimum score must be between zero and one")

    resolved = []
    for box in boxes:
        if box.label == "WaterTruck" and box.score < watertruck_min_score:
            resolved.append(
                box.model_copy(
                    update={
                        "label": "Truck",
                        "raw_label": "WaterTruck",
                        "postprocess_reason": "ambiguous_watertruck_fallback_to_truck",
                    }
                )
            )
        else:
            resolved.append(box)
    return sorted(
        resolved,
        key=lambda box: (-box.score, box.label, box.x1, box.y1, box.x2, box.y2),
    )
