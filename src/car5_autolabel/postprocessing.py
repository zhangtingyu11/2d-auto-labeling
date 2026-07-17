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


def suppress_cross_class_overlaps(
    boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float | None,
    watertruck_ambiguity_margin: float = 0.0,
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
        return ordered
    if not 0 < iou_threshold <= 1:
        raise ValueError("cross-class NMS IoU threshold must be in (0, 1]")

    kept: list[BoundingBox] = []
    for candidate in ordered:
        duplicate = any(
            candidate.label != accepted.label
            and box_iou(candidate, accepted) >= iou_threshold
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
