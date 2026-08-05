"""Minimum visible-box size policy used by inference and evaluation."""

from collections.abc import Iterable

from car5_autolabel.schemas import BoundingBox


def long_side_px(box: BoundingBox) -> int:
    """Return the rounded longest visible side of a pixel-space box."""
    return max(round(box.x2 - box.x1), round(box.y2 - box.y1))


def filter_minimum_long_side(
    boxes: Iterable[BoundingBox],
    *,
    minimum_px: int = 70,
) -> tuple[list[BoundingBox], list[BoundingBox]]:
    """Split boxes into formal outputs and below-policy review candidates."""
    if minimum_px < 0:
        raise ValueError("minimum long side must be non-negative")
    kept: list[BoundingBox] = []
    below_policy: list[BoundingBox] = []
    for box in boxes:
        destination = kept if long_side_px(box) >= minimum_px else below_policy
        destination.append(box)
    return kept, below_policy
