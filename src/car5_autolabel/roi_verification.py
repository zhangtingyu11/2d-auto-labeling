from __future__ import annotations

from collections.abc import Sequence

from car5_autolabel.postprocessing import (
    box_intersection_over_smaller,
    box_iou,
)
from car5_autolabel.schemas import BoundingBox
from car5_autolabel.tiling import TileWindow, map_box_from_tile


def make_context_crop(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    context_scale: float = 4.0,
    minimum_size: int = 320,
) -> TileWindow:
    """Build a square native-image crop around a low-confidence proposal."""
    if context_scale < 1:
        raise ValueError("context scale must be at least one")
    if minimum_size <= 0:
        raise ValueError("minimum crop size must be positive")
    center_x = (box.x1 + box.x2) / 2
    center_y = (box.y1 + box.y2) / 2
    object_size = max(box.x2 - box.x1, box.y2 - box.y1)
    side = min(
        max(round(object_size * context_scale), minimum_size),
        image_width,
        image_height,
    )
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    return TileWindow(left=left, top=top, right=left + side, bottom=top + side)


def expand_context_crop(
    crop: TileWindow,
    *,
    image_width: int,
    image_height: int,
    scale: float = 2.0,
) -> TileWindow:
    """Expand a crop around its center while staying inside the source image."""
    if scale <= 1:
        raise ValueError("crop expansion scale must be greater than one")
    center_x = (crop.left + crop.right) / 2
    center_y = (crop.top + crop.bottom) / 2
    side = min(
        round(max(crop.width, crop.height) * scale),
        image_width,
        image_height,
    )
    left = min(max(round(center_x - side / 2), 0), image_width - side)
    top = min(max(round(center_y - side / 2), 0), image_height - side)
    return TileWindow(left=left, top=top, right=left + side, bottom=top + side)


def touches_context_crop_edge(
    box: BoundingBox,
    crop: TileWindow,
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float = 0.03,
) -> bool:
    """Flag a mapped detection that may be a fragment clipped by an ROI edge."""
    if not 0 <= margin_ratio < 0.5:
        raise ValueError("crop edge margin ratio must be in [0, 0.5)")
    margin_x = crop.width * margin_ratio
    margin_y = crop.height * margin_ratio
    return (
        (crop.left > 0 and box.x1 <= crop.left + margin_x)
        or (
            crop.right < image_width
            and box.x2 >= crop.right - margin_x
        )
        or (crop.top > 0 and box.y1 <= crop.top + margin_y)
        or (
            crop.bottom < image_height
            and box.y2 >= crop.bottom - margin_y
        )
    )


def is_small_proposal(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    maximum_side: float = 180.0,
    maximum_area_ratio: float = 0.02,
) -> bool:
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    area_ratio = width * height / (image_width * image_height)
    return max(width, height) <= maximum_side or area_ratio <= maximum_area_ratio


def select_verified_detection(
    proposal: BoundingBox,
    local_detections: Sequence[BoundingBox],
    crop: TileWindow,
    *,
    confirmation_threshold: float,
    minimum_iou: float = 0.05,
    minimum_intersection_over_smaller: float = 0.50,
    allow_reclassification: bool = False,
    reclassification_threshold: float = 0.80,
) -> BoundingBox | None:
    """Select a crop detection that spatially confirms a global proposal."""
    mapped = [map_box_from_tile(box, crop) for box in local_detections]
    def spatial_match(candidate: BoundingBox) -> bool:
        return (
            box_iou(proposal, candidate) >= minimum_iou
            or box_intersection_over_smaller(proposal, candidate)
            >= minimum_intersection_over_smaller
        )

    eligible = [
        candidate
        for candidate in mapped
        if candidate.score >= confirmation_threshold
        and spatial_match(candidate)
        and (
            candidate.label == proposal.label
            or (
                allow_reclassification
                and candidate.score >= reclassification_threshold
            )
        )
    ]
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda candidate: (
            candidate.score + (0.03 if candidate.label == proposal.label else 0),
            box_iou(proposal, candidate),
        ),
    )
    reason = (
        "native_crop_verified_same_class"
        if chosen.label == proposal.label
        else f"native_crop_reclassified_from_{proposal.label}"
    )
    return chosen.model_copy(update={"postprocess_reason": reason})
