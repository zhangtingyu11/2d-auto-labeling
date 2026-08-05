from __future__ import annotations

from dataclasses import dataclass

from car5_autolabel.schemas import BoundingBox


@dataclass(frozen=True)
class TileWindow:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tile_windows(
    width: int,
    height: int,
    *,
    tile_size: int,
    overlap: float,
) -> list[TileWindow]:
    """Cover an image with deterministic overlapping square tiles."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("tile overlap must be in [0, 1)")

    windows = [
        TileWindow(
            left=x,
            top=y,
            right=min(x + tile_size, width),
            bottom=min(y + tile_size, height),
        )
        for y in _axis_starts(height, tile_size, overlap)
        for x in _axis_starts(width, tile_size, overlap)
    ]
    full_image = TileWindow(0, 0, width, height)
    return [window for window in windows if window != full_image]


def touches_internal_tile_edge(
    box: BoundingBox,
    window: TileWindow,
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> bool:
    """Detect likely crop fragments while preserving real image-edge objects."""
    if not 0 <= margin_ratio < 0.5:
        raise ValueError("tile edge margin ratio must be in [0, 0.5)")
    margin_x = window.width * margin_ratio
    margin_y = window.height * margin_ratio
    center_x = (box.x1 + box.x2) / 2
    center_y = (box.y1 + box.y2) / 2
    return (
        (window.left > 0 and center_x <= margin_x)
        or (window.right < image_width and center_x >= window.width - margin_x)
        or (window.top > 0 and center_y <= margin_y)
        or (window.bottom < image_height and center_y >= window.height - margin_y)
    )


def map_box_from_tile(box: BoundingBox, window: TileWindow) -> BoundingBox:
    return box.model_copy(
        update={
            "x1": box.x1 + window.left,
            "y1": box.y1 + window.top,
            "x2": box.x2 + window.left,
            "y2": box.y2 + window.top,
            "postprocess_reason": "high_resolution_tile_candidate",
        }
    )
