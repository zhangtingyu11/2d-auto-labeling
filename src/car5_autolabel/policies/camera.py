from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator


class IgnoreRegion(BaseModel):
    name: str
    polygon_xy: list[tuple[float, float]] = Field(min_length=3)


class CameraPolicy(BaseModel):
    version: int = Field(ge=1)
    camera_id: str
    image_size: tuple[int, int]
    ignore_regions: list[IgnoreRegion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_geometry(self) -> CameraPolicy:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("camera image size must be positive")
        for region in self.ignore_regions:
            for x, y in region.polygon_xy:
                if not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError(
                        f"ignore polygon {region.name!r} is outside {width}x{height}"
                    )
        return self


def load_camera_policy(path: str | Path) -> CameraPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CameraPolicy.model_validate(payload)


def build_ignore_mask(policy: CameraPolicy) -> np.ndarray:
    width, height = policy.image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in policy.ignore_regions:
        polygon = np.rint(np.asarray(region.polygon_xy, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=1)
    return mask.astype(bool)


def intersection_over_box_area(mask: np.ndarray, box_xyxy: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError("ignore mask must have shape (height, width)")
    height, width = mask.shape
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float64)
    left = min(max(int(np.floor(x1)), 0), width)
    top = min(max(int(np.floor(y1)), 0), height)
    right = min(max(int(np.ceil(x2)), 0), width)
    bottom = min(max(int(np.ceil(y2)), 0), height)
    if right <= left or bottom <= top:
        return 0.0
    crop = mask[top:bottom, left:right]
    return float(np.count_nonzero(crop) / crop.size)
