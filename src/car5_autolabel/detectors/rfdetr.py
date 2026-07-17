from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from car5_autolabel.datasets.constants import ACTIVE_CLASSES
from car5_autolabel.detectors.base import Detector
from car5_autolabel.schemas import BoundingBox, ImagePrediction


class RFDETRDetector(Detector):
    """RF-DETR inference adapter that emits the project prediction contract."""

    def __init__(
        self,
        weights: Path,
        *,
        threshold: float = 0.2,
        model_version: str = "rfdetr-small-car5-v1",
        optimize: bool = True,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self._threshold = threshold
        self._model_version = model_version
        if model is not None:
            self._model = model
            return

        from rfdetr import RFDETRSmall

        self._model = RFDETRSmall(
            pretrain_weights=str(weights.resolve()),
            num_classes=len(ACTIVE_CLASSES),
        )
        if optimize:
            import torch

            self._model.optimize_for_inference(dtype=torch.float16)

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, image_path: Path) -> ImagePrediction:
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode image: {image_path}")
        height, width = image.shape[:2]
        detections = self._model.predict(str(image_path), threshold=self._threshold)

        boxes: list[BoundingBox] = []
        for xyxy, confidence, class_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            strict=True,
        ):
            class_index = int(class_id)
            if not 0 <= class_index < len(ACTIVE_CLASSES):
                raise ValueError(f"RF-DETR returned an unknown class ID: {class_index}")
            x1 = min(max(float(xyxy[0]), 0.0), float(width))
            y1 = min(max(float(xyxy[1]), 0.0), float(height))
            x2 = min(max(float(xyxy[2]), 0.0), float(width))
            y2 = min(max(float(xyxy[3]), 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=ACTIVE_CLASSES[class_index],
                    score=float(confidence),
                )
            )

        return ImagePrediction(
            image_id=image_path.stem,
            width=width,
            height=height,
            model_version=self.model_version,
            boxes=boxes,
        )
