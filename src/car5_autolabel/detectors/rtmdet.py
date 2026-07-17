from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from car5_autolabel.detectors.base import Detector
from car5_autolabel.schemas import BoundingBox, ImagePrediction

ACTIVE_CLASS_NAMES = ("Car", "Truck", "Bulldozer", "Excavator", "WaterTruck", "Sign")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instances_to_boxes(
    bboxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    width: int,
    height: int,
    score_threshold: float,
    class_names: Sequence[str] = ACTIVE_CLASS_NAMES,
) -> list[BoundingBox]:
    """Convert model arrays to clipped, deterministic internal pixel boxes."""
    if not (len(bboxes) == len(scores) == len(labels)):
        raise ValueError("bboxes, scores, and labels must have identical lengths")

    converted: list[BoundingBox] = []
    for raw_box, raw_score, raw_label in zip(bboxes, scores, labels, strict=True):
        score = float(raw_score)
        label_id = int(raw_label)
        if score < score_threshold:
            continue
        if label_id < 0 or label_id >= len(class_names):
            raise ValueError(f"model returned unknown class id {label_id}")
        if len(raw_box) != 4:
            raise ValueError("each predicted box must contain four xyxy coordinates")
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        converted.append(
            BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=class_names[label_id],
                score=score,
            )
        )
    return sorted(
        converted,
        key=lambda box: (-box.score, box.label, box.x1, box.y1, box.x2, box.y2),
    )


class RTMDetDetector(Detector):
    """Lazy MMDetection RTMDet adapter for the separate GPU worker environment."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        dataset_manifest_id: str,
        device: str = "cuda:0",
        score_threshold: float = 0.1,
        model_version: str | None = None,
    ) -> None:
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.dataset_manifest_id = dataset_manifest_id
        self.device = device
        self.score_threshold = score_threshold
        self._model_version = model_version or self.checkpoint_path.stem
        self.config_sha256 = sha256_file(self.config_path)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

        # MMDetection is intentionally imported only inside the GPU process.
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        self._model = init_detector(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )
        self._test_pipeline = Compose(get_test_pipeline_cfg(self._model.cfg.copy()))
        checkpoint_classes = tuple((self._model.dataset_meta or {}).get("classes", ()))
        if checkpoint_classes and checkpoint_classes != ACTIVE_CLASS_NAMES:
            raise ValueError(
                "checkpoint class order does not match the immutable v1 taxonomy: "
                f"{checkpoint_classes!r}"
            )

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, image_path: Path, image_id: str | None = None) -> ImagePrediction:
        return self.predict_batch([image_path], [image_id or Path(image_path).stem])[0]

    def predict_batch(
        self,
        image_paths: Sequence[str | Path],
        image_ids: Sequence[str],
    ) -> list[ImagePrediction]:
        """Run one true GPU batch instead of looping over single-image calls."""
        import torch

        if len(image_paths) != len(image_ids):
            raise ValueError("image_paths and image_ids must have identical lengths")
        prepared: list[dict[str, Any]] = []
        for index, image_path in enumerate(image_paths):
            resolved = Path(image_path).resolve()
            if not resolved.is_file():
                raise ValueError(f"cannot read image: {resolved}")
            prepared.append(self._test_pipeline(dict(img_path=str(resolved), img_id=index)))

        batch = {
            "inputs": [item["inputs"] for item in prepared],
            "data_samples": [item["data_samples"] for item in prepared],
        }
        with torch.no_grad():
            samples = self._model.test_step(batch)

        predictions: list[ImagePrediction] = []
        for image_id, sample in zip(image_ids, samples, strict=True):
            height, width = (int(value) for value in sample.ori_shape[:2])
            instances = sample.pred_instances.cpu()
            boxes = instances_to_boxes(
                instances.bboxes.tolist(),
                instances.scores.tolist(),
                instances.labels.tolist(),
                width=width,
                height=height,
                score_threshold=self.score_threshold,
            )
            predictions.append(
                ImagePrediction(
                    image_id=image_id,
                    width=width,
                    height=height,
                    model_version=self.model_version,
                    config_sha256=self.config_sha256,
                    checkpoint_sha256=self.checkpoint_sha256,
                    dataset_manifest_id=self.dataset_manifest_id,
                    source="detector",
                    review_status="unreviewed_prediction",
                    boxes=boxes,
                )
            )
        return predictions
