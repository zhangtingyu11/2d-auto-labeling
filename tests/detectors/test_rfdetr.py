from pathlib import Path

import cv2
import numpy as np

from car5_autolabel.detectors.rfdetr import RFDETRDetector


class _Detections:
    xyxy = np.array([[-10.0, 5.0, 80.0, 70.0], [20.0, 20.0, 20.0, 30.0]])
    confidence = np.array([0.91, 0.8])
    class_id = np.array([1, 3])


class _Model:
    def predict(self, image_path: str, *, threshold: float) -> _Detections:
        assert Path(image_path).is_file()
        assert threshold == 0.25
        return _Detections()


def test_rfdetr_adapter_clips_boxes_and_uses_active_class_order(tmp_path: Path) -> None:
    image_path = tmp_path / "frame-001.jpg"
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    image_path.write_bytes(encoded.tobytes())
    detector = RFDETRDetector(
        tmp_path / "unused.pth",
        threshold=0.25,
        model_version="test-rfdetr",
        model=_Model(),
    )

    prediction = detector.predict(image_path)

    assert prediction.image_id == "frame-001"
    assert prediction.width == 100
    assert prediction.height == 60
    assert prediction.model_version == "test-rfdetr"
    assert len(prediction.boxes) == 1
    assert prediction.boxes[0].label == "Truck"
    assert prediction.boxes[0].model_dump() == {
        "x1": 0.0,
        "y1": 5.0,
        "x2": 80.0,
        "y2": 60.0,
        "label": "Truck",
        "score": 0.91,
        "track_id": None,
    }


def test_rfdetr_adapter_rejects_invalid_threshold(tmp_path: Path) -> None:
    try:
        RFDETRDetector(tmp_path / "unused.pth", threshold=1.1, model=_Model())
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("invalid threshold should fail")
