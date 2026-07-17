from car5_autolabel.integrations.label_studio import (
    box_to_label_studio,
    prediction_task_payload,
    prediction_to_label_studio_payload,
)
from car5_autolabel.schemas import BoundingBox, ImagePrediction


def test_box_to_label_studio_percentage_coordinates() -> None:
    box = BoundingBox(x1=192, y1=108, x2=960, y2=540, label="Truck", score=0.9)

    result = box_to_label_studio(box, width=1920, height=1080)

    assert result["value"] == {
        "x": 10.0,
        "y": 10.0,
        "width": 40.0,
        "height": 40.0,
        "rotation": 0,
        "rectanglelabels": ["Truck"],
    }


def test_prediction_payload_preserves_provenance() -> None:
    prediction = ImagePrediction(
        image_id="dataset:timestamp:front",
        width=1920,
        height=1080,
        model_version="rtmdet-s-v1",
        config_sha256="config-hash",
        checkpoint_sha256="checkpoint-hash",
        dataset_manifest_id="manifest-hash",
        boxes=[BoundingBox(x1=10, y1=20, x2=110, y2=220, label="Truck", score=0.8)],
    )

    payload = prediction_to_label_studio_payload(prediction)

    assert payload["model_version"] == "rtmdet-s-v1"
    assert payload["result"][0]["meta"] == {
        "image_id": "dataset:timestamp:front",
        "source": "detector",
        "review_status": "unreviewed_prediction",
        "model_version": "rtmdet-s-v1",
        "config_sha256": "config-hash",
        "checkpoint_sha256": "checkpoint-hash",
        "dataset_manifest_id": "manifest-hash",
    }

    task = prediction_task_payload(
        prediction,
        image_url="/data/local-files/?d=camera/front/example.jpg",
        camera_id="front",
        frame_id=7,
        capture_timestamp="20260717_120000_000000",
    )
    assert task["data"]["image_id"] == prediction.image_id
    assert task["data"]["manifest_frame_id"] == 7
    assert task["predictions"][0]["model_version"] == "rtmdet-s-v1"
