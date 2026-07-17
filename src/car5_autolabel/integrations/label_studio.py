from car5_autolabel.schemas import BoundingBox, ImagePrediction


def box_to_label_studio(box: BoundingBox, width: int, height: int) -> dict:
    """Convert an internal pixel-space box to a Label Studio rectangle result."""
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "prediction",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": box.x1 * 100.0 / width,
            "y": box.y1 * 100.0 / height,
            "width": (box.x2 - box.x1) * 100.0 / width,
            "height": (box.y2 - box.y1) * 100.0 / height,
            "rotation": 0,
            "rectanglelabels": [box.label],
        },
        "score": box.score,
    }


def prediction_to_label_studio(prediction: ImagePrediction) -> list[dict]:
    results = [
        box_to_label_studio(box, prediction.width, prediction.height) for box in prediction.boxes
    ]
    for result in results:
        result["meta"] = {
            "image_id": prediction.image_id,
            "source": prediction.source,
            "review_status": prediction.review_status,
            "model_version": prediction.model_version,
            "config_sha256": prediction.config_sha256,
            "checkpoint_sha256": prediction.checkpoint_sha256,
            "dataset_manifest_id": prediction.dataset_manifest_id,
        }
    return results


def prediction_to_label_studio_payload(prediction: ImagePrediction) -> dict:
    """Build one Label Studio prediction object with auditable provenance."""
    return {
        "model_version": prediction.model_version,
        "score": max((box.score for box in prediction.boxes), default=0.0),
        "result": prediction_to_label_studio(prediction),
    }


def prediction_task_payload(
    prediction: ImagePrediction,
    *,
    image_url: str,
    camera_id: str,
    frame_id: int,
    capture_timestamp: str,
) -> dict:
    """Build an idempotent Label Studio task import with one prediction."""
    return {
        "data": {
            "image": image_url,
            "image_id": prediction.image_id,
            "camera": camera_id,
            "frame_id": str(capture_timestamp),
            "manifest_frame_id": frame_id,
        },
        "predictions": [prediction_to_label_studio_payload(prediction)],
    }
