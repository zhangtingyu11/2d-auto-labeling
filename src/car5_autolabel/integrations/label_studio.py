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
    return [
        box_to_label_studio(box, prediction.width, prediction.height) for box in prediction.boxes
    ]
