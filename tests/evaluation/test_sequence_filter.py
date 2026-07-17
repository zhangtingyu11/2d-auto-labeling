import json

from car5_autolabel.datasets.models import ReviewedTask
from car5_autolabel.evaluation.detection import evaluate_prediction_jsonl


def test_evaluation_can_be_limited_to_held_out_sequence(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "camera": "front",
            "timestamp": "one",
            "sequence_id": "s00",
            "detections": [],
        },
        {
            "camera": "front",
            "timestamp": "two",
            "sequence_id": "s01",
            "detections": [],
        },
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    reviewed = {
        ("front", "one"): ReviewedTask(
            image_id="s00:one:front",
            frame_id="one",
            camera_id="front",
            review_status="verified_empty",
            boxes=(),
        ),
        ("front", "two"): ReviewedTask(
            image_id="s01:two:front",
            frame_id="two",
            camera_id="front",
            review_status="verified_empty",
            boxes=(),
        ),
    }

    report = evaluate_prediction_jsonl(predictions, reviewed, sequence_ids={"s00"})

    assert report["evaluated_tasks"] == 1
    assert report["sequence_ids"] == ["s00"]
