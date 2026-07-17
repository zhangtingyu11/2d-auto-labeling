import json
from pathlib import Path

from car5_autolabel.datasets.models import ReviewedBox, ReviewedTask
from car5_autolabel.evaluation.detection import evaluate_prediction_jsonl


def test_proposal_and_class_aware_metrics_are_separate(tmp_path: Path) -> None:
    reviewed = {
        ("front", "frame-1"): ReviewedTask(
            image_id="front:frame-1",
            frame_id="frame-1",
            camera_id="front",
            review_status="human_verified",
            boxes=(
                ReviewedBox(3, "excavator", (0, 0, 100, 100), "human"),
                ReviewedBox(5, "sign", (200, 200, 250, 250), "human"),
            ),
        )
    }
    prediction = {
        "camera": "front",
        "timestamp": "frame-1",
        "detections": [
            {"label": "vehicle", "score": 0.9, "bbox_xyxy": [0, 0, 100, 100]},
            {"label": "traffic sign", "score": 0.8, "bbox_xyxy": [200, 200, 250, 250]},
            {"label": "person", "score": 0.7, "bbox_xyxy": [400, 400, 450, 500]},
        ],
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    report = evaluate_prediction_jsonl(path, reviewed)

    assert report["proposal"] == {
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "precision": 2 / 3,
        "recall": 1,
        "f1": 0.8,
    }
    assert report["class_aware"]["tp"] == 1
    assert report["class_aware"]["fp"] == 2
    assert report["class_aware"]["fn"] == 1
    assert report["proposal_recall_by_gt_class"]["excavator"]["recall"] == 1
    assert report["proposal_recall_by_gt_class"]["sign"]["recall"] == 1
