from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from car5_autolabel.datasets.models import ReviewedBox, ReviewedTask

VEHICLE_CLASSES = {"car", "truck", "bulldozer", "excavator", "watertruck"}


@dataclass(frozen=True)
class PredictionBox:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def normalize_prediction_label(label: str) -> str:
    value = label.strip().lower().replace("_", " ")
    if "traffic sign" in value or value == "sign" or "sign candidate" in value:
        return "sign"
    if "water" in value and ("truck" in value or "tanker" in value):
        return "watertruck"
    if "excavator" in value:
        return "excavator"
    if "bulldozer" in value:
        return "bulldozer"
    if value == "car":
        return "car"
    if "truck" in value or value == "bus" or "haul" in value:
        return "truck"
    if value in {"vehicle", "vehicle candidate", "motorcycle"}:
        return "vehicle"
    if "person" in value or "pedestrian" in value:
        return "inactive_person"
    return "unknown"


def _proposal_compatible(ground_truth: ReviewedBox, prediction: PredictionBox) -> bool:
    predicted = normalize_prediction_label(prediction.label)
    if ground_truth.class_name in VEHICLE_CLASSES:
        return predicted in VEHICLE_CLASSES | {"vehicle"}
    return ground_truth.class_name == "sign" and predicted == "sign"


def _class_compatible(ground_truth: ReviewedBox, prediction: PredictionBox) -> bool:
    return ground_truth.class_name == normalize_prediction_label(prediction.label)


def _match(
    ground_truth: tuple[ReviewedBox, ...],
    predictions: list[PredictionBox],
    compatibility: Callable[[ReviewedBox, PredictionBox], bool],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (ground_truth_index, prediction_index, box_iou(gt.xyxy, pred.xyxy))
            for ground_truth_index, gt in enumerate(ground_truth)
            for prediction_index, pred in enumerate(predictions)
            if compatibility(gt, pred)
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for ground_truth_index, prediction_index, iou in candidates:
        if iou < iou_threshold:
            break
        if ground_truth_index in matched_ground_truth or prediction_index in matched_predictions:
            continue
        matched_ground_truth.add(ground_truth_index)
        matched_predictions.add(prediction_index)
        matches.append((ground_truth_index, prediction_index, iou))
    return matches


def _metrics(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def evaluate_prediction_jsonl(
    prediction_path: str | Path,
    reviewed_tasks: dict[tuple[str, str], ReviewedTask],
    *,
    iou_threshold: float = 0.5,
    sequence_ids: set[str] | None = None,
) -> dict[str, Any]:
    rows = _load_predictions(Path(prediction_path))
    if sequence_ids is not None:
        rows = [row for row in rows if str(row.get("sequence_id")) in sequence_ids]
    seen_keys: set[tuple[str, str]] = set()
    totals = {"proposal": Counter(), "class_aware": Counter()}
    per_camera: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"proposal": Counter(), "class_aware": Counter()}
    )
    gt_class_total: Counter[str] = Counter()
    gt_class_matched: Counter[str] = Counter()
    gt_class_matched_class_aware: Counter[str] = Counter()
    unknown_tasks: list[str] = []
    duplicate_tasks: list[str] = []
    prediction_labels: Counter[str] = Counter()

    for row in rows:
        camera = str(row["camera"])
        timestamp = str(row["timestamp"])
        key = (camera, timestamp)
        if key in seen_keys:
            duplicate_tasks.append(f"{camera}/{timestamp}")
            continue
        seen_keys.add(key)
        reviewed = reviewed_tasks.get(key)
        if reviewed is None or reviewed.review_status not in {"human_verified", "verified_empty"}:
            unknown_tasks.append(f"{camera}/{timestamp}")
            continue

        predictions = [
            PredictionBox(
                label=str(item["label"]),
                score=float(item["score"]),
                xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            )
            for item in row.get("detections", [])
        ]
        prediction_labels.update(normalize_prediction_label(item.label) for item in predictions)
        gt_class_total.update(box.class_name for box in reviewed.boxes)

        for mode, compatibility in (
            ("proposal", _proposal_compatible),
            ("class_aware", _class_compatible),
        ):
            matches = _match(reviewed.boxes, predictions, compatibility, iou_threshold)
            tp = len(matches)
            fp = len(predictions) - tp
            fn = len(reviewed.boxes) - tp
            totals[mode].update(tp=tp, fp=fp, fn=fn)
            per_camera[camera][mode].update(tp=tp, fp=fp, fn=fn)
            if mode == "proposal":
                gt_class_matched.update(reviewed.boxes[index].class_name for index, _, _ in matches)
            else:
                gt_class_matched_class_aware.update(
                    reviewed.boxes[index].class_name for index, _, _ in matches
                )

    result = {
        "prediction_path": str(Path(prediction_path)),
        "iou_threshold": iou_threshold,
        "sequence_ids": sorted(sequence_ids) if sequence_ids is not None else None,
        "evaluated_tasks": len(seen_keys) - len(unknown_tasks),
        "duplicate_tasks": sorted(duplicate_tasks),
        "unknown_or_unreviewed_tasks": sorted(unknown_tasks),
        "ground_truth_boxes": sum(gt_class_total.values()),
        "prediction_boxes": sum(prediction_labels.values()),
        "prediction_label_counts": dict(sorted(prediction_labels.items())),
        "proposal": _metrics(
            **{
                "true_positive": totals["proposal"]["tp"],
                "false_positive": totals["proposal"]["fp"],
                "false_negative": totals["proposal"]["fn"],
            }
        ),
        "class_aware": _metrics(
            **{
                "true_positive": totals["class_aware"]["tp"],
                "false_positive": totals["class_aware"]["fp"],
                "false_negative": totals["class_aware"]["fn"],
            }
        ),
        "per_camera": {
            camera: {
                mode: _metrics(
                    counts["tp"],
                    counts["fp"],
                    counts["fn"],
                )
                for mode, counts in modes.items()
            }
            for camera, modes in sorted(per_camera.items())
        },
        "proposal_recall_by_gt_class": {
            class_name: {
                "matched": gt_class_matched[class_name],
                "total": total,
                "recall": gt_class_matched[class_name] / total if total else 0,
            }
            for class_name, total in sorted(gt_class_total.items())
        },
        "class_aware_by_class": {
            class_name: _metrics(
                gt_class_matched_class_aware[class_name],
                prediction_labels[class_name] - gt_class_matched_class_aware[class_name],
                total - gt_class_matched_class_aware[class_name],
            )
            for class_name, total in sorted(gt_class_total.items())
        },
    }
    return result
