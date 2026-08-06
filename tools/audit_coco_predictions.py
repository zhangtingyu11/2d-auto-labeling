"""Compare COCO predictions with GT and write metrics plus annotation-error candidates."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from car5_autolabel.kfold_audit import (
    box_iou_xywh,
    maximum_iou_match_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--operating-score-threshold", type=float, default=0.20)
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def coco_ap(
    coco_gt: COCO, predictions: list[dict[str, Any]], category_ids: list[int]
) -> tuple[float | None, float | None]:
    """Return official COCO AP50 and AP50:95 using standard bbox settings."""

    if not predictions:
        return 0.0, 0.0
    with redirect_stdout(io.StringIO()):
        coco_predictions = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
        evaluator.params.catIds = category_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    map50_95 = float(evaluator.stats[0])
    map50 = float(evaluator.stats[1])
    return (map50 if map50 >= 0 else None, map50_95 if map50_95 >= 0 else None)


def main() -> None:
    args = parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8-sig"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    if not 0 <= args.operating_score_threshold <= 1:
        raise ValueError("operating-score-threshold must be between zero and one")
    images = {int(image["id"]): image for image in ground_truth["images"]}
    categories = {
        int(category["id"]): str(category["name"]) for category in ground_truth["categories"]
    }
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(args.ground_truth))
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        gt_by_image[int(annotation["image_id"])].append(annotation)
    unknown_image_ids = sorted(
        {int(prediction["image_id"]) for prediction in predictions} - set(images)
    )
    if unknown_image_ids:
        raise ValueError(f"predictions reference unknown image IDs: {unknown_image_ids[:10]}")
    operating_predictions = [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= args.operating_score_threshold
    ]
    for prediction in operating_predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    counts: dict[int, Counter[str]] = defaultdict(Counter)
    candidates: list[dict[str, Any]] = []
    for image_id, image in images.items():
        gt = gt_by_image[image_id]
        predicted = predictions_by_image[image_id]
        matches = maximum_iou_match_indices(gt, predicted, iou_threshold=args.iou_threshold)
        matched_gt = {match[0] for match in matches}
        matched_predictions = {match[1] for match in matches}
        for gt_index, _, _ in matches:
            counts[int(gt[gt_index]["category_id"])]["tp"] += 1

        remaining_gt = [(index, row) for index, row in enumerate(gt) if index not in matched_gt]
        remaining_predictions = [
            (index, row) for index, row in enumerate(predicted) if index not in matched_predictions
        ]
        mismatch_pairs = []
        for gt_index, gt_row in remaining_gt:
            for prediction_index, prediction in remaining_predictions:
                if int(gt_row["category_id"]) == int(prediction["category_id"]):
                    continue
                iou = box_iou_xywh(gt_row["bbox"], prediction["bbox"])
                if iou >= args.iou_threshold:
                    mismatch_pairs.append((iou, gt_index, prediction_index))
        used_mismatch_gt: set[int] = set()
        used_mismatch_predictions: set[int] = set()
        for iou, gt_index, prediction_index in sorted(mismatch_pairs, reverse=True):
            if gt_index in used_mismatch_gt or prediction_index in used_mismatch_predictions:
                continue
            gt_row, prediction = gt[gt_index], predicted[prediction_index]
            used_mismatch_gt.add(gt_index)
            used_mismatch_predictions.add(prediction_index)
            candidates.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "issue": "class_mismatch",
                    "gt_class": categories[int(gt_row["category_id"])],
                    "predicted_class": categories.get(int(prediction["category_id"]), "unknown"),
                    "score": prediction.get("score", ""),
                    "iou": iou,
                    "gt_bbox_xywh": json.dumps(gt_row["bbox"]),
                    "prediction_bbox_xywh": json.dumps(prediction["bbox"]),
                }
            )

        localization_gt = [
            (index, row) for index, row in remaining_gt if index not in used_mismatch_gt
        ]
        localization_predictions = [
            (index, row)
            for index, row in remaining_predictions
            if index not in used_mismatch_predictions
        ]
        localization_matches = maximum_iou_match_indices(
            [row for _, row in localization_gt],
            [row for _, row in localization_predictions],
            iou_threshold=0.1,
        )
        used_localization_gt: set[int] = set()
        used_localization_predictions: set[int] = set()
        for local_gt_index, local_prediction_index, iou in localization_matches:
            gt_index, gt_row = localization_gt[local_gt_index]
            prediction_index, prediction = localization_predictions[local_prediction_index]
            used_localization_gt.add(gt_index)
            used_localization_predictions.add(prediction_index)
            candidates.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "issue": "localization_error",
                    "gt_class": categories[int(gt_row["category_id"])],
                    "predicted_class": categories[int(prediction["category_id"])],
                    "score": prediction.get("score", ""),
                    "iou": iou,
                    "gt_bbox_xywh": json.dumps(gt_row["bbox"]),
                    "prediction_bbox_xywh": json.dumps(prediction["bbox"]),
                }
            )

        for gt_index, gt_row in remaining_gt:
            category_id = int(gt_row["category_id"])
            counts[category_id]["fn"] += 1
            if gt_index not in used_mismatch_gt and gt_index not in used_localization_gt:
                candidates.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "issue": "unmatched_gt_possible_missed_or_bad_label",
                        "gt_class": categories[category_id],
                        "predicted_class": "",
                        "score": "",
                        "iou": "",
                        "gt_bbox_xywh": json.dumps(gt_row["bbox"]),
                        "prediction_bbox_xywh": "",
                    }
                )
        for prediction_index, prediction in remaining_predictions:
            category_id = int(prediction["category_id"])
            counts[category_id]["fp"] += 1
            if (
                prediction_index not in used_mismatch_predictions
                and prediction_index not in used_localization_predictions
            ):
                candidates.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "issue": "unmatched_prediction_possible_missing_gt",
                        "gt_class": "",
                        "predicted_class": categories.get(category_id, "unknown"),
                        "score": prediction.get("score", ""),
                        "iou": "",
                        "gt_bbox_xywh": "",
                        "prediction_bbox_xywh": json.dumps(prediction["bbox"]),
                    }
                )

    rows = []
    total = Counter()
    average_precision = {
        category_id: coco_ap(coco_gt, predictions, [category_id])
        for category_id in categories
        if counts[category_id]["tp"] + counts[category_id]["fn"] > 0
    }
    for category_id, name in categories.items():
        count = counts[category_id]
        total.update(count)
        precision = safe_divide(count["tp"], count["tp"] + count["fp"])
        recall = safe_divide(count["tp"], count["tp"] + count["fn"])
        rows.append(
            {
                "category_id": category_id,
                "category": name,
                "tp": count["tp"],
                "fp": count["fp"],
                "fn": count["fn"],
                "precision": precision,
                "recall": recall,
                "f1": safe_divide(2 * precision * recall, precision + recall),
                "map50": average_precision.get(category_id, (None, None))[0],
                "map50_95": average_precision.get(category_id, (None, None))[1],
            }
        )
    precision = safe_divide(total["tp"], total["tp"] + total["fp"])
    recall = safe_divide(total["tp"], total["tp"] + total["fn"])
    overall_map50, overall_map50_95 = coco_ap(coco_gt, predictions, list(categories))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(
            {
                "iou_threshold": args.iou_threshold,
                "operating_score_threshold": args.operating_score_threshold,
                "overall": {
                    "tp": total["tp"],
                    "fp": total["fp"],
                    "fn": total["fn"],
                    "precision": precision,
                    "recall": recall,
                    "f1": safe_divide(2 * precision * recall, precision + recall),
                    "map50": overall_map50,
                    "map50_95": overall_map50_95,
                },
                "per_class": rows,
                "candidate_rows": len(candidates),
                "ignore_regions_applied": False,
                "ignored_host_region_predictions": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "image_id",
        "file_name",
        "issue",
        "gt_class",
        "predicted_class",
        "score",
        "iou",
        "gt_bbox_xywh",
        "prediction_bbox_xywh",
    ]
    with args.candidates.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


if __name__ == "__main__":
    main()
