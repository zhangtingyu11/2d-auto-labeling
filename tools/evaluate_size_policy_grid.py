"""Evaluate fixed predictions under bbox-size and truncation policies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rounded_long_side(box):
    return max(round(float(box[2])), round(float(box[3])))


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def edge_touch(pred, image, fraction=0.01):
    x, y, w, h = map(float, pred["bbox"])
    width, height = float(image["width"]), float(image["height"])
    mx, my = width * fraction, height * fraction
    return x <= mx or y <= my or x + w >= width - mx or y + h >= height - my


def fixed_metrics(gt, predictions):
    names = {int(row["id"]): str(row["name"]) for row in gt["categories"]}
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    for row in gt["annotations"]:
        gt_by_image[int(row["image_id"])].append(row)
    for row in predictions:
        pred_by_image[int(row["image_id"])].append(row)
    overall = Counter()
    per_class = {name: Counter() for name in names.values()}
    for image in gt["images"]:
        image_id = int(image["id"])
        truth = gt_by_image[image_id]
        detected = sorted(pred_by_image[image_id], key=lambda x: float(x["score"]), reverse=True)
        candidates = []
        for pi, pred in enumerate(detected):
            for gi, ann in enumerate(truth):
                if int(pred["category_id"]) != int(ann["category_id"]):
                    continue
                overlap = iou(pred["bbox"], ann["bbox"])
                if overlap >= 0.5:
                    candidates.append((overlap, pi, gi))
        used_p, used_g = set(), set()
        for _, pi, gi in sorted(candidates, reverse=True):
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
        for gi, ann in enumerate(truth):
            key = "tp" if gi in used_g else "fn"
            overall[key] += 1
            per_class[names[int(ann["category_id"])]][key] += 1
        for pi, pred in enumerate(detected):
            if pi not in used_p:
                overall["fp"] += 1
                per_class[names[int(pred["category_id"])]]["fp"] += 1

    def calculate(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    return calculate(overall), {name: calculate(counts) for name, counts in per_class.items()}


def coco_metrics(gt_path, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--policy-predictions", type=Path)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--policy-name")
    parser.add_argument(
        "--size-thresholds",
        type=int,
        nargs="+",
        default=[70],
        help="Longest-side thresholds to evaluate; the formal project default is 70.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_gt = load(args.gt)
    image_by_id = {int(row["id"]): row for row in original_gt["images"]}
    valid_image_ids = set(image_by_id)
    prediction_sets = {
        args.model_name: [
            row for row in load(args.raw_predictions) if int(row["image_id"]) in valid_image_ids
        ],
    }
    if args.policy_predictions:
        policy_name = args.policy_name or f"{args.model_name}_policy"
        prediction_sets[policy_name] = [
            row for row in load(args.policy_predictions) if int(row["image_id"]) in valid_image_ids
        ]
    sign_ids = {int(row["id"]) for row in original_gt["categories"] if str(row["name"]) == "Sign"}
    results = []
    for model_name, source_predictions in prediction_sets.items():
        for threshold in args.size_thresholds:
            if threshold < 0:
                raise ValueError("size thresholds must be non-negative")
            for include_sign in (True, False):
                for mode, confidence in (
                    ("hard_size", None),
                    ("truncation_edge_exception", 0.7),
                    ("truncation_edge_exception", 0.8),
                    ("truncation_edge_exception", 0.9),
                ):
                    categories = [
                        row
                        for row in original_gt["categories"]
                        if include_sign or int(row["id"]) not in sign_ids
                    ]
                    allowed_categories = {int(row["id"]) for row in categories}
                    annotations = []
                    small_truncated_gt = 0
                    for ann in original_gt["annotations"]:
                        if int(ann["category_id"]) not in allowed_categories:
                            continue
                        size_ok = rounded_long_side(ann["bbox"]) >= threshold
                        truncated = bool(ann.get("attributes", {}).get("truncation"))
                        if truncated and not size_ok:
                            small_truncated_gt += 1
                        if size_ok or (mode == "truncation_edge_exception" and truncated):
                            annotations.append(ann)
                    predictions = []
                    exception_predictions = 0
                    for pred in source_predictions:
                        if int(pred["category_id"]) not in allowed_categories:
                            continue
                        size_ok = rounded_long_side(pred["bbox"]) >= threshold
                        exception = (
                            mode == "truncation_edge_exception"
                            and not size_ok
                            and float(pred["score"]) >= float(confidence)
                            and edge_touch(pred, image_by_id[int(pred["image_id"])])
                        )
                        if size_ok or exception:
                            predictions.append(pred)
                            exception_predictions += int(exception)
                    gt = {**original_gt, "categories": categories, "annotations": annotations}
                    sign_slug = "with_sign" if include_sign else "without_sign"
                    confidence_slug = (
                        "" if confidence is None else str(int(confidence * 100))
                    )
                    slug = (
                        f"{model_name}_px{threshold}_{sign_slug}_{mode}{confidence_slug}"
                    )
                    gt_path = args.output_dir / f"{slug}_gt.json"
                    # pycocotools opens annotation files without an explicit
                    # encoding on Windows.  ASCII escaping keeps human Chinese
                    # attributes intact while avoiding the host GBK default.
                    gt_path.write_text(json.dumps(gt, ensure_ascii=True), encoding="utf-8")
                    overall, per_class = fixed_metrics(gt, predictions)
                    coco = coco_metrics(gt_path, predictions)
                    row = {
                        "model": model_name,
                        "size_threshold": threshold,
                        "include_sign": include_sign,
                        "mode": mode,
                        "exception_confidence": confidence,
                        "gt_boxes": len(annotations),
                        "prediction_boxes": len(predictions),
                        "small_truncated_gt": small_truncated_gt,
                        "exception_predictions": exception_predictions,
                        **overall,
                        **coco,
                        "per_class": per_class,
                    }
                    results.append(row)
    (args.output_dir / "grid_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_fields = [key for key in results[0] if key != "per_class"]
    with (args.output_dir / "grid_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: value for key, value in row.items() if key != "per_class"})
    best = sorted(results, key=lambda row: (row["f1"], row["map_50"]), reverse=True)[:12]
    print(json.dumps({"configurations": len(results), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
