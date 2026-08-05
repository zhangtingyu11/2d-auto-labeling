"""Split finalized COCO predictions into automatic and review-only outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--automatic-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    names = {
        int(row["id"]): str(row["name"]) for row in annotations["categories"]
    }
    default_threshold = float(policy["default_threshold"])
    thresholds = {
        str(name): float(value)
        for name, value in policy.get("class_thresholds", {}).items()
    }
    review_only = set(policy.get("review_only_labels", {}))

    automatic = []
    review = []
    automatic_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    for prediction in predictions:
        label = names[int(prediction["category_id"])]
        score = float(prediction["score"])
        threshold = thresholds.get(label, default_threshold)
        if label in review_only:
            review.append(
                {
                    **prediction,
                    "review_reason": "review_only_class",
                    "policy_threshold": threshold,
                }
            )
            review_counts[label] += 1
        elif score >= threshold:
            automatic.append(prediction)
            automatic_counts[label] += 1
        else:
            review.append(
                {
                    **prediction,
                    "review_reason": "below_class_threshold",
                    "policy_threshold": threshold,
                }
            )
            review_counts[label] += 1

    dump(args.automatic_output, automatic)
    dump(args.review_output, review)
    summary = {
        "policy": str(args.policy.resolve()),
        "input_boxes": len(predictions),
        "automatic_boxes": len(automatic),
        "review_boxes": len(review),
        "automatic_by_class": dict(sorted(automatic_counts.items())),
        "review_by_class": dict(sorted(review_counts.items())),
    }
    dump(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
