from __future__ import annotations

import argparse
import json
from pathlib import Path

from car5_autolabel.datasets.label_studio_export import parse_label_studio_export
from car5_autolabel.evaluation.detection import evaluate_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONL against reviewed 2D GT")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("label_studio_export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="Evaluate only these sequence IDs; repeat for more than one",
    )
    args = parser.parse_args()

    reviewed, _ = parse_label_studio_export(args.label_studio_export)
    report = evaluate_prediction_jsonl(
        args.predictions,
        reviewed,
        iou_threshold=args.iou,
        sequence_ids=set(args.sequences) if args.sequences else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
