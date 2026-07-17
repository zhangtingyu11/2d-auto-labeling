from __future__ import annotations

import argparse
import json
from pathlib import Path

from car5_autolabel.datasets.label_studio_export import parse_label_studio_export
from car5_autolabel.datasets.manifest import build_manifest
from car5_autolabel.datasets.release import build_coco_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reviewed COCO release and sequence folds")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("label_studio_export", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", default="car5_20260611_day_02_v1")
    parser.add_argument("--hash-scope", choices=("none", "keyframes", "all"), default="keyframes")
    parser.add_argument(
        "--dimension-scope", choices=("none", "sample", "keyframes", "all"), default="sample"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = args.output_dir / "manifest"
    rows, _ = build_manifest(
        args.dataset_root,
        manifest_dir,
        dataset_id=args.dataset_id,
        label_studio_export=args.label_studio_export,
        hash_scope=args.hash_scope,
        dimension_scope=args.dimension_scope,
    )
    reviewed_tasks, _ = parse_label_studio_export(args.label_studio_export)
    summary = build_coco_release(
        rows,
        reviewed_tasks,
        args.output_dir / "coco",
        dataset_id=args.dataset_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
