from __future__ import annotations

import argparse
import json
from pathlib import Path

from car5_autolabel.datasets.manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate the car5 six-camera manifest")
    parser.add_argument("dataset_root", type=Path, help="Directory containing camera/ and label/")
    parser.add_argument("output_dir", type=Path, help="Generated artifacts directory")
    parser.add_argument("--dataset-id")
    parser.add_argument("--label-studio-export", type=Path)
    parser.add_argument("--hash-scope", choices=("none", "keyframes", "all"), default="keyframes")
    parser.add_argument(
        "--dimension-scope", choices=("none", "sample", "keyframes", "all"), default="sample"
    )
    parser.add_argument(
        "--allow-errors", action="store_true", help="Write report and exit successfully"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, report = build_manifest(
        args.dataset_root,
        args.output_dir,
        dataset_id=args.dataset_id,
        label_studio_export=args.label_studio_export,
        hash_scope=args.hash_scope,
        dimension_scope=args.dimension_scope,
        strict=not args.allow_errors,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
