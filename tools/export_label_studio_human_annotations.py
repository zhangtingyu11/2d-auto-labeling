"""Export only editable human boxes from a Label Studio SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from car5_autolabel.integrations.label_studio_human_export import (
    export_human_annotations,
    summary_as_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Label Studio SQLite database")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--label-studio-json", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--default-width", type=int, default=1920)
    parser.add_argument("--default-height", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_human_annotations(
        database=args.database,
        project_id=args.project_id,
        label_studio_json=args.label_studio_json,
        coco_json=args.coco,
        default_width=args.default_width,
        default_height=args.default_height,
    )
    print(json.dumps(summary_as_dict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
