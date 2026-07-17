from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from car5_autolabel.datasets.car5 import Car5DatasetBuilder
from car5_autolabel.datasets.coco import build_coco_dataset
from car5_autolabel.datasets.gold import build_gold_set
from car5_autolabel.datasets.manifest import (
    read_canonical_annotations_json,
    read_manifest_jsonl,
    write_canonical_annotations_json,
    write_json,
    write_manifest_jsonl,
    write_validation_report,
)
from car5_autolabel.integrations.label_studio import parse_label_studio_export


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car5-autolabel",
        description="Build and validate deterministic car5 dataset artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "build-manifest",
        help="scan the car5 source tree and write a deterministic JSONL manifest",
    )
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--report", type=Path, required=True)
    manifest.add_argument("--dataset-id", default="car5_20260611_day_02")
    manifest.add_argument("--include-context", action="store_true")
    manifest.add_argument("--skip-hashes", action="store_true")
    manifest.add_argument("--verify-dimensions", action="store_true")
    manifest.add_argument("--expected-keyframes", type=int, default=928)
    manifest.add_argument("--expected-camera-frames", type=int, default=4640)
    manifest.add_argument("--gap-threshold-ms", type=float, default=1000.0)

    label_studio = subparsers.add_parser(
        "validate-label-studio",
        help="validate a Label Studio export and write canonical annotations",
    )
    label_studio.add_argument("--export", type=Path, required=True)
    label_studio.add_argument("--manifest", type=Path, required=True)
    label_studio.add_argument("--output", type=Path, required=True)
    label_studio.add_argument("--report", type=Path, required=True)
    label_studio.add_argument("--dataset-id", default="car5_20260611_day_02")
    label_studio.add_argument("--allow-unreviewed", action="store_true")

    gold = subparsers.add_parser(
        "build-gold-set",
        help="select deterministic complete six-camera timestamp groups for evaluation",
    )
    gold.add_argument("--manifest", type=Path, required=True)
    gold.add_argument("--annotations", type=Path, required=True)
    gold.add_argument("--output", type=Path, required=True)
    gold.add_argument("--report", type=Path, required=True)
    gold.add_argument("--dataset-release-id", required=True)
    gold.add_argument("--timestamp-groups", type=int, default=120)

    coco = subparsers.add_parser(
        "export-coco",
        help="export deterministic train/validation COCO files for a sequence fold",
    )
    coco.add_argument("--manifest", type=Path, required=True)
    coco.add_argument("--annotations", type=Path, required=True)
    coco.add_argument("--output-dir", type=Path, required=True)
    coco.add_argument("--held-out-fold", required=True)
    return parser


def _print_summary(valid: bool, errors: int, warnings: int, stats: dict) -> None:
    print(
        json.dumps(
            {
                "valid": valid,
                "error_count": errors,
                "warning_count": warnings,
                "stats": stats,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def _run_build_manifest(args: argparse.Namespace) -> int:
    result = Car5DatasetBuilder(
        args.data_root,
        dataset_id=args.dataset_id,
        include_context=args.include_context,
        hash_files=not args.skip_hashes,
        verify_dimensions=args.verify_dimensions,
        expected_keyframes=args.expected_keyframes,
        expected_camera_frames=args.expected_camera_frames,
        gap_threshold_ms=args.gap_threshold_ms,
    ).build()
    write_manifest_jsonl(result.rows, args.output)
    write_validation_report(result.report, args.report)
    _print_summary(
        result.report.valid,
        len(result.report.errors),
        len(result.report.warnings),
        result.report.stats,
    )
    return 0 if result.report.valid else 2


def _run_validate_label_studio(args: argparse.Namespace) -> int:
    manifest_rows = read_manifest_jsonl(args.manifest)
    result = parse_label_studio_export(
        args.export,
        manifest_rows=manifest_rows,
        dataset_id=args.dataset_id,
        require_reviewed=not args.allow_unreviewed,
    )
    write_canonical_annotations_json(result.annotations, args.output)
    write_validation_report(result.report, args.report)
    _print_summary(
        result.report.valid,
        len(result.report.errors),
        len(result.report.warnings),
        result.report.stats,
    )
    return 0 if result.report.valid else 2


def _run_build_gold_set(args: argparse.Namespace) -> int:
    result = build_gold_set(
        read_manifest_jsonl(args.manifest),
        read_canonical_annotations_json(args.annotations),
        dataset_release_id=args.dataset_release_id,
        timestamp_groups=args.timestamp_groups,
    )
    write_json([row.model_dump(mode="json") for row in result.rows], args.output)
    write_validation_report(result.report, args.report)
    _print_summary(
        result.report.valid,
        len(result.report.errors),
        len(result.report.warnings),
        result.report.stats,
    )
    return 0 if result.report.valid else 2


def _run_export_coco(args: argparse.Namespace) -> int:
    manifest_rows = read_manifest_jsonl(args.manifest)
    annotations = read_canonical_annotations_json(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        payload = build_coco_dataset(
            manifest_rows,
            annotations,
            held_out_fold=args.held_out_fold,
            split=split,
        )
        write_json(payload, args.output_dir / f"instances_{split}.json", compact=True)
        summaries[split] = {
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
        }
    print(json.dumps({"held_out_fold": args.held_out_fold, "splits": summaries}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-manifest":
        return _run_build_manifest(args)
    if args.command == "validate-label-studio":
        return _run_validate_label_studio(args)
    if args.command == "build-gold-set":
        return _run_build_gold_set(args)
    if args.command == "export-coco":
        return _run_export_coco(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
