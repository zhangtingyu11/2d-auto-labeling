from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from car5_autolabel.detectors.rtmdet import RTMDetDetector


def _manifest_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(str(json.loads(line)["image_id"]))
    return keys


def _rows(path: Path, include_context: bool) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not include_context:
        rows = [row for row in rows if row["is_keyframe"]]
    return sorted(
        rows,
        key=lambda row: (row["sequence_id"], row["frame_id"], row["camera_order"]),
    )


def _output_stats(path: Path) -> dict[str, Any]:
    image_ids: list[str] = []
    class_counts: Counter[str] = Counter()
    empty_images = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_ids.append(str(row["image_id"]))
        detections = row.get("detections", [])
        if not detections:
            empty_images += 1
        class_counts.update(str(item["label"]) for item in detections)
    return {
        "output_rows": len(image_ids),
        "unique_image_ids": len(set(image_ids)),
        "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
        "detection_boxes": sum(class_counts.values()),
        "empty_images": empty_images,
        "class_counts": dict(sorted(class_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditable RTMDet batch inference")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--cross-class-nms-iou",
        type=float,
        help="Suppress overlapping boxes with different labels; disabled when omitted",
    )
    parser.add_argument(
        "--watertruck-min-score",
        type=float,
        help="Below this score, relabel ambiguous WaterTruck predictions as Truck",
    )
    parser.add_argument(
        "--watertruck-ambiguity-margin",
        type=float,
        default=0.0,
        help="WaterTruck must beat an overlapping class by this score margin",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-version")
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_id = _manifest_id(args.manifest)
    detector = RTMDetDetector(
        args.config,
        args.checkpoint,
        dataset_manifest_id=manifest_id,
        device=args.device,
        score_threshold=args.score_threshold,
        cross_class_nms_iou=args.cross_class_nms_iou,
        watertruck_min_score=args.watertruck_min_score,
        watertruck_ambiguity_margin=args.watertruck_ambiguity_margin,
        model_version=args.model_version,
    )
    rows = _rows(args.manifest, args.include_context)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = _existing_keys(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    pending = [row for row in rows if row["image_id"] not in existing]
    skipped = len(rows) - len(pending)

    def write_prediction(stream: Any, row: dict[str, Any], prediction: Any) -> None:
        payload = {
            "image_id": prediction.image_id,
            "camera": row["camera_id"],
            "timestamp": row["capture_timestamp"],
            "sequence_id": row["sequence_id"],
            "frame_id": row["frame_id"],
            "width": prediction.width,
            "height": prediction.height,
            "model_version": prediction.model_version,
            "config_sha256": prediction.config_sha256,
            "checkpoint_sha256": prediction.checkpoint_sha256,
            "dataset_manifest_id": prediction.dataset_manifest_id,
            "source": prediction.source,
            "review_status": prediction.review_status,
            "postprocess": {
                "score_threshold": detector.score_threshold,
                "cross_class_nms_iou": detector.cross_class_nms_iou,
                "watertruck_min_score": detector.watertruck_min_score,
                "watertruck_ambiguity_margin": detector.watertruck_ambiguity_margin,
            },
            "detections": [
                {
                    "label": box.label,
                    "score": box.score,
                    "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                    "source": prediction.source,
                    "review_status": prediction.review_status,
                    **(
                        {"raw_label": box.raw_label}
                        if box.raw_label is not None
                        else {}
                    ),
                    **(
                        {"postprocess_reason": box.postprocess_reason}
                        if box.postprocess_reason is not None
                        else {}
                    ),
                }
                for box in prediction.boxes
            ],
        }
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()

    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for offset in range(0, len(pending), args.batch_size):
            batch_rows = pending[offset : offset + args.batch_size]
            try:
                predictions = detector.predict_batch(
                    [args.dataset_root / row["relative_path"] for row in batch_rows],
                    [row["image_id"] for row in batch_rows],
                )
                for row, prediction in zip(batch_rows, predictions, strict=True):
                    write_prediction(stream, row, prediction)
                    processed += 1
            except Exception:
                # A failed batch is retried image-by-image so one corrupt image
                # cannot discard the rest of a long resumable job.
                for row in batch_rows:
                    try:
                        prediction = detector.predict(
                            args.dataset_root / row["relative_path"], row["image_id"]
                        )
                        write_prediction(stream, row, prediction)
                        processed += 1
                    except Exception as exc:
                        failures.append({"image_id": row["image_id"], "error": str(exc)})

    elapsed = time.perf_counter() - started
    run = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "config_sha256": detector.config_sha256,
        "checkpoint_sha256": detector.checkpoint_sha256,
        "dataset_manifest_id": manifest_id,
        "model_version": detector.model_version,
        "score_threshold": args.score_threshold,
        "cross_class_nms_iou": args.cross_class_nms_iou,
        "watertruck_min_score": args.watertruck_min_score,
        "watertruck_ambiguity_margin": args.watertruck_ambiguity_margin,
        "batch_size": args.batch_size,
        "selected_images": len(rows),
        "processed": processed,
        "skipped_existing": skipped,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "processed_fps": processed / elapsed if elapsed else 0.0,
    }
    run.update(_output_stats(args.output))
    run_path = args.output.with_suffix(args.output.suffix + ".run.json")
    run_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
