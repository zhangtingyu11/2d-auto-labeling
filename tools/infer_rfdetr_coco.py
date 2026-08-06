"""Run an RF-DETR checkpoint on a COCO split and export size-filtered predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rf-home", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--minimum-long-side-px", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--data-manifest-id", required=True)
    parser.add_argument("--source", default="rfdetr-kfold-oof")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if args.minimum_long_side_px <= 0:
        raise ValueError("minimum-long-side-px must be positive")
    os.environ["RF_HOME"] = str(args.rf_home.resolve())

    from rfdetr import RFDETR

    split_dir = args.split_dir.resolve()
    dataset = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8-sig"))
    category_id_by_name = {
        str(category["name"]): int(category["id"]) for category in dataset["categories"]
    }
    model = RFDETR.from_checkpoint(args.checkpoint.resolve())
    checkpoint_digest = file_sha256(args.checkpoint)
    class_names = list(model.class_names)
    predictions: list[dict[str, object]] = []
    ignored_below_size = 0
    ignored_unknown_class = 0
    invalid_geometry = 0

    images = dataset["images"]
    for offset in range(0, len(images), args.batch_size):
        batch_meta = images[offset : offset + args.batch_size]
        batch_images = []
        for image_meta in batch_meta:
            with Image.open(split_dir / image_meta["file_name"]) as image_handle:
                batch_images.append(image_handle.convert("RGB"))
        detections = model.predict(
            batch_images,
            threshold=args.threshold,
            include_source_image=False,
        )
        if not isinstance(detections, list):
            detections = [detections]
        for image_meta, image_detections in zip(batch_meta, detections, strict=True):
            for xyxy, score, class_id in zip(
                image_detections.xyxy,
                image_detections.confidence,
                image_detections.class_id,
                strict=True,
            ):
                index = int(class_id)
                if not 0 <= index < len(class_names):
                    ignored_unknown_class += 1
                    continue
                label = class_names[index]
                category_id = category_id_by_name.get(label)
                if category_id is None:
                    ignored_unknown_class += 1
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    invalid_geometry += 1
                    continue
                x1 = max(0.0, min(x1, float(image_meta["width"])))
                y1 = max(0.0, min(y1, float(image_meta["height"])))
                x2 = max(0.0, min(x2, float(image_meta["width"])))
                y2 = max(0.0, min(y2, float(image_meta["height"])))
                if x2 <= x1 or y2 <= y1:
                    invalid_geometry += 1
                    continue
                width, height = x2 - x1, y2 - y1
                if max(width, height) < args.minimum_long_side_px:
                    ignored_below_size += 1
                    continue
                prediction_id = f"{checkpoint_digest[:12]}:{image_meta['id']}:{len(predictions)}"
                predictions.append(
                    {
                        "id": len(predictions) + 1,
                        "prediction_id": prediction_id,
                        "image_id": int(image_meta["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, width, height],
                        "score": float(score),
                        "model_version": checkpoint_digest,
                        "config_hash": args.config_hash,
                        "data_manifest_id": args.data_manifest_id,
                        "source": args.source,
                        "review_state": "unreviewed",
                    }
                )
        print(
            json.dumps(
                {"processed": min(offset + args.batch_size, len(images)), "total": len(images)}
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(
            {
                "images": len(images),
                "predictions": len(predictions),
                "threshold": args.threshold,
                "minimum_long_side_px": args.minimum_long_side_px,
                "ignored_below_size": ignored_below_size,
                "ignored_unknown_class": ignored_unknown_class,
                "invalid_geometry": invalid_geometry,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_classes": class_names,
                "model_version": checkpoint_digest,
                "config_hash": args.config_hash,
                "data_manifest_id": args.data_manifest_id,
                "source": args.source,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
