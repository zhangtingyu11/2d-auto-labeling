"""Evaluate low-confidence small proposals with native-image crop verification."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rf-home", type=Path, required=True)
    parser.add_argument("--main-threshold", type=float, default=0.50)
    parser.add_argument("--proposal-threshold", type=float, default=0.20)
    parser.add_argument("--confirmation-threshold", type=float, default=0.50)
    parser.add_argument("--allow-reclassification", action="store_true")
    parser.add_argument("--reclassification-threshold", type=float, default=0.80)
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--minimum-crop-size", type=int, default=320)
    parser.add_argument("--maximum-object-side", type=float, default=180.0)
    parser.add_argument("--maximum-proposals", type=int, default=12)
    parser.add_argument("--crop-edge-margin", type=float, default=0.03)
    parser.add_argument("--crop-edge-retry-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.proposal_threshold >= args.main_threshold:
        raise ValueError("proposal threshold must be lower than main threshold")
    os.environ["RF_HOME"] = str(args.rf_home.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rfdetr import RFDETR

    from car5_autolabel.postprocessing import (
        suppress_cross_class_overlaps,
        suppress_same_class_duplicates,
    )
    from car5_autolabel.roi_verification import (
        expand_context_crop,
        is_small_proposal,
        make_context_crop,
        select_verified_detection,
        touches_context_crop_edge,
    )
    from car5_autolabel.schemas import BoundingBox

    split_dir = args.split_dir.resolve()
    dataset = json.loads(
        (split_dir / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    images = dataset["images"][: args.max_images]
    category_id_by_name = {
        str(item["name"]): int(item["id"]) for item in dataset["categories"]
    }
    model = RFDETR.from_checkpoint(args.checkpoint.resolve())
    effective_names = [
        "Truck" if name == "BoxTruck" else name for name in model.class_names
    ]
    vehicle_labels = {
        "Car",
        "Truck",
        "Bulldozer",
        "Excavator",
        "WaterTruck",
    }
    exported = []
    global_main = 0
    proposals = 0
    confirmed = 0
    reclassified = 0
    edge_retries = 0
    edge_rejected = 0
    invalid_class_queries = 0
    invalid_geometry_queries = 0
    started = time.time()

    def convert(detections) -> list[BoundingBox]:
        nonlocal invalid_class_queries, invalid_geometry_queries
        boxes = []
        for xyxy, score, class_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            strict=True,
        ):
            index = int(class_id)
            if not 0 <= index < len(effective_names):
                invalid_class_queries += 1
                continue
            label = effective_names[index]
            if label not in category_id_by_name:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            if (
                not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or x1 < 0
                or y1 < 0
                or x2 <= x1
                or y2 <= y1
            ):
                invalid_geometry_queries += 1
                continue
            boxes.append(
                BoundingBox(
                    label=label,
                    score=float(score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return boxes

    for image_index, image_meta in enumerate(images, start=1):
        image_path = split_dir / image_meta["file_name"]
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            width, height = image.size
            global_detections = model.predict(
                image,
                threshold=args.proposal_threshold,
                include_source_image=False,
            )
            all_boxes = convert(global_detections)
            boxes = [
                box for box in all_boxes if box.score >= args.main_threshold
            ]
            global_main += len(boxes)
            low_proposals = [
                box
                for box in all_boxes
                if args.proposal_threshold <= box.score < args.main_threshold
                and is_small_proposal(
                    box,
                    image_width=width,
                    image_height=height,
                    maximum_side=args.maximum_object_side,
                )
            ]
            low_proposals = suppress_same_class_duplicates(
                low_proposals,
                iou_threshold=0.55,
                containment_threshold=0.90,
            )[: args.maximum_proposals]
            proposals += len(low_proposals)
            windows = [
                make_context_crop(
                    proposal,
                    image_width=width,
                    image_height=height,
                    context_scale=args.context_scale,
                    minimum_size=args.minimum_crop_size,
                )
                for proposal in low_proposals
            ]
            retry_queue = []
            for offset in range(0, len(windows), args.batch_size):
                proposal_batch = low_proposals[offset : offset + args.batch_size]
                window_batch = windows[offset : offset + args.batch_size]
                crop_images = [
                    image.crop(
                        (
                            window.left,
                            window.top,
                            window.right,
                            window.bottom,
                        )
                    )
                    for window in window_batch
                ]
                crop_detections = model.predict(
                    crop_images,
                    threshold=args.confirmation_threshold,
                    include_source_image=False,
                )
                if not isinstance(crop_detections, list):
                    crop_detections = [crop_detections]
                for proposal, window, detections in zip(
                    proposal_batch,
                    window_batch,
                    crop_detections,
                    strict=True,
                ):
                    verified = select_verified_detection(
                        proposal,
                        convert(detections),
                        window,
                        confirmation_threshold=args.confirmation_threshold,
                        allow_reclassification=args.allow_reclassification,
                        reclassification_threshold=args.reclassification_threshold,
                    )
                    if verified is None:
                        continue
                    if touches_context_crop_edge(
                        verified,
                        window,
                        image_width=width,
                        image_height=height,
                        margin_ratio=args.crop_edge_margin,
                    ):
                        retry_queue.append(
                            (
                                proposal,
                                expand_context_crop(
                                    window,
                                    image_width=width,
                                    image_height=height,
                                    scale=args.crop_edge_retry_scale,
                                ),
                            )
                        )
                        continue
                    confirmed += 1
                    if verified.label != proposal.label:
                        reclassified += 1
                    boxes.append(verified)
            edge_retries += len(retry_queue)
            for offset in range(0, len(retry_queue), args.batch_size):
                retry_batch = retry_queue[offset : offset + args.batch_size]
                retry_detections = model.predict(
                    [
                        image.crop(
                            (
                                window.left,
                                window.top,
                                window.right,
                                window.bottom,
                            )
                        )
                        for _, window in retry_batch
                    ],
                    threshold=args.confirmation_threshold,
                    include_source_image=False,
                )
                if not isinstance(retry_detections, list):
                    retry_detections = [retry_detections]
                for (proposal, window), detections in zip(
                    retry_batch, retry_detections, strict=True
                ):
                    verified = select_verified_detection(
                        proposal,
                        convert(detections),
                        window,
                        confirmation_threshold=args.confirmation_threshold,
                        allow_reclassification=args.allow_reclassification,
                        reclassification_threshold=args.reclassification_threshold,
                    )
                    if verified is None or touches_context_crop_edge(
                        verified,
                        window,
                        image_width=width,
                        image_height=height,
                        margin_ratio=args.crop_edge_margin,
                    ):
                        edge_rejected += 1
                        continue
                    confirmed += 1
                    if verified.label != proposal.label:
                        reclassified += 1
                    boxes.append(
                        verified.model_copy(
                            update={
                                "postprocess_reason": (
                                    "expanded_native_crop_verified"
                                )
                            }
                        )
                    )

        boxes = suppress_same_class_duplicates(
            boxes,
            iou_threshold=0.55,
            containment_threshold=0.90,
        )
        boxes = suppress_cross_class_overlaps(
            boxes,
            iou_threshold=0.70,
            watertruck_ambiguity_margin=0.10,
        )
        boxes = suppress_cross_class_overlaps(
            boxes,
            iou_threshold=None,
            containment_threshold=0.90,
            comparable_labels=vehicle_labels,
        )
        for box in boxes:
            exported.append(
                {
                    "image_id": int(image_meta["id"]),
                    "category_id": category_id_by_name[box.label],
                    "bbox": [
                        box.x1,
                        box.y1,
                        box.x2 - box.x1,
                        box.y2 - box.y1,
                    ],
                    "score": box.score,
                }
            )
        if image_index % 50 == 0 or image_index == len(images):
            elapsed = max(time.time() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "processed": image_index,
                        "total": len(images),
                        "images_per_second": image_index / elapsed,
                        "proposals": proposals,
                        "confirmed": confirmed,
                    }
                ),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "images": len(images),
        "main_threshold": args.main_threshold,
        "proposal_threshold": args.proposal_threshold,
        "confirmation_threshold": args.confirmation_threshold,
        "allow_reclassification": args.allow_reclassification,
        "reclassification_threshold": args.reclassification_threshold,
        "context_scale": args.context_scale,
        "minimum_crop_size": args.minimum_crop_size,
        "maximum_object_side": args.maximum_object_side,
        "maximum_proposals": args.maximum_proposals,
        "global_main_boxes": global_main,
        "low_confidence_proposals": proposals,
        "crop_confirmed": confirmed,
        "crop_reclassified": reclassified,
        "crop_edge_margin": args.crop_edge_margin,
        "crop_edge_retry_scale": args.crop_edge_retry_scale,
        "crop_edge_retries": edge_retries,
        "crop_edge_rejected": edge_rejected,
        "invalid_class_queries_ignored": invalid_class_queries,
        "invalid_geometry_queries_ignored": invalid_geometry_queries,
        "merged_detections": len(exported),
        "elapsed_seconds": time.time() - started,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
