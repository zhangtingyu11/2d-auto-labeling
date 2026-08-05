"""Run an RF-DETR checkpoint over an unlabelled multi-camera image folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rf-home", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--label-studio-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument(
        "--minimum-long-side-px",
        type=int,
        default=70,
        help=(
            "Only export formal boxes whose rounded longest side in the original "
            "image is at least this many pixels."
        ),
    )
    parser.add_argument("--same-class-nms-iou", type=float, default=0.55)
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    parser.add_argument("--cross-class-nms-iou", type=float, default=0.70)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Enable a high-resolution small-object pass with this tile size; 0 disables it.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-edge-margin", type=float, default=0.04)
    parser.add_argument(
        "--proposal-threshold",
        type=float,
        help=(
            "Enable native-image ROI verification for small proposals between "
            "this score and --threshold."
        ),
    )
    parser.add_argument("--crop-confirmation-threshold", type=float, default=0.50)
    parser.add_argument("--crop-allow-reclassification", action="store_true")
    parser.add_argument("--crop-reclassification-threshold", type=float, default=0.80)
    parser.add_argument("--crop-context-scale", type=float, default=4.0)
    parser.add_argument("--crop-minimum-size", type=int, default=320)
    parser.add_argument("--crop-maximum-object-side", type=float, default=180.0)
    parser.add_argument("--crop-maximum-proposals", type=int, default=12)
    parser.add_argument("--crop-edge-margin", type=float, default=0.03)
    parser.add_argument("--crop-edge-retry-scale", type=float, default=2.0)
    parser.add_argument(
        "--watertruck-min-score",
        type=float,
        default=0.55,
        help="Below this score, conservatively relabel WaterTruck as Truck.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Sample each camera independently to this rate using filename timestamps.",
    )
    parser.add_argument("--model-version", default="rfdetr_m_mining7_v2_20260723")
    return parser.parse_args()


def timestamp_seconds(stem: str) -> float:
    parsed = datetime.strptime(stem, "%Y%m%d_%H%M%S_%f")
    return parsed.timestamp()


def sample_by_camera(paths: list[Path], target_fps: float) -> list[Path]:
    interval = 1.0 / target_fps
    selected: list[Path] = []
    cameras = sorted({path.parent.name for path in paths})
    for camera in cameras:
        camera_paths = sorted(
            (path for path in paths if path.parent.name == camera),
            key=lambda path: path.stem,
        )
        last_selected: float | None = None
        for path in camera_paths:
            current = timestamp_seconds(path.stem)
            if last_selected is None or current - last_selected >= interval - 1e-6:
                selected.append(path)
                last_selected = current
    return sorted(selected, key=lambda path: (path.parent.name, path.stem))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RF_HOME"] = str(args.rf_home.resolve())
    repo_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(repo_src))

    from rfdetr import RFDETR

    from car5_autolabel.policies.size import filter_minimum_long_side
    from car5_autolabel.postprocessing import (
        apply_mining_label_policy,
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
    from car5_autolabel.tiling import (
        make_tile_windows,
        map_box_from_tile,
        touches_internal_tile_edge,
    )

    image_paths = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: (path.parent.name, path.stem, path.suffix.lower()),
    )
    if not image_paths:
        raise RuntimeError(f"No images found below {image_root}")
    source_image_count = len(image_paths)
    if args.target_fps is not None:
        if args.target_fps <= 0:
            raise ValueError("--target-fps must be positive")
        image_paths = sample_by_camera(image_paths, args.target_fps)

    manifest_seed = "\n".join(
        f"{path.relative_to(image_root).as_posix()}:{path.stat().st_size}" for path in image_paths
    )
    manifest_id = hashlib.sha256(manifest_seed.encode("utf-8")).hexdigest()
    checkpoint_hash = sha256(checkpoint)
    model = RFDETR.from_checkpoint(checkpoint)
    class_names = list(model.class_names)
    started = time.time()
    counts: Counter[str] = Counter()
    boxes_total = 0
    tile_candidates_total = 0
    crop_proposals_total = 0
    crop_confirmed_total = 0
    crop_reclassified_total = 0
    invalid_class_queries_total = 0
    invalid_geometry_queries_total = 0
    crop_edge_retries_total = 0
    crop_edge_rejected_total = 0
    below_size_policy_total = 0

    def detections_to_boxes(detections, *, offset_window=None) -> list[BoundingBox]:
        nonlocal invalid_class_queries_total, invalid_geometry_queries_total
        converted = []
        for box, score, class_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            strict=True,
        ):
            class_index = int(class_id)
            # Some RF-DETR releases may expose the no-object/background query
            # in low-threshold output. It is not a dataset class and must not
            # be indexed into class_names.
            if not 0 <= class_index < len(class_names):
                invalid_class_queries_total += 1
                continue
            label = class_names[class_index]
            x1, y1, x2, y2 = (float(value) for value in box)
            if (
                not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or x1 < 0
                or y1 < 0
                or x2 <= x1
                or y2 <= y1
            ):
                invalid_geometry_queries_total += 1
                continue
            candidate = BoundingBox(
                label=label,
                score=float(score),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            if offset_window is not None:
                candidate = map_box_from_tile(candidate, offset_window)
            converted.append(candidate)
        return converted

    predictions_tmp = output_dir / "predictions.jsonl.tmp"
    manifest_tmp = output_dir / "manifest.jsonl.tmp"
    with (
        predictions_tmp.open("w", encoding="utf-8", newline="\n") as pred_handle,
        manifest_tmp.open("w", encoding="utf-8", newline="\n") as manifest_handle,
    ):
        for offset in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[offset : offset + args.batch_size]
            inference_threshold = (
                min(args.threshold, args.proposal_threshold)
                if args.proposal_threshold is not None
                else args.threshold
            )
            batch_predictions = model.predict(
                [str(path) for path in batch_paths],
                threshold=inference_threshold,
                include_source_image=False,
            )
            if not isinstance(batch_predictions, list):
                batch_predictions = [batch_predictions]

            for path, detections in zip(batch_paths, batch_predictions, strict=True):
                camera = path.parent.name
                frame_id = path.stem
                image_id = f"{args.dataset_name}__{frame_id}__{camera}"
                with Image.open(path) as image_handle:
                    image = image_handle.convert("RGB")
                    width, height = image.size
                    all_global_boxes = detections_to_boxes(detections)
                    raw_boxes = [box for box in all_global_boxes if box.score >= args.threshold]
                    if args.proposal_threshold is not None:
                        low_proposals = [
                            box
                            for box in all_global_boxes
                            if args.proposal_threshold <= box.score < args.threshold
                            and is_small_proposal(
                                box,
                                image_width=width,
                                image_height=height,
                                maximum_side=args.crop_maximum_object_side,
                            )
                        ]
                        low_proposals = suppress_same_class_duplicates(
                            low_proposals,
                            iou_threshold=args.same_class_nms_iou,
                            containment_threshold=args.containment_threshold,
                        )[: args.crop_maximum_proposals]
                        crop_proposals_total += len(low_proposals)
                        crop_windows = [
                            make_context_crop(
                                proposal,
                                image_width=width,
                                image_height=height,
                                context_scale=args.crop_context_scale,
                                minimum_size=args.crop_minimum_size,
                            )
                            for proposal in low_proposals
                        ]
                        edge_retries = []
                        for crop_offset in range(0, len(crop_windows), args.batch_size):
                            proposal_batch = low_proposals[
                                crop_offset : crop_offset + args.batch_size
                            ]
                            window_batch = crop_windows[crop_offset : crop_offset + args.batch_size]
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
                            crop_predictions = model.predict(
                                crop_images,
                                threshold=args.crop_confirmation_threshold,
                                include_source_image=False,
                            )
                            if not isinstance(crop_predictions, list):
                                crop_predictions = [crop_predictions]
                            for proposal, window, crop_detections in zip(
                                proposal_batch,
                                window_batch,
                                crop_predictions,
                                strict=True,
                            ):
                                verified = select_verified_detection(
                                    proposal,
                                    detections_to_boxes(crop_detections),
                                    window,
                                    confirmation_threshold=(args.crop_confirmation_threshold),
                                    allow_reclassification=(args.crop_allow_reclassification),
                                    reclassification_threshold=(
                                        args.crop_reclassification_threshold
                                    ),
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
                                    edge_retries.append(
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
                                crop_confirmed_total += 1
                                if verified.label != proposal.label:
                                    crop_reclassified_total += 1
                                raw_boxes.append(verified)
                        crop_edge_retries_total += len(edge_retries)
                        for retry_offset in range(0, len(edge_retries), args.batch_size):
                            retry_batch = edge_retries[
                                retry_offset : retry_offset + args.batch_size
                            ]
                            retry_predictions = model.predict(
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
                                threshold=args.crop_confirmation_threshold,
                                include_source_image=False,
                            )
                            if not isinstance(retry_predictions, list):
                                retry_predictions = [retry_predictions]
                            for (
                                (proposal, window),
                                retry_detections,
                            ) in zip(
                                retry_batch,
                                retry_predictions,
                                strict=True,
                            ):
                                verified = select_verified_detection(
                                    proposal,
                                    detections_to_boxes(retry_detections),
                                    window,
                                    confirmation_threshold=(args.crop_confirmation_threshold),
                                    allow_reclassification=(args.crop_allow_reclassification),
                                    reclassification_threshold=(
                                        args.crop_reclassification_threshold
                                    ),
                                )
                                if verified is None or touches_context_crop_edge(
                                    verified,
                                    window,
                                    image_width=width,
                                    image_height=height,
                                    margin_ratio=args.crop_edge_margin,
                                ):
                                    crop_edge_rejected_total += 1
                                    continue
                                crop_confirmed_total += 1
                                if verified.label != proposal.label:
                                    crop_reclassified_total += 1
                                raw_boxes.append(
                                    verified.model_copy(
                                        update={
                                            "postprocess_reason": ("expanded_native_crop_verified")
                                        }
                                    )
                                )
                    if args.tile_size > 0:
                        windows = make_tile_windows(
                            width,
                            height,
                            tile_size=args.tile_size,
                            overlap=args.tile_overlap,
                        )
                        for tile_offset in range(0, len(windows), args.batch_size):
                            tile_windows = windows[tile_offset : tile_offset + args.batch_size]
                            tile_images = [
                                image.crop(
                                    (
                                        window.left,
                                        window.top,
                                        window.right,
                                        window.bottom,
                                    )
                                )
                                for window in tile_windows
                            ]
                            tile_predictions = model.predict(
                                tile_images,
                                threshold=args.threshold,
                                include_source_image=False,
                            )
                            if not isinstance(tile_predictions, list):
                                tile_predictions = [tile_predictions]
                            for window, tile_detections in zip(
                                tile_windows,
                                tile_predictions,
                                strict=True,
                            ):
                                local_boxes = detections_to_boxes(tile_detections)
                                local_boxes = [
                                    box
                                    for box in local_boxes
                                    if not touches_internal_tile_edge(
                                        box,
                                        window,
                                        image_width=width,
                                        image_height=height,
                                        margin_ratio=args.tile_edge_margin,
                                    )
                                ]
                                tile_candidates_total += len(local_boxes)
                                raw_boxes.extend(
                                    map_box_from_tile(box, window) for box in local_boxes
                                )

                resolved_boxes = suppress_same_class_duplicates(
                    raw_boxes,
                    iou_threshold=args.same_class_nms_iou,
                    containment_threshold=args.containment_threshold,
                )
                resolved_boxes = suppress_cross_class_overlaps(
                    resolved_boxes,
                    iou_threshold=args.cross_class_nms_iou,
                    watertruck_ambiguity_margin=0.10,
                )
                resolved_boxes = suppress_cross_class_overlaps(
                    resolved_boxes,
                    iou_threshold=None,
                    containment_threshold=args.containment_threshold,
                    comparable_labels={
                        "Car",
                        "Truck",
                        "BoxTruck",
                        "Bulldozer",
                        "Excavator",
                        "WaterTruck",
                    },
                )
                resolved_boxes = apply_mining_label_policy(
                    resolved_boxes,
                    watertruck_min_score=args.watertruck_min_score,
                )
                resolved_boxes = suppress_same_class_duplicates(
                    resolved_boxes,
                    iou_threshold=args.same_class_nms_iou,
                    containment_threshold=args.containment_threshold,
                )
                resolved_boxes, below_size_policy = filter_minimum_long_side(
                    resolved_boxes,
                    minimum_px=args.minimum_long_side_px,
                )
                below_size_policy_total += len(below_size_policy)
                exported = []
                for box in resolved_boxes:
                    counts[box.label] += 1
                    boxes_total += 1
                    exported.append(
                        {
                            "label": box.label,
                            "raw_label": box.raw_label,
                            "postprocess_reason": box.postprocess_reason,
                            "score": box.score,
                            "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                        }
                    )
                pred_handle.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "sequence_id": f"{args.dataset_name}_{camera}",
                            "camera": camera,
                            "timestamp": frame_id,
                            "width": width,
                            "height": height,
                            "model_version": args.model_version,
                            "checkpoint_sha256": checkpoint_hash,
                            "dataset_manifest_id": manifest_id,
                            "source": "rfdetr",
                            "review_status": "unreviewed_prediction",
                            "detections": exported,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                relative_path = (
                    Path(args.label_studio_root) / "camera" / camera / path.name
                ).as_posix()
                manifest_handle.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "relative_path": relative_path,
                            "camera_id": camera,
                            "frame_id": frame_id,
                            "capture_timestamp": frame_id,
                            "sequence_id": f"{args.dataset_name}_{camera}",
                            "width": width,
                            "height": height,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            completed = min(offset + len(batch_paths), len(image_paths))
            if completed == len(image_paths) or completed % 200 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "processed": completed,
                            "total": len(image_paths),
                            "images_per_second": completed / elapsed,
                            "eta_seconds": (len(image_paths) - completed)
                            / max(completed / elapsed, 1e-9),
                            "boxes": boxes_total,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "manifest.jsonl"
    predictions_tmp.replace(predictions_path)
    manifest_tmp.replace(manifest_path)
    summary = {
        "dataset": args.dataset_name,
        "image_root": str(image_root),
        "images": len(image_paths),
        "source_images": source_image_count,
        "target_fps": args.target_fps,
        "model_version": args.model_version,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest_id": manifest_id,
        "threshold": args.threshold,
        "minimum_long_side_px": args.minimum_long_side_px,
        "below_size_policy_boxes": below_size_policy_total,
        "same_class_nms_iou": args.same_class_nms_iou,
        "cross_class_nms_iou": args.cross_class_nms_iou,
        "containment_threshold": args.containment_threshold,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_edge_margin": args.tile_edge_margin,
        "tile_candidates_before_merge": tile_candidates_total,
        "proposal_threshold": args.proposal_threshold,
        "crop_confirmation_threshold": args.crop_confirmation_threshold,
        "crop_allow_reclassification": args.crop_allow_reclassification,
        "crop_reclassification_threshold": args.crop_reclassification_threshold,
        "crop_context_scale": args.crop_context_scale,
        "crop_minimum_size": args.crop_minimum_size,
        "crop_maximum_object_side": args.crop_maximum_object_side,
        "crop_proposals": crop_proposals_total,
        "crop_confirmed": crop_confirmed_total,
        "crop_reclassified": crop_reclassified_total,
        "crop_edge_margin": args.crop_edge_margin,
        "crop_edge_retry_scale": args.crop_edge_retry_scale,
        "crop_edge_retries": crop_edge_retries_total,
        "crop_edge_rejected": crop_edge_rejected_total,
        "invalid_class_queries_ignored": invalid_class_queries_total,
        "invalid_geometry_queries_ignored": invalid_geometry_queries_total,
        "watertruck_min_score": args.watertruck_min_score,
        "prediction_boxes": boxes_total,
        "class_counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.time() - started,
        "images_per_second": len(image_paths) / max(time.time() - started, 1e-9),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
