from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CAMERA_ORDER = ("front", "front_left", "front_right", "back", "back_left", "back_right")
COLORS = {
    "Car": (80, 220, 80),
    "Truck": (255, 120, 40),
    "Bulldozer": (0, 150, 255),
    "Excavator": (220, 80, 220),
    "WaterTruck": (220, 220, 40),
    "Sign": (40, 220, 255),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def draw_prediction(image: np.ndarray, row: dict) -> np.ndarray:
    canvas = image.copy()
    for detection in row["detections"]:
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        label = detection["label"]
        score = detection["score"]
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        text = f"{label} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        text_y = max(text_height + baseline + 2, y1)
        cv2.rectangle(
            canvas,
            (x1, text_y - text_height - baseline - 4),
            (x1 + text_width + 6, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x1 + 3, text_y - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def make_panel(image: np.ndarray, row: dict, width: int = 640, height: int = 360) -> np.ndarray:
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    title = f"{row['camera']} | predictions: {len(row['detections'])}"
    cv2.rectangle(panel, (0, 0), (width, 36), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def render_group(
    rows: list[dict],
    manifest_by_id: dict[str, dict],
    dataset_root: Path,
    output: Path,
) -> None:
    by_camera = {row["camera"]: row for row in rows}
    panels = []
    for camera in CAMERA_ORDER:
        row = by_camera[camera]
        image_path = dataset_root / manifest_by_id[row["image_id"]]["relative_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        panels.append(make_panel(draw_prediction(image, row), row))

    collage = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), collage, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Could not write image: {output}")


def choose_groups(groups: list[list[dict]]) -> dict[str, list[dict]]:
    nonempty = [group for group in groups if sum(len(row["detections"]) for row in group) > 0]
    if not nonempty:
        raise ValueError("No non-empty prediction groups found")
    ordered = sorted(nonempty, key=lambda group: sum(len(row["detections"]) for row in group))
    return {
        "sparse": ordered[0],
        "typical": ordered[len(ordered) // 2],
        "busy": ordered[round((len(ordered) - 1) * 0.9)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render honest six-camera prediction samples")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--sequence",
        default="s00",
        help="Sequence to sample; s00 is the held-out validation sequence by default",
    )
    parser.add_argument(
        "--timestamp",
        action="append",
        dest="timestamps",
        help="Render an exact timestamp; repeat to create directly comparable samples",
    )
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    manifest = load_jsonl(args.manifest)
    manifest_by_id = {row["image_id"]: row for row in manifest}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        if row["sequence_id"] == args.sequence:
            grouped[row["timestamp"]].append(row)
    complete_groups = [
        rows
        for rows in grouped.values()
        if {row["camera"] for row in rows} == set(CAMERA_ORDER)
    ]

    if args.timestamps:
        by_timestamp = {rows[0]["timestamp"]: rows for rows in complete_groups}
        missing = [timestamp for timestamp in args.timestamps if timestamp not in by_timestamp]
        if missing:
            raise ValueError(f"Timestamps not found in sequence {args.sequence}: {missing}")
        selected = {
            f"selected_{index:02d}": by_timestamp[timestamp]
            for index, timestamp in enumerate(args.timestamps, start=1)
        }
    else:
        selected = choose_groups(complete_groups)

    summary = {}
    for sample_type, rows in selected.items():
        timestamp = rows[0]["timestamp"]
        output = args.output_dir / f"{sample_type}_{args.sequence}_{timestamp}_6cams.jpg"
        render_group(rows, manifest_by_id, args.dataset_root, output)
        summary[sample_type] = {
            "timestamp": timestamp,
            "detections": sum(len(row["detections"]) for row in rows),
            "output": str(output.resolve()),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
