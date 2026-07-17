from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _uniform_indices(total: int, count: int) -> list[int]:
    if count <= 0 or count > total:
        raise ValueError(f"group count must be between 1 and {total}")
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select reviewed six-camera keyframe benchmark")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=int, default=40)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["is_keyframe"] and row["annotation_status"] in {
            "human_verified",
            "verified_empty",
        }:
            grouped[row["capture_timestamp"]].append(row)
    timestamps = sorted(grouped)
    selected = [timestamps[index] for index in _uniform_indices(len(timestamps), args.groups)]

    output_rows = []
    for benchmark_group, timestamp in enumerate(selected):
        camera_rows = sorted(grouped[timestamp], key=lambda row: row["camera_order"])
        if len(camera_rows) != 6:
            raise ValueError(f"keyframe {timestamp} has {len(camera_rows)} cameras, expected 6")
        for row in camera_rows:
            output_rows.append(
                {
                    "scene_id": row["dataset_id"],
                    "time_index": row["raw_sequence_index"],
                    "timestamp": timestamp,
                    "camera": row["camera_id"],
                    "image_path": str((args.dataset_root / row["relative_path"]).resolve()),
                    "width": row["width"],
                    "height": row["height"],
                    "benchmark_group": benchmark_group,
                    "selection_reason": "uniform_reviewed_keyframes_synchronized_six_camera",
                    "annotation_status": row["annotation_status"],
                    "sequence_id": row["sequence_id"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "available_keyframes": len(timestamps),
                "selected_groups": len(selected),
                "images": len(output_rows),
                "first_timestamp": selected[0],
                "last_timestamp": selected[-1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
