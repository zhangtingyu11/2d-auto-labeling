from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from car5_autolabel.integrations.label_studio import prediction_task_payload
from car5_autolabel.schemas import BoundingBox, ImagePrediction


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RTMDet JSONL to Label Studio tasks")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local-files-prefix", default="/data/local-files/?d=")
    args = parser.parse_args()

    manifest = {row["image_id"]: row for row in _jsonl(args.manifest)}
    tasks: list[dict] = []
    seen: set[str] = set()
    for row in _jsonl(args.predictions):
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate prediction image_id: {image_id}")
        seen.add(image_id)
        source = manifest.get(image_id)
        if source is None:
            raise ValueError(f"prediction is absent from manifest: {image_id}")
        prediction = ImagePrediction(
            image_id=image_id,
            width=row["width"],
            height=row["height"],
            model_version=row["model_version"],
            config_sha256=row.get("config_sha256"),
            checkpoint_sha256=row.get("checkpoint_sha256"),
            dataset_manifest_id=row.get("dataset_manifest_id"),
            source=row.get("source", "detector"),
            review_status=row.get("review_status", "unreviewed_prediction"),
            boxes=[
                BoundingBox(
                    x1=item["bbox_xyxy"][0],
                    y1=item["bbox_xyxy"][1],
                    x2=item["bbox_xyxy"][2],
                    y2=item["bbox_xyxy"][3],
                    label=item["label"],
                    score=item["score"],
                )
                for item in row.get("detections", [])
            ],
        )
        image_url = args.local_files_prefix + quote(source["relative_path"], safe="/")
        tasks.append(
            prediction_task_payload(
                prediction,
                image_url=image_url,
                camera_id=source["camera_id"],
                frame_id=source["frame_id"],
                capture_timestamp=source["capture_timestamp"],
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(tasks), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
