from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage exported car5 COCO folds in the directory layout RF-DETR expects."
    )
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-valid-images", type=int)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def _load_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing COCO keys: {sorted(missing)}")
    return payload


def _limit_coco(payload: dict[str, Any], limit: int | None) -> dict[str, Any]:
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    if limit is not None:
        if limit <= 0:
            raise ValueError("image limits must be positive")
        images = images[:limit]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    return {
        key: value
        for key, value in payload.items()
        if key not in {"images", "annotations"}
    } | {"images": images, "annotations": annotations}


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe COCO file name: {value}")
    return relative


def _stage_images(
    payload: dict[str, Any],
    *,
    split_dir: Path,
    images_root: Path,
    copy_images: bool,
) -> None:
    camera_source = images_root / "camera"
    if not camera_source.is_dir():
        raise FileNotFoundError(camera_source)

    for image in payload["images"]:
        relative = _safe_relative_path(str(image["file_name"]))
        source = images_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)

    split_dir.mkdir(parents=True, exist_ok=True)
    camera_destination = split_dir / "camera"
    if copy_images:
        import shutil

        for image in payload["images"]:
            relative = _safe_relative_path(str(image["file_name"]))
            source = images_root / relative
            destination = split_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
        return

    if camera_destination.is_symlink():
        if camera_destination.resolve() == camera_source.resolve():
            return
        camera_destination.unlink()
    elif camera_destination.exists():
        raise FileExistsError(
            f"refusing to replace non-symlink image directory: {camera_destination}"
        )
    os.symlink(camera_source, camera_destination, target_is_directory=True)


def _stage_split(
    source: Path,
    *,
    split_dir: Path,
    images_root: Path,
    limit: int | None,
    copy_images: bool,
) -> dict[str, int]:
    payload = _limit_coco(_load_coco(source), limit)
    _stage_images(
        payload,
        split_dir=split_dir,
        images_root=images_root,
        copy_images=copy_images,
    )
    annotation_path = split_dir / "_annotations.coco.json"
    annotation_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
    }


def main() -> int:
    args = _parse_args()
    coco_root = args.coco_root.resolve()
    images_root = args.images_root.resolve()
    output_dir = args.output_dir.resolve()
    report = {
        "train": _stage_split(
            coco_root / "instances_train.json",
            split_dir=output_dir / "train",
            images_root=images_root,
            limit=args.max_train_images,
            copy_images=args.copy_images,
        ),
        "valid": _stage_split(
            coco_root / "instances_val.json",
            split_dir=output_dir / "valid",
            images_root=images_root,
            limit=args.max_valid_images,
            copy_images=args.copy_images,
        ),
        "output_dir": str(output_dir),
        "image_mode": "copy" if args.copy_images else "symlink",
    }
    (output_dir / "stage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
