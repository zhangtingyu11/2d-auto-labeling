from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy COCO-referenced images to a Linux-native training cache."
    )
    parser.add_argument("--coco", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    return parser.parse_args()


def _read_file_names(paths: list[Path]) -> list[Path]:
    names: set[Path] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for image in payload["images"]:
            relative = Path(image["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe COCO file name: {relative}")
            names.add(relative)
    return sorted(names, key=lambda path: path.as_posix())


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    copied = 0
    reused = 0
    for relative in _read_file_names(args.coco):
        source = source_root / relative
        destination = destination_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            reused += 1
            continue
        shutil.copy2(source, destination)
        copied += 1
    print(
        json.dumps(
            {
                "copied": copied,
                "reused": reused,
                "total": copied + reused,
                "destination_root": str(destination_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
