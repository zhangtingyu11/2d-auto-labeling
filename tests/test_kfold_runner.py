import argparse
import csv
import json
import os

import pytest
from pycocotools.coco import COCO

from tools.audit_coco_predictions import coco_ap
from tools.run_rfdetr_kfold_label_audit import (
    Source,
    docker_base,
    parse_source,
    prepare,
    stable_hash,
)


def test_parse_source_rejects_path_traversal_name(tmp_path) -> None:
    coco = tmp_path / "annotations.json"
    coco.write_text("{}", encoding="utf-8")

    with pytest.raises(argparse.ArgumentTypeError):
        parse_source(f"../outside={coco}::{tmp_path}")


def test_docker_command_sets_host_user_and_optional_gpu(tmp_path) -> None:
    args = argparse.Namespace(docker_image="audit:test")

    command = docker_base(args, 3, [(tmp_path, "/data", True)])

    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert command[command.index("--gpus") + 1] == "device=3"
    assert f"{tmp_path.resolve()}:/data:ro" in command


def test_preparation_hash_changes_with_policy() -> None:
    first = stable_hash({"minimum_long_side_px": 70, "purge_seconds": 3})
    second = stable_hash({"minimum_long_side_px": 70, "purge_seconds": 4})

    assert first != second


def test_official_coco_ap_is_one_for_perfect_prediction(tmp_path) -> None:
    ground_truth = {
        "info": {},
        "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 0, "name": "Car"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [0, 0, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            }
        ],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(ground_truth), encoding="utf-8")
    coco = COCO(str(path))
    prediction = [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100], "score": 1.0}]

    map50, map50_95 = coco_ap(coco, prediction, [0])

    assert map50 == pytest.approx(1.0)
    assert map50_95 == pytest.approx(1.0)


def test_prepare_purges_nearby_frames_across_cameras_and_checks_fingerprint(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    for name in ("valid.jpg", "near.jpg", "far.jpg"):
        (source_root / name).write_bytes(b"image")
    categories = [
        {"id": index + 1, "name": name}
        for index, name in enumerate(
            ("Car", "Truck", "Bulldozer", "Excavator", "WaterTruck", "Sign")
        )
    ]
    coco = {
        "images": [
            {"id": 1, "file_name": "valid.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "near.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "far.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": index, "image_id": index, "category_id": 1, "bbox": [0, 0, 70, 70]}
            for index in range(1, 4)
        ],
        "categories": categories,
    }
    coco_path = source_root / "annotations.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    assignments = tmp_path / "assignments.csv"
    with assignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "package",
                "source_image_id",
                "source_dataset",
                "camera",
                "timestamp",
                "validation_fold",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "package": "source",
                    "source_image_id": 1,
                    "source_dataset": "route",
                    "camera": "front",
                    "timestamp": "20260101_000000_000000",
                    "validation_fold": 0,
                },
                {
                    "package": "source",
                    "source_image_id": 2,
                    "source_dataset": "route",
                    "camera": "back",
                    "timestamp": "20260101_000002_000000",
                    "validation_fold": 1,
                },
                {
                    "package": "source",
                    "source_image_id": 3,
                    "source_dataset": "route",
                    "camera": "back",
                    "timestamp": "20260101_000010_000000",
                    "validation_fold": 1,
                },
            ]
        )
    args = argparse.Namespace(
        workspace=tmp_path / "workspace",
        assignments=assignments,
        folds=2,
        minimum_long_side_px=70,
        purge_seconds=3.0,
    )
    source = Source("source", coco_path, source_root)

    report = prepare(args, [source])

    assert report["fold_reports"]["0"]["train"]["images"] == 1
    assert report["fold_reports"]["0"]["purged_near_validation"] == 1
    prepared = json.loads(
        (args.workspace / "folds/fold_0/dataset/train/_annotations.coco.json").read_text()
    )
    assert prepared["images"][0]["source_image_id"] == 3
    args.purge_seconds = 4.0
    with pytest.raises(RuntimeError, match="different inputs or policy"):
        prepare(args, [source])
