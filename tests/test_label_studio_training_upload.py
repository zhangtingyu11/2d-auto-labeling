import csv
import json
from pathlib import Path

import pytest

from car5_autolabel.integrations.label_studio_training_upload import (
    convert_training_export,
)


def _rectangle(result_id: str, label: str, *, from_name: str = "label") -> dict:
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": "image",
        "original_width": 200,
        "original_height": 100,
        "image_rotation": 0,
        "readonly": from_name != "label",
        "value": {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def _write_assignments(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "package",
                "source_image_id",
                "source_dataset",
                "camera",
                "timestamp",
                "file_name",
                "block_key",
                "validation_fold",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "package": "old",
                    "source_image_id": "10",
                    "source_dataset": "sequence-a",
                    "camera": "front",
                    "timestamp": "20260611_120000_000000",
                    "file_name": "images/sequence-a/front/a.jpg",
                    "block_key": "sequence-a:1",
                    "validation_fold": "2",
                },
                {
                    "package": "old",
                    "source_image_id": "11",
                    "source_dataset": "sequence-b",
                    "camera": "back",
                    "timestamp": "20260611_130000_000000",
                    "file_name": "images/sequence-b/back/b.jpg",
                    "block_key": "sequence-b:1",
                    "validation_fold": "4",
                },
            ]
        )


def _tasks() -> list[dict]:
    return [
        {
            "id": 101,
            "data": {
                "image": "/data/local-files/?d=images/sequence-a/front/a.jpg",
                "candidate_summary": "可能漏标 PRED=Car",
                "review_tag": ["可能漏标"],
                "kfold_model_version": "old-model",
            },
            "annotations": [
                {
                    "id": 1,
                    "updated_at": "2026-08-11T10:00:00Z",
                    "result": [
                        _rectangle("human-1", "Car"),
                        _rectangle(
                            "hint-1", "可能漏标（预测=Truck）", from_name="audit_hint"
                        ),
                    ],
                }
            ],
        },
        {
            "id": 102,
            "data": {"image": "/data/local-files/?d=images/sequence-b/back/b.jpg"},
            "annotations": [{"id": 2, "result": []}],
        },
    ]


def test_training_upload_keeps_only_human_boxes_and_inherits_folds(tmp_path: Path) -> None:
    exported = tmp_path / "export.json"
    template = tmp_path / "old.csv"
    coco = tmp_path / "human.coco.json"
    assignments = tmp_path / "assignments.csv"
    summary_path = tmp_path / "summary.json"
    exported.write_text(json.dumps(_tasks()), encoding="utf-8")
    _write_assignments(template)

    summary = convert_training_export(
        label_studio_export=exported,
        assignment_template=template,
        coco_output=coco,
        assignments_output=assignments,
        summary_output=summary_path,
    )

    converted = json.loads(coco.read_text())
    assert summary.task_count == 2
    assert summary.human_box_count == 1
    assert summary.removed_non_human_result_count == 1
    assert summary.empty_task_count == 1
    assert summary.fold_counts == {"2": 1, "4": 1}
    assert [item["name"] for item in converted["categories"]] == [
        "Car",
        "Truck",
        "BoxTruck",
        "Bulldozer",
        "Excavator",
        "WaterTruck",
        "Sign",
        "Pedestrian",
    ]
    assert len(converted["annotations"]) == 1
    assert "可能漏标" not in coco.read_text(encoding="utf-8")
    with assignments.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["package"] for row in rows] == ["reviewed1085", "reviewed1085"]
    assert [row["validation_fold"] for row in rows] == ["2", "4"]


def test_training_upload_rejects_unsubmitted_tasks(tmp_path: Path) -> None:
    tasks = _tasks()
    tasks[1]["annotations"] = []
    exported = tmp_path / "export.json"
    template = tmp_path / "old.csv"
    exported.write_text(json.dumps(tasks), encoding="utf-8")
    _write_assignments(template)

    with pytest.raises(ValueError, match="no submitted annotation"):
        convert_training_export(
            label_studio_export=exported,
            assignment_template=template,
            coco_output=tmp_path / "human.coco.json",
            assignments_output=tmp_path / "assignments.csv",
            summary_output=tmp_path / "summary.json",
        )


def test_training_upload_rejects_audit_label_in_human_control(tmp_path: Path) -> None:
    tasks = _tasks()
    tasks[0]["annotations"][0]["result"][0] = _rectangle("bad", "可能漏标")
    exported = tmp_path / "export.json"
    template = tmp_path / "old.csv"
    exported.write_text(json.dumps(tasks), encoding="utf-8")
    _write_assignments(template)

    with pytest.raises(ValueError, match="unsupported human labels"):
        convert_training_export(
            label_studio_export=exported,
            assignment_template=template,
            coco_output=tmp_path / "human.coco.json",
            assignments_output=tmp_path / "assignments.csv",
            summary_output=tmp_path / "summary.json",
        )
