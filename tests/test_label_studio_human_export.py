import json
import sqlite3
from pathlib import Path

import pytest

from car5_autolabel.integrations.label_studio_human_export import export_human_annotations


LABEL_CONFIG = """<View>
  <RectangleLabels name="prediction_reference" toName="image">
    <Label value="【可能漏标】Car"/>
  </RectangleLabels>
  <RectangleLabels name="label" toName="image">
    <Label value="Car"/><Label value="Truck"/>
  </RectangleLabels>
</View>"""


def _create_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE project (
            id INTEGER PRIMARY KEY, title TEXT, label_config TEXT, deleted_at TEXT
        );
        CREATE TABLE task (
            id INTEGER PRIMARY KEY, data TEXT, is_labeled INTEGER, project_id INTEGER
        );
        CREATE TABLE task_completion (
            id INTEGER PRIMARY KEY, result TEXT, was_cancelled INTEGER, created_at TEXT,
            updated_at TEXT, task_id INTEGER, ground_truth INTEGER
        );
        """
    )
    connection.execute("INSERT INTO project VALUES (3, '2D数据集', ?, NULL)", (LABEL_CONFIG,))
    task_one_data = {
        "image": "/data/local-files/?d=images/one.jpg",
        "image_id": "image-one",
        "camera": "front",
        "review_tag": "可能漏标",
        "review_tag_label": "可能漏标",
        "review_summary": "可能漏标:Car",
        "kfold_candidate_count": 1,
    }
    task_two_data = {"image": "/data/local-files/?d=images/two.jpg", "image_id": "image-two"}
    connection.execute("INSERT INTO task VALUES (11, ?, 1, 3)", (json.dumps(task_one_data),))
    connection.execute("INSERT INTO task VALUES (12, ?, 1, 3)", (json.dumps(task_two_data),))
    human = {
        "id": "human-1",
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": 200,
        "original_height": 100,
        "value": {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "rotation": 0,
            "rectanglelabels": ["Car"],
        },
        "origin": "manual",
    }
    locked = {
        "id": "locked-1",
        "from_name": "prediction_reference",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": 200,
        "original_height": 100,
        "readonly": True,
        "value": {
            "x": 50,
            "y": 50,
            "width": 20,
            "height": 20,
            "rectanglelabels": ["【可能漏标】Car"],
        },
    }
    issue = {
        "id": "issue-1",
        "from_name": "review_issue",
        "to_name": "image",
        "type": "choices",
        "readonly": True,
        "value": {"choices": ["可能漏标"]},
    }
    connection.execute(
        "INSERT INTO task_completion VALUES (21, ?, 0, '2026-01-01', '2026-01-01', 11, 0)",
        (json.dumps([human, locked, issue]),),
    )
    connection.execute(
        "INSERT INTO task_completion VALUES (22, '[]', 0, '2026-01-01', '2026-01-01', 12, 0)"
    )
    connection.commit()
    connection.close()


def test_export_keeps_only_human_boxes_and_preserves_empty_tasks(tmp_path: Path) -> None:
    database = tmp_path / "label_studio.sqlite3"
    label_studio_json = tmp_path / "human.json"
    coco_json = tmp_path / "human.coco.json"
    _create_database(database)

    summary = export_human_annotations(
        database=database,
        project_id=3,
        label_studio_json=label_studio_json,
        coco_json=coco_json,
        default_width=1920,
        default_height=1080,
    )

    label_studio = json.loads(label_studio_json.read_text())
    coco = json.loads(coco_json.read_text())
    assert summary.task_count == 2
    assert summary.empty_task_count == 1
    assert summary.human_box_count == 1
    assert summary.removed_non_human_result_count == 2
    assert len(label_studio) == 2
    assert label_studio[0]["data"] == {
        "image": "/data/local-files/?d=images/one.jpg",
        "image_id": "image-one",
        "camera": "front",
    }
    assert label_studio[0]["annotations"][0]["result"][0]["from_name"] == "label"
    assert label_studio[1]["annotations"][0]["result"] == []
    serialized = label_studio_json.read_text(encoding="utf-8")
    assert "可能漏标" not in serialized
    assert "prediction_reference" not in serialized
    assert "review_issue" not in serialized
    assert coco["images"][0]["file_name"] == "images/one.jpg"
    assert coco["images"][1]["width"] == 1920
    assert coco["annotations"][0]["bbox"] == pytest.approx([20, 20, 60, 40])
    assert [category["name"] for category in coco["categories"]] == ["Car", "Truck"]
