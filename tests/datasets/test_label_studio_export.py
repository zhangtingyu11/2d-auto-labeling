import json
from pathlib import Path

import pytest

from car5_autolabel.datasets.label_studio_export import (
    LabelStudioExportError,
    parse_label_studio_export,
)


def test_export_distinguishes_empty_skipped_and_unreviewed(tmp_path: Path) -> None:
    tasks = [
        {
            "id": 1,
            "data": {"camera": "front", "frame_id": "frame-1"},
            "annotations": [{"id": 1, "was_cancelled": False, "result": []}],
        },
        {
            "id": 2,
            "data": {"camera": "front", "frame_id": "frame-2"},
            "annotations": [{"id": 2, "was_cancelled": True, "result": []}],
        },
        {"id": 3, "data": {"camera": "front", "frame_id": "frame-3"}, "annotations": []},
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")

    parsed, summary = parse_label_studio_export(path)

    assert parsed[("front", "frame-1")].review_status == "verified_empty"
    assert parsed[("front", "frame-2")].review_status == "skipped"
    assert parsed[("front", "frame-3")].review_status == "unreviewed"
    assert summary["status_counts"] == {"skipped": 1, "unreviewed": 1, "verified_empty": 1}


def test_export_rejects_unknown_class(tmp_path: Path) -> None:
    tasks = [
        {
            "id": 1,
            "data": {"camera": "front", "frame_id": "frame-1"},
            "annotations": [
                {
                    "id": 1,
                    "was_cancelled": False,
                    "result": [
                        {
                            "id": "box-1",
                            "type": "rectanglelabels",
                            "original_width": 1920,
                            "original_height": 1080,
                            "value": {
                                "x": 10,
                                "y": 10,
                                "width": 10,
                                "height": 10,
                                "rectanglelabels": ["BoxTruck"],
                            },
                        }
                    ],
                }
            ],
        }
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")

    with pytest.raises(LabelStudioExportError, match="unknown active class"):
        parse_label_studio_export(path)
