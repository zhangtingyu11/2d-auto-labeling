import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from car5_autolabel.datasets.manifest import build_manifest

CAMERAS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _make_dataset(root: Path) -> list[str]:
    stems = [
        "20260101_000000_000000",
        "20260101_000000_100000",
        "20260101_000002_000000",
    ]
    for camera in CAMERAS:
        for stem in stems:
            _write_image(root / "camera" / camera / f"{stem}.jpg")
    (root / "label").mkdir()
    for stem in (stems[0], stems[2]):
        (root / "label" / f"{stem}.json").write_text("[]\n", encoding="utf-8")
    return stems


def _write_export(path: Path, stems: list[str]) -> None:
    tasks = []
    task_id = 1
    for stem in (stems[0], stems[2]):
        for camera in CAMERAS:
            result = []
            if camera == "front" and stem == stems[0]:
                result = [
                    {
                        "id": "box-1",
                        "type": "rectanglelabels",
                        "original_width": 1920,
                        "original_height": 1080,
                        "value": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "rectanglelabels": ["truck"],
                        },
                    }
                ]
            tasks.append(
                {
                    "id": task_id,
                    "data": {"camera": camera, "frame_id": stem},
                    "annotations": [{"id": task_id, "was_cancelled": False, "result": result}],
                }
            )
            task_id += 1
    path.write_text(json.dumps(tasks), encoding="utf-8")


def test_build_manifest_preserves_context_and_review_states(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    stems = _make_dataset(dataset)
    export = tmp_path / "export.json"
    _write_export(export, stems)

    rows, report = build_manifest(
        dataset,
        tmp_path / "output",
        label_studio_export=export,
        hash_scope="none",
        dimension_scope="sample",
    )

    assert report.valid
    assert report.manifest_rows == 18
    assert report.supervised_rows == 12
    assert report.temporal_context_rows == 6
    assert report.sequence_counts == {"s00": 1, "s01": 1}
    assert report.annotation_status_counts == {
        "human_verified": 1,
        "temporal_context": 6,
        "verified_empty": 11,
    }
    assert report.reviewed_box_counts == {"truck": 1}
    assert [row.camera_id for row in rows[:6]] == list(CAMERAS)
    assert rows[6].annotation_status == "temporal_context"
    assert rows[12].sequence_id == "s01"


def test_missing_camera_fails_release_validation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    stems = _make_dataset(dataset)
    (dataset / "camera" / "back_right" / f"{stems[0]}.jpg").unlink()

    with pytest.raises(ValueError, match="missing"):
        build_manifest(
            dataset,
            tmp_path / "output",
            hash_scope="none",
            dimension_scope="none",
        )

    report = json.loads((tmp_path / "output" / "validation_report.json").read_text())
    assert not report["valid"]
    assert report["missing_camera_stems"]["back_right"] == [stems[0]]
