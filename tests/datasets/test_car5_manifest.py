from pathlib import Path

import cv2
import numpy as np
import pytest

from car5_autolabel.datasets.car5 import Car5DatasetBuilder, resolve_dataset_root
from car5_autolabel.datasets.constants import CAMERA_ORDER
from car5_autolabel.datasets.manifest import write_manifest_jsonl
from car5_autolabel.datasets.models import AnnotationStatus


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    wrapper = tmp_path / "unicode_path_数据"
    root = wrapper / "car5_test"
    stems = [
        "20260611_165130_000000",
        "20260611_165130_100000",
        "20260611_165132_000000",
        "20260611_165132_100000",
    ]
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    image_bytes = encoded.tobytes()
    for camera in CAMERA_ORDER:
        camera_dir = root / "camera" / camera
        camera_dir.mkdir(parents=True)
        for stem in stems:
            (camera_dir / f"{stem}.jpg").write_bytes(image_bytes)
    label_dir = root / "label"
    label_dir.mkdir(parents=True)
    for stem in (stems[0], stems[2]):
        (label_dir / f"{stem}.json").write_text("{}\n", encoding="utf-8")
    return wrapper, root, stems


def test_resolves_single_nested_dataset_root(
    synthetic_dataset: tuple[Path, Path, list[str]],
) -> None:
    wrapper, root, _ = synthetic_dataset
    assert resolve_dataset_root(wrapper) == root.resolve()


def test_builds_deterministic_frame_major_rows_with_sequences(
    synthetic_dataset: tuple[Path, Path, list[str]],
    tmp_path: Path,
) -> None:
    wrapper, _, stems = synthetic_dataset
    result = Car5DatasetBuilder(
        wrapper,
        dataset_id="synthetic",
        include_context=True,
        hash_files=True,
        verify_dimensions=True,
        expected_keyframes=2,
        expected_camera_frames=4,
        gap_threshold_ms=1000,
    ).build()

    assert result.report.valid
    assert len(result.rows) == 24
    assert result.report.stats["selected_keyframe_image_count"] == 12
    assert result.report.stats["selected_context_image_count"] == 12
    assert result.report.stats["sequence_count"] == 2
    assert [row.camera_id for row in result.rows[:6]] == list(CAMERA_ORDER)
    assert {row.frame_id for row in result.rows[:6]} == {stems[0]}
    assert all(row.sha256 for row in result.rows)

    keyframe = result.rows[0]
    context = next(row for row in result.rows if not row.is_keyframe)
    assert keyframe.annotation_status is AnnotationStatus.UNREVIEWED
    assert context.annotation_status is AnnotationStatus.TEMPORAL_CONTEXT
    assert keyframe.fold_id == "s00"
    assert result.rows[-1].fold_id == "s01"

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest_jsonl(result.rows, first)
    write_manifest_jsonl(result.rows, second)
    assert first.read_bytes() == second.read_bytes()


def test_reports_missing_camera_keyframe(
    synthetic_dataset: tuple[Path, Path, list[str]],
) -> None:
    wrapper, root, stems = synthetic_dataset
    (root / "camera" / "back_right" / f"{stems[2]}.jpg").unlink()

    result = Car5DatasetBuilder(
        wrapper,
        dataset_id="synthetic",
        include_context=False,
        hash_files=False,
        expected_keyframes=2,
        expected_camera_frames=4,
    ).build()

    assert not result.report.valid
    codes = {issue.code for issue in result.report.errors}
    assert "camera_timeline_mismatch" in codes
    assert "missing_keyframe_image" in codes
    assert "unexpected_camera_frame_count" in codes
