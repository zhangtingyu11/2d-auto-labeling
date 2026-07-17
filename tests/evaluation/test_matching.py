from pathlib import Path

import numpy as np

from car5_autolabel.evaluation.matching import (
    maximum_iou_match,
    maximum_iou_match_with_ignore,
    pairwise_iou,
)
from car5_autolabel.policies.camera import build_ignore_mask, load_camera_policy


def test_pairwise_iou_and_one_to_one_assignment() -> None:
    ground_truth = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    predictions = np.array(
        [[1, 1, 11, 11], [20, 20, 30, 30], [100, 100, 110, 110]], dtype=float
    )

    ious = pairwise_iou(ground_truth, predictions)
    result = maximum_iou_match(ground_truth, predictions, iou_threshold=0.5)

    assert ious.shape == (2, 3)
    assert [(item.ground_truth_index, item.prediction_index) for item in result.matches] == [
        (0, 0),
        (1, 1),
    ]
    assert result.unmatched_ground_truth == ()
    assert result.unmatched_predictions == (2,)


def test_camera_policy_filters_only_boxes_mostly_inside_ignore_region(
    tmp_path: Path,
) -> None:
    config = tmp_path / "camera.yaml"
    config.write_text(
        """version: 1
camera_id: test
image_size: [100, 100]
ignore_regions:
  - name: host
    polygon_xy: [[0, 0], [20, 0], [20, 100], [0, 100]]
""",
        encoding="utf-8",
    )
    mask = build_ignore_mask(load_camera_policy(config))
    ground_truth = np.array([[40, 40, 60, 60]], dtype=float)
    predictions = np.array(
        [[0, 10, 15, 30], [40, 40, 60, 60], [15, 10, 35, 30]], dtype=float
    )

    result = maximum_iou_match_with_ignore(
        ground_truth,
        predictions,
        mask,
        iou_threshold=0.5,
        ignore_overlap_threshold=0.8,
    )

    assert result.ignored_predictions == (0,)
    assert [(item.ground_truth_index, item.prediction_index) for item in result.matches] == [
        (0, 1)
    ]
    assert result.unmatched_predictions == (2,)
