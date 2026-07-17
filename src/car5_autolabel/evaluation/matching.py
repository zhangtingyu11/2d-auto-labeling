from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from car5_autolabel.policies.camera import intersection_over_box_area


@dataclass(frozen=True)
class BoxMatch:
    ground_truth_index: int
    prediction_index: int
    iou: float


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[BoxMatch, ...]
    unmatched_ground_truth: tuple[int, ...]
    unmatched_predictions: tuple[int, ...]
    ignored_ground_truth: tuple[int, ...] = ()
    ignored_predictions: tuple[int, ...] = ()


def _as_boxes(value: np.ndarray) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("every box must satisfy x2>x1 and y2>y1")
    return boxes


def pairwise_iou(ground_truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    ground_truth = _as_boxes(ground_truth)
    predictions = _as_boxes(predictions)
    if len(ground_truth) == 0 or len(predictions) == 0:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)

    top_left = np.maximum(ground_truth[:, None, :2], predictions[None, :, :2])
    bottom_right = np.minimum(ground_truth[:, None, 2:], predictions[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    ground_truth_area = np.prod(ground_truth[:, 2:] - ground_truth[:, :2], axis=1)
    prediction_area = np.prod(predictions[:, 2:] - predictions[:, :2], axis=1)
    union = ground_truth_area[:, None] + prediction_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def maximum_iou_match(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between zero and one")
    ground_truth = _as_boxes(ground_truth)
    predictions = _as_boxes(predictions)
    ious = pairwise_iou(ground_truth, predictions)
    if ious.size == 0:
        return MatchResult(
            matches=(),
            unmatched_ground_truth=tuple(range(len(ground_truth))),
            unmatched_predictions=tuple(range(len(predictions))),
        )

    ground_truth_indices, prediction_indices = linear_sum_assignment(ious, maximize=True)
    matches = tuple(
        BoxMatch(int(gt_index), int(pred_index), float(ious[gt_index, pred_index]))
        for gt_index, pred_index in zip(
            ground_truth_indices, prediction_indices, strict=True
        )
        if ious[gt_index, pred_index] >= iou_threshold
    )
    matched_ground_truth = {match.ground_truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}
    return MatchResult(
        matches=matches,
        unmatched_ground_truth=tuple(
            index for index in range(len(ground_truth)) if index not in matched_ground_truth
        ),
        unmatched_predictions=tuple(
            index for index in range(len(predictions)) if index not in matched_predictions
        ),
    )


def maximum_iou_match_with_ignore(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    ignore_mask: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    ignore_overlap_threshold: float = 0.8,
) -> MatchResult:
    ground_truth = _as_boxes(ground_truth)
    predictions = _as_boxes(predictions)
    ignored_ground_truth = tuple(
        index
        for index, box in enumerate(ground_truth)
        if intersection_over_box_area(ignore_mask, box) >= ignore_overlap_threshold
    )
    ignored_predictions = tuple(
        index
        for index, box in enumerate(predictions)
        if intersection_over_box_area(ignore_mask, box) >= ignore_overlap_threshold
    )
    active_ground_truth = [
        index for index in range(len(ground_truth)) if index not in ignored_ground_truth
    ]
    active_predictions = [
        index for index in range(len(predictions)) if index not in ignored_predictions
    ]
    matched = maximum_iou_match(
        ground_truth[active_ground_truth],
        predictions[active_predictions],
        iou_threshold=iou_threshold,
    )
    return MatchResult(
        matches=tuple(
            BoxMatch(
                active_ground_truth[item.ground_truth_index],
                active_predictions[item.prediction_index],
                item.iou,
            )
            for item in matched.matches
        ),
        unmatched_ground_truth=tuple(
            active_ground_truth[index] for index in matched.unmatched_ground_truth
        ),
        unmatched_predictions=tuple(
            active_predictions[index] for index in matched.unmatched_predictions
        ),
        ignored_ground_truth=ignored_ground_truth,
        ignored_predictions=ignored_predictions,
    )
