"""Shared policies for leakage-safe K-fold annotation auditing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

KFOLD_TRAINING_CLASSES = (
    "Car",
    "Truck",
    "BoxTruck",
    "Bulldozer",
    "Excavator",
    "WaterTruck",
    "Sign",
    "Pedestrian",
)


def long_side(box_xywh: list[float] | tuple[float, ...]) -> float:
    """Return the original floating-point longest side of a COCO ``xywh`` box."""

    if len(box_xywh) != 4:
        raise ValueError("COCO bbox must contain exactly four values")
    width = float(box_xywh[2])
    height = float(box_xywh[3])
    if width <= 0 or height <= 0:
        raise ValueError("COCO bbox width and height must be positive")
    return max(width, height)


@dataclass(frozen=True)
class EmptyFrameFilterReport:
    kept_images: int
    kept_boxes: int
    excluded_empty_images: int


@dataclass(frozen=True)
class PredictionNMSReport:
    input_predictions: int
    kept_predictions: int
    suppressed_same_class: int
    suppressed_cross_class: int


def clip_ground_truth_boxes(coco: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Clip COCO GT boxes to image bounds and return the number changed."""

    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations = []
    clipped = 0
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"annotation references unknown image ID: {image_id}")
        image = images[image_id]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if width <= 0 or height <= 0:
            raise ValueError(f"annotation has invalid bbox: {annotation.get('id')}")
        within_bounds = (
            x >= 0
            and y >= 0
            and x + width <= float(image["width"])
            and y + height <= float(image["height"])
        )
        if within_bounds:
            annotations.append(annotation)
            continue
        x1 = max(0.0, min(x, float(image["width"])))
        y1 = max(0.0, min(y, float(image["height"])))
        x2 = max(0.0, min(x + width, float(image["width"])))
        y2 = max(0.0, min(y + height, float(image["height"])))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"annotation is outside image bounds: {annotation.get('id')}")
        bbox = [x1, y1, x2 - x1, y2 - y1]
        clipped += 1
        annotations.append({**annotation, "bbox": bbox, "area": bbox[2] * bbox[3]})
    return {**coco, "annotations": annotations}, clipped


def normalize_kfold_training_taxonomy(coco: dict[str, Any]) -> dict[str, Any]:
    """Keep all eight labeled classes and remap their IDs deterministically to 0..7."""

    name_by_source_id = {
        int(category["id"]): str(category["name"]) for category in coco.get("categories", [])
    }
    expected_names = set(KFOLD_TRAINING_CLASSES)
    source_names = set(name_by_source_id.values())
    if source_names != expected_names:
        raise ValueError(
            "K-fold source categories differ from the required eight classes; "
            f"missing={sorted(expected_names - source_names)}, "
            f"unexpected={sorted(source_names - expected_names)}"
        )
    target_id_by_name = {name: index for index, name in enumerate(KFOLD_TRAINING_CLASSES)}
    annotations = []
    for annotation in coco.get("annotations", []):
        source_category_id = int(annotation["category_id"])
        if source_category_id not in name_by_source_id:
            raise ValueError(
                f"annotation references unknown category ID: {source_category_id}"
            )
        annotations.append(
            {
                **annotation,
                "category_id": target_id_by_name[name_by_source_id[source_category_id]],
            }
        )
    return {
        **coco,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(KFOLD_TRAINING_CLASSES)
        ],
    }


def filter_empty_training_frames(
    coco: dict[str, Any],
) -> tuple[dict[str, Any], EmptyFrameFilterReport]:
    """Remove only images with no human boxes; retain every box regardless of size."""

    annotated_image_ids = {
        int(annotation["image_id"]) for annotation in coco.get("annotations", [])
    }
    kept_images = [
        image for image in coco.get("images", []) if int(image["id"]) in annotated_image_ids
    ]
    filtered = {**coco, "images": kept_images}
    return filtered, EmptyFrameFilterReport(
        kept_images=len(kept_images),
        kept_boxes=len(coco.get("annotations", [])),
        excluded_empty_images=len(coco.get("images", [])) - len(kept_images),
    )


def filter_validation_predictions(
    predictions: list[dict[str, Any]],
    *,
    minimum_long_side_px: int = 70,
) -> list[dict[str, Any]]:
    """Ignore validation predictions below the formal size threshold."""

    if minimum_long_side_px <= 0:
        raise ValueError("minimum_long_side_px must be positive")
    return [
        prediction
        for prediction in predictions
        if long_side(prediction["bbox"]) >= minimum_long_side_px
    ]


def box_iou_xywh(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    """Return intersection-over-union for two COCO ``xywh`` boxes."""

    if len(first) != 4 or len(second) != 4:
        raise ValueError("COCO bbox must contain exactly four values")
    ax1, ay1, aw, ah = (float(value) for value in first)
    bx1, by1, bw, bh = (float(value) for value in second)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        raise ValueError("COCO bbox width and height must be positive")
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    return intersection / (aw * ah + bw * bh - intersection)


def deduplicate_operating_predictions(
    predictions: list[dict[str, Any]],
    *,
    same_class_iou_threshold: float = 0.7,
    cross_class_iou_threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], PredictionNMSReport]:
    """Deduplicate review-point predictions without changing raw COCO AP input.

    Same-class boxes use conventional greedy NMS. Nearly identical boxes from
    different classes retain the highest-score geometry and record the other
    classes in ``audit_nms.category_conflicts`` so audit consumers can expose
    the ambiguity instead of drawing duplicate boxes or silently hiding it.
    """

    for name, threshold in (
        ("same_class_iou_threshold", same_class_iou_threshold),
        ("cross_class_iou_threshold", cross_class_iou_threshold),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if cross_class_iou_threshold < same_class_iou_threshold:
        raise ValueError("cross-class NMS threshold must be at least the same-class threshold")

    def order_key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, prediction = item
        bbox = tuple(float(value) for value in prediction["bbox"])
        return (
            int(prediction["image_id"]),
            -float(prediction.get("score", 0.0)),
            int(prediction["category_id"]),
            bbox,
            index,
        )

    by_image_and_class: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        by_image_and_class[key].append((index, prediction))

    class_survivors: list[tuple[int, dict[str, Any]]] = []
    same_class_suppressed = 0
    for group in by_image_and_class.values():
        kept: list[tuple[int, dict[str, Any]]] = []
        for indexed_prediction in sorted(group, key=order_key):
            prediction = indexed_prediction[1]
            if any(
                box_iou_xywh(prediction["bbox"], existing[1]["bbox"])
                >= same_class_iou_threshold
                for existing in kept
            ):
                same_class_suppressed += 1
                continue
            kept.append(indexed_prediction)
        class_survivors.extend(kept)

    by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for indexed_prediction in class_survivors:
        by_image[int(indexed_prediction[1]["image_id"])].append(indexed_prediction)

    deduplicated: list[tuple[int, dict[str, Any]]] = []
    cross_class_suppressed = 0
    for group in by_image.values():
        kept: list[tuple[int, dict[str, Any]]] = []
        for index, source_prediction in sorted(group, key=order_key):
            conflict_winner: tuple[int, dict[str, Any]] | None = None
            for existing in kept:
                if int(existing[1]["category_id"]) == int(source_prediction["category_id"]):
                    continue
                if (
                    box_iou_xywh(source_prediction["bbox"], existing[1]["bbox"])
                    >= cross_class_iou_threshold
                ):
                    conflict_winner = existing
                    break
            if conflict_winner is None:
                kept.append((index, dict(source_prediction)))
                continue
            winner = conflict_winner[1]
            metadata = winner.setdefault("audit_nms", {})
            metadata.setdefault("category_conflicts", []).append(
                {
                    "category_id": int(source_prediction["category_id"]),
                    "score": float(source_prediction.get("score", 0.0)),
                    "bbox": [float(value) for value in source_prediction["bbox"]],
                }
            )
            cross_class_suppressed += 1
        deduplicated.extend(kept)

    result = [prediction for _, prediction in sorted(deduplicated, key=order_key)]
    return result, PredictionNMSReport(
        input_predictions=len(predictions),
        kept_predictions=len(result),
        suppressed_same_class=same_class_suppressed,
        suppressed_cross_class=cross_class_suppressed,
    )


def _minimum_cost_assignment(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Solve a rectangular linear assignment without an optional SciPy dependency."""

    if not costs or not costs[0]:
        return []
    rows, columns = len(costs), len(costs[0])
    transposed = rows > columns
    matrix = (
        [[costs[row][column] for row in range(rows)] for column in range(columns)]
        if transposed
        else costs
    )
    row_count, column_count = len(matrix), len(matrix[0])
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break
    pairs = [(matched_row[column] - 1, column - 1) for column in range(1, column_count + 1)]
    pairs = [(row, column) for row, column in pairs if row >= 0]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


def maximum_iou_match_indices(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    require_same_category: bool = True,
) -> list[tuple[int, int, float]]:
    """Return the one-to-one assignment that maximizes total compatible IoU."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    ious: list[list[float]] = []
    for gt in ground_truth:
        row = []
        for prediction in predictions:
            compatible = not require_same_category or int(gt["category_id"]) == int(
                prediction["category_id"]
            )
            row.append(box_iou_xywh(gt["bbox"], prediction["bbox"]) if compatible else 0.0)
        ious.append(row)
    pairs = _minimum_cost_assignment([[1.0 - iou for iou in row] for row in ious])
    return [
        (gt_index, prediction_index, ious[gt_index][prediction_index])
        for gt_index, prediction_index in pairs
        if ious[gt_index][prediction_index] >= iou_threshold
    ]


@dataclass(frozen=True)
class GPUStatus:
    index: int
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


def parse_nvidia_smi_csv(output: str) -> list[GPUStatus]:
    """Parse the stable no-unit CSV requested by the K-fold runner."""

    statuses: list[GPUStatus] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        statuses.append(GPUStatus(*(int(value) for value in values)))
    return statuses


def select_idle_gpu(
    statuses: list[GPUStatus],
    *,
    minimum_free_memory_mib: int,
    maximum_utilization_percent: int,
    allowed_indices: set[int] | None = None,
) -> GPUStatus | None:
    """Choose the freest GPU that satisfies both idle conditions."""

    candidates = [
        status
        for status in statuses
        if (allowed_indices is None or status.index in allowed_indices)
        and status.memory_free_mib >= minimum_free_memory_mib
        and status.utilization_percent <= maximum_utilization_percent
    ]
    return max(candidates, key=lambda status: status.memory_free_mib, default=None)
