"""Idempotent K-fold audit updates for an existing Label Studio project."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from car5_autolabel.kfold_audit import (
    KFOLD_TRAINING_CLASSES,
    box_iou_xywh,
    deduplicate_operating_predictions,
    long_side,
    maximum_iou_match_indices,
)


ISSUE_LABELS = {
    "unmatched_prediction_possible_missing_gt": "可能漏标",
    "unmatched_gt_possible_missed_or_bad_label": "可能多标",
    "class_mismatch": "可能类别错误",
    "localization_error": "可能框不够准",
}
ISSUE_COLORS = {
    "可能漏标": "#E53935",
    "可能多标": "#FDD835",
    "可能类别错误": "#8E24AA",
    "可能框不够准": "#1E88E5",
}
AUDIT_RESULT_CONTROLS = {"audit_hint", "prediction_reference", "review_issue"}
AUDIT_DATA_KEYS = {
    "candidate_count",
    "candidate_summary",
    "review_summary",
    "review_tag",
    "review_tag_label",
}


@dataclass(frozen=True)
class CandidateGenerationReport:
    input_predictions: int
    score_and_size_filtered_predictions: int
    kept_predictions: int
    suppressed_same_class: int
    suppressed_cross_class: int
    candidate_count: int


def _category_names(ground_truth: dict[str, Any]) -> dict[int, str]:
    names = {
        int(category["id"]): str(category["name"])
        for category in ground_truth.get("categories", [])
    }
    if not names:
        raise ValueError("ground truth has no categories")
    return names


def _candidate(
    *,
    image_id: int,
    issue: str,
    bbox: Iterable[float],
    gt_class: str = "",
    predicted_class: str = "",
    score: float | None = None,
    iou: float | None = None,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "issue": issue,
        "bbox": [float(value) for value in bbox],
        "gt_class": gt_class,
        "predicted_class": predicted_class,
        "score": score,
        "iou": iou,
    }


def generate_audit_candidates(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    operating_score_threshold: float = 0.2,
    possible_missing_score_threshold: float = 0.4,
    match_iou_threshold: float = 0.5,
    localization_iou_threshold: float = 0.1,
    minimum_long_side_px: int = 70,
    nms_iou_threshold: float = 0.7,
    cross_class_nms_iou_threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], CandidateGenerationReport]:
    """Build one-to-one audit candidates after score, size, and NMS filtering."""

    if not 0 <= operating_score_threshold <= 1:
        raise ValueError("operating_score_threshold must be between zero and one")
    if not 0 <= possible_missing_score_threshold <= 1:
        raise ValueError(
            "possible_missing_score_threshold must be between zero and one"
        )
    if minimum_long_side_px <= 0:
        raise ValueError("minimum_long_side_px must be positive")
    images = {int(image["id"]): image for image in ground_truth.get("images", [])}
    categories = _category_names(ground_truth)
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"ground-truth annotation references unknown image {image_id}")
        gt_by_image[image_id].append(annotation)

    filtered_predictions = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if image_id not in images:
            raise ValueError(f"prediction references unknown image {image_id}")
        if float(prediction.get("score", 0.0)) < operating_score_threshold:
            continue
        if long_side(prediction["bbox"]) < minimum_long_side_px:
            continue
        filtered_predictions.append(prediction)
    operating_predictions, nms_report = deduplicate_operating_predictions(
        filtered_predictions,
        same_class_iou_threshold=nms_iou_threshold,
        cross_class_iou_threshold=cross_class_nms_iou_threshold,
    )
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in operating_predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    candidates: list[dict[str, Any]] = []
    for image_id in sorted(images):
        gt = gt_by_image[image_id]
        predicted = predictions_by_image[image_id]
        matches = maximum_iou_match_indices(
            gt, predicted, iou_threshold=match_iou_threshold
        )
        matched_gt = {match[0] for match in matches}
        matched_predictions = {match[1] for match in matches}
        remaining_gt = [(index, row) for index, row in enumerate(gt) if index not in matched_gt]
        remaining_predictions = [
            (index, row)
            for index, row in enumerate(predicted)
            if index not in matched_predictions
        ]

        mismatch_pairs: list[tuple[float, int, int]] = []
        for gt_index, gt_row in remaining_gt:
            for prediction_index, prediction in remaining_predictions:
                if int(gt_row["category_id"]) == int(prediction["category_id"]):
                    continue
                iou = box_iou_xywh(gt_row["bbox"], prediction["bbox"])
                if iou >= match_iou_threshold:
                    mismatch_pairs.append((iou, gt_index, prediction_index))
        used_mismatch_gt: set[int] = set()
        used_mismatch_predictions: set[int] = set()
        for iou, gt_index, prediction_index in sorted(mismatch_pairs, reverse=True):
            if gt_index in used_mismatch_gt or prediction_index in used_mismatch_predictions:
                continue
            gt_row = gt[gt_index]
            prediction = predicted[prediction_index]
            used_mismatch_gt.add(gt_index)
            used_mismatch_predictions.add(prediction_index)
            candidates.append(
                _candidate(
                    image_id=image_id,
                    issue="class_mismatch",
                    bbox=prediction["bbox"],
                    gt_class=categories[int(gt_row["category_id"])],
                    predicted_class=categories[int(prediction["category_id"])],
                    score=float(prediction.get("score", 0.0)),
                    iou=iou,
                )
            )

        localization_gt = [
            (index, row) for index, row in remaining_gt if index not in used_mismatch_gt
        ]
        localization_predictions = [
            (index, row)
            for index, row in remaining_predictions
            if index not in used_mismatch_predictions
        ]
        localization_matches = maximum_iou_match_indices(
            [row for _, row in localization_gt],
            [row for _, row in localization_predictions],
            iou_threshold=localization_iou_threshold,
        )
        used_localization_gt: set[int] = set()
        used_localization_predictions: set[int] = set()
        for local_gt_index, local_prediction_index, iou in localization_matches:
            gt_index, gt_row = localization_gt[local_gt_index]
            prediction_index, prediction = localization_predictions[local_prediction_index]
            used_localization_gt.add(gt_index)
            used_localization_predictions.add(prediction_index)
            candidates.append(
                _candidate(
                    image_id=image_id,
                    issue="localization_error",
                    bbox=prediction["bbox"],
                    gt_class=categories[int(gt_row["category_id"])],
                    predicted_class=categories[int(prediction["category_id"])],
                    score=float(prediction.get("score", 0.0)),
                    iou=iou,
                )
            )

        for gt_index, gt_row in remaining_gt:
            if gt_index in used_mismatch_gt or gt_index in used_localization_gt:
                continue
            if long_side(gt_row["bbox"]) < minimum_long_side_px:
                continue
            candidates.append(
                _candidate(
                    image_id=image_id,
                    issue="unmatched_gt_possible_missed_or_bad_label",
                    bbox=gt_row["bbox"],
                    gt_class=categories[int(gt_row["category_id"])],
                )
            )
        for prediction_index, prediction in remaining_predictions:
            if (
                prediction_index in used_mismatch_predictions
                or prediction_index in used_localization_predictions
            ):
                continue
            if float(prediction.get("score", 0.0)) < possible_missing_score_threshold:
                continue
            candidates.append(
                _candidate(
                    image_id=image_id,
                    issue="unmatched_prediction_possible_missing_gt",
                    bbox=prediction["bbox"],
                    predicted_class=categories[int(prediction["category_id"])],
                    score=float(prediction.get("score", 0.0)),
                )
            )

    candidates.sort(
        key=lambda row: (
            int(row["image_id"]),
            row["issue"],
            tuple(round(value, 6) for value in row["bbox"]),
            row["gt_class"],
            row["predicted_class"],
        )
    )
    return candidates, CandidateGenerationReport(
        input_predictions=len(predictions),
        score_and_size_filtered_predictions=len(filtered_predictions),
        kept_predictions=nms_report.kept_predictions,
        suppressed_same_class=nms_report.suppressed_same_class,
        suppressed_cross_class=nms_report.suppressed_cross_class,
        candidate_count=len(candidates),
    )


def _stable_hint_id(candidate: dict[str, Any], model_version: str) -> str:
    identity = {
        "model_version": model_version,
        "image_id": int(candidate["image_id"]),
        "issue": candidate["issue"],
        "bbox": [round(float(value), 6) for value in candidate["bbox"]],
        "gt_class": candidate.get("gt_class", ""),
        "predicted_class": candidate.get("predicted_class", ""),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"kfold-hint-{digest}"


def candidate_display_label(candidate: dict[str, Any]) -> str:
    """Return the visible issue label, including the model class for mismatches."""

    issue = str(candidate["issue"])
    if issue not in ISSUE_LABELS:
        raise ValueError(f"unsupported audit issue: {issue}")
    label = ISSUE_LABELS[issue]
    if issue not in {"class_mismatch", "unmatched_prediction_possible_missing_gt"}:
        return label
    predicted_class = str(candidate.get("predicted_class", "")).strip()
    if not predicted_class:
        raise ValueError(f"{issue} candidate has no predicted_class")
    return f"{label}（预测={predicted_class}）"


def candidate_to_label_studio_result(
    candidate: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    model_version: str,
    hint_control: str = "audit_hint",
    image_control: str = "image",
) -> dict[str, Any]:
    """Convert one candidate into a deterministic locked Label Studio rectangle."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    issue = str(candidate["issue"])
    display_label = candidate_display_label(candidate)
    x, y, width, height = (float(value) for value in candidate["bbox"])
    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("candidate box is empty after clipping")
    return {
        "id": _stable_hint_id(candidate, model_version),
        "type": "rectanglelabels",
        "from_name": hint_control,
        "to_name": image_control,
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "value": {
            "x": x1 / image_width * 100.0,
            "y": y1 / image_height * 100.0,
            "width": (x2 - x1) / image_width * 100.0,
            "height": (y2 - y1) / image_height * 100.0,
            "rotation": 0,
            "rectanglelabels": [display_label],
        },
        "score": candidate.get("score"),
        "readonly": True,
        "interactive_mode": "none",
        "hidden": False,
        "origin": "prediction",
        "meta": {
            "locked_reference": True,
            "source": "kfold_oof_audit",
            "review_status": "unreviewed_prediction",
            "model_version": model_version,
            "issue": issue,
            "gt_class": candidate.get("gt_class", ""),
            "predicted_class": candidate.get("predicted_class", ""),
            "iou": candidate.get("iou"),
        },
    }


def replace_audit_results(
    existing_results: list[dict[str, Any]], new_hints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace every historical audit result and keep each new hint exactly once."""

    preserved = [
        result
        for result in existing_results
        if result.get("from_name") not in AUDIT_RESULT_CONTROLS
    ]
    unique_hints: dict[str, dict[str, Any]] = {}
    for hint in new_hints:
        if hint.get("from_name") not in AUDIT_RESULT_CONTROLS:
            raise ValueError("new hint does not use an audit result control")
        unique_hints[str(hint["id"])] = hint
    # Hints are underneath editable human rectangles, so overlapping GT remains clickable.
    return [unique_hints[key] for key in sorted(unique_hints)] + preserved


def update_task_audit_data(
    data: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    audited: bool,
    model_version: str,
) -> dict[str, Any]:
    """Replace task-level audit metadata without touching task identity or human state."""

    updated = {
        key: value
        for key, value in data.items()
        if key not in AUDIT_DATA_KEYS and not key.startswith("kfold_")
    }
    if not audited:
        updated.update(
            {
                "candidate_count": 0,
                "candidate_summary": (
                    "未参与本次五折 OOF 审计（训练时排除的空图）"
                ),
                "review_tag": [],
                "kfold_audit_state": "excluded",
                "kfold_candidate_count": 0,
                "kfold_model_version": model_version,
            }
        )
        return updated
    labels = sorted({ISSUE_LABELS[str(candidate["issue"])] for candidate in candidates})
    summaries = []
    for candidate in candidates:
        label = ISSUE_LABELS[str(candidate["issue"])]
        details = []
        if candidate.get("gt_class"):
            details.append(f"GT={candidate['gt_class']}")
        if candidate.get("predicted_class"):
            details.append(f"PRED={candidate['predicted_class']}")
        if candidate.get("score") is not None:
            details.append(f"score={float(candidate['score']):.3f}")
        summaries.append(" ".join([label, *details]))
    updated.update(
        {
            "candidate_count": len(candidates),
            "candidate_summary": (
                "；".join(summaries) if summaries else "本帧未发现疑似问题"
            ),
            "review_tag": labels,
            "kfold_audit_state": "candidate" if candidates else "no_candidate",
            "kfold_candidate_count": len(candidates),
            "kfold_model_version": model_version,
        }
    )
    return updated


def update_label_config(
    label_config: str,
    *,
    hint_control: str = "audit_hint",
    image_control: str = "image",
    remove_legacy_controls: bool = True,
) -> str:
    """Install one canonical audit UI, removing older audit controls first."""

    root = ET.fromstring(label_config)
    removed_names = {hint_control, "candidate_info"}
    if remove_legacy_controls:
        removed_names.update({"prediction_reference", "review_issue"})
    for parent in root.iter():
        for child in list(parent):
            name = child.attrib.get("name")
            value = child.attrib.get("value", "")
            if name in removed_names or value.startswith("五折 OOF 提示："):
                parent.remove(child)
    header = ET.Element(
        "Header",
        {
            "value": (
                "五折 OOF 提示：红=可能漏标，黄=可能多标，"
                "紫=可能类别错误，蓝=可能框不够准；提示框只读"
            )
        },
    )
    text = ET.Element("Text", {"name": "candidate_info", "value": "$candidate_summary"})
    control = ET.Element(
        "RectangleLabels",
        {
            "name": hint_control,
            "toName": image_control,
            "strokeWidth": "5",
            "opacity": "0.15",
            "snap": "pixel",
        },
    )
    for label, color in ISSUE_COLORS.items():
        if label in {"可能漏标", "可能类别错误"}:
            for predicted_class in KFOLD_TRAINING_CLASSES:
                ET.SubElement(
                    control,
                    "Label",
                    {
                        "value": f"{label}（预测={predicted_class}）",
                        "background": color,
                    },
                )
        else:
            ET.SubElement(control, "Label", {"value": label, "background": color})
    root.insert(0, control)
    root.insert(0, text)
    root.insert(0, header)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
