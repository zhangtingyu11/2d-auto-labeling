import importlib.util
from pathlib import Path

from car5_autolabel.integrations.label_studio_kfold_sync import (
    candidate_to_label_studio_result,
    generate_audit_candidates,
    replace_audit_results,
    update_label_config,
)


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "sync_kfold_audit_to_label_studio.py"
SPEC = importlib.util.spec_from_file_location("sync_kfold_audit_to_label_studio", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _ground_truth() -> dict:
    return {
        "images": [
            {"id": index, "width": 200, "height": 100, "label_studio_task_id": 10 + index}
            for index in range(1, 6)
        ],
        "categories": [{"id": 0, "name": "Car"}, {"id": 1, "name": "Truck"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 80, 80]},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [10, 10, 80, 80]},
            {"id": 3, "image_id": 3, "category_id": 0, "bbox": [10, 10, 80, 80]},
            {"id": 4, "image_id": 4, "category_id": 0, "bbox": [10, 10, 80, 80]},
        ],
    }


def test_candidate_generation_is_one_to_one_and_nms_deduplicated() -> None:
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 80, 80], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [11, 10, 80, 80], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [10, 10, 80, 80], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [11, 10, 80, 80], "score": 0.8},
        {"image_id": 4, "category_id": 0, "bbox": [55, 10, 80, 80], "score": 0.9},
        {"image_id": 5, "category_id": 0, "bbox": [10, 10, 80, 80], "score": 0.9},
    ]

    candidates, report = generate_audit_candidates(_ground_truth(), predictions)

    assert report.suppressed_same_class == 2
    assert [candidate["issue"] for candidate in candidates] == [
        "class_mismatch",
        "unmatched_gt_possible_missed_or_bad_label",
        "localization_error",
        "unmatched_prediction_possible_missing_gt",
    ]


def test_possible_missing_uses_higher_score_without_changing_matching() -> None:
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 80, 80], "score": 0.3},
        {"image_id": 5, "category_id": 0, "bbox": [10, 10, 80, 80], "score": 0.3},
    ]

    candidates, _ = generate_audit_candidates(
        _ground_truth(),
        predictions,
        operating_score_threshold=0.2,
        possible_missing_score_threshold=0.4,
    )

    assert not any(candidate["image_id"] == 1 for candidate in candidates)
    assert not any(
        candidate["issue"] == "unmatched_prediction_possible_missing_gt"
        for candidate in candidates
    )


def test_result_replacement_removes_historical_duplicates_and_is_idempotent() -> None:
    candidate = {
        "image_id": 1,
        "issue": "unmatched_prediction_possible_missing_gt",
        "bbox": [10, 10, 80, 80],
        "gt_class": "",
        "predicted_class": "Car",
        "score": 0.9,
        "iou": None,
    }
    hint = candidate_to_label_studio_result(
        candidate, image_width=200, image_height=100, model_version="model-v1"
    )
    human = {"id": "human-1", "from_name": "label", "type": "rectanglelabels"}
    old = [
        human,
        {"id": "duplicate-a", "from_name": "audit_hint"},
        {"id": "duplicate-b", "from_name": "prediction_reference"},
        {"id": "old-choice", "from_name": "review_issue"},
    ]

    once = replace_audit_results(old, [hint, hint])
    twice = replace_audit_results(once, [hint, hint])

    assert once == twice
    assert [result["id"] for result in once] == [hint["id"], "human-1"]
    assert once[0]["readonly"] is True
    assert once[0]["interactive_mode"] == "none"
    assert once[0]["value"]["rectanglelabels"] == ["可能漏标（预测=Car）"]


def test_class_mismatch_box_label_shows_predicted_class() -> None:
    candidate = {
        "image_id": 1,
        "issue": "class_mismatch",
        "bbox": [10, 10, 80, 80],
        "gt_class": "Truck",
        "predicted_class": "Car",
        "score": 0.9,
        "iou": 0.8,
    }

    result = candidate_to_label_studio_result(
        candidate, image_width=200, image_height=100, model_version="model-v1"
    )

    assert result["value"]["rectanglelabels"] == ["可能类别错误（预测=Car）"]
    assert result["meta"]["gt_class"] == "Truck"


def test_label_config_update_is_idempotent() -> None:
    original = """<View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image"><Label value="Car"/></RectangleLabels>
      <RectangleLabels name="prediction_reference" toName="image"/>
    </View>"""

    once = update_label_config(original)
    twice = update_label_config(once)

    assert once == twice
    assert once.count('name="audit_hint"') == 1
    assert "prediction_reference" not in once
    assert 'name="label"' in once
    assert "可能漏标（预测=Excavator）" in once
    assert "可能类别错误（预测=Car）" in once


def test_build_updates_never_creates_tasks_or_annotations() -> None:
    state = {
        "tasks": {task_id: {"image": f"{task_id}.jpg"} for task_id in range(11, 17)},
        "annotations": {
            task_id: [(100 + task_id, [{"id": f"human-{task_id}", "from_name": "label"}])]
            for task_id in range(11, 16)
        },
        "drafts": {},
    }
    predictions = [
        {"image_id": 5, "category_id": 0, "bbox": [10, 10, 80, 80], "score": 0.9}
    ]

    updates, report = SCRIPT.build_updates(
        state=state,
        ground_truth=_ground_truth(),
        predictions=predictions,
        model_version="model-v1",
        operating_score_threshold=0.2,
        possible_missing_score_threshold=0.4,
        minimum_long_side_px=70,
        nms_iou_threshold=0.7,
        cross_class_nms_iou_threshold=0.95,
    )

    assert report["created_tasks"] == 0
    assert report["created_annotations"] == 0
    assert report["project_tasks"] == 6
    assert report["audited_tasks"] == 5
    assert report["candidate_tasks"] == 5
    assert report["tasks_without_annotations"] == 1
    assert all(resource in {"tasks", "annotations", "drafts"} for resource, _, _ in updates)
