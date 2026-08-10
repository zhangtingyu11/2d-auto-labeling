from car5_autolabel.kfold_audit import (
    _minimum_cost_assignment,
    box_iou_xywh,
    clip_ground_truth_boxes,
    deduplicate_operating_predictions,
    enforce_v1_active_taxonomy,
    filter_training_frames,
    filter_validation_predictions,
    maximum_iou_match_indices,
    parse_nvidia_smi_csv,
    select_idle_gpu,
)


def test_nms_suppresses_same_class_duplicate_at_operating_point() -> None:
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100], "score": 0.8},
        {"image_id": 1, "category_id": 0, "bbox": [5, 5, 100, 100], "score": 0.7},
        {"image_id": 2, "category_id": 0, "bbox": [5, 5, 100, 100], "score": 0.6},
    ]

    kept, report = deduplicate_operating_predictions(predictions)

    assert [(row["image_id"], row["score"]) for row in kept] == [(1, 0.8), (2, 0.6)]
    assert report.suppressed_same_class == 1
    assert report.suppressed_cross_class == 0


def test_nms_consolidates_identical_cross_class_boxes_and_records_conflict() -> None:
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "score": 0.7},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 100, 100], "score": 0.9},
    ]

    kept, report = deduplicate_operating_predictions(predictions)

    assert len(kept) == 1
    assert kept[0]["category_id"] == 2
    assert kept[0]["audit_nms"]["category_conflicts"][0]["category_id"] == 1
    assert report.suppressed_cross_class == 1


def test_nms_keeps_overlapping_different_objects_below_cross_class_threshold() -> None:
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [5, 5, 100, 100], "score": 0.8},
    ]

    kept, report = deduplicate_operating_predictions(predictions)

    assert len(kept) == 2
    assert report.suppressed_cross_class == 0


def _fixture() -> dict:
    return {
        "images": [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in range(1, 5)],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 100, 80]},
            {"id": 2, "image_id": 1, "bbox": [0, 0, 69.4, 20]},
            {"id": 3, "image_id": 2, "bbox": [0, 0, 70, 10]},
            {"id": 4, "image_id": 3, "bbox": [0, 0, 80, 20]},
        ],
        "categories": [{"id": 1, "name": "Truck"}],
    }


def test_filter_training_frames_removes_whole_frame_for_one_small_box_and_empty() -> None:
    filtered, report = filter_training_frames(_fixture(), minimum_long_side_px=70)

    assert [image["id"] for image in filtered["images"]] == [2, 3]
    assert [annotation["id"] for annotation in filtered["annotations"]] == [3, 4]
    assert report.kept_images == 2
    assert report.kept_boxes == 2
    assert report.excluded_below_size_images == 1
    assert report.excluded_empty_images == 1
    assert report.excluded_boxes_with_removed_images == 2


def test_filter_validation_predictions_uses_strict_float_70px_boundary() -> None:
    predictions = [
        {"bbox": [0, 0, 69.4, 10], "score": 0.9},
        {"bbox": [0, 0, 69.6, 10], "score": 0.8},
        {"bbox": [0, 0, 70.0, 10], "score": 0.7},
    ]

    kept = filter_validation_predictions(predictions, minimum_long_side_px=70)

    assert [row["score"] for row in kept] == [0.7]


def test_select_idle_gpu_requires_both_memory_and_utilization() -> None:
    statuses = parse_nvidia_smi_csv("0, 24564, 500, 2\n1, 24564, 100, 90\n")

    selected = select_idle_gpu(
        statuses,
        minimum_free_memory_mib=22000,
        maximum_utilization_percent=5,
    )

    assert selected is not None
    assert selected.index == 0


def test_iou_and_matching_require_same_class_and_one_to_one_pairs() -> None:
    gt = [
        {"bbox": [0, 0, 100, 100], "category_id": 1},
        {"bbox": [200, 0, 100, 100], "category_id": 2},
    ]
    predictions = [
        {"bbox": [0, 0, 100, 100], "category_id": 1},
        {"bbox": [0, 0, 100, 100], "category_id": 1},
        {"bbox": [200, 0, 100, 100], "category_id": 1},
    ]

    assert box_iou_xywh(gt[0]["bbox"], predictions[0]["bbox"]) == 1.0
    matches = maximum_iou_match_indices(gt, predictions, iou_threshold=0.5)

    assert len(matches) == 1
    assert matches[0][0] == 0
    assert matches[0][1] in {0, 1}


def test_assignment_maximizes_total_score_instead_of_greedy_first_pair() -> None:
    costs = [[0.10, 0.20], [0.11, 0.90]]

    pairs = _minimum_cost_assignment(costs)

    assert set(pairs) == {(0, 1), (1, 0)}


def test_v1_taxonomy_excludes_whole_inactive_frame_and_remaps_ids() -> None:
    coco = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100]},
            {"id": 2, "image_id": 2, "category_id": 3, "bbox": [0, 0, 100, 100]},
        ],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Truck"},
            {"id": 3, "name": "BoxTruck"},
            {"id": 4, "name": "Bulldozer"},
            {"id": 5, "name": "Excavator"},
            {"id": 6, "name": "WaterTruck"},
            {"id": 7, "name": "Sign"},
        ],
    }

    filtered, report = enforce_v1_active_taxonomy(coco)

    assert [image["id"] for image in filtered["images"]] == [1]
    assert filtered["annotations"][0]["category_id"] == 0
    assert [category["name"] for category in filtered["categories"]] == [
        "Car",
        "Truck",
        "Bulldozer",
        "Excavator",
        "WaterTruck",
        "Sign",
    ]
    assert report.excluded_inactive_class_images == 1


def test_clip_ground_truth_boxes_updates_bbox_and_area() -> None:
    coco = {
        "images": [{"id": 1, "width": 100, "height": 80}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [-1, 10, 102, 80]}],
    }

    clipped, count = clip_ground_truth_boxes(coco)

    assert count == 1
    assert clipped["annotations"][0]["bbox"] == [0.0, 10.0, 100.0, 70.0]
    assert clipped["annotations"][0]["area"] == 7000.0
