from car5_autolabel.detectors.rtmdet import ACTIVE_CLASS_NAMES, instances_to_boxes


def test_instances_to_boxes_filters_clips_and_orders() -> None:
    boxes = instances_to_boxes(
        [[10, 20, 110, 220], [-5, -7, 40, 50], [4, 4, 3, 8], [0, 0, 5, 5]],
        [0.8, 0.95, 0.99, 0.09],
        [1, 3, 0, 5],
        width=100,
        height=200,
        score_threshold=0.1,
    )

    assert [box.label for box in boxes] == ["Excavator", "Truck"]
    assert boxes[0].model_dump() == {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 40.0,
        "y2": 50.0,
        "label": "Excavator",
        "score": 0.95,
        "raw_label": None,
        "postprocess_reason": None,
        "track_id": None,
    }
    assert boxes[1].x2 == 100
    assert boxes[1].y2 == 200


def test_instances_to_boxes_rejects_unknown_class() -> None:
    try:
        instances_to_boxes(
            [[0, 0, 10, 10]],
            [0.9],
            [len(ACTIVE_CLASS_NAMES)],
            width=20,
            height=20,
            score_threshold=0.1,
        )
    except ValueError as exc:
        assert "unknown class id" in str(exc)
    else:
        raise AssertionError("unknown model class must fail closed")
