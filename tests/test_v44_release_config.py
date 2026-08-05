import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = [
    "Car",
    "Truck",
    "BoxTruck",
    "Bulldozer",
    "Excavator",
    "WaterTruck",
    "Sign",
    "Pedestrian",
]


def _load_json(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_v44_release_declares_all_checkpoint_classes() -> None:
    model = _load_json("configs/models/rfdetr/mining8_v44_70px.json")

    assert model["classes"] == EXPECTED_CLASSES


def test_v44_threshold_policy_covers_every_release_class() -> None:
    model = _load_json("configs/models/rfdetr/mining8_v44_70px.json")
    policy = _load_json("configs/policies/mining8_v44_class_thresholds_v1.json")
    configured = set(policy["class_thresholds"])
    review_only = set(policy["review_only_labels"])

    assert set(model["classes"]) == configured | review_only
