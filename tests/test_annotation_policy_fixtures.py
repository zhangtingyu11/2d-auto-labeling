import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "annotations"
CAMERA_ORDER = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "back": 3,
    "back_left": 4,
    "back_right": 5,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_synthetic_gold_manifest_is_complete_and_deterministic() -> None:
    rows = _read_jsonl(FIXTURE_DIR / "gold_manifest.synthetic.jsonl")

    assert len(rows) == 6
    assert [row["camera_id"] for row in rows] == list(CAMERA_ORDER)
    assert len({row["image_id"] for row in rows}) == len(rows)
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert {row["capture_timestamp"] for row in rows} == {"2026-01-01T00:00:00.000Z"}
    assert all(row["camera_order"] == CAMERA_ORDER[row["camera_id"]] for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["selection_status"] == "frozen" for row in rows)


def test_synthetic_disagreements_preserve_blocking_state() -> None:
    rows = _read_jsonl(FIXTURE_DIR / "disagreements.synthetic.jsonl")

    assert {row["category"] for row in rows} == {
        "class_mismatch",
        "host_ignore_disagreement",
    }
    assert rows[0]["status"] == "resolved"
    assert rows[0]["resolution"]["class"] == "Truck"
    assert rows[1]["status"] == "release_blocking"
    assert rows[1]["resolution"] is None
