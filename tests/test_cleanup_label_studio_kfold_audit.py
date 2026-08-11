import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "cleanup_label_studio_kfold_audit.py"
SPEC = importlib.util.spec_from_file_location("cleanup_label_studio_kfold_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cleanup_removes_only_kfold_audit_fields() -> None:
    data = {
        "image": "one.jpg",
        "review_tag": "可能漏标",
        "review_summary": "可能漏标:Car",
        "kfold_model_version": "model",
        "camera": "front",
    }
    results = [
        {"from_name": "label", "type": "rectanglelabels"},
        {"from_name": "truncation", "type": "choices"},
        {"from_name": "occlusion", "type": "choices"},
        {"from_name": "review_issue", "type": "choices"},
        {"from_name": "prediction_reference", "type": "rectanglelabels"},
    ]

    assert MODULE.clean_task_data(data) == {
        "image": "one.jpg",
        "camera": "front",
    }
    assert [row["from_name"] for row in MODULE.clean_results(results)] == [
        "label",
        "truncation",
        "occlusion",
    ]


def test_cleanup_removes_audit_controls_but_keeps_human_controls() -> None:
    config = """<View>
      <Header value="数据提示：$review_tag_label"/>
      <Text name="review_tag" value="$review_tag"/>
      <Choices name="review_issue" toName="image"><Choice value="可能漏标"/></Choices>
      <RectangleLabels name="prediction_reference" toName="image"/>
      <RectangleLabels name="label" toName="image"><Label value="Car"/></RectangleLabels>
      <Choices name="truncation" toName="image"/>
      <Choices name="occlusion" toName="image"/>
    </View>"""

    cleaned = MODULE.clean_label_config(config)

    assert "可能" not in cleaned
    assert "review_" not in cleaned
    assert "prediction_reference" not in cleaned
    assert 'name="label"' in cleaned
    assert 'name="truncation"' in cleaned
    assert 'name="occlusion"' in cleaned
