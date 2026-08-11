from car5_autolabel.integrations.label_studio_project_api import (
    LabelStudioProjectSnapshot,
    snapshot_to_sync_state,
)


def test_snapshot_to_sync_state_keeps_existing_objects_without_creating_any() -> None:
    snapshot = LabelStudioProjectSnapshot(
        project={"id": 3, "label_config": "<View/>"},
        tasks=[
            {
                "id": 11,
                "project": 3,
                "data": {"image": "/data/local-files/?d=images/a.jpg"},
                "annotations": [
                    {"id": 21, "result": [{"id": "human"}], "was_cancelled": False},
                    {"id": 22, "result": [{"id": "cancelled"}], "was_cancelled": True},
                ],
                "drafts": [{"id": 31, "result": [{"id": "draft"}]}],
            }
        ],
    )

    state = snapshot_to_sync_state(snapshot)

    assert state["tasks"] == {11: {"image": "/data/local-files/?d=images/a.jpg"}}
    assert state["annotations"] == {11: [(21, [{"id": "human"}])]}
    assert state["drafts"] == {11: [(31, [{"id": "draft"}])]}


def test_snapshot_to_sync_state_rejects_duplicate_task_ids() -> None:
    task = {"id": 11, "project": 3, "data": {}, "annotations": [], "drafts": []}
    snapshot = LabelStudioProjectSnapshot(
        project={"id": 3, "label_config": "<View/>"}, tasks=[task, task]
    )

    try:
        snapshot_to_sync_state(snapshot)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate Label Studio tasks were accepted")
