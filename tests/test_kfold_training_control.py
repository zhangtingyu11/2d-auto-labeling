import json
from pathlib import Path

from fastapi.testclient import TestClient

from car5_autolabel.api.kfold_training_control import (
    TrainingControlConfig,
    TrainingJobManager,
    create_training_control_app,
)


def _config(tmp_path: Path) -> TrainingControlConfig:
    assignment = tmp_path / "assignments.csv"
    assignment.write_text("placeholder", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    repository = Path(__file__).parents[1]
    return TrainingControlConfig(
        assignment_template=assignment,
        image_root=image_root,
        workspace_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "artifacts",
        repository_root=repository,
        username="reviewer",
        password="secret-password",
        gpu_indices=(3, 4),
    )


def test_training_page_requires_auth_and_launcher_is_public(tmp_path: Path) -> None:
    app = create_training_control_app(_config(tmp_path))
    client = TestClient(app)

    assert client.get("/kfold-training/").status_code == 401
    page = client.get("/kfold-training/", auth=("reviewer", "secret-password"))
    assert page.status_code == 200
    assert "一键读取、训练并更新" in page.text
    assert 'type="file"' not in page.text
    launcher = client.get("/kfold-training/launcher.js")
    assert launcher.status_code == 200
    assert "/projects/3/" in launcher.text


def test_manager_starts_with_no_jobs(tmp_path: Path) -> None:
    manager = TrainingJobManager(_config(tmp_path))

    assert manager.list_jobs() == []


def test_one_click_job_queues_label_studio_source(tmp_path: Path) -> None:
    app = create_training_control_app(_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/kfold-training/api/jobs",
        auth=("reviewer", "secret-password"),
        data={"run_name": "reviewed-v3"},
    )

    assert response.status_code == 201
    metadata = response.json()
    assert metadata["state"] == "queued"
    assert metadata["source_mode"] == "label_studio_api"
    assert metadata["label_studio_project_id"] == 3
    assert not (Path(metadata["log_path"]).parent / "label_studio_export.json").exists()


def test_one_click_rejects_second_active_job(tmp_path: Path) -> None:
    app = create_training_control_app(_config(tmp_path))
    client = TestClient(app)
    auth = ("reviewer", "secret-password")

    assert client.post(
        "/kfold-training/api/jobs", auth=auth, data={"run_name": "first"}
    ).status_code == 201
    second = client.post(
        "/kfold-training/api/jobs", auth=auth, data={"run_name": "second"}
    )

    assert second.status_code == 400
    assert "only one job is allowed" in second.json()["detail"]


def test_manager_does_not_probe_worker_pid_from_web_process(tmp_path: Path) -> None:
    manager = TrainingJobManager(_config(tmp_path))
    metadata = manager.create_job(run_name="host-worker")
    metadata.update({"state": "running", "pid": 2_147_483_647})
    metadata_path = Path(metadata["log_path"]).parent / "job.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    refreshed = manager.get_job(metadata["job_id"])

    assert refreshed["state"] == "running"
    assert refreshed["pid"] == 2_147_483_647
    assert "finished_at" not in refreshed
