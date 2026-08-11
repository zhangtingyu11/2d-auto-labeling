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
    assert "Label Studio JSON" in page.text
    launcher = client.get("/kfold-training/launcher.js")
    assert launcher.status_code == 200
    assert "/projects/3/" in launcher.text


def test_manager_starts_with_no_jobs(tmp_path: Path) -> None:
    manager = TrainingJobManager(_config(tmp_path))

    assert manager.list_jobs() == []
