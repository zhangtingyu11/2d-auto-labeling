import argparse
import importlib.util
import json
from pathlib import Path

from car5_autolabel.integrations.label_studio_project_api import LabelStudioProjectSnapshot

SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "run_kfold_training_worker.py"
SPEC = importlib.util.spec_from_file_location("run_kfold_training_worker", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_worker_command_has_fixed_training_policy(tmp_path: Path) -> None:
    job_id = "20260811-reviewed-v2-deadbeef"
    jobs_root = tmp_path / "control_jobs"
    job_directory = jobs_root / job_id
    job_directory.mkdir(parents=True)
    workspace_root = tmp_path / "workspaces"
    artifact_root = tmp_path / "artifacts"
    metadata = {
        "job_id": job_id,
        "workspace_directory": str(workspace_root / job_id),
        "artifact_directory": str(artifact_root / "runs" / job_id),
    }

    command = SCRIPT._command(
        repository_root=Path(__file__).parents[1],
        image_root=tmp_path / "images",
        docker_image="fixed-image",
        gpu_indices=(3, 4),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        job_directory=job_directory,
        metadata=metadata,
    )

    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--grad-accum-steps") + 1] == "1"
    assert command[command.index("--docker-image") + 1] == "fixed-image"
    assert command.count("--gpu") == 2
    assert "--pretrain-weights" not in command


def test_worker_rejects_output_path_outside_fixed_roots(tmp_path: Path) -> None:
    job_id = "job-1"
    job_directory = tmp_path / "control_jobs" / job_id
    job_directory.mkdir(parents=True)
    metadata = {
        "job_id": job_id,
        "workspace_directory": str(tmp_path / "wrong" / job_id),
        "artifact_directory": str(tmp_path / "artifacts" / "runs" / job_id),
    }

    try:
        SCRIPT._command(
            repository_root=Path(__file__).parents[1],
            image_root=tmp_path / "images",
            docker_image="fixed-image",
            gpu_indices=(3,),
            workspace_root=tmp_path / "workspaces",
            artifact_root=tmp_path / "artifacts",
            job_directory=job_directory,
            metadata=metadata,
        )
    except ValueError as error:
        assert "outside the fixed workspace root" in str(error)
    else:
        raise AssertionError("unsafe workspace path was accepted")


def test_worker_sync_command_uses_api_without_exposing_token(tmp_path: Path) -> None:
    command = SCRIPT._sync_command(
        repository_root=Path(__file__).parents[1],
        artifact_directory=tmp_path / "artifacts",
        job_directory=tmp_path / "job",
        job_id="job-1",
        label_studio_url="http://127.0.0.1:8092",
        label_studio_project_id=3,
    )

    assert "--database" not in command
    assert command[command.index("--url") + 1] == "http://127.0.0.1:8092"
    assert command[command.index("--project-id") + 1] == "3"
    assert "--apply" in command
    assert not any("TOKEN" in part or "token" in part for part in command)


def test_input_digest_changes_with_human_training_input(tmp_path: Path) -> None:
    coco = tmp_path / "human.coco.json"
    assignments = tmp_path / "assignments.csv"
    coco.write_text('{"annotations": []}\n', encoding="utf-8")
    assignments.write_text("source_image_id,validation_fold\n1,0\n", encoding="utf-8")
    before = SCRIPT._input_digest(coco, assignments)

    coco.write_text('{"annotations": [{"id": 1}]}\n', encoding="utf-8")

    assert SCRIPT._input_digest(coco, assignments) != before


def test_worker_reads_label_studio_and_strips_locked_hints(
    tmp_path: Path, monkeypatch
) -> None:
    assignments = tmp_path / "assignment-template.csv"
    assignments.write_text(
        "package,source_image_id,source_dataset,camera,timestamp,file_name,block_key,validation_fold\n"
        "old,10,dataset,front,1,images/front/a.jpg,block-a,0\n",
        encoding="utf-8",
    )
    human = {
        "id": "human-1",
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": 1920,
        "original_height": 1080,
        "value": {
            "x": 10,
            "y": 10,
            "width": 20,
            "height": 20,
            "rotation": 0,
            "rectanglelabels": ["Car"],
        },
    }
    hint = {
        **human,
        "id": "hint-1",
        "from_name": "audit_hint",
        "readonly": True,
        "value": {**human["value"], "rectanglelabels": ["可能漏标（预测=Car）"]},
    }
    snapshot = LabelStudioProjectSnapshot(
        project={"id": 3, "label_config": "<View/>"},
        tasks=[
            {
                "id": 10,
                "project": 3,
                "data": {"image": "/data/local-files/?d=images/front/a.jpg"},
                "annotations": [
                    {
                        "id": 20,
                        "updated_at": "2026-08-11T00:00:00Z",
                        "was_cancelled": False,
                        "result": [human, hint],
                    }
                ],
                "drafts": [],
            }
        ],
    )

    class FakeApi:
        def snapshot(self, project_id: int) -> LabelStudioProjectSnapshot:
            assert project_id == 3
            return snapshot

    monkeypatch.setattr(SCRIPT, "_label_studio_api", lambda _args: FakeApi())
    job_id = "job-1"
    jobs_root = tmp_path / "jobs"
    job_directory = jobs_root / job_id
    job_directory.mkdir(parents=True)
    metadata_path = job_directory / "job.json"
    metadata = {
        "job_id": job_id,
        "state": "queued",
        "source_mode": "label_studio_api",
        "label_studio_project_id": 3,
    }
    SCRIPT._write(metadata_path, metadata)
    args = argparse.Namespace(
        label_studio_project_id=3,
        assignment_template=assignments,
    )

    prepared = SCRIPT._prepare_label_studio_source(
        args, metadata_path, metadata, job_directory
    )

    coco = json.loads((job_directory / "human.coco.json").read_text(encoding="utf-8"))
    assert prepared["conversion"]["human_box_count"] == 1
    assert prepared["conversion"]["removed_non_human_result_count"] == 1
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["label_studio_result_id"] == "human-1"


def test_worker_does_not_sync_when_labels_change_during_training(
    tmp_path: Path, monkeypatch
) -> None:
    job_id = "job-stale"
    jobs_root = tmp_path / "jobs"
    job_directory = jobs_root / job_id
    job_directory.mkdir(parents=True)
    (job_directory / "human.coco.json").write_text("{}\n", encoding="utf-8")
    (job_directory / "assignments.csv").write_text("header\n", encoding="utf-8")
    workspace_root = tmp_path / "workspaces"
    artifact_root = tmp_path / "artifacts"
    metadata_path = job_directory / "job.json"
    SCRIPT._write(
        metadata_path,
        {
            "job_id": job_id,
            "state": "queued",
            "source_mode": "legacy_upload",
            "input_digest": "before",
            "workspace_directory": str(workspace_root / job_id),
            "artifact_directory": str(artifact_root / "runs" / job_id),
        },
    )

    class FakeProcess:
        pid = 12345

        def wait(self) -> int:
            return 0

    calls = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(SCRIPT.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(SCRIPT, "_current_label_studio_input_digest", lambda *_: "after")
    args = argparse.Namespace(
        jobs_root=jobs_root,
        repository_root=Path(__file__).parents[1],
        image_root=tmp_path / "images",
        docker_image="fixed-image",
        gpus=[3, 4],
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        label_studio_url="http://127.0.0.1:8092",
        label_studio_project_id=3,
    )

    SCRIPT.process_job(args, metadata_path)

    finished = SCRIPT._read(metadata_path)
    assert finished["state"] == "stale"
    assert "未回写" in finished["error"]
    assert len(calls) == 1
