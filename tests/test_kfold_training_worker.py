import importlib.util
from pathlib import Path


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
