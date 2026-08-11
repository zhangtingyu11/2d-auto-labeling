"""Serve the authenticated K-fold training page used behind Label Studio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from car5_autolabel.api.kfold_training_control import (
    TrainingControlConfig,
    create_training_control_app,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-template", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--docker-image", default="car5-rfdetr:v4.4-70px-train-user")
    parser.add_argument("--project-id", type=int, default=3)
    parser.add_argument("--gpu", type=int, action="append", dest="gpus")
    parser.add_argument("--username", default="kfold-admin")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get("CAR5_KFOLD_TRAINING_PASSWORD", "")
    if not password:
        raise ValueError("CAR5_KFOLD_TRAINING_PASSWORD is required")
    config = TrainingControlConfig(
        assignment_template=args.assignment_template.resolve(),
        image_root=args.image_root.resolve(),
        workspace_root=args.workspace_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        repository_root=args.repository_root.resolve(),
        docker_image=args.docker_image,
        project_id=args.project_id,
        gpu_indices=tuple(args.gpus or (3, 4)),
        username=args.username,
        password=password,
    )
    app = create_training_control_app(config)
    uvicorn.run(app, host=args.host, port=args.port, access_log=True)


if __name__ == "__main__":
    main()
