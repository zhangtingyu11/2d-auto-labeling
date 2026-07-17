from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pinned car5 RF-DETR-S training stage.")
    parser.add_argument(
        "--stage",
        choices=("smoke", "pilot", "pilot_tuned", "long"),
        required=True,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("model") != "rfdetr_small":
        raise ValueError("RF-DETR runner requires model: rfdetr_small")
    stages = payload.get("stages", {})
    if stage not in stages:
        raise ValueError(f"missing RF-DETR stage configuration: {stage}")
    train_config = dict(payload.get("common", {}))
    train_config.update(stages[stage])
    return payload, train_config


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "models" / "rfdetr" / "car5_rfdetr_s.yaml"
    model_config, train_config = _load_config(config_path, args.stage)

    dataset_dir = args.dataset_dir.resolve()
    required = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "valid" / "_annotations.coco.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RF-DETR dataset is not staged: {missing}")

    train_config["dataset_dir"] = str(dataset_dir)
    train_config["output_dir"] = str(args.output_dir.resolve())
    train_config["class_names"] = list(model_config["classes"])
    if args.resume is not None:
        train_config["resume"] = str(args.resume.resolve())
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        train_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num workers cannot be negative")
        train_config["num_workers"] = args.num_workers

    from rfdetr import RFDETRSmall

    print(
        json.dumps(
            {
                "model": model_config["model"],
                "stage": args.stage,
                "resolution": model_config["resolution"],
                "train": train_config,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    model = RFDETRSmall(resolution=int(model_config["resolution"]))
    model.train(**train_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
