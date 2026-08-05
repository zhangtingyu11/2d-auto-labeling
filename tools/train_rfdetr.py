"""Reproducible RF-DETR training entry point for the mining 2D release."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MINING_DAY_AUGMENTATIONS: dict[str, dict[str, Any]] = {
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
    "ColorJitter": {
        "brightness": 0.12,
        "contrast": 0.12,
        "saturation": 0.12,
        "hue": 0.03,
        "p": 0.25,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.15},
    "GaussNoise": {"std_range": (0.01, 0.03), "p": 0.15},
}

# The verified RF-DETR workstation environment still uses Python 3.10, where
# ``datetime.UTC`` is unavailable. Keep this alias until that runtime is
# upgraded to the repository's Python 3.11 target.
UTC = timezone.utc  # noqa: UP017 - required by the verified Python 3.10 runtime


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium"), default="medium")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rf-home", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pretrain-weights", type=Path)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--release", default="combined-v2-20260722-similarity-pruned")
    parser.add_argument("--taxonomy", default="mining7-candidate-v2")
    parser.add_argument("--frozen-test", default="car7")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    rf_home = args.rf_home.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rf_home.mkdir(parents=True, exist_ok=True)
    os.environ["RF_HOME"] = str(rf_home)
    os.environ.setdefault("PYTHONHASHSEED", "42")

    import torch
    import torchvision
    from rfdetr import RFDETRMedium, RFDETRSmall

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start an overnight CPU training run")

    model_type = RFDETRMedium if args.model == "medium" else RFDETRSmall
    model_kwargs: dict[str, Any] = {
        "gradient_checkpointing": not args.disable_gradient_checkpointing
    }
    if args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = str(args.pretrain_weights.resolve())
    model = model_type(**model_kwargs)

    config: dict[str, Any] = {
        "model": args.model,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "rf_home": str(rf_home),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "resolution": args.resolution,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "pretrain_weights": (
            str(args.pretrain_weights.resolve()) if args.pretrain_weights else None
        ),
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "release": args.release,
        "taxonomy": args.taxonomy,
        "frozen_test": args.frozen_test,
        "gradient_checkpointing": not args.disable_gradient_checkpointing,
        "augmentations": MINING_DAY_AUGMENTATIONS,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    dump_json(output_dir / "run_config.json", config)

    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "early_stopping": not args.smoke,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": 0.001,
        "early_stopping_use_ema": True,
        "skip_best_epochs": 1 if not args.smoke else 0,
        "warmup_epochs": 2.0 if not args.smoke else 0.0,
        "lr_scheduler": "cosine",
        "lr_min_factor": 0.05,
        "use_ema": True,
        "amp_dtype": "auto",
        "multi_scale": False,
        "expanded_scales": False,
        "square_resize_div_64": True,
        "aug_config": MINING_DAY_AUGMENTATIONS,
        "augmentation_backend": "cpu",
        "seed": 42,
        "progress_bar": "tqdm" if args.smoke else None,
        "tensorboard": False,
        "wandb": False,
        "run_test": False,
        "log_per_class_metrics": True,
        "compute_val_loss": True,
        "pin_memory": True,
        # RF-DETR 1.8.3 maps ``cuda:0`` to a list, while its Windows Lightning
        # trainer currently expects an int/string.  ``cuda`` selects the only
        # visible GPU and avoids that upstream compatibility bug.
        "device": "cuda",
        "notes": {
            "project": "mining-day-6cam-2d-autolabel",
            "release": args.release,
            "taxonomy": args.taxonomy,
            "frozen_test": args.frozen_test,
            "evaluation_policy": (
                "development validation during training; frozen test once "
                "after model selection"
            ),
        },
    }
    if args.resolution is not None:
        train_kwargs["resolution"] = args.resolution
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume.resolve())

    try:
        model.train(**train_kwargs)
    except BaseException as error:
        dump_json(
            output_dir / "failure.json",
            {
                "failed_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    else:
        dump_json(
            output_dir / "completed.json",
            {"completed_at": datetime.now(UTC).isoformat()},
        )


if __name__ == "__main__":
    main()
