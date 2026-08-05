"""Build a Git-clean RF-DETR source-and-weight handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from datetime import timezone as _timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO / "configs" / "models" / "rfdetr" / "mining8_v44_70px.json"
ARCHIVE_PREFIX = "car5-rfdetr-v44-70px/"
UTC = _timezone.utc  # noqa: UP017 -- the reference deployment uses Python 3.10.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    actual_hash = sha256(checkpoint)
    expected_hash = str(config["checkpoint_sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    commit = git("rev-parse", "HEAD")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            "--format=zip",
            f"--prefix={ARCHIVE_PREFIX}",
            f"--output={output}",
            "HEAD",
        ],
        check=True,
    )

    release = {
        "release": config["model_version"],
        "source_commit": commit,
        "source_branch": git("branch", "--show-current"),
        "checkpoint_file": config["checkpoint_file"],
        "checkpoint_sha256": actual_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "formal_minimum_long_side_px": config["formal_minimum_long_side_px"],
        "created_at": datetime.now(UTC).isoformat(),
        "contains_company_images": False,
        "contains_annotations": False,
        "quick_start": "docs/rfdetr-v44-70px.md",
    }
    with zipfile.ZipFile(output, "a", allowZip64=True) as archive:
        archive.write(
            checkpoint,
            ARCHIVE_PREFIX + "weights/" + str(config["checkpoint_file"]),
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr(
            ARCHIVE_PREFIX + "MODEL_RELEASE.json",
            json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        )

    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt archive member: {corrupt}")
    print(json.dumps({"archive": str(output), **release}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
