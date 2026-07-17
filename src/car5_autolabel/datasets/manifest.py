import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from car5_autolabel.datasets.models import (
    CanonicalImageAnnotation,
    ManifestRow,
    ValidationReport,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_manifest_jsonl(rows: Iterable[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            payload = row.model_dump(mode="json")
            serialized = json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write(serialized)
            stream.write("\n")


def read_manifest_jsonl(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(ManifestRow.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid manifest row at line {line_number}: {exc}") from exc
    return rows


def read_canonical_annotations_json(path: Path) -> list[CanonicalImageAnnotation]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid canonical annotation JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("canonical annotation root must be a list")
    try:
        return [CanonicalImageAnnotation.model_validate(item) for item in payload]
    except ValueError as exc:
        raise ValueError(f"invalid canonical annotation payload: {exc}") from exc


def write_validation_report(report: ValidationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_canonical_annotations_json(
    annotations: Iterable[CanonicalImageAnnotation],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [annotation.model_dump(mode="json") for annotation in annotations]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json(payload: object, output_path: Path, *, compact: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = (
        {"separators": (",", ":")}
        if compact
        else {"indent": 2}
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, **kwargs) + "\n",
        encoding="utf-8",
    )
