from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from car5_autolabel.datasets.constants import (
    CAMERA_INDEX,
    CAMERA_ORDER,
    EXPECTED_IMAGE_HEIGHT,
    EXPECTED_IMAGE_WIDTH,
    parse_capture_timestamp,
)
from car5_autolabel.datasets.manifest import sha256_file
from car5_autolabel.datasets.models import (
    AnnotationStatus,
    ManifestBuildResult,
    ManifestRow,
    ValidationReport,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def _is_dataset_root(path: Path) -> bool:
    return (path / "camera").is_dir() and (path / "label").is_dir()


def resolve_dataset_root(path: str | Path) -> Path:
    """Resolve either the car5 dataset root or its single outer wrapper directory."""
    candidate = Path(path).expanduser().resolve()
    if _is_dataset_root(candidate):
        return candidate

    nested = [
        child
        for child in candidate.iterdir()
        if child.is_dir() and child.name != "__MACOSX" and _is_dataset_root(child)
    ]
    if len(nested) == 1:
        return nested[0]
    if not nested:
        raise ValueError(f"no car5 dataset root found below {candidate}")
    raise ValueError(f"multiple car5 dataset roots found below {candidate}")


def _collect_files_by_stem(
    directory: Path,
    extensions: set[str],
    report: ValidationReport,
    source: str,
) -> dict[str, Path]:
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    stem_counts = Counter(path.stem for path in files)
    for stem, count in sorted(stem_counts.items()):
        if count > 1:
            report.add_error(
                "duplicate_stem",
                "multiple files share the same timestamp stem",
                source=source,
                frame_id=stem,
                count=count,
            )
    return {path.stem: path for path in sorted(files)}


def _parse_timeline(
    files: dict[str, Path],
    report: ValidationReport,
    source: str,
) -> list[tuple[str, datetime]]:
    parsed: list[tuple[str, datetime]] = []
    for stem in files:
        try:
            parsed.append((stem, parse_capture_timestamp(stem)))
        except ValueError as exc:
            report.add_error(
                "invalid_timestamp",
                str(exc),
                source=source,
                frame_id=stem,
            )
    return sorted(parsed, key=lambda item: (item[1], item[0]))


class Car5DatasetBuilder:
    def __init__(
        self,
        data_root: str | Path,
        *,
        dataset_id: str = "car5_20260611_day_02",
        include_context: bool = False,
        hash_files: bool = True,
        verify_dimensions: bool = False,
        expected_keyframes: int | None = 928,
        expected_camera_frames: int | None = 4640,
        gap_threshold_ms: float = 1000.0,
    ) -> None:
        self.data_root = resolve_dataset_root(data_root)
        self.dataset_id = dataset_id
        self.include_context = include_context
        self.hash_files = hash_files
        self.verify_dimensions = verify_dimensions
        self.expected_keyframes = expected_keyframes
        self.expected_camera_frames = expected_camera_frames
        self.gap_threshold_ms = gap_threshold_ms

    def build(self) -> ManifestBuildResult:
        report = ValidationReport(dataset_id=self.dataset_id)
        camera_files = self._collect_camera_files(report)
        front_files = camera_files.get("front", {})
        timeline = _parse_timeline(front_files, report, "camera/front")
        self._validate_camera_alignment(camera_files, front_files, report)

        label_files = _collect_files_by_stem(
            self.data_root / "label",
            {".json"},
            report,
            "label",
        )
        label_timeline = _parse_timeline(label_files, report, "label")
        keyframe_stems = {stem for stem, _ in label_timeline}
        self._validate_expected_counts(camera_files, label_files, report)
        self._validate_keyframe_coverage(camera_files, keyframe_stems, report)

        timeline_metadata = self._build_timeline_metadata(timeline)
        selected_timeline = [
            item for item in timeline if self.include_context or item[0] in keyframe_stems
        ]
        rows = self._build_rows(
            selected_timeline,
            timeline_metadata,
            camera_files,
            keyframe_stems,
            report,
        )
        report.stats = self._build_stats(
            rows,
            camera_files,
            label_files,
            timeline_metadata,
        )
        if not self.hash_files:
            report.add_warning(
                "hashes_skipped",
                "file hashes were skipped; this manifest is not release-ready",
            )
        if not self.verify_dimensions:
            report.add_warning(
                "dimensions_assumed",
                "image dimensions were taken from the v1 contract without decoding files",
                width=EXPECTED_IMAGE_WIDTH,
                height=EXPECTED_IMAGE_HEIGHT,
            )
        return ManifestBuildResult(rows=rows, report=report)

    def _collect_camera_files(
        self,
        report: ValidationReport,
    ) -> dict[str, dict[str, Path]]:
        result: dict[str, dict[str, Path]] = {}
        camera_root = self.data_root / "camera"
        for camera in CAMERA_ORDER:
            directory = camera_root / camera
            if not directory.is_dir():
                report.add_error(
                    "missing_camera_directory",
                    "required camera directory is missing",
                    camera_id=camera,
                )
                result[camera] = {}
                continue
            result[camera] = _collect_files_by_stem(
                directory,
                IMAGE_EXTENSIONS,
                report,
                f"camera/{camera}",
            )
        return result

    def _validate_camera_alignment(
        self,
        camera_files: dict[str, dict[str, Path]],
        front_files: dict[str, Path],
        report: ValidationReport,
    ) -> None:
        reference = set(front_files)
        for camera in CAMERA_ORDER:
            stems = set(camera_files.get(camera, {}))
            missing = sorted(reference - stems)
            extra = sorted(stems - reference)
            if missing or extra:
                report.add_error(
                    "camera_timeline_mismatch",
                    "camera timestamp set does not match the front camera",
                    camera_id=camera,
                    missing_count=len(missing),
                    extra_count=len(extra),
                    missing_sample=missing[:10],
                    extra_sample=extra[:10],
                )

    def _validate_expected_counts(
        self,
        camera_files: dict[str, dict[str, Path]],
        label_files: dict[str, Path],
        report: ValidationReport,
    ) -> None:
        if self.expected_camera_frames is not None:
            for camera in CAMERA_ORDER:
                actual = len(camera_files.get(camera, {}))
                if actual != self.expected_camera_frames:
                    report.add_error(
                        "unexpected_camera_frame_count",
                        "camera frame count differs from the dataset contract",
                        camera_id=camera,
                        expected=self.expected_camera_frames,
                        actual=actual,
                    )
        if self.expected_keyframes is not None and len(label_files) != self.expected_keyframes:
            report.add_error(
                "unexpected_keyframe_count",
                "label timestamp count differs from the dataset contract",
                expected=self.expected_keyframes,
                actual=len(label_files),
            )

    def _validate_keyframe_coverage(
        self,
        camera_files: dict[str, dict[str, Path]],
        keyframe_stems: set[str],
        report: ValidationReport,
    ) -> None:
        for frame_id in sorted(keyframe_stems):
            for camera in CAMERA_ORDER:
                if frame_id not in camera_files.get(camera, {}):
                    report.add_error(
                        "missing_keyframe_image",
                        "labeled timestamp has no matching camera image",
                        frame_id=frame_id,
                        camera_id=camera,
                    )

    def _build_timeline_metadata(
        self,
        timeline: list[tuple[str, datetime]],
    ) -> dict[str, tuple[int, str, int]]:
        metadata: dict[str, tuple[int, str, int]] = {}
        sequence_index = 0
        previous: datetime | None = None
        for raw_index, (frame_id, timestamp) in enumerate(timeline):
            if previous is not None:
                gap_ms = (timestamp - previous).total_seconds() * 1000.0
                if gap_ms > self.gap_threshold_ms:
                    sequence_index += 1
            sequence_id = f"{self.dataset_id}_s{sequence_index:02d}"
            metadata[frame_id] = (raw_index, sequence_id, sequence_index)
            previous = timestamp
        return metadata

    def _build_rows(
        self,
        selected_timeline: list[tuple[str, datetime]],
        timeline_metadata: dict[str, tuple[int, str, int]],
        camera_files: dict[str, dict[str, Path]],
        keyframe_stems: set[str],
        report: ValidationReport,
    ) -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        for frame_id, timestamp in selected_timeline:
            raw_index, sequence_id, sequence_index = timeline_metadata[frame_id]
            is_keyframe = frame_id in keyframe_stems
            for camera in CAMERA_ORDER:
                path = camera_files.get(camera, {}).get(frame_id)
                if path is None:
                    continue
                dimensions = self._read_dimensions(path, report)
                if dimensions is None:
                    continue
                width, height = dimensions
                rows.append(
                    ManifestRow(
                        dataset_id=self.dataset_id,
                        image_id=f"{self.dataset_id}__{frame_id}__{camera}",
                        relative_path=path.relative_to(self.data_root).as_posix(),
                        sha256=sha256_file(path) if self.hash_files else None,
                        width=width,
                        height=height,
                        sequence_id=sequence_id,
                        raw_sequence_index=raw_index,
                        frame_id=frame_id,
                        capture_timestamp=timestamp.isoformat(timespec="microseconds"),
                        keyframe_id=frame_id if is_keyframe else None,
                        is_keyframe=is_keyframe,
                        sync_delta_ms=0.0,
                        camera_id=camera,
                        camera_order=CAMERA_INDEX[camera],
                        fold_id=f"s{sequence_index:02d}",
                        annotation_status=(
                            AnnotationStatus.UNREVIEWED
                            if is_keyframe
                            else AnnotationStatus.TEMPORAL_CONTEXT
                        ),
                    )
                )
        return rows

    def _read_dimensions(
        self,
        path: Path,
        report: ValidationReport,
    ) -> tuple[int, int] | None:
        if not self.verify_dimensions:
            return EXPECTED_IMAGE_WIDTH, EXPECTED_IMAGE_HEIGHT

        import cv2
        import numpy as np

        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            report.add_error(
                "unreadable_image",
                "OpenCV could not decode image",
                relative_path=path.relative_to(self.data_root).as_posix(),
            )
            return None
        height, width = image.shape[:2]
        if width != EXPECTED_IMAGE_WIDTH or height != EXPECTED_IMAGE_HEIGHT:
            report.add_error(
                "unexpected_image_dimensions",
                "image dimensions differ from the v1 contract",
                relative_path=path.relative_to(self.data_root).as_posix(),
                expected_width=EXPECTED_IMAGE_WIDTH,
                expected_height=EXPECTED_IMAGE_HEIGHT,
                actual_width=width,
                actual_height=height,
            )
        return width, height

    def _build_stats(
        self,
        rows: list[ManifestRow],
        camera_files: dict[str, dict[str, Path]],
        label_files: dict[str, Path],
        timeline_metadata: dict[str, tuple[int, str, int]],
    ) -> dict[str, object]:
        sequence_ids = sorted({metadata[1] for metadata in timeline_metadata.values()})
        source_timestamps_by_sequence = Counter(
            metadata[1] for metadata in timeline_metadata.values()
        )
        selected_images_by_sequence = Counter(row.sequence_id for row in rows)
        return {
            "camera_order": list(CAMERA_ORDER),
            "camera_frame_counts": {
                camera: len(camera_files.get(camera, {})) for camera in CAMERA_ORDER
            },
            "source_keyframe_count": len(label_files),
            "source_timestamp_count": len(timeline_metadata),
            "selected_image_count": len(rows),
            "selected_keyframe_image_count": sum(row.is_keyframe for row in rows),
            "selected_context_image_count": sum(not row.is_keyframe for row in rows),
            "sequence_count": len(sequence_ids),
            "sequence_ids": sequence_ids,
            "source_timestamp_counts_by_sequence": {
                sequence_id: source_timestamps_by_sequence[sequence_id]
                for sequence_id in sequence_ids
            },
            "selected_image_counts_by_sequence": {
                sequence_id: selected_images_by_sequence[sequence_id]
                for sequence_id in sequence_ids
            },
        }
