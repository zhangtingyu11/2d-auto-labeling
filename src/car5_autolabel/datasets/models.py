from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CAMERA_ORDER = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "back": 3,
    "back_left": 4,
    "back_right": 5,
}
ACTIVE_CLASSES = {
    "car": 0,
    "truck": 1,
    "bulldozer": 2,
    "excavator": 3,
    "watertruck": 4,
    "sign": 5,
}


@dataclass(frozen=True)
class ReviewedBox:
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    source: str
    result_id: str | None = None


@dataclass(frozen=True)
class ReviewedTask:
    image_id: str
    frame_id: str
    camera_id: str
    review_status: str
    boxes: tuple[ReviewedBox, ...] = ()
    task_id: int | str | None = None


@dataclass(frozen=True)
class ManifestRow:
    dataset_id: str
    image_id: str
    relative_path: str
    sha256: str | None
    width: int
    height: int
    sequence_id: str
    frame_id: int
    capture_timestamp: str
    raw_sequence_index: int
    keyframe_id: str | None
    is_keyframe: bool
    sync_delta_ms: float | None
    camera_id: str
    camera_order: int
    fold_id: str
    annotation_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    dataset_id: str
    source_root: str
    camera_counts: dict[str, int] = field(default_factory=dict)
    source_timestamps: int = 0
    keyframe_timestamps: int = 0
    manifest_rows: int = 0
    supervised_rows: int = 0
    temporal_context_rows: int = 0
    sequence_counts: dict[str, int] = field(default_factory=dict)
    annotation_status_counts: dict[str, int] = field(default_factory=dict)
    reviewed_box_counts: dict[str, int] = field(default_factory=dict)
    missing_camera_stems: dict[str, list[str]] = field(default_factory=dict)
    extra_camera_stems: dict[str, list[str]] = field(default_factory=dict)
    keyframes_missing_images: dict[str, list[str]] = field(default_factory=dict)
    image_dimension_errors: list[dict[str, Any]] = field(default_factory=list)
    export_duplicate_keys: list[str] = field(default_factory=list)
    export_missing_keys: list[str] = field(default_factory=list)
    export_extra_keys: list[str] = field(default_factory=list)
    export_unknown_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest_sha256: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result
