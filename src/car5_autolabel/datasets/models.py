from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised by the Python 3.10 training runtime
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042
        __str__ = str.__str__

from pydantic import BaseModel, Field, model_validator


class AnnotationStatus(StrEnum):
    REVIEWED = "reviewed"
    VERIFIED_EMPTY = "verified_empty"
    UNREVIEWED = "unreviewed"
    TEMPORAL_CONTEXT = "temporal_context"


class ReviewStatus(StrEnum):
    UNREVIEWED_PREDICTION = "unreviewed_prediction"
    HUMAN_VERIFIED = "human_verified"
    HUMAN_MODIFIED = "human_modified"
    HUMAN_CREATED = "human_created"
    VERIFIED_EMPTY = "verified_empty"
    REJECTED_PREDICTION = "rejected_prediction"


class ManifestRow(BaseModel):
    dataset_id: str
    image_id: str
    relative_path: str
    sha256: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sequence_id: str
    raw_sequence_index: int = Field(ge=0)
    frame_id: str
    capture_timestamp: str
    keyframe_id: str | None = None
    is_keyframe: bool
    sync_delta_ms: float | None = None
    camera_id: str
    camera_order: int = Field(ge=0)
    fold_id: str
    annotation_status: AnnotationStatus

    @model_validator(mode="after")
    def validate_keyframe_state(self) -> "ManifestRow":
        if self.is_keyframe and not self.keyframe_id:
            raise ValueError("keyframe rows require keyframe_id")
        if not self.is_keyframe and self.keyframe_id is not None:
            raise ValueError("context rows cannot have keyframe_id")
        if self.is_keyframe and self.annotation_status is AnnotationStatus.TEMPORAL_CONTEXT:
            raise ValueError("keyframe rows cannot be temporal_context")
        if not self.is_keyframe and self.annotation_status is not AnnotationStatus.TEMPORAL_CONTEXT:
            raise ValueError("non-keyframe rows must be temporal_context")
        return self


class CanonicalBox(BaseModel):
    annotation_id: str
    image_id: str
    class_id: int = Field(ge=0)
    label: str
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)
    source: str = "human"
    review_status: ReviewStatus
    track_id: str | None = None
    occluded: bool = False
    truncated: bool = False
    model_version: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_corners(self) -> "CanonicalBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("x2/y2 must be greater than x1/y1")
        return self


class CanonicalImageAnnotation(BaseModel):
    image_id: str
    camera_id: str
    frame_id: str
    review_status: ReviewStatus
    boxes: list[CanonicalBox] = Field(default_factory=list)
    source_project_id: int | None = None
    source_task_id: int | None = None
    source_annotation_id: int | None = None


class ValidationIssue(BaseModel):
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    dataset_id: str
    valid: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.valid = False
        self.errors.append(ValidationIssue(code=code, message=message, context=context))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(ValidationIssue(code=code, message=message, context=context))


class ManifestBuildResult(BaseModel):
    rows: list[ManifestRow]
    report: ValidationReport


class LabelStudioParseResult(BaseModel):
    annotations: list[CanonicalImageAnnotation]
    report: ValidationReport


class GoldSetRow(BaseModel):
    dataset_release_id: str
    image_id: str
    sha256: str = Field(min_length=64, max_length=64)
    frame_id: str
    camera_id: str
    sequence_id: str
    sampling_reasons: list[str] = Field(min_length=1)
    review_state: str = "frozen"


class GoldSetBuildResult(BaseModel):
    rows: list[GoldSetRow]
    report: ValidationReport
