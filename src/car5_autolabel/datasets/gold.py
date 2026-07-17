from __future__ import annotations

from collections import Counter, defaultdict

from car5_autolabel.datasets.constants import CAMERA_ORDER
from car5_autolabel.datasets.models import (
    CanonicalImageAnnotation,
    GoldSetBuildResult,
    GoldSetRow,
    ManifestRow,
    ReviewStatus,
    ValidationReport,
)

SMALL_BOX_AREA = 32 * 32


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return values.copy()
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _group_features(
    rows: list[ManifestRow],
    annotations: dict[str, CanonicalImageAnnotation],
    class_counts: Counter[str],
) -> tuple[float, list[str]]:
    labels: set[str] = set()
    has_small = False
    has_edge = False
    empty_count = 0
    score = 0.0
    total_boxes = max(sum(class_counts.values()), 1)

    for row in rows:
        annotation = annotations[row.image_id]
        if not annotation.boxes:
            empty_count += 1
        for box in annotation.boxes:
            labels.add(box.label)
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            has_small |= width * height < SMALL_BOX_AREA
            has_edge |= (
                box.x1 <= 1
                or box.y1 <= 1
                or box.x2 >= row.width - 1
                or box.y2 >= row.height - 1
            )

    reasons = [f"class:{label}" for label in sorted(labels)]
    for label in labels:
        score += total_boxes / max(class_counts[label], 1)
    if has_small:
        reasons.append("small_box")
        score += 2.0
    if has_edge:
        reasons.append("edge_truncation")
        score += 2.0
    if 0 < empty_count < len(rows):
        reasons.append("mixed_positive_empty")
        score += 1.0
    if empty_count == len(rows):
        reasons.append("all_cameras_verified_empty")
        score += 0.5
    return score, reasons


def build_gold_set(
    manifest_rows: list[ManifestRow],
    annotations: list[CanonicalImageAnnotation],
    *,
    dataset_release_id: str,
    timestamp_groups: int = 120,
) -> GoldSetBuildResult:
    report = ValidationReport(dataset_id=dataset_release_id)
    annotation_by_image = {annotation.image_id: annotation for annotation in annotations}
    if len(annotation_by_image) != len(annotations):
        report.add_error("duplicate_annotations", "canonical image IDs must be unique")

    keyframe_rows = [row for row in manifest_rows if row.is_keyframe]
    missing_annotations = sorted(
        row.image_id for row in keyframe_rows if row.image_id not in annotation_by_image
    )
    missing_hashes = sorted(row.image_id for row in keyframe_rows if not row.sha256)
    if missing_annotations:
        report.add_error(
            "missing_annotations",
            "gold set requires one canonical annotation per keyframe image",
            count=len(missing_annotations),
            sample=missing_annotations[:10],
        )
    if missing_hashes:
        report.add_error(
            "missing_hashes",
            "gold set requires release-grade image hashes",
            count=len(missing_hashes),
            sample=missing_hashes[:10],
        )

    invalid_review = sorted(
        annotation.image_id
        for annotation in annotations
        if annotation.review_status
        not in {ReviewStatus.HUMAN_VERIFIED, ReviewStatus.VERIFIED_EMPTY}
    )
    if invalid_review:
        report.add_error(
            "unreviewed_annotations",
            "gold set cannot contain unreviewed images",
            count=len(invalid_review),
            sample=invalid_review[:10],
        )
    if not report.valid:
        return GoldSetBuildResult(rows=[], report=report)

    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in keyframe_rows:
        groups[row.frame_id].append(row)
    for frame_id, rows in sorted(groups.items()):
        cameras = [row.camera_id for row in sorted(rows, key=lambda item: item.camera_order)]
        if cameras != list(CAMERA_ORDER):
            report.add_error(
                "incomplete_camera_group",
                "each gold timestamp must contain the canonical six-camera order",
                frame_id=frame_id,
                cameras=cameras,
            )
    if timestamp_groups > len(groups):
        report.add_error(
            "gold_target_too_large",
            "requested gold timestamp groups exceed available keyframes",
            requested=timestamp_groups,
            available=len(groups),
        )
    if not report.valid:
        return GoldSetBuildResult(rows=[], report=report)

    class_counts: Counter[str] = Counter(
        box.label for annotation in annotations for box in annotation.boxes
    )
    sequence_frames: dict[str, list[str]] = defaultdict(list)
    for frame_id, rows in groups.items():
        sequence_frames[rows[0].sequence_id].append(frame_id)
    for values in sequence_frames.values():
        values.sort()

    sequence_ids = sorted(sequence_frames)
    base_quota, remainder = divmod(timestamp_groups, len(sequence_ids))
    selected_reasons: dict[str, list[str]] = {}
    for sequence_index, sequence_id in enumerate(sequence_ids):
        quota = base_quota + (1 if sequence_index < remainder else 0)
        frame_ids = sequence_frames[sequence_id]
        scored: list[tuple[float, str, list[str]]] = []
        for frame_id in frame_ids:
            score, reasons = _group_features(
                groups[frame_id], annotation_by_image, class_counts
            )
            scored.append((score, frame_id, reasons))

        challenge_count = min(quota // 2, len(scored))
        challenge = sorted(scored, key=lambda item: (-item[0], item[1]))[:challenge_count]
        for _, frame_id, reasons in challenge:
            selected_reasons[frame_id] = reasons or ["challenge_coverage"]

        remaining = [
            frame_id for frame_id in frame_ids if frame_id not in selected_reasons
        ]
        for frame_id in _evenly_spaced(remaining, quota - challenge_count):
            _, reasons = _group_features(
                groups[frame_id], annotation_by_image, class_counts
            )
            selected_reasons[frame_id] = sorted(set([*reasons, "temporal_coverage"]))

    rows: list[GoldSetRow] = []
    for frame_id in sorted(selected_reasons):
        for manifest_row in sorted(groups[frame_id], key=lambda item: item.camera_order):
            annotation = annotation_by_image[manifest_row.image_id]
            reasons = list(selected_reasons[frame_id])
            reasons.append(f"camera:{manifest_row.camera_id}")
            if not annotation.boxes:
                reasons.append("verified_empty")
            rows.append(
                GoldSetRow(
                    dataset_release_id=dataset_release_id,
                    image_id=manifest_row.image_id,
                    sha256=manifest_row.sha256 or "",
                    frame_id=frame_id,
                    camera_id=manifest_row.camera_id,
                    sequence_id=manifest_row.sequence_id,
                    sampling_reasons=sorted(set(reasons)),
                )
            )

    selected_image_ids = {row.image_id for row in rows}
    selected_annotations = [
        annotation
        for annotation in annotations
        if annotation.image_id in selected_image_ids
    ]
    selected_class_counts = Counter(
        box.label for annotation in selected_annotations for box in annotation.boxes
    )
    report.stats = {
        "timestamp_group_count": len(selected_reasons),
        "image_count": len(rows),
        "sequence_group_counts": dict(
            sorted(Counter(row.sequence_id for row in rows[:: len(CAMERA_ORDER)]).items())
        ),
        "camera_image_counts": dict(sorted(Counter(row.camera_id for row in rows).items())),
        "source_class_box_counts": dict(sorted(class_counts.items())),
        "gold_class_box_counts": dict(sorted(selected_class_counts.items())),
        "gold_verified_empty_image_count": sum(
            not annotation.boxes for annotation in selected_annotations
        ),
        "sampling_reason_counts": dict(
            sorted(Counter(reason for row in rows for reason in row.sampling_reasons).items())
        ),
        "small_box_area_threshold": SMALL_BOX_AREA,
    }
    return GoldSetBuildResult(rows=rows, report=report)
