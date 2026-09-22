"""Validation for canonical UA-DETRAC annotations without source normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from datasets.schema import BoundingBox, SequenceAnnotation

ALLOWED_VEHICLE_CLASSES = frozenset(("car", "bus", "van", "others"))
UA_DETRAC_BORDER_TOLERANCE = 1.0


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    frame_number: int | None = None
    track_id: int | None = None
    severity: str = "error"


@dataclass(frozen=True)
class ValidationReport:
    sequence_name: str
    partition: str
    image_size: tuple[int, int]
    frame_count: int
    issue_count: int
    error_count: int
    warning_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0


@dataclass(frozen=True)
class FrameAlignmentDiagnostic:
    """Raw XML/image frame-set comparison; it does not alter validation semantics."""

    sequence_name: str
    image_frame_count: int
    image_frame_minimum: int | None
    image_frame_maximum: int | None
    xml_frame_count: int
    xml_frame_minimum: int | None
    xml_frame_maximum: int | None
    image_frames_absent_from_xml: int
    xml_frames_without_images: int
    first_20_image_frames_absent_from_xml: tuple[int, ...]
    last_20_image_frames_absent_from_xml: tuple[int, ...]
    has_leading_missing_xml_frames: bool
    has_trailing_missing_xml_frames: bool
    has_interior_missing_xml_gaps: bool
    contiguous_missing_xml_blocks: tuple[tuple[int, int], ...]


def _contiguous_blocks(frame_numbers: list[int]) -> tuple[tuple[int, int], ...]:
    if not frame_numbers:
        return ()
    blocks = []
    start = previous = frame_numbers[0]
    for current in frame_numbers[1:]:
        if current != previous + 1:
            blocks.append((start, previous))
            start = current
        previous = current
    blocks.append((start, previous))
    return tuple(blocks)


def diagnose_frame_alignment(sequence: SequenceAnnotation) -> FrameAlignmentDiagnostic:
    """Describe raw frame-set differences without inferring empty XML frames."""

    image_numbers = sorted(number for number, _name in sequence.image_frames)
    xml_numbers = sorted(frame.number for frame in sequence.frames)
    image_set = set(image_numbers)
    xml_set = set(xml_numbers)
    missing_xml = sorted(image_set - xml_set)
    blocks = _contiguous_blocks(missing_xml)
    image_minimum = min(image_numbers) if image_numbers else None
    image_maximum = max(image_numbers) if image_numbers else None
    return FrameAlignmentDiagnostic(
        sequence_name=sequence.name,
        image_frame_count=len(image_numbers),
        image_frame_minimum=image_minimum,
        image_frame_maximum=image_maximum,
        xml_frame_count=len(xml_numbers),
        xml_frame_minimum=min(xml_numbers) if xml_numbers else None,
        xml_frame_maximum=max(xml_numbers) if xml_numbers else None,
        image_frames_absent_from_xml=len(missing_xml),
        xml_frames_without_images=len(xml_set - image_set),
        first_20_image_frames_absent_from_xml=tuple(missing_xml[:20]),
        last_20_image_frames_absent_from_xml=tuple(missing_xml[-20:]),
        has_leading_missing_xml_frames=bool(missing_xml and missing_xml[0] == image_minimum),
        has_trailing_missing_xml_frames=bool(missing_xml and missing_xml[-1] == image_maximum),
        has_interior_missing_xml_gaps=any(
            block[0] != image_minimum and block[1] != image_maximum for block in blocks
        ),
        contiguous_missing_xml_blocks=blocks,
    )


def _box_issues(
    box: BoundingBox, width: int, height: int, frame_number: int | None
) -> list[ValidationIssue]:
    x1, y1, x2, y2 = box.xyxy
    issues = []
    if box.source_xywh[2] <= 0 or box.source_xywh[3] <= 0:
        issues.append(
            ValidationIssue(
                "non_positive_box_area", "Box width and height must be positive", frame_number
            )
        )
    if (
        x1 < 0
        or y1 < 0
        or x2 > width + UA_DETRAC_BORDER_TOLERANCE
        or y2 > height + UA_DETRAC_BORDER_TOLERANCE
    ):
        issues.append(
            ValidationIssue(
                "box_out_of_image_range",
                f"Box {box.xyxy} exceeds image range 0..{width}, 0..{height}",
                frame_number,
            )
        )
    return issues


def validate_sequence(sequence: SequenceAnnotation) -> ValidationReport:
    """Return all protocol-relevant defects found in one parsed sequence."""

    width, height = sequence.image_size
    issues: list[ValidationIssue] = []
    image_numbers = [number for number, _name in sequence.image_frames]
    if Path(sequence.image_directory).name != sequence.name:
        issues.append(
            ValidationIssue(
                "xml_image_sequence_mismatch",
                "XML sequence name does not match the image directory name",
            )
        )
    expected_images = list(range(image_numbers[0], image_numbers[-1] + 1))
    if image_numbers != expected_images:
        issues.append(
            ValidationIssue("non_contiguous_image_frames", "Image frame numbers are not contiguous")
        )
    frame_numbers = [frame.number for frame in sequence.frames]
    if frame_numbers != sorted(frame_numbers) or len(frame_numbers) != len(set(frame_numbers)):
        issues.append(
            ValidationIssue(
                "invalid_frame_order", "XML frame numbers are not unique and increasing"
            )
        )
    if sequence.annotation_coverage.unannotated_image_frame_numbers:
        issues.append(
            ValidationIssue(
                "image_frame_absent_from_xml",
                "Image frames have no XML <frame> element; annotation coverage is unknown",
                severity="warning",
            )
        )
    if sequence.annotation_coverage.xml_frames_without_images:
        issues.append(
            ValidationIssue(
                "xml_frame_without_image",
                "XML <frame> elements have no corresponding image frame",
            )
        )
    for ignored in sequence.ignored_regions:
        issues.extend(_box_issues(ignored, width, height, None))
    for frame in sequence.frames:
        seen_ids: set[int] = set()
        for target in frame.targets:
            if target.track_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        "duplicate_track_id",
                        "Track ID is repeated within one frame",
                        frame.number,
                        target.track_id,
                    )
                )
            seen_ids.add(target.track_id)
            if target.vehicle_type not in ALLOWED_VEHICLE_CLASSES:
                issues.append(
                    ValidationIssue(
                        "unknown_vehicle_type",
                        f"Unsupported vehicle type: {target.vehicle_type!r}",
                        frame.number,
                        target.track_id,
                    )
                )
            issues.extend(_box_issues(target.box, width, height, frame.number))
            for overlap in target.occlusions:
                issues.extend(_box_issues(overlap.box, width, height, frame.number))
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    return ValidationReport(
        sequence_name=sequence.name,
        partition=sequence.partition,
        image_size=sequence.image_size,
        frame_count=len(sequence.frames),
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        issues=tuple(issues),
    )


def report_as_dict(report: ValidationReport) -> dict[str, object]:
    """Serialize a report using only JSON-compatible immutable values."""

    return asdict(report)
