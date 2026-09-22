"""Immutable canonical UA-DETRAC annotation data structures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

RawAttributes: TypeAlias = tuple[tuple[str, str], ...]


def freeze_attributes(attributes: Iterable[tuple[str, str]] = ()) -> RawAttributes:
    """Return XML attributes as an immutable, deterministically ordered tuple."""

    return tuple(sorted((str(key), str(value)) for key, value in attributes))


@dataclass(frozen=True)
class BoundingBox:
    """A source xywh box and its derived half-open xyxy coordinates."""

    source_xywh: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    raw_attributes: RawAttributes


def xywh_to_xyxy(
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    raw_attributes: RawAttributes = (),
) -> BoundingBox:
    """Build a canonical box without clipping or changing source coordinates."""

    source = (float(left), float(top), float(width), float(height))
    return BoundingBox(
        source_xywh=source,
        xyxy=(source[0], source[1], source[0] + source[2], source[1] + source[3]),
        raw_attributes=freeze_attributes(raw_attributes),
    )


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Convert a half-open xyxy box to xywh without rounding."""

    return (float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1))


def clip_box_to_image(
    box: BoundingBox, *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Derive clipped image-space xyxy coordinates without changing ``box``."""

    x1, y1, x2, y2 = box.xyxy
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


@dataclass(frozen=True)
class OcclusionOverlap:
    box: BoundingBox
    occlusion_id: int | None
    occlusion_status: int | None
    raw_attributes: RawAttributes


@dataclass(frozen=True)
class TargetAnnotation:
    track_id: int
    vehicle_type: str | None
    box: BoundingBox
    raw_attributes: RawAttributes
    attribute_values: RawAttributes
    occlusion_attributes: tuple[RawAttributes, ...]
    occlusions: tuple[OcclusionOverlap, ...]


@dataclass(frozen=True)
class FrameAnnotation:
    number: int
    density: int | None
    raw_attributes: RawAttributes
    targets: tuple[TargetAnnotation, ...]


@dataclass(frozen=True)
class AnnotationCoverage:
    """Immutable XML/image frame-set coverage metadata without label inference."""

    image_frame_numbers: tuple[int, ...]
    xml_frame_numbers: tuple[int, ...]
    unannotated_image_frame_numbers: tuple[int, ...]
    xml_frames_without_images: tuple[int, ...]


@dataclass(frozen=True)
class SequenceAnnotation:
    name: str
    partition: str
    annotation_path: str
    image_directory: str
    image_size: tuple[int, int]
    image_frames: tuple[tuple[int, str], ...]
    raw_attributes: RawAttributes
    sequence_attributes: RawAttributes
    ignored_region_attributes: tuple[RawAttributes, ...]
    ignored_regions: tuple[BoundingBox, ...]
    frames: tuple[FrameAnnotation, ...]
    annotation_coverage: AnnotationCoverage
