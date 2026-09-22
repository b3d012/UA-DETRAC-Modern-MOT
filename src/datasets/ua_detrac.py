"""Faithful UA-DETRAC XML parsing into the canonical M2 schema."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from datasets.schema import (
    AnnotationCoverage,
    BoundingBox,
    FrameAnnotation,
    OcclusionOverlap,
    SequenceAnnotation,
    TargetAnnotation,
    freeze_attributes,
    xywh_to_xyxy,
)

_FRAME_NUMBER = re.compile(r".*?(\d+)(?:\.[^.]+)$")


class AnnotationParseError(ValueError):
    """Raised when an XML file cannot be represented faithfully."""


def _integer(value: str | None, label: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AnnotationParseError(f"Missing or invalid integer {label}: {value!r}") from exc


def _optional_integer(value: str | None, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _box(element: ET.Element, label: str) -> BoundingBox:
    try:
        left = float(element.attrib["left"])
        top = float(element.attrib["top"])
        width = float(element.attrib["width"])
        height = float(element.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise AnnotationParseError(f"Invalid {label} box attributes: {element.attrib}") from exc
    if not all(math.isfinite(value) for value in (left, top, width, height)):
        raise AnnotationParseError(f"Invalid non-finite {label} box attributes: {element.attrib}")
    return xywh_to_xyxy(
        left, top, width, height, raw_attributes=freeze_attributes(element.attrib.items())
    )


def _image_frames(image_directory: Path) -> tuple[tuple[int, str], ...]:
    entries: list[tuple[int, str]] = []
    for path in sorted(image_directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        match = _FRAME_NUMBER.fullmatch(path.name)
        if match is None:
            continue
        entries.append((int(match.group(1)), path.name))
    if not entries:
        raise AnnotationParseError(
            f"Image directory has no numbered image frames: {image_directory}"
        )
    return tuple(entries)


def _target(element: ET.Element) -> TargetAnnotation:
    box_element = element.find("box")
    if box_element is None:
        raise AnnotationParseError("Target is missing <box>")
    attribute = element.find("attribute")
    attribute_values = freeze_attributes(attribute.attrib.items()) if attribute is not None else ()
    overlaps = []
    occlusion_attributes = []
    for occlusion in element.findall("occlusion"):
        occlusion_attributes.append(freeze_attributes(occlusion.attrib.items()))
        for overlap in occlusion.findall("region_overlap"):
            overlaps.append(
                OcclusionOverlap(
                    box=_box(overlap, "occlusion overlap"),
                    occlusion_id=_optional_integer(
                        overlap.attrib.get("occlusion_id"), "occlusion_id"
                    ),
                    occlusion_status=_optional_integer(
                        overlap.attrib.get("occlusion_status"), "occlusion_status"
                    ),
                    raw_attributes=freeze_attributes(overlap.attrib.items()),
                )
            )
    return TargetAnnotation(
        track_id=_integer(element.attrib.get("id"), "target id"),
        vehicle_type=None if attribute is None else attribute.attrib.get("vehicle_type"),
        box=_box(box_element, "target"),
        raw_attributes=freeze_attributes(element.attrib.items()),
        attribute_values=attribute_values,
        occlusion_attributes=tuple(occlusion_attributes),
        occlusions=tuple(overlaps),
    )


def parse_sequence(
    annotation_path: str | Path, image_directory: str | Path, partition: str
) -> SequenceAnnotation:
    """Parse one XML file and its corresponding image directory without mutation."""

    xml_path = Path(annotation_path).resolve()
    frames_directory = Path(image_directory).resolve()
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise AnnotationParseError(f"Cannot parse annotation XML {xml_path}: {exc}") from exc
    if root.tag != "sequence" or not root.attrib.get("name"):
        raise AnnotationParseError(f"Annotation root must be named <sequence>: {xml_path}")
    image_frames = _image_frames(frames_directory)
    try:
        with Image.open(frames_directory / image_frames[0][1]) as image:
            image_size = image.size
    except OSError as exc:
        raise AnnotationParseError(f"Cannot open first image in {frames_directory}: {exc}") from exc

    ignored_regions = root.findall("ignored_region")
    ignored = tuple(
        _box(box, "ignored region") for region in ignored_regions for box in region.findall("box")
    )
    sequence_attribute = root.find("sequence_attribute")
    parsed_frames = []
    for frame in root.findall("frame"):
        parsed_frames.append(
            FrameAnnotation(
                number=_integer(frame.attrib.get("num"), "frame num"),
                density=_optional_integer(frame.attrib.get("density"), "frame density"),
                raw_attributes=freeze_attributes(frame.attrib.items()),
                targets=tuple(_target(target) for target in frame.findall("./target_list/target")),
            )
        )
    image_frame_numbers = tuple(sorted(number for number, _name in image_frames))
    xml_frame_numbers = tuple(sorted({frame.number for frame in parsed_frames}))
    image_frame_set = set(image_frame_numbers)
    xml_frame_set = set(xml_frame_numbers)
    return SequenceAnnotation(
        name=root.attrib["name"],
        partition=partition,
        annotation_path=str(xml_path),
        image_directory=str(frames_directory),
        image_size=image_size,
        image_frames=image_frames,
        raw_attributes=freeze_attributes(root.attrib.items()),
        sequence_attributes=(
            ()
            if sequence_attribute is None
            else freeze_attributes(sequence_attribute.attrib.items())
        ),
        ignored_region_attributes=tuple(
            freeze_attributes(region.attrib.items()) for region in ignored_regions
        ),
        ignored_regions=ignored,
        frames=tuple(parsed_frames),
        annotation_coverage=AnnotationCoverage(
            image_frame_numbers=image_frame_numbers,
            xml_frame_numbers=xml_frame_numbers,
            unannotated_image_frame_numbers=tuple(sorted(image_frame_set - xml_frame_set)),
            xml_frames_without_images=tuple(sorted(xml_frame_set - image_frame_set)),
        ),
    )
