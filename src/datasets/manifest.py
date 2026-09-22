"""Deterministic, M1-only UA-DETRAC dataset manifest generation."""

from __future__ import annotations

import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from datasets.paths import DatasetLayout


class DatasetValidationError(ValueError):
    """Raised when the raw dataset violates the M1 layout contract."""


class DatasetOutputError(ValueError):
    """Raised when a requested manifest destination is unsafe or conflicts."""


_FRAME_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>\.[^.]+)$")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _annotation_files(directory: Path) -> dict[str, Path]:
    files = sorted(directory.glob("*.xml"), key=lambda path: path.name)
    result = {path.stem: path for path in files}
    if len(result) != len(files):
        raise DatasetValidationError(f"Duplicate annotation names in {directory.name}")
    return result


def _sequence_metadata(annotation_path: Path, expected_name: str) -> dict[str, str]:
    root_name: str | None = None
    attributes: dict[str, str] = {}
    try:
        with annotation_path.open("rb") as source:
            for _event, element in ET.iterparse(source, events=("start",)):
                if root_name is None:
                    if element.tag != "sequence":
                        raise DatasetValidationError(
                            f"Annotation root must be <sequence>: {annotation_path.name}"
                        )
                    root_name = element.attrib.get("name")
                    continue
                if element.tag == "sequence_attribute":
                    attributes = dict(sorted(element.attrib.items()))
                    break
                if element.tag in {"frame", "ignored_region"}:
                    break
    except ET.ParseError as exc:
        raise DatasetValidationError(
            f"Cannot read top-level XML metadata from {annotation_path.name}: {exc}"
        ) from exc

    if root_name != expected_name:
        raise DatasetValidationError(
            f"XML sequence name {root_name!r} does not match annotation filename {expected_name!r}"
        )
    return attributes


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.verify()
            return dimensions
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetValidationError(f"Unreadable image frame: {path}") from exc


def _read_image_dimensions(
    paths: list[Path], *, max_workers: int | None = None
) -> list[tuple[int, int]]:
    workers = max_workers or min(32, max(4, (os.cpu_count() or 1) + 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_image_dimensions, paths))


def _frame_inventory(sequence_dir: Path) -> dict[str, Any]:
    files = sorted(
        (path for path in sequence_dir.iterdir() if path.is_file()), key=lambda p: p.name
    )
    if not files:
        raise DatasetValidationError(f"Sequence has no image frames: {sequence_dir.name}")

    matches = [_FRAME_NAME.fullmatch(path.name) for path in files]
    if any(match is None for match in matches):
        raise DatasetValidationError(f"Sequence has inconsistent frame naming: {sequence_dir.name}")

    parsed = [match.groupdict() for match in matches if match is not None]
    conventions = {(item["prefix"], len(item["number"]), item["suffix"]) for item in parsed}
    if len(conventions) != 1:
        raise DatasetValidationError(f"Sequence has inconsistent frame naming: {sequence_dir.name}")
    prefix, width, suffix = conventions.pop()
    numbers = [int(item["number"]) for item in parsed]
    expected_numbers = list(range(numbers[0], numbers[-1] + 1))
    missing = sorted(set(expected_numbers) - set(numbers))
    if missing:
        raise DatasetValidationError(
            f"Sequence {sequence_dir.name} has missing frame numbers: {missing}"
        )
    if len(numbers) != len(set(numbers)):
        raise DatasetValidationError(f"Sequence has duplicate frame numbers: {sequence_dir.name}")

    dimensions: tuple[int, int] | None = None
    for path, current in zip(files, _read_image_dimensions(files), strict=True):
        if dimensions is None:
            dimensions = current
        elif current != dimensions:
            raise DatasetValidationError(
                f"Sequence {sequence_dir.name} has inconsistent image dimensions: "
                f"expected {dimensions}, found {current} in {path.name}"
            )

    assert dimensions is not None
    return {
        "first_frame": files[0].name,
        "frame_count": len(files),
        "frame_naming_pattern": f"{prefix}{{frame:0{width}d}}{suffix}",
        "image_dimensions": {"height": dimensions[1], "width": dimensions[0]},
        "last_frame": files[-1].name,
    }


def _partition_entries(
    layout: DatasetLayout,
    annotations: dict[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(annotations):
        sequence_dir = layout.images_dir / name
        if not sequence_dir.is_dir():
            raise DatasetValidationError(f"Sequence without images: {name}")
        annotation_path = annotations[name]
        entry = {
            "annotation_file": _relative(annotation_path, layout.root),
            "image_directory": _relative(sequence_dir, layout.root),
            "name": name,
            "sequence_attributes": _sequence_metadata(annotation_path, name),
        }
        entry.update(_frame_inventory(sequence_dir))
        entries.append(entry)
    return entries


def build_manifest(
    layout: DatasetLayout,
    *,
    expected_train_sequences: int,
    expected_test_sequences: int,
) -> dict[str, Any]:
    """Validate the raw layout and return a deterministic manifest dictionary."""

    train_annotations = _annotation_files(layout.train_annotations_dir)
    test_annotations = _annotation_files(layout.test_annotations_dir)
    overlap = sorted(set(train_annotations) & set(test_annotations))
    if overlap:
        raise DatasetValidationError(f"Sequences appear in both partitions: {overlap}")

    if len(train_annotations) != expected_train_sequences:
        raise DatasetValidationError(
            f"Train partition expected {expected_train_sequences} annotations, "
            f"found {len(train_annotations)}"
        )
    if len(test_annotations) != expected_test_sequences:
        raise DatasetValidationError(
            f"Test partition expected {expected_test_sequences} annotations, "
            f"found {len(test_annotations)}"
        )

    image_directories = {path.name for path in layout.images_dir.iterdir() if path.is_dir()}
    annotated = set(train_annotations) | set(test_annotations)
    without_annotations = sorted(image_directories - annotated)
    without_images = sorted(annotated - image_directories)
    if without_annotations:
        raise DatasetValidationError(f"Image sequences without annotations: {without_annotations}")
    if without_images:
        raise DatasetValidationError(f"Annotations without image sequences: {without_images}")

    train_entries = _partition_entries(layout, train_annotations)
    test_entries = _partition_entries(layout, test_annotations)
    return {
        "dataset": "UA-DETRAC",
        "layout": {
            "images": _relative(layout.images_dir, layout.root),
            "test_annotations": _relative(layout.test_annotations_dir, layout.root),
            "toolkit": _relative(layout.toolkit_dir, layout.root),
            "train_annotations": _relative(layout.train_annotations_dir, layout.root),
        },
        "partitions": {"test": test_entries, "train": train_entries},
        "schema_version": 1,
        "summary": {
            "sequences_with_missing_frames": 0,
            "sequences_without_annotations": 0,
            "test_sequences": len(test_entries),
            "total_sequences": len(test_entries) + len(train_entries),
            "train_sequences": len(train_entries),
        },
    }


def manifest_json_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize a manifest using the repository's deterministic JSON format."""

    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_manifest(
    manifest: dict[str, Any], output_path: str | Path, *, dataset_root: str | Path
) -> Path:
    """Atomically write a deterministic manifest and return its resolved path."""

    destination = Path(output_path).resolve()
    raw_root = Path(dataset_root).resolve()
    if destination == raw_root or destination.is_relative_to(raw_root):
        raise DatasetOutputError(f"Manifest destination is inside the raw dataset: {destination}")
    content = manifest_json_bytes(manifest)
    if destination.exists():
        if not destination.is_file():
            raise DatasetOutputError(f"Manifest destination is not a file: {destination}")
        if destination.read_bytes() == content:
            return destination
        raise DatasetOutputError(
            f"Existing manifest differs and will not be overwritten: {destination}. "
            "Use a new output path for an approved protocol revision."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination
