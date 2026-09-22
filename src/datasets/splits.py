"""M3 deterministic, annotation-availability-aware UA-DETRAC split support."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

from datasets.ua_detrac import parse_sequence

SplitRole = Literal["development_train", "validation", "official_test"]


class SplitError(ValueError):
    """Raised when a split input, frozen artifact, or access request is invalid."""


@dataclass(frozen=True, order=True)
class SequenceSplitCandidate:
    """The complete, deliberately narrow input contract for the split optimizer."""

    name: str
    sence_weather: str
    camera_state: str
    annotated_frame_count: int
    frame_count: int


def _category(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "<missing>"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _tie(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:".encode() + _canonical(value)).hexdigest()


def _hamilton(counts: dict[str, int], total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    if denominator <= 0:
        raise SplitError("Cannot allocate validation quotas from no sequences")
    floors = {name: total * count // denominator for name, count in counts.items()}
    remaining = total - sum(floors.values())
    order = sorted(
        counts,
        key=lambda name: (-(total * counts[name] % denominator), name),
    )
    for name in order[:remaining]:
        floors[name] += 1
    return floors


def _quota_matrices(candidates: tuple[SequenceSplitCandidate, ...], validation_count: int):
    weather = sorted({item.sence_weather for item in candidates})
    camera = sorted({item.camera_state for item in candidates})
    cells = [(w, c) for w in weather for c in camera]
    capacities = {
        cell: sum(item.sence_weather == cell[0] and item.camera_state == cell[1] for item in candidates)
        for cell in cells
    }
    weather_counts = {w: sum(capacities[w, c] for c in camera) for w in weather}
    camera_counts = {c: sum(capacities[w, c] for w in weather) for c in camera}
    weather_target = _hamilton(weather_counts, validation_count)
    camera_target = _hamilton(camera_counts, validation_count)
    matrices: list[dict[tuple[str, str], int]] = []

    def visit_exact(index: int, row_left: dict[str, int], column_left: dict[str, int], matrix: dict[tuple[str, str], int]) -> None:
        if index == len(cells):
            if not any(row_left.values()) and not any(column_left.values()):
                matrices.append(matrix.copy())
            return
        w, c = cells[index]
        maximum = min(capacities[w, c], row_left[w], column_left[c])
        for quota in range(maximum + 1):
            matrix[w, c] = quota
            next_rows, next_columns = row_left.copy(), column_left.copy()
            next_rows[w] -= quota
            next_columns[c] -= quota
            visit_exact(index + 1, next_rows, next_columns, matrix)
        matrix.pop((w, c), None)

    visit_exact(0, weather_target.copy(), camera_target.copy(), {})
    if not matrices:
        raise SplitError("Hamilton weather/camera quotas have no feasible joint allocation")

    def categorical_score(matrix: dict[tuple[str, str], int]) -> tuple[int, int]:
        marginal = sum(
            abs(sum(matrix[w, c] for c in camera) - weather_target[w]) for w in weather
        ) + sum(abs(sum(matrix[w, c] for w in weather) - camera_target[c]) for c in camera)
        joint = sum(abs(5 * matrix[cell] - capacities[cell]) for cell in cells)
        return marginal, joint

    best = min(categorical_score(matrix) for matrix in matrices)
    selected = [matrix for matrix in matrices if categorical_score(matrix) == best]
    return selected, capacities, weather_target, camera_target, cells


def _subsets(items: tuple[SequenceSplitCandidate, ...], quota: int):
    return tuple(itertools.combinations(items, quota))


def _partial_combinations(groups: list[tuple[tuple[SequenceSplitCandidate, ...], ...]]):
    partial = [(0, 0, ())]
    for options in groups:
        partial = [
            (
                annotated + sum(item.annotated_frame_count for item in option),
                raw + sum(item.frame_count for item in option),
                names + tuple(item.name for item in option),
            )
            for annotated, raw, names in partial
            for option in options
        ]
    return partial


def _choose_for_matrix(
    candidates: tuple[SequenceSplitCandidate, ...],
    matrix: dict[tuple[str, str], int],
    cells: list[tuple[str, str]],
    seed: int,
) -> tuple[tuple[int, int, str], tuple[str, ...]]:
    groups = []
    for cell in cells:
        members = tuple(item for item in candidates if (item.sence_weather, item.camera_state) == cell)
        groups.append(_subsets(members, matrix[cell]))
    # Balance the Cartesian product before materializing the two halves.
    indexed = sorted(enumerate(groups), key=lambda item: len(item[1]), reverse=True)
    left: list[tuple[tuple[SequenceSplitCandidate, ...], ...]] = []
    right: list[tuple[tuple[SequenceSplitCandidate, ...], ...]] = []
    left_size = right_size = 1
    for _index, group in indexed:
        if left_size <= right_size:
            left.append(group)
            left_size *= len(group)
        else:
            right.append(group)
            right_size *= len(group)
    first, second = _partial_combinations(left), _partial_combinations(right)
    second_by_annotated: dict[int, list[tuple[int, int, tuple[str, ...]]]] = {}
    for annotated, raw, names in second:
        second_by_annotated.setdefault(annotated, []).append((annotated, raw, names))
    annotated_values = sorted(second_by_annotated)
    target_annotated = sum(item.annotated_frame_count for item in candidates)
    target_raw = sum(item.frame_count for item in candidates)
    best: tuple[tuple[int, int, str], tuple[str, ...]] | None = None
    for annotated, raw, names in first:
        desired = target_annotated - annotated
        position = bisect_left(annotated_values, desired)
        for value in annotated_values[max(0, position - 1) : position + 1]:
            for other_annotated, other_raw, other_names in second_by_annotated[value]:
                selected = tuple(sorted(names + other_names))
                score = (
                    abs(5 * (annotated + other_annotated) - target_annotated),
                    abs(5 * (raw + other_raw) - target_raw),
                    _tie(seed, selected),
                )
                if best is None or score < best[0]:
                    best = score, selected
    if best is None:
        raise SplitError("Could not choose validation sequences")
    return best


def select_validation_sequences(
    candidates: Iterable[SequenceSplitCandidate], *, validation_count: int = 12, seed: int = 42
) -> tuple[str, ...]:
    """Return the globally lexicographically optimal validation sequence names."""

    values = tuple(sorted(candidates))
    if not values or len({item.name for item in values}) != len(values):
        raise SplitError("Candidates must be nonempty and have unique names")
    if not 0 < validation_count < len(values):
        raise SplitError("validation_count must be strictly between zero and candidate count")
    matrices, _capacities, _weather, _camera, cells = _quota_matrices(values, validation_count)
    scored = [_choose_for_matrix(values, matrix, cells, seed) for matrix in matrices]
    return min(scored)[1]


def derive_training_candidates(manifest: dict[str, Any], dataset_root: str | Path):
    """Extract coverage availability and expose only split-safe candidate records."""

    root = Path(dataset_root)
    entries = manifest.get("partitions", {}).get("train")
    if not isinstance(entries, list):
        raise SplitError("Manifest has no training partition")
    candidates = []
    coverage = []
    for entry in sorted(entries, key=lambda item: str(item["name"])):
        sequence = parse_sequence(root / entry["annotation_file"], root / entry["image_directory"], "train")
        if sequence.annotation_coverage.xml_frames_without_images:
            raise SplitError(f"XML-present frame lacks image: {sequence.name}")
        attributes = entry.get("sequence_attributes", {})
        candidate = SequenceSplitCandidate(
            name=str(entry["name"]),
            sence_weather=_category(attributes.get("sence_weather")),
            camera_state=_category(attributes.get("camera_state")),
            annotated_frame_count=len(sequence.annotation_coverage.xml_frame_numbers),
            frame_count=int(entry["frame_count"]),
        )
        candidates.append(candidate)
        coverage.append((candidate.name, candidate.annotated_frame_count))
    return tuple(candidates), hashlib.sha256(_canonical(coverage)).hexdigest()


def _proportion(
    source_category_count: int,
    validation_category_count: int,
    source_total: int,
    validation_total: int,
) -> dict[str, float | int]:
    target_proportion = source_category_count / source_total if source_total else 0.0
    achieved_proportion = validation_category_count / validation_total if validation_total else 0.0
    return {
        "source_count": source_category_count,
        "validation_count": validation_category_count,
        "target_proportion": target_proportion,
        "achieved_proportion": achieved_proportion,
        "percentage_point_deviation": 100 * (achieved_proportion - target_proportion),
    }


def _exposure(source: list[SequenceSplitCandidate], validation: list[SequenceSplitCandidate], field: str):
    total = sum(getattr(item, field) for item in source)
    achieved = sum(getattr(item, field) for item in validation)
    proportion = achieved / total if total else 0.0
    return {
        "source_total": total,
        "validation_total": achieved,
        "target_proportion": 0.2,
        "achieved_proportion": proportion,
        "percentage_point_deviation": 100 * (proportion - 0.2),
    }


def _summary(items: list[SequenceSplitCandidate], field: str) -> dict[str, float | int]:
    values = sorted(getattr(item, field) for item in items)
    return {
        "count": len(values),
        "minimum": values[0],
        "median": median(values),
        "mean": sum(values) / len(values),
        "maximum": values[-1],
    }


def build_split(manifest: dict[str, Any], dataset_root: str | Path, *, seed: int = 42) -> dict[str, Any]:
    candidates, coverage_fingerprint = derive_training_candidates(manifest, dataset_root)
    tests = manifest.get("partitions", {}).get("test")
    if len(candidates) != 60 or not isinstance(tests, list) or len(tests) != 40:
        raise SplitError("M3 requires exactly 60 official training and 40 official test sequences")
    validation_names = set(select_validation_sequences(candidates, seed=seed))
    validation = [item for item in candidates if item.name in validation_names]
    development = [item for item in candidates if item.name not in validation_names]
    roles = {
        "development_train": [asdict(item) for item in development],
        "validation": [asdict(item) for item in validation],
        "official_test": [{"name": str(item["name"])} for item in sorted(tests, key=lambda item: item["name"])],
    }
    weather = sorted({item.sence_weather for item in candidates})
    camera = sorted({item.camera_state for item in candidates})
    audit = {
        "role_counts": {role: len(items) for role, items in roles.items()},
        "intersections": {"development_validation": [], "development_official_test": [], "validation_official_test": []},
        "weather": {key: _proportion(sum(item.sence_weather == key for item in candidates), sum(item.sence_weather == key for item in validation), len(candidates), len(validation)) for key in weather},
        "camera_state": {key: _proportion(sum(item.camera_state == key for item in candidates), sum(item.camera_state == key for item in validation), len(candidates), len(validation)) for key in camera},
        "joint_strata": {},
        "raw_image_frame_exposure": _exposure(list(candidates), validation, "frame_count"),
        "annotated_frame_exposure": _exposure(list(candidates), validation, "annotated_frame_count"),
        "frame_summaries": {
            "official_train": {
                "raw_image_frames": _summary(list(candidates), "frame_count"),
                "annotated_frames": _summary(list(candidates), "annotated_frame_count"),
            },
            "validation": {
                "raw_image_frames": _summary(validation, "frame_count"),
                "annotated_frames": _summary(validation, "annotated_frame_count"),
            },
        },
    }
    for w in weather:
        for c in camera:
            source = [item for item in candidates if (item.sence_weather, item.camera_state) == (w, c)]
            if source:
                selected = sum(item in validation for item in source)
                data = _proportion(len(source), selected, len(candidates), len(validation))
                data["zero_validation_sequences"] = selected == 0
                audit["joint_strata"][f"{w}|{c}"] = data
    return {
        "schema_version": 1,
        "algorithm": {"name": "m3_annotation_aware_lexicographic", "seed": seed},
        "source_manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "annotation_coverage_fingerprint": coverage_fingerprint,
        "roles": roles,
        "audit": audit,
    }


def split_json_bytes(split: dict[str, Any]) -> bytes:
    return (json.dumps(split, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_split(split: dict[str, Any], output_path: str | Path, *, dataset_root: str | Path) -> Path:
    destination, raw_root = Path(output_path).resolve(), Path(dataset_root).resolve()
    if destination == raw_root or destination.is_relative_to(raw_root):
        raise SplitError(f"Split destination is inside raw dataset: {destination}")
    content = split_json_bytes(split)
    if destination.exists():
        if destination.read_bytes() == content:
            return destination
        raise SplitError(f"Existing split differs and will not be overwritten: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def validate_split_access(split: dict[str, Any], *, mode: str, role: str, sequences: Iterable[str] = (), allow_official_test: bool = False) -> tuple[str, ...]:
    roles = split.get("roles", {})
    if role not in {"development_train", "validation", "official_test"}:
        raise SplitError(f"Unknown split role: {role}")
    memberships = {name: assigned for assigned, entries in roles.items() for name in (str(entry["name"]) for entry in entries)}
    requested = tuple(sequences) or tuple(str(entry["name"]) for entry in roles.get(role, []))
    if any(name not in memberships for name in requested) or any(memberships[name] != role for name in requested):
        raise SplitError("Requested sequences do not belong to the requested role")
    if mode == "tuning" and (role == "official_test" or any(memberships[name] == "official_test" for name in requested)):
        raise SplitError("Tuning-mode access to official-test sequences is prohibited")
    if role == "official_test" and (mode != "evaluation" or not allow_official_test):
        raise SplitError("Official-test access requires evaluation mode and explicit acknowledgement")
    return requested
