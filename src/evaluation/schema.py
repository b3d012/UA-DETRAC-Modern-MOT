"""Canonical M4 tracker-result interchange validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass


class EvaluationInputError(ValueError):
    """Raised when a canonical evaluation input is malformed."""


@dataclass(frozen=True)
class TrackObservation:
    sequence: str
    frame_number: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float | None = None


def load_result_json(
    path: str, *, allowed_sequences: set[str]
) -> tuple[dict[str, object], tuple[TrackObservation, ...]]:
    """Load the version-1 project interchange without accepting implicit defaults."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version") != 1
        or payload.get("coordinate_convention") != "xyxy_half_open"
    ):
        raise EvaluationInputError("Expected schema_version=1 and xyxy_half_open coordinates")
    for key in ("producer", "role", "observations"):
        if key not in payload:
            raise EvaluationInputError(f"Missing required result field: {key}")
    if not isinstance(payload["producer"], dict) or not isinstance(payload["role"], str):
        raise EvaluationInputError("producer must be an object and role must be a string")
    points = payload.get("pr_operating_points")
    if points is not None:
        required = {"name", "threshold", "recall", "precision", "result_file"}
        if (
            not isinstance(points, list)
            or not points
            or any(not isinstance(point, dict) or not required <= set(point) for point in points)
        ):
            raise EvaluationInputError("Invalid PR operating-point metadata")
    observations = tuple(
        TrackObservation(
            str(item["sequence"]),
            int(item["frame_number"]),
            int(item["track_id"]),
            tuple(float(x) for x in item["bbox_xyxy"]),
            None if item.get("score") is None else float(item["score"]),
        )
        for item in payload["observations"]
    )
    return payload, validate_observations(observations, allowed_sequences=allowed_sequences)


def validate_observations(
    observations: Iterable[TrackObservation], *, allowed_sequences: set[str]
) -> tuple[TrackObservation, ...]:
    values = tuple(observations)
    seen: set[tuple[str, int, int]] = set()
    for item in values:
        if item.sequence not in allowed_sequences:
            raise EvaluationInputError(f"Unknown sequence: {item.sequence}")
        if item.frame_number < 1 or item.track_id < 1:
            raise EvaluationInputError("frame_number and track_id must be positive integers")
        x1, y1, x2, y2 = item.bbox_xyxy
        if not all(math.isfinite(value) for value in item.bbox_xyxy) or x2 <= x1 or y2 <= y1:
            raise EvaluationInputError("bbox_xyxy must be finite with positive area")
        if item.score is not None and (not math.isfinite(item.score) or not 0 <= item.score <= 1):
            raise EvaluationInputError("score must be in [0, 1]")
        key = (item.sequence, item.frame_number, item.track_id)
        if key in seen:
            raise EvaluationInputError(f"Duplicate observation: {key}")
        seen.add(key)
    return tuple(sorted(values, key=lambda item: (item.sequence, item.frame_number, item.track_id)))
