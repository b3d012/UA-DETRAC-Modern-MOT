"""Verified UA-DETRAC legacy ignored-region filtering."""

from __future__ import annotations

import math

from evaluation.schema import TrackObservation

LEGACY_IGNORE_POLICY = "ua_detrac_legacy_drop_tracks"


def _matlab_round(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def filter_ignored_observations(
    observations: tuple[TrackObservation, ...],
    *,
    ignored_regions: tuple[tuple[float, float, float, float], ...],
    image_size: tuple[int, int],
    policy: str,
) -> tuple[tuple[TrackObservation, ...], int]:
    """Apply the configured observation-level ignored-region policy.

    The legacy threshold and coordinate rounding mirror ``dropTracks.m``. M4
    uses this observation-level routine symmetrically for modern GT and tracker
    inputs; native ``parseGT.m`` has additional trajectory-span behavior and is
    therefore not claimed to be identical to this modern policy.
    """

    if policy == "none":
        return observations, 0
    if policy != LEGACY_IGNORE_POLICY:
        raise ValueError(f"Unknown ignore policy: {policy}")
    _width, _height = image_size
    kept: list[TrackObservation] = []
    suppressed = 0
    for item in observations:
        x1, y1, x2, y2 = item.bbox_xyxy
        left, top = _matlab_round(x1), _matlab_round(y1)
        box_width, box_height = _matlab_round(x2 - x1), _matlab_round(y2 - y1)
        if box_width <= 0 or box_height <= 0:
            kept.append(item)
            continue
        covered = 0.0
        for rx1, ry1, rx2, ry2 in ignored_regions:
            ix1, iy1 = max(left, rx1), max(top, ry1)
            ix2, iy2 = min(left + box_width, rx2), min(top + box_height, ry2)
            covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if covered / (box_width * box_height) >= 0.5:
            suppressed += 1
        else:
            kept.append(item)
    return tuple(kept), suppressed


def filter_ignored_predictions(
    observations: tuple[TrackObservation, ...],
    *,
    ignored_regions: tuple[tuple[float, float, float, float], ...],
    image_size: tuple[int, int],
    policy: str,
) -> tuple[tuple[TrackObservation, ...], int]:
    """Compatibility name for native tracker-output filtering callers."""
    return filter_ignored_observations(
        observations, ignored_regions=ignored_regions, image_size=image_size, policy=policy
    )
