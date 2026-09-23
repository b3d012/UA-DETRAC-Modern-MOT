from __future__ import annotations

import json

import pytest

from evaluation.coverage import segment_annotated_frames
from evaluation.ignores import filter_ignored_predictions
from evaluation.schema import (
    EvaluationInputError,
    TrackObservation,
    load_result_json,
    validate_observations,
)


def test_observation_validation_rejects_duplicate_track_frame() -> None:
    observations = (
        TrackObservation("SEQ", 1, 7, (0, 0, 10, 10)),
        TrackObservation("SEQ", 1, 7, (1, 1, 11, 11)),
    )

    with pytest.raises(EvaluationInputError, match="Duplicate"):
        validate_observations(observations, allowed_sequences={"SEQ"})


def test_canonical_json_loader_requires_versioned_metadata(tmp_path) -> None:
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_convention": "xyxy_half_open",
                "producer": {"name": "fixture", "version": "1"},
                "role": "validation",
                "observations": [
                    {"sequence": "SEQ", "frame_number": 1, "track_id": 1, "bbox_xyxy": [0, 0, 1, 1]}
                ],
            }
        ),
        encoding="utf-8",
    )
    payload, observations = load_result_json(str(source), allowed_sequences={"SEQ"})
    assert payload["producer"]["name"] == "fixture" and observations[0].track_id == 1


def test_coverage_segmentation_keeps_gap_as_two_deterministic_segments() -> None:
    segments = segment_annotated_frames("SEQ", (1, 2, 4, 5))

    assert [item.segment_id for item in segments] == [
        "SEQ__annseg_001_00001_00002",
        "SEQ__annseg_002_00004_00005",
    ]
    assert [item.frame_numbers for item in segments] == [(1, 2), (4, 5)]


def test_gap_prevents_cross_gap_identity_continuity() -> None:
    segments = segment_annotated_frames("SEQ", (1, 3))
    assert [segment.segment_id for segment in segments] == [
        "SEQ__annseg_001_00001_00001",
        "SEQ__annseg_002_00003_00003",
    ]


def test_verified_ignore_filter_removes_box_once_and_preserves_partial_overlap() -> None:
    observations = (
        TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),
        TrackObservation("SEQ", 1, 2, (8, 0, 18, 10)),
    )

    filtered, suppressed = filter_ignored_predictions(
        observations,
        ignored_regions=((0, 0, 10, 10),),
        image_size=(20, 20),
        policy="ua_detrac_legacy_drop_tracks",
    )

    assert [item.track_id for item in filtered] == [2]
    assert suppressed == 1
