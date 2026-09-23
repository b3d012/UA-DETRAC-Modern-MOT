from __future__ import annotations

import pytest

from evaluation.native import NativePrerequisiteError, export_native_matrices
from evaluation.schema import TrackObservation


def test_native_export_applies_ignore_once(tmp_path) -> None:
    result = export_native_matrices(
        tmp_path,
        sequence="SEQ",
        frame_count=2,
        observations=(TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),),
        ignored_regions=((0, 0, 10, 10),),
        image_size=(20, 20),
        native_ignore_filter_state="not_applied",
    )
    assert result["suppressed_observations"] == 1
    assert (tmp_path / "SEQ_LX.txt").read_text() == "\n\n"


def test_native_export_requires_filter_provenance(tmp_path) -> None:
    with pytest.raises(NativePrerequisiteError, match="native_ignore_filter_state"):
        export_native_matrices(
            tmp_path,
            sequence="SEQ",
            frame_count=1,
            observations=(),
            ignored_regions=(),
            image_size=(1, 1),
            native_ignore_filter_state="",
        )


def test_already_filtered_requires_provenance_and_never_filters_again(tmp_path) -> None:
    observation = TrackObservation("SEQ", 1, 1, (0, 0, 10, 10))
    with pytest.raises(NativePrerequisiteError, match="upstream"):
        export_native_matrices(
            tmp_path,
            sequence="SEQ",
            frame_count=1,
            observations=(observation,),
            ignored_regions=((0, 0, 10, 10),),
            image_size=(20, 20),
            native_ignore_filter_state="already_applied",
        )
    result = export_native_matrices(
        tmp_path,
        sequence="SEQ",
        frame_count=1,
        observations=(observation,),
        ignored_regions=((0, 0, 10, 10),),
        image_size=(20, 20),
        native_ignore_filter_state="already_applied",
        already_applied_provenance={"component": "official_trackingEvaluation"},
    )
    assert result["suppressed_observations"] == 0
