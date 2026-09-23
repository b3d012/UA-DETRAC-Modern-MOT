from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.coverage import segment_annotated_frames
from evaluation.ignores import filter_ignored_observations, filter_ignored_predictions
from evaluation.modern import run_trackeval, stage_trackeval_mot
from evaluation.schema import TrackObservation


def test_trackeval_staging_uses_segment_ids_and_compact_frames(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1, 2, 4))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    pred = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)

    metadata = stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)

    assert metadata["annotation_gap_policy"] == "contiguous_annotated_segments_reset_identity"
    assert (tmp_path / "gt" / segment.segment_id / "gt" / "gt.txt").read_text().startswith("1,1")


def test_trackeval_executes_perfect_single_frame_fixture(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    observations = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    stage_trackeval_mot(
        tmp_path, segments=(segment,), ground_truth=observations, predictions=observations
    )

    result = run_trackeval(tmp_path)

    assert result["metrics"]["HOTA"] == 1.0
    assert result["metrics"]["MOTA"] == 1.0


def test_trackeval_empty_prediction_has_one_false_negative(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=())

    result = run_trackeval(tmp_path)

    assert result["metrics"]["CLR_FN"] == 1
    assert result["metrics"]["CLR_FP"] == 0
    assert result["metrics"]["MOTA"] == 0.0


def test_trackeval_false_positive_is_counted(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    pred = gt + (TrackObservation("SEQ", 1, 2, (20, 20, 30, 30)),)
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)

    result = run_trackeval(tmp_path)

    assert result["metrics"]["CLR_FP"] == 1
    assert result["metrics"]["MOTA"] == 0.0


def test_trackeval_miss_has_exact_count_and_mota(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1, 2, 3))[0]
    gt = tuple(TrackObservation("SEQ", frame, 1, (0, 0, 10, 10)) for frame in (1, 2, 3))
    pred = (gt[0], gt[2])
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)
    result = run_trackeval(tmp_path)["metrics"]
    assert (result["CLR_FN"], result["CLR_FP"], result["IDSW"]) == (1, 0, 0)
    assert result["MOTA"] == pytest.approx(1 - 1 / 3)
    assert result["HOTA"] < 1 and result["DetA"] < 1 and result["IDF1"] < 1


def test_trackeval_identity_change_is_one_switch(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1, 2, 3))[0]
    gt = tuple(TrackObservation("SEQ", frame, 1, (0, 0, 10, 10)) for frame in (1, 2, 3))
    pred = (
        gt[0],
        TrackObservation("SEQ", 2, 2, (0, 0, 10, 10)),
        TrackObservation("SEQ", 3, 2, (0, 0, 10, 10)),
    )
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)
    result = run_trackeval(tmp_path)["metrics"]
    assert (result["IDSW"], result["CLR_FP"], result["CLR_FN"]) == (1, 0, 0)
    assert result["MOTA"] < 1 and result["AssA"] < 1 and result["IDF1"] < 1 and result["HOTA"] < 1


def test_detection_gap_with_same_id_is_not_an_identity_switch(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1, 2, 3))[0]
    gt = tuple(TrackObservation("SEQ", frame, 1, (0, 0, 10, 10)) for frame in (1, 2, 3))
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=(gt[0], gt[2]))
    result = run_trackeval(tmp_path)["metrics"]
    assert result["CLR_FN"] == 1
    assert result["IDSW"] == 0


def test_annotation_gap_resets_identity_association_and_keeps_segment_metadata(
    tmp_path: Path,
) -> None:
    # Same physical identity changes tracker ID across unknown frame 2. M4's
    # explicit segment policy must not infer an ID switch or association there.
    segments = segment_annotated_frames("SEQ", (1, 3))
    gt = (
        TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),
        TrackObservation("SEQ", 3, 1, (0, 0, 10, 10)),
    )
    pred = (
        TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),
        TrackObservation("SEQ", 3, 2, (0, 0, 10, 10)),
    )
    metadata = stage_trackeval_mot(tmp_path, segments=segments, ground_truth=gt, predictions=pred)
    result = run_trackeval(tmp_path)["metrics"]
    assert result["IDSW"] == 0
    assert metadata["segments"] == [
        {
            "segment_id": "SEQ__annseg_001_00001_00001",
            "sequence": "SEQ",
            "original_frame_numbers": [1],
        },
        {
            "segment_id": "SEQ__annseg_002_00003_00003",
            "sequence": "SEQ",
            "original_frame_numbers": [3],
        },
    ]


def test_imperfect_localization_lowers_loca_without_miss(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    pred = (TrackObservation("SEQ", 1, 1, (2, 0, 12, 10)),)
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)
    result = run_trackeval(tmp_path)["metrics"]
    assert result["CLR_FN"] == result["CLR_FP"] == 0
    assert 0 < result["LocA"] < 1


def test_localization_below_clear_threshold_creates_miss_and_fp(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    pred = (TrackObservation("SEQ", 1, 1, (8, 0, 18, 10)),)
    stage_trackeval_mot(tmp_path, segments=(segment,), ground_truth=gt, predictions=pred)
    result = run_trackeval(tmp_path)["metrics"]
    assert (result["CLR_FN"], result["CLR_FP"]) == (1, 1)


def test_legacy_ignore_policy_suppresses_fp_while_none_does_not(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1,))[0]
    gt = (TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),)
    predictions = gt + (TrackObservation("SEQ", 1, 2, (20, 0, 30, 10)),)
    suppressed, count = filter_ignored_predictions(
        predictions,
        ignored_regions=((20, 0, 30, 10),),
        image_size=(40, 20),
        policy="ua_detrac_legacy_drop_tracks",
    )
    untouched, untouched_count = filter_ignored_predictions(
        predictions, ignored_regions=((20, 0, 30, 10),), image_size=(40, 20), policy="none"
    )
    stage_trackeval_mot(
        tmp_path / "legacy", segments=(segment,), ground_truth=gt, predictions=suppressed
    )
    stage_trackeval_mot(
        tmp_path / "none", segments=(segment,), ground_truth=gt, predictions=untouched
    )
    assert count == 1 and untouched_count == 0
    assert run_trackeval(tmp_path / "legacy")["metrics"]["CLR_FP"] == 0
    assert run_trackeval(tmp_path / "none")["metrics"]["CLR_FP"] == 1


def test_symmetric_modern_ignore_removes_ignored_gt_and_prediction_together(tmp_path: Path) -> None:
    segment = segment_annotated_frames("SEQ", (1, 2))[0]
    gt = (
        TrackObservation("SEQ", 1, 1, (0, 0, 10, 10)),
        TrackObservation("SEQ", 2, 1, (20, 0, 30, 10)),
    )
    predictions = gt
    kwargs = {
        "ignored_regions": ((20, 0, 30, 10),),
        "image_size": (40, 20),
        "policy": "ua_detrac_legacy_drop_tracks",
    }
    filtered_gt, gt_count = filter_ignored_observations(gt, **kwargs)
    filtered_predictions, prediction_count = filter_ignored_observations(predictions, **kwargs)
    stage_trackeval_mot(
        tmp_path / "legacy",
        segments=(segment,),
        ground_truth=filtered_gt,
        predictions=filtered_predictions,
    )
    legacy = run_trackeval(tmp_path / "legacy")["metrics"]
    untouched_gt, _ = filter_ignored_observations(gt, **{**kwargs, "policy": "none"})
    untouched_predictions, _ = filter_ignored_observations(
        predictions, **{**kwargs, "policy": "none"}
    )
    stage_trackeval_mot(
        tmp_path / "none",
        segments=(segment,),
        ground_truth=untouched_gt,
        predictions=untouched_predictions,
    )
    assert (gt_count, prediction_count) == (1, 1)
    assert legacy["MOTA"] == legacy["HOTA"] == 1.0
    assert run_trackeval(tmp_path / "none")["metrics"]["MOTA"] == 1.0
