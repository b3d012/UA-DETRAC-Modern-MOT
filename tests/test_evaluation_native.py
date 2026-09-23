from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.native import NativePrerequisiteError, inspect_native_toolkit


def test_native_inspector_requires_executable(tmp_path: Path) -> None:
    with pytest.raises(NativePrerequisiteError, match="DETRAC_MOT_EVAL.exe"):
        inspect_native_toolkit(tmp_path)


def test_native_inspector_fingerprints_required_assets(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    ignored = evaluation / "igrs"
    ignored.mkdir(parents=True)
    (evaluation / "DETRAC_MOT_EVAL.exe").write_bytes(b"exe")
    (evaluation / "trackingEvaluation.m").write_text(
        "dropTracks(stateInfo, curSequence);", encoding="utf-8"
    )
    (evaluation / "printFinalEvaluation.m").write_text("DETRAC_MOT_EVAL.exe", encoding="utf-8")
    (ignored / "SEQ_IgR.txt").write_text("1,2,3,4\n", encoding="utf-8")

    report = inspect_native_toolkit(tmp_path)

    assert report["native_ground_truth"] == "resolved_by_official_executable_or_workflow"
    assert report["ignored_region_file_count"] == 1
    assert report["drop_tracks_stage"] == "pre_executable"
    assert report["native_pr_execution_prerequisite"]["status"] == "blocked_external_prerequisite"
