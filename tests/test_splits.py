from __future__ import annotations

from pathlib import Path

import pytest

from datasets.splits import (
    SequenceSplitCandidate,
    SplitError,
    build_split,
    split_json_bytes,
    validate_split_access,
    write_split,
)


def test_optimizer_accepts_only_sequence_level_split_candidates() -> None:
    from datasets.splits import SequenceSplitCandidate, select_validation_sequences

    candidates = (
        SequenceSplitCandidate("A", "sunny", "stable", 10, 12),
        SequenceSplitCandidate("B", "sunny", "unstable", 10, 12),
        SequenceSplitCandidate("C", "rainy", "stable", 10, 12),
        SequenceSplitCandidate("D", "rainy", "unstable", 10, 12),
        SequenceSplitCandidate("E", "sunny", "stable", 10, 12),
    )

    selected = select_validation_sequences(candidates, validation_count=1, seed=42)

    assert len(selected) == 1
    assert selected[0] in {candidate.name for candidate in candidates}


def test_optimizer_prefers_annotated_coverage_before_raw_frame_balance() -> None:
    from datasets.splits import select_validation_sequences

    candidates = (
        SequenceSplitCandidate("A", "sunny", "stable", 1, 100),
        SequenceSplitCandidate("B", "sunny", "stable", 9, 1),
        SequenceSplitCandidate("C", "sunny", "stable", 10, 1),
        SequenceSplitCandidate("D", "sunny", "stable", 10, 1),
        SequenceSplitCandidate("E", "sunny", "stable", 10, 1),
    )

    assert select_validation_sequences(candidates, validation_count=1, seed=42) == ("B",)


def test_split_audit_reports_target_achieved_and_zero_joint_stratum(
    monkeypatch, tmp_path: Path
) -> None:
    candidates = tuple(
        SequenceSplitCandidate(
            f"TRAIN_{index:03d}",
            "sunny" if index < 38 else "rainy",
            "stable" if index % 2 == 0 else "unstable",
            1,
            1,
        )
        for index in range(60)
    )
    from datasets import splits

    monkeypatch.setattr(
        splits, "derive_training_candidates", lambda _manifest, _root: (candidates, "coverage")
    )
    monkeypatch.setattr(
        splits,
        "select_validation_sequences",
        lambda _candidates, **_kwargs: tuple(item.name for item in candidates[:12]),
    )
    manifest = {
        "partitions": {
            "train": [{}] * 60,
            "test": [{"name": f"TEST_{index:03d}"} for index in range(40)],
        }
    }

    split = build_split(manifest, tmp_path)
    sunny = split["audit"]["weather"]["sunny"]
    rainy_joint = split["audit"]["joint_strata"]["rainy|unstable"]
    assert sunny["target_proportion"] == pytest.approx(38 / 60)
    assert sunny["achieved_proportion"] == 1.0
    assert sunny["percentage_point_deviation"] == pytest.approx(100 * (1 - 38 / 60))
    assert rainy_joint["zero_validation_sequences"] is True


def test_access_guard_rejects_tuning_official_test() -> None:
    split = {
        "roles": {
            "development_train": [{"name": "TRAIN"}],
            "validation": [{"name": "VALID"}],
            "official_test": [{"name": "TEST"}],
        }
    }

    with pytest.raises(SplitError, match="Tuning-mode"):
        validate_split_access(split, mode="tuning", role="official_test")
    with pytest.raises(SplitError, match="acknowledgement"):
        validate_split_access(split, mode="evaluation", role="official_test")
    assert validate_split_access(
        split, mode="evaluation", role="official_test", allow_official_test=True
    ) == ("TEST",)


def test_frozen_split_refuses_different_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    split = {"roles": {"development_train": [], "validation": [], "official_test": []}}
    assert write_split(split, output, dataset_root=tmp_path / "raw") == output.resolve()
    assert output.read_bytes() == split_json_bytes(split)
    with pytest.raises(SplitError, match="will not be overwritten"):
        write_split(
            {"roles": {"development_train": [{"name": "changed"}]}},
            output,
            dataset_root=tmp_path / "raw",
        )
