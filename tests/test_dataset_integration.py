from __future__ import annotations

import os
from pathlib import Path

import pytest

from datasets.manifest import build_manifest
from datasets.paths import detect_layout


def test_local_dataset_satisfies_m1_contract() -> None:
    repository = Path(__file__).resolve().parents[1]
    configured = os.environ.get("UA_DETRAC_ROOT")
    root = Path(configured) if configured else repository / "data" / "ua_detrac"
    if not root.is_dir():
        pytest.skip("UA-DETRAC raw dataset is not available")

    manifest = build_manifest(
        detect_layout(root), expected_train_sequences=60, expected_test_sequences=40
    )

    assert manifest["summary"]["train_sequences"] == 60
    assert manifest["summary"]["test_sequences"] == 40
    assert manifest["summary"]["sequences_with_missing_frames"] == 0
    assert manifest["summary"]["sequences_without_annotations"] == 0
    for partition in manifest["partitions"].values():
        for sequence in partition:
            assert not Path(sequence["image_directory"]).is_absolute()
            assert not Path(sequence["annotation_file"]).is_absolute()
            assert sequence["image_dimensions"]["width"] > 0
            assert sequence["image_dimensions"]["height"] > 0
