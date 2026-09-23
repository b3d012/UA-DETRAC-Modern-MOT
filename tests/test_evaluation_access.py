from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_rejects_official_test_without_explicit_acknowledgement(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "roles": {
                    "development_train": [],
                    "validation": [],
                    "official_test": [{"name": "TEST"}],
                }
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "evaluate_tracking.py"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "trackeval",
            "--role",
            "official_test",
            "--sequence",
            "TEST",
            "--split-path",
            str(split),
            "--output",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert done.returncode == 2
    assert "Official-test access requires" in done.stderr


def test_synthetic_all_artifact_contains_every_approved_fixture(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "evaluate_tracking.py"
    output = tmp_path / "synthetic"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "trackeval",
            "--synthetic-fixtures",
            "--fixture-suite",
            "all",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    fixtures = json.loads((output / "result.json").read_text())["synthetic_fixture_metrics"]
    assert set(fixtures) == {
        "perfect",
        "empty",
        "false_positive",
        "miss",
        "id_switch",
        "detection_gap_fragmentation_diagnostic",
        "matched_imperfect_localization",
        "localization_crosses_matching_threshold",
        "ignored_region_ua_detrac_legacy_drop_tracks",
        "ignored_region_none",
        "interior_annotation_gap_identity_reset",
    }
    assert "Synthetic fixtures" in (output / "summary.md").read_text()
