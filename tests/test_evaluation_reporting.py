from __future__ import annotations

import json

import pytest

from evaluation.reporting import write_result


def test_result_documents_are_deterministic_and_collision_safe(tmp_path) -> None:
    output = tmp_path / "result"
    write_result(
        output,
        {
            "result_family": "modern_mot",
            "metric_scale": "fraction_0_1",
            "metrics": {"MOTA": 1.0, "CLR_FP": 0},
        },
    )
    data = json.loads((output / "result.json").read_text())
    assert data["metric_scale"] == "fraction_0_1"
    assert "Result family" in (output / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        write_result(output, {"result_family": "modern_mot"})
