from __future__ import annotations

from pathlib import Path

from evaluation.native import invoke_native_pr, parse_pr_result


def test_parse_official_pr_result_fixture(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    result.write_text("-1 10 20 0.1 1 2 3 4 5 6 7 8 9 10\n", encoding="utf-8")
    assert parse_pr_result(result)["pr_mota"] == 8


def test_native_runner_invokes_stub_and_parses_output(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    output.write_text("-1 10 20 0.1 1 2 3 4 5 6 7 8 9 10\n", encoding="utf-8")
    executable = tmp_path / "stub.cmd"
    executable.write_text("@exit /b 0\n", encoding="utf-8")
    result = invoke_native_pr(executable, [], output)
    assert result["result_family"] == "ua_detrac_pr"
