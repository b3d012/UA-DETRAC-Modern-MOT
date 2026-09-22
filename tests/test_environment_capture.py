from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _environment_module():
    try:
        return importlib.import_module("reproducibility.environment")
    except ModuleNotFoundError as exc:
        pytest.fail(f"reproducibility.environment must exist: {exc}")


def test_capture_environment_contains_reproducibility_contract(tmp_path: Path) -> None:
    environment = _environment_module()
    repo = Path(__file__).resolve().parents[1]
    first_config = tmp_path / "a.yaml"
    second_config = tmp_path / "b.yaml"
    first_config.write_text("value: 1\n", encoding="utf-8")
    second_config.write_text("value: 2\n", encoding="utf-8")
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    captured = environment.capture_environment(
        repo,
        config_paths=[second_config, first_config],
        seed=42,
        command=["python", "scripts/capture_environment.py"],
        now=instant,
    )

    assert captured["timestamp_utc"] == "2026-01-02T03:04:05Z"
    assert captured["seed"] == 42
    assert captured["command"] == ["python", "scripts/capture_environment.py"]
    assert len(captured["config_hash_sha256"]) == 64
    assert [Path(path).name for path in captured["config_files"]] == ["a.yaml", "b.yaml"]
    assert captured["experiment_id"].endswith(captured["config_hash_sha256"][:12])
    assert set(captured["git"]) == {"commit", "dirty"}
    assert captured["python"]["version"]
    assert captured["system"]["os"]
    assert captured["cpu"]["logical_cores"]
    assert captured["cpu"]["physical_cores"]
    assert "torch" in captured["packages"]
    assert captured["packages"]["psutil"]
    assert set(captured["accelerator"]) >= {
        "cuda_available",
        "cuda_runtime",
        "cuda_toolkit",
        "cudnn_version",
        "gpus",
        "nvidia_driver",
    }


def test_environment_config_hash_is_order_independent(tmp_path: Path) -> None:
    environment = _environment_module()
    repo = Path(__file__).resolve().parents[1]
    first = tmp_path / "one.yaml"
    second = tmp_path / "two.yaml"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")

    assert environment.hash_configs([first, second], repo) == environment.hash_configs(
        [second, first], repo
    )


def test_capture_environment_script_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    completed = subprocess.run(
        [sys.executable, "scripts/capture_environment.py", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"Environment: {output.resolve()}\n"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["git"]["commit"]
    assert payload["config_files"] == [
        "configs/dataset/ua_detrac.yaml",
        "configs/experiments/m1_foundation.yaml",
        "configs/reproducibility.yaml",
    ]


def test_write_environment_rejects_destination_inside_protected_root(tmp_path: Path) -> None:
    environment = _environment_module()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    destination = raw_root / "annotation.xml"

    with pytest.raises(environment.EnvironmentOutputError, match="protected raw-data"):
        environment.write_environment(
            {"schema_version": 1}, destination, protected_roots=[raw_root]
        )

    assert not destination.exists()


def test_write_environment_rejects_symlink_to_protected_root(tmp_path: Path) -> None:
    environment = _environment_module()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    link = tmp_path / "raw-link"
    try:
        link.symlink_to(raw_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(environment.EnvironmentOutputError, match="protected raw-data"):
        environment.write_environment(
            {"schema_version": 1}, link / "environment.json", protected_roots=[raw_root]
        )


def test_write_environment_requires_explicit_overwrite(tmp_path: Path) -> None:
    environment = _environment_module()
    output = tmp_path / "environment.json"
    environment.write_environment({"value": 1}, output)

    with pytest.raises(environment.EnvironmentOutputError, match="already exists"):
        environment.write_environment({"value": 2}, output)

    environment.write_environment({"value": 2}, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
