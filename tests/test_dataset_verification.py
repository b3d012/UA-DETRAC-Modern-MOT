from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def _verification_module():
    try:
        return importlib.import_module("datasets.verification")
    except ModuleNotFoundError as exc:
        pytest.fail(f"datasets.verification must exist: {exc}")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_verify_git_safety_accepts_ignored_untracked_raw_data(tmp_path: Path) -> None:
    verification = _verification_module()
    repo = tmp_path / "repo"
    raw = repo / "data" / "ua_detrac"
    raw.mkdir(parents=True)
    (raw / "frame.jpg").write_bytes(b"raw")
    (repo / ".gitignore").write_text("/data/ua_detrac/\n", encoding="utf-8")
    _git(repo, "init")

    verification.verify_git_safety(repo, raw)


def test_verify_git_safety_accepts_absent_ignored_raw_data_directory(tmp_path: Path) -> None:
    verification = _verification_module()
    repo = tmp_path / "repo"
    raw = repo / "data" / "ua_detrac"
    repo.mkdir()
    (repo / ".gitignore").write_text("/data/ua_detrac/\n", encoding="utf-8")
    _git(repo, "init")

    verification.verify_git_safety(repo, raw)
    assert not raw.exists()


def test_verify_git_safety_rejects_absent_non_ignored_raw_data_directory(tmp_path: Path) -> None:
    verification = _verification_module()
    repo = tmp_path / "repo"
    raw = repo / "data" / "ua_detrac"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    _git(repo, "init")

    with pytest.raises(verification.GitSafetyError, match="not ignored"):
        verification.verify_git_safety(repo, raw)


def test_verify_git_safety_rejects_force_added_raw_data(tmp_path: Path) -> None:
    verification = _verification_module()
    repo = tmp_path / "repo"
    raw = repo / "data" / "ua_detrac"
    raw.mkdir(parents=True)
    (raw / "frame.jpg").write_bytes(b"raw")
    (repo / ".gitignore").write_text("/data/ua_detrac/\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", "-f", "data/ua_detrac/frame.jpg")

    with pytest.raises(verification.GitSafetyError, match="tracked or staged"):
        verification.verify_git_safety(repo, raw)


def test_verify_dataset_cli_writes_manifest_and_summary(
    raw_dataset_factory, tmp_path: Path
) -> None:
    train = tuple(f"TRAIN_{number:03d}" for number in range(60))
    test = tuple(f"TEST_{number:03d}" for number in range(40))
    root = raw_dataset_factory(train=train, test=test, frame_numbers=(1,))
    config = tmp_path / "dataset.yaml"
    output = tmp_path / "manifest.json"
    config.write_text(
        """dataset:
  root_env: UA_DETRAC_ROOT
  root: null
  layout:
    images_dir: DETRAC-Images
    train_annotations_dir: DETRAC-Train-Annotations-XML
    test_annotations_dir: DETRAC-Test-Annotations-XML
    toolkit_glob: DETRAC-Toolkit*
  expected:
    train_sequences: 60
    test_sequences: 40
  manifest_path: unused.json
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_dataset.py",
            "--root",
            str(root),
            "--config",
            str(config),
            "--manifest",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "Train sequences: 60",
        "Test sequences: 40",
        "Sequences with missing frames: 0",
        "Sequences without annotations: 0",
        f"Manifest: {output.resolve()}",
        "STATUS: PASS",
    ]
    assert output.is_file()


def test_verify_dataset_cli_rejects_non_official_configured_counts(
    raw_dataset_factory, tmp_path: Path
) -> None:
    root = raw_dataset_factory()
    config = tmp_path / "dataset.yaml"
    config.write_text(
        """dataset:
  root_env: UA_DETRAC_ROOT
  expected:
    train_sequences: 1
    test_sequences: 1
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "scripts/verify_dataset.py", "--root", str(root), "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "exactly 60 train and 40 test" in completed.stderr
    assert "STATUS: FAIL" in completed.stderr
