from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _paths_module():
    try:
        return importlib.import_module("datasets.paths")
    except ModuleNotFoundError as exc:
        pytest.fail(f"datasets.paths must exist: {exc}")


def _make_layout(root: Path, *, toolkit_names: tuple[str, ...] = ("DETRAC-Toolkits",)) -> None:
    (root / "DETRAC-Images").mkdir(parents=True)
    (root / "DETRAC-Train-Annotations-XML").mkdir()
    (root / "DETRAC-Test-Annotations-XML").mkdir()
    for name in toolkit_names:
        (root / name).mkdir()


def test_resolve_dataset_root_uses_cli_then_config_then_environment(tmp_path: Path) -> None:
    paths = _paths_module()
    cli_root = tmp_path / "cli"
    config_root = tmp_path / "config"
    env_root = tmp_path / "env"
    for root in (cli_root, config_root, env_root):
        root.mkdir()

    assert (
        paths.resolve_dataset_root(cli_root, config_root, environ={"UA_DETRAC_ROOT": str(env_root)})
        == cli_root.resolve()
    )
    assert (
        paths.resolve_dataset_root(None, config_root, environ={"UA_DETRAC_ROOT": str(env_root)})
        == config_root.resolve()
    )
    assert (
        paths.resolve_dataset_root(None, None, environ={"UA_DETRAC_ROOT": str(env_root)})
        == env_root.resolve()
    )


def test_resolve_dataset_root_rejects_missing_or_invalid_root(tmp_path: Path) -> None:
    paths = _paths_module()

    with pytest.raises(paths.DatasetLayoutError, match="UA_DETRAC_ROOT"):
        paths.resolve_dataset_root(None, None, environ={})
    with pytest.raises(paths.DatasetLayoutError, match="does not exist"):
        paths.resolve_dataset_root(tmp_path / "missing", None, environ={})


def test_detect_layout_accepts_downloaded_toolkits_name(tmp_path: Path) -> None:
    paths = _paths_module()
    _make_layout(tmp_path)

    layout = paths.detect_layout(tmp_path)

    assert layout.root == tmp_path.resolve()
    assert layout.images_dir.name == "DETRAC-Images"
    assert layout.train_annotations_dir.name == "DETRAC-Train-Annotations-XML"
    assert layout.test_annotations_dir.name == "DETRAC-Test-Annotations-XML"
    assert layout.toolkit_dir.name == "DETRAC-Toolkits"


def test_detect_layout_matches_toolkit_case_insensitively(tmp_path: Path) -> None:
    paths = _paths_module()
    _make_layout(tmp_path, toolkit_names=("DETRAC-toolkit",))

    assert paths.detect_layout(tmp_path).toolkit_dir.name == "DETRAC-toolkit"


def test_detect_layout_rejects_missing_and_ambiguous_assets(tmp_path: Path) -> None:
    paths = _paths_module()
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(paths.DatasetLayoutError, match="DETRAC-Images"):
        paths.detect_layout(missing_root)

    ambiguous_root = tmp_path / "ambiguous"
    _make_layout(ambiguous_root, toolkit_names=("DETRAC-Toolkit", "DETRAC-Toolkits"))
    with pytest.raises(paths.DatasetLayoutError, match="multiple toolkit"):
        paths.detect_layout(ambiguous_root)
