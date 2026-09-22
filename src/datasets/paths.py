"""Dataset-root resolution and read-only UA-DETRAC layout discovery."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path


class DatasetLayoutError(ValueError):
    """Raised when the configured dataset root or layout is invalid."""


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved locations of the raw UA-DETRAC assets used by M1."""

    root: Path
    images_dir: Path
    train_annotations_dir: Path
    test_annotations_dir: Path
    toolkit_dir: Path


def resolve_dataset_root(
    cli_root: str | Path | None,
    config_root: str | Path | None,
    *,
    env_var: str = "UA_DETRAC_ROOT",
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the dataset root using CLI, config, then environment precedence."""

    environment = os.environ if environ is None else environ
    selected = cli_root or config_root or environment.get(env_var)
    if selected is None or not str(selected).strip():
        raise DatasetLayoutError(
            f"Dataset root is required; pass --root, configure dataset.root, or set {env_var}."
        )

    root = Path(selected).expanduser().resolve()
    if not root.exists():
        raise DatasetLayoutError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise DatasetLayoutError(f"Dataset root is not a directory: {root}")
    return root


def _required_directory(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_dir():
        raise DatasetLayoutError(f"Required dataset directory is missing: {name}")
    return path


def detect_layout(
    root: str | Path,
    *,
    images_dir_name: str = "DETRAC-Images",
    train_annotations_dir_name: str = "DETRAC-Train-Annotations-XML",
    test_annotations_dir_name: str = "DETRAC-Test-Annotations-XML",
    toolkit_glob: str = "DETRAC-Toolkit*",
) -> DatasetLayout:
    """Detect the supported raw UA-DETRAC layout without changing it."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise DatasetLayoutError(f"Dataset root is not a directory: {resolved}")

    images = _required_directory(resolved, images_dir_name)
    train_annotations = _required_directory(resolved, train_annotations_dir_name)
    test_annotations = _required_directory(resolved, test_annotations_dir_name)
    folded_glob = toolkit_glob.casefold()
    toolkit_candidates = sorted(
        path
        for path in resolved.iterdir()
        if path.is_dir() and fnmatchcase(path.name.casefold(), folded_glob)
    )
    if not toolkit_candidates:
        raise DatasetLayoutError("Required dataset toolkit directory is missing: DETRAC-Toolkit*")
    if len(toolkit_candidates) > 1:
        names = ", ".join(path.name for path in toolkit_candidates)
        raise DatasetLayoutError(f"Found multiple toolkit directories: {names}")

    return DatasetLayout(
        root=resolved,
        images_dir=images,
        train_annotations_dir=train_annotations,
        test_annotations_dir=test_annotations,
        toolkit_dir=toolkit_candidates[0],
    )
