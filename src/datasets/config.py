"""Configuration loading for M1 dataset verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DatasetConfigError(ValueError):
    """Raised when the dataset configuration is missing required M1 values."""


OFFICIAL_TRAIN_SEQUENCES = 60
OFFICIAL_TEST_SEQUENCES = 40


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the M1 dataset configuration."""

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetConfigError(f"Cannot read dataset config {config_path}: {exc}") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get("dataset"), dict):
        raise DatasetConfigError("Dataset config must contain a 'dataset' mapping")
    dataset = loaded["dataset"]
    expected = dataset.get("expected")
    if not isinstance(expected, dict):
        raise DatasetConfigError("Dataset config must contain dataset.expected")
    for field in ("train_sequences", "test_sequences"):
        if not isinstance(expected.get(field), int) or expected[field] < 0:
            raise DatasetConfigError(f"dataset.expected.{field} must be a non-negative integer")
    if (
        expected["train_sequences"] != OFFICIAL_TRAIN_SEQUENCES
        or expected["test_sequences"] != OFFICIAL_TEST_SEQUENCES
    ):
        raise DatasetConfigError(
            "UA-DETRAC verification requires exactly 60 train and 40 test sequences"
        )
    return dataset
