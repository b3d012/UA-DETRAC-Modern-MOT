"""Capture machine-readable M1 reproducibility and environment metadata."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from reproducibility.environment import (
    EnvironmentOutputError,
    capture_environment,
    write_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "dataset" / "ua_detrac.yaml",
    PROJECT_ROOT / "configs" / "experiments" / "m1_foundation.yaml",
    PROJECT_ROOT / "configs" / "reproducibility.yaml",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "environment" / "m1_environment.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        dest="configs",
        help="Config to hash; repeat for multiple files (defaults to the three M1 configs)",
    )
    parser.add_argument("--seed", type=int, help="Override the configured reproducibility seed")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    try:
        seed = args.seed
        if seed is None:
            reproducibility = yaml.safe_load(DEFAULT_CONFIGS[-1].read_text(encoding="utf-8"))
            seed = reproducibility["reproducibility"]["seed"]
        dataset_document = yaml.safe_load(DEFAULT_CONFIGS[0].read_text(encoding="utf-8"))
        dataset_config = dataset_document["dataset"]
        protected_roots = [PROJECT_ROOT / "data" / "ua_detrac"]
        configured_root = os.environ.get(dataset_config.get("root_env", "UA_DETRAC_ROOT"))
        configured_root = configured_root or dataset_config.get("root")
        if configured_root:
            configured_path = Path(configured_root)
            if not configured_path.is_absolute():
                configured_path = PROJECT_ROOT / configured_path
            protected_roots.append(configured_path)
        metadata = capture_environment(
            PROJECT_ROOT,
            config_paths=config_paths,
            seed=seed,
            command=[sys.executable, *sys.argv],
        )
        output = write_environment(
            metadata,
            args.output,
            protected_roots=protected_roots,
            overwrite=args.force,
        )
    except (EnvironmentOutputError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Environment: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
