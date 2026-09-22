"""Verify the raw UA-DETRAC layout and write the deterministic M1 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets.config import DatasetConfigError, load_dataset_config
from datasets.manifest import (
    DatasetOutputError,
    DatasetValidationError,
    build_manifest,
    write_manifest,
)
from datasets.paths import DatasetLayoutError, detect_layout, resolve_dataset_root
from datasets.verification import GitSafetyError, verify_git_safety

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset" / "ua_detrac.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Explicit raw UA-DETRAC root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, help="Manifest output path")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        dataset_config = load_dataset_config(args.config)
        root = resolve_dataset_root(
            args.root,
            dataset_config.get("root"),
            env_var=dataset_config.get("root_env", "UA_DETRAC_ROOT"),
        )
        layout_config = dataset_config.get("layout", {})
        layout = detect_layout(
            root,
            images_dir_name=layout_config.get("images_dir", "DETRAC-Images"),
            train_annotations_dir_name=layout_config.get(
                "train_annotations_dir", "DETRAC-Train-Annotations-XML"
            ),
            test_annotations_dir_name=layout_config.get(
                "test_annotations_dir", "DETRAC-Test-Annotations-XML"
            ),
            toolkit_glob=layout_config.get("toolkit_glob", "DETRAC-Toolkit*"),
        )
        expected = dataset_config["expected"]
        manifest = build_manifest(
            layout,
            expected_train_sequences=expected["train_sequences"],
            expected_test_sequences=expected["test_sequences"],
        )
        verify_git_safety(PROJECT_ROOT, root)
        configured_output = dataset_config.get(
            "manifest_path", "data/manifests/ua_detrac_manifest.json"
        )
        output = args.manifest or PROJECT_ROOT / configured_output
        output = write_manifest(manifest, output, dataset_root=root)
    except (
        DatasetConfigError,
        DatasetLayoutError,
        DatasetOutputError,
        DatasetValidationError,
        GitSafetyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("STATUS: FAIL", file=sys.stderr)
        return 1

    summary = manifest["summary"]
    print(f"Train sequences: {summary['train_sequences']}")
    print(f"Test sequences: {summary['test_sequences']}")
    print(f"Sequences with missing frames: {summary['sequences_with_missing_frames']}")
    print(f"Sequences without annotations: {summary['sequences_without_annotations']}")
    print(f"Manifest: {output}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
