"""Create the deterministic, frozen M3 UA-DETRAC sequence split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets.config import DatasetConfigError, load_dataset_config
from datasets.paths import DatasetLayoutError, resolve_dataset_root
from datasets.splits import SplitError, build_split, write_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Raw UA-DETRAC root; otherwise UA_DETRAC_ROOT")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/ua_detrac_manifest.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/splits/ua_detrac_m3_split.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dataset/ua_detrac.yaml")
    args = parser.parse_args(arguments)
    try:
        config = load_dataset_config(args.config)
        root = resolve_dataset_root(args.root, config.get("root"), env_var=config.get("root_env", "UA_DETRAC_ROOT"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        split = build_split(manifest, root, seed=int(config.get("split_policy", {}).get("seed", 42)))
        output = write_split(split, args.output, dataset_root=root)
    except (OSError, ValueError, DatasetConfigError, DatasetLayoutError, SplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("STATUS: FAIL", file=sys.stderr)
        return 1
    counts = split["audit"]["role_counts"]
    print(f"Development-train sequences: {counts['development_train']}")
    print(f"Validation sequences: {counts['validation']}")
    print(f"Official-test sequences: {counts['official_test']}")
    print(f"Split: {output}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
