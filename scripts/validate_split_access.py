"""Validate a role/sequence request against the frozen M3 split lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets.splits import SplitError, validate_split_access

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data/splits/ua_detrac_m3_split.json")
    parser.add_argument("--mode", choices=("tuning", "evaluation"), required=True)
    parser.add_argument("--role", choices=("development_train", "validation", "official_test"), required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--allow-official-test", action="store_true")
    args = parser.parse_args(arguments)
    try:
        split = json.loads(args.split.read_text(encoding="utf-8"))
        names = validate_split_access(split, mode=args.mode, role=args.role, sequences=args.sequence, allow_official_test=args.allow_official_test)
    except (OSError, ValueError, SplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("STATUS: FAIL", file=sys.stderr)
        return 1
    print(f"Sequences: {len(names)}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
