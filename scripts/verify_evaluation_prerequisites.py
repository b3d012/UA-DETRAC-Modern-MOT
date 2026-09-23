"""Inspect the immutable native UA-DETRAC evaluator."""

from __future__ import annotations

import argparse
import json

from evaluation.native import inspect_native_toolkit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit-root", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_native_toolkit(args.toolkit_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
