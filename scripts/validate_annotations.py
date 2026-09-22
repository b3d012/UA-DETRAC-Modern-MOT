"""Validate selected UA-DETRAC annotations; official-test access is deliberate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets.paths import DatasetLayoutError, detect_layout, resolve_dataset_root
from datasets.ua_detrac import AnnotationParseError, parse_sequence
from datasets.validation import diagnose_frame_alignment, report_as_dict, validate_sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Raw UA-DETRAC root; defaults to UA_DETRAC_ROOT")
    parser.add_argument("--partition", choices=("train", "test"), default="train")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_sequences")
    parser.add_argument("--diagnose-frame-alignment", action="store_true")
    parser.add_argument("--allow-official-test", action="store_true")
    return parser


def _sequence_names(directory: Path, requested: list[str], all_sequences: bool) -> list[str]:
    if all_sequences:
        return [path.stem for path in sorted(directory.glob("*.xml"))]
    if requested:
        return sorted(set(requested))
    raise ValueError(
        "Pass --sequence for lightweight validation or --all for a partition-wide scan"
    )


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.partition == "test" and not args.allow_official_test:
        print("ERROR: Official-test access requires --allow-official-test", file=sys.stderr)
        return 1
    if args.diagnose_frame_alignment and (args.all_sequences or not args.sequence):
        print(
            "ERROR: --diagnose-frame-alignment requires one or more explicit --sequence values",
            file=sys.stderr,
        )
        return 1
    try:
        layout = detect_layout(resolve_dataset_root(args.root, None))
        annotations = (
            layout.train_annotations_dir
            if args.partition == "train"
            else layout.test_annotations_dir
        )
        reports = []
        diagnostics = []
        for name in _sequence_names(annotations, args.sequence, args.all_sequences):
            annotation = annotations / f"{name}.xml"
            if not annotation.is_file():
                raise ValueError(f"Annotation does not exist in {args.partition}: {name}")
            sequence = parse_sequence(annotation, layout.images_dir / name, args.partition)
            if args.diagnose_frame_alignment:
                diagnostics.append(report_as_dict(diagnose_frame_alignment(sequence)))
            else:
                reports.append(report_as_dict(validate_sequence(sequence)))
    except (AnnotationParseError, DatasetLayoutError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.diagnose_frame_alignment:
        print(
            json.dumps(
                {"partition": args.partition, "diagnostics": diagnostics}, indent=2, sort_keys=True
            )
        )
        return 0
    print(json.dumps({"partition": args.partition, "reports": reports}, indent=2, sort_keys=True))
    return 0 if all(report["error_count"] == 0 for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
