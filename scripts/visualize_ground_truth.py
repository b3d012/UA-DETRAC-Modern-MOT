"""Render UA-DETRAC ground truth with deliberate official-test safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets.paths import DatasetLayoutError, detect_layout, resolve_dataset_root
from datasets.ua_detrac import AnnotationParseError, parse_sequence
from datasets.visualization import render_sequence_frames, select_audit_frames

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Raw UA-DETRAC root; defaults to UA_DETRAC_ROOT")
    parser.add_argument("--partition", choices=("train", "test"), default="train")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--frames", type=int, nargs="+", default=[])
    parser.add_argument("--all", action="store_true", dest="all_sequences")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _audit_by_sequence(manifest_path: Path) -> dict[str, list[int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_audit_frames(manifest["partitions"]["train"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for item in selected:
        grouped[item.sequence_name].append(item.frame_number)
    if len(selected) < 30:
        print(
            f"WARNING: audit has only {len(selected)} eligible official-train frames",
            file=sys.stderr,
        )
    return dict(grouped)


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.partition == "test" and not args.allow_official_test:
        print("ERROR: Official-test access requires --allow-official-test", file=sys.stderr)
        return 1
    if args.audit and args.partition != "train":
        print("ERROR: --audit is official-train only", file=sys.stderr)
        return 1
    if args.all_sequences and not args.frames:
        print("ERROR: Pass --frames with --all to limit rendered frames", file=sys.stderr)
        return 1
    try:
        layout = detect_layout(resolve_dataset_root(args.root, None))
        annotation_dir = (
            layout.train_annotations_dir
            if args.partition == "train"
            else layout.test_annotations_dir
        )
        if args.audit:
            selections = _audit_by_sequence(
                PROJECT_ROOT / "data" / "manifests" / "ua_detrac_manifest.json"
            )
        elif args.all_sequences:
            selections = {path.stem: args.frames for path in sorted(annotation_dir.glob("*.xml"))}
        elif args.sequence:
            if not args.frames:
                raise ValueError("Pass --frames with --sequence, or use --audit")
            selections = {name: args.frames for name in sorted(set(args.sequence))}
        else:
            raise ValueError("Pass --sequence with --frames, --all, or --audit")
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()) and not args.force:
            raise FileExistsError(f"Visualization output directory is nonempty: {output}")
        written = []
        for name, frames in selections.items():
            sequence = parse_sequence(
                annotation_dir / f"{name}.xml", layout.images_dir / name, args.partition
            )
            # The CLI checked the destination once above; individual sequence renders may share it.
            written.extend(render_sequence_frames(sequence, frames, output, force=True))
    except (
        AnnotationParseError,
        DatasetLayoutError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered frames: {len(written)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
