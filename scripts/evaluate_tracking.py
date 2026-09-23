"""M4 guarded, reproducible tracking-evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from datasets.paths import detect_layout
from datasets.splits import validate_split_access
from datasets.ua_detrac import parse_sequence
from evaluation.coverage import ANNOTATION_GAP_POLICY, segment_annotated_frames
from evaluation.ignores import filter_ignored_observations
from evaluation.modern import run_trackeval, stage_trackeval_mot
from evaluation.native import (
    NativePrerequisiteError,
    export_native_matrices,
    inspect_native_toolkit,
    invoke_native_pr,
)
from evaluation.reporting import environment_provenance, git_state, sha256_file, write_result
from evaluation.schema import TrackObservation, load_result_json, validate_observations
from evaluation.trackeval_adapter import trackeval_provenance

ROOT = Path(__file__).resolve().parents[1]


def _split(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _names(split: dict[str, object], role: str) -> set[str]:
    return {str(x["name"]) for x in split["roles"][role]}  # type: ignore[index]


def _gt(root: Path, names: tuple[str, ...], role: str):
    layout = detect_layout(root)
    annotations = (
        layout.test_annotations_dir if role == "official_test" else layout.train_annotations_dir
    )
    values: list[TrackObservation] = []
    segments = []
    excluded: dict[str, list[int]] = {}
    for name in names:
        seq = parse_sequence(
            annotations / f"{name}.xml",
            layout.images_dir / name,
            "test" if role == "official_test" else "train",
        )
        values.extend(
            TrackObservation(name, frame.number, target.track_id, target.box.xyxy)
            for frame in seq.frames
            for target in frame.targets
        )
        segments.extend(segment_annotated_frames(name, seq.annotation_coverage.xml_frame_numbers))
        excluded[name] = list(seq.annotation_coverage.unannotated_image_frame_numbers)
    return (
        tuple(values),
        tuple(segments),
        {
            "annotation_gap_policy": ANNOTATION_GAP_POLICY,
            "excluded_unannotated_frame_ids": excluded,
        },
    )


def _stage_native(
    root: Path, names: tuple[str, ...], role: str, predictions, stage: Path, state: str, upstream
):
    """Export only disposable LX/LY/W/H files; never touch toolkit/data assets."""
    layout = detect_layout(root)
    annotations = (
        layout.test_annotations_dir if role == "official_test" else layout.train_annotations_dir
    )
    records = {}
    for name in names:
        seq = parse_sequence(
            annotations / f"{name}.xml",
            layout.images_dir / name,
            "test" if role == "official_test" else "train",
        )
        records[name] = export_native_matrices(
            stage,
            sequence=name,
            frame_count=max(seq.annotation_coverage.image_frame_numbers),
            observations=tuple(item for item in predictions if item.sequence == name),
            ignored_regions=tuple(box.xyxy for box in seq.ignored_regions),
            image_size=seq.image_size,
            native_ignore_filter_state=state,
            already_applied_provenance=upstream,
        )
    return records


def _apply_modern_ignore(root: Path, names: tuple[str, ...], role: str, predictions, policy: str):
    if policy == "none":
        return predictions, 0
    layout = detect_layout(root)
    annotations = (
        layout.test_annotations_dir if role == "official_test" else layout.train_annotations_dir
    )
    retained = []
    suppressed = 0
    for name in names:
        seq = parse_sequence(
            annotations / f"{name}.xml",
            layout.images_dir / name,
            "test" if role == "official_test" else "train",
        )
        values, count = filter_ignored_observations(
            tuple(item for item in predictions if item.sequence == name),
            ignored_regions=tuple(box.xyxy for box in seq.ignored_regions),
            image_size=seq.image_size,
            policy=policy,
        )
        retained.extend(values)
        suppressed += count
    return tuple(retained), suppressed


def _run_synthetic_case(segments, ground_truth, predictions) -> dict[str, object]:
    """Use the production TrackEval staging boundary for synthetic fixtures."""
    with tempfile.TemporaryDirectory(prefix="m4-fixture-") as stage:
        metadata = stage_trackeval_mot(
            stage, segments=segments, ground_truth=ground_truth, predictions=predictions
        )
        return {"metrics": run_trackeval(stage)["metrics"], "segments": metadata["segments"]}


def _synthetic_fixture_results(suite: str) -> dict[str, dict[str, object]]:
    if suite not in {"all", "core"}:
        raise ValueError("fixture-suite must be all or core")
    gt = tuple(TrackObservation("SYN", frame, 1, (0, 0, 10, 10)) for frame in (1, 2, 3))
    contiguous = segment_annotated_frames("SYN", (1, 2, 3))
    cases = {
        "perfect": gt,
        "empty": (),
        "false_positive": gt + (TrackObservation("SYN", 1, 2, (20, 0, 30, 10)),),
        "miss": (gt[0], gt[2]),
        "id_switch": (
            gt[0],
            TrackObservation("SYN", 2, 2, (0, 0, 10, 10)),
            TrackObservation("SYN", 3, 2, (0, 0, 10, 10)),
        ),
    }
    results = {
        name: _run_synthetic_case(contiguous, gt, prediction) for name, prediction in cases.items()
    }
    if suite == "core":
        return results
    results["detection_gap_fragmentation_diagnostic"] = _run_synthetic_case(
        contiguous, gt, (gt[0], gt[2])
    )
    results["matched_imperfect_localization"] = _run_synthetic_case(
        contiguous[:1], (gt[0],), (TrackObservation("SYN", 1, 1, (2, 0, 12, 10)),)
    )
    results["localization_crosses_matching_threshold"] = _run_synthetic_case(
        contiguous[:1], (gt[0],), (TrackObservation("SYN", 1, 1, (8, 0, 18, 10)),)
    )
    ignore_prediction = gt + (TrackObservation("SYN", 1, 2, (20, 0, 30, 10)),)
    legacy, legacy_count = filter_ignored_observations(
        ignore_prediction,
        ignored_regions=((20, 0, 30, 10),),
        image_size=(40, 20),
        policy="ua_detrac_legacy_drop_tracks",
    )
    none, none_count = filter_ignored_observations(
        ignore_prediction, ignored_regions=((20, 0, 30, 10),), image_size=(40, 20), policy="none"
    )
    results["ignored_region_ua_detrac_legacy_drop_tracks"] = {
        **_run_synthetic_case(contiguous, gt, legacy),
        "suppressed_observations": legacy_count,
    }
    results["ignored_region_none"] = {
        **_run_synthetic_case(contiguous, gt, none),
        "suppressed_observations": none_count,
    }
    gap_segments = segment_annotated_frames("SYN", (1, 3))
    gap_gt = (gt[0], gt[2])
    gap_prediction = (gt[0], TrackObservation("SYN", 3, 2, (0, 0, 10, 10)))
    results["interior_annotation_gap_identity_reset"] = {
        **_run_synthetic_case(gap_segments, gap_gt, gap_prediction),
        "excluded_original_frame_ids": [2],
        "expected_cross_gap_idsw": 0,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("trackeval", "ua-detrac-pr"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--input", type=Path, help="Canonical tracker-result JSON")
    parser.add_argument("--synthetic-fixtures", action="store_true")
    parser.add_argument("--fixture-suite", default="all")
    parser.add_argument("--ground-truth-as-prediction", action="store_true")
    parser.add_argument(
        "--role", default="validation", choices=("development_train", "validation", "official_test")
    )
    parser.add_argument("--mode", default="evaluation", choices=("evaluation", "tuning"))
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument(
        "--split-path", type=Path, default=ROOT / "data/splits/ua_detrac_m3_split.json"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/evaluation/m4_trackeval.yaml"
    )
    parser.add_argument("--ignore-policy", default="ua_detrac_legacy_drop_tracks")
    parser.add_argument("--toolkit-root", type=Path)
    parser.add_argument("--diagnostic-pr-context", type=Path)
    parser.add_argument("--native-ignore-filter-state", choices=("not_applied", "already_applied"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic_fixtures:
        if args.output.exists():
            parser.error(f"Output already exists; refusing collision: {args.output}")
        collected = _synthetic_fixture_results(args.fixture_suite)
        result = {
            "result_family": "modern_mot",
            "metric_scale": "fraction_0_1",
            "metrics": collected["perfect"]["metrics"],
            "synthetic_fixture_metrics": collected,
            "provenance": {
                "synthetic": True,
                "fixture_suite": args.fixture_suite,
                "environment": environment_provenance(
                    ROOT / "requirements/evaluation.lock", trackeval_provenance()
                ),
            },
        }
        write_result(args.output, result)
        print(
            json.dumps(
                {
                    "result": str(args.output / "result.json"),
                    "result_family": result["result_family"],
                },
                sort_keys=True,
            )
        )
        return 0
    split = _split(args.split_path)
    sequences = validate_split_access(
        split,
        mode=args.mode,
        role=args.role,
        sequences=args.sequence,
        allow_official_test=args.allow_official_test,
    )
    if args.output.exists():
        parser.error(f"Output already exists; refusing collision: {args.output}")
    if not args.input and not args.ground_truth_as_prediction:
        parser.error("Provide --input or --ground-truth-as-prediction")
    if not args.root:
        parser.error("--root is required for ground truth and evaluation staging")
    ground_truth, segments, coverage = _gt(args.root, sequences, args.role)
    allowed = _names(split, args.role)
    if args.ground_truth_as_prediction:
        predictions, input_metadata = (
            ground_truth,
            {"producer": {"name": "canonical_m2_ground_truth"}},
        )
    else:
        input_metadata, predictions = load_result_json(str(args.input), allowed_sequences=allowed)
        if input_metadata["role"] != args.role:
            parser.error("Canonical result role does not match --role")
        predictions = validate_observations(
            (x for x in predictions if x.sequence in sequences), allowed_sequences=set(sequences)
        )
    evaluable: dict[str, set[int]] = {}
    for segment in segments:
        evaluable.setdefault(segment.sequence, set()).update(segment.frame_numbers)
    excluded_predictions = [
        {
            "sequence": item.sequence,
            "original_frame_number": item.frame_number,
            "track_id": item.track_id,
        }
        for item in predictions
        if item.frame_number not in evaluable[item.sequence]
    ]
    predictions = tuple(
        item for item in predictions if item.frame_number in evaluable[item.sequence]
    )
    coverage["excluded_prediction_observations"] = excluded_predictions
    provenance = {
        "git": git_state(ROOT),
        "m3_split_sha256": sha256_file(args.split_path),
        "config_sha256": sha256_file(args.config),
        "environment": environment_provenance(
            ROOT / "requirements/evaluation.lock", trackeval_provenance()
        ),
        "command_line": sys.argv,
        "annotation_gap_policy": ANNOTATION_GAP_POLICY,
        "ignore_policy": args.ignore_policy,
        "input": input_metadata,
    }
    if args.backend == "trackeval":
        ground_truth, modern_gt_suppressed = _apply_modern_ignore(
            args.root, sequences, args.role, ground_truth, args.ignore_policy
        )
        predictions, modern_prediction_suppressed = _apply_modern_ignore(
            args.root, sequences, args.role, predictions, args.ignore_policy
        )
        with tempfile.TemporaryDirectory(prefix="m4-trackeval-") as stage:
            staged = stage_trackeval_mot(
                stage, segments=segments, ground_truth=ground_truth, predictions=predictions
            )
            result = run_trackeval(stage)
        result.update(
            {
                "provenance": provenance,
                "coverage": coverage,
                "trackeval_staging": staged,
                "ignore_filter_suppressed_observations": {
                    "ground_truth": modern_gt_suppressed,
                    "predictions": modern_prediction_suppressed,
                },
            }
        )
    else:
        if (
            not args.toolkit_root
            or not args.diagnostic_pr_context
            or not args.native_ignore_filter_state
        ):
            parser.error(
                "Native PR requires --toolkit-root, --diagnostic-pr-context, and --native-ignore-filter-state"
            )
        toolkit = inspect_native_toolkit(args.toolkit_root)
        native_pr_prerequisite = toolkit["native_pr_execution_prerequisite"]
        if native_pr_prerequisite["status"] != "ready":
            parser.error(
                f"Native PR execution blocked by external prerequisite: {native_pr_prerequisite['limitation']}"
            )
        context = yaml.safe_load(args.diagnostic_pr_context.read_text(encoding="utf-8"))
        upstream = (
            input_metadata.get("native_ignore_filter_provenance")
            if isinstance(input_metadata, dict)
            else None
        )
        with tempfile.TemporaryDirectory(prefix="m4-native-") as temporary:
            stage = Path(temporary)
            matrix_metadata = _stage_native(
                args.root,
                sequences,
                args.role,
                predictions,
                stage,
                args.native_ignore_filter_state,
                upstream,
            )
            replacements = {
                "{tracking_results_root}": str(stage),
                "{output_dir}": str(stage / "official-output"),
            }
            arguments = [replacements.get(str(value), str(value)) for value in context["arguments"]]
            result_path = Path(
                str(context["official_result_file"]).replace(
                    "{output_dir}", str(stage / "official-output")
                )
            )
            native = invoke_native_pr(
                args.toolkit_root / "evaluation" / "DETRAC_MOT_EVAL.exe", arguments, result_path
            )
        native["provenance"] = {
            **provenance,
            "native_toolkit": toolkit,
            "native_ignore_filter_state": args.native_ignore_filter_state,
            "native_matrix_export": matrix_metadata,
            "diagnostic_pr_context_sha256": sha256_file(args.diagnostic_pr_context),
        }
        result = native
    write_result(args.output, result)
    print(
        json.dumps(
            {"result": str(args.output / "result.json"), "result_family": result["result_family"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, NativePrerequisiteError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
