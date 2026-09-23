"""Read-only inspection and provenance for the official UA-DETRAC toolkit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from evaluation.ignores import LEGACY_IGNORE_POLICY, filter_ignored_predictions
from evaluation.schema import TrackObservation


class NativePrerequisiteError(RuntimeError):
    """Raised when the immutable official toolkit is incomplete."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_native_toolkit(toolkit_root: str | Path) -> dict[str, object]:
    """Validate and fingerprint assets without modifying the toolkit."""

    root = Path(toolkit_root).resolve()
    evaluation = root / "evaluation"
    executable = evaluation / "DETRAC_MOT_EVAL.exe"
    if not executable.is_file():
        raise NativePrerequisiteError(f"Missing official evaluator: {executable}")
    required = ("trackingEvaluation.m", "printFinalEvaluation.m")
    missing = [name for name in required if not (evaluation / name).is_file()]
    if missing:
        raise NativePrerequisiteError(f"Missing official workflow scripts: {', '.join(missing)}")
    ignore_files = sorted((evaluation / "igrs").glob("*_IgR.txt"))
    discovered = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and any(token in path.name.casefold() for token in ("sequence", "groundtruth", "gt"))
    )
    sequence_file = root / "sequences.txt"
    sequence_names = (
        [
            line.strip()
            for line in sequence_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if sequence_file.is_file()
        else []
    )
    pr_files = sorted((evaluation / "curve_results").glob("*_detection_PR.txt"))
    detector_names = [path.name.removesuffix("_detection_PR.txt") for path in pr_files]
    # The normal supplied layout is <UA_DETRAC_ROOT>/DETRAC-Toolkits/DETRAC-MOT-toolkit.
    candidate_root = root.parent.parent
    threshold_files = sorted(candidate_root.rglob("*_thres.txt")) if candidate_root.is_dir() else []
    native_pr = {
        "sequence_file": str(sequence_file) if sequence_file.is_file() else None,
        "sequence_count": len(sequence_names),
        "includes_MVI_20011": "MVI_20011" in sequence_names,
        "historical_detector_pr_files": {path.name: _sha256(path) for path in pr_files},
        "historical_detector_names": detector_names,
        "detector_specific_threshold_files": [str(path) for path in threshold_files],
    }
    native_pr["status"] = "ready" if threshold_files else "blocked_external_prerequisite"
    native_pr["limitation"] = (
        "No authentic detector-specific *_thres.txt artifact is available; "
        "printFinalEvaluation.m requires <detectorName>_thres.txt and "
        "<detectorName>_detection_PR.txt. Do not fabricate or rename generic thresh.txt."
        if not threshold_files
        else None
    )
    return {
        "toolkit_root": str(root),
        "executable_sha256": _sha256(executable),
        "workflow_scripts": {name: _sha256(evaluation / name) for name in required},
        "ignored_region_file_count": len(ignore_files),
        "ignored_region_sha256": {path.name: _sha256(path) for path in ignore_files},
        "discovered_native_sequence_or_gt_assets": {
            str(path.relative_to(root)): _sha256(path) for path in discovered
        },
        "native_ground_truth": "resolved_by_official_executable_or_workflow",
        "drop_tracks_stage": "pre_executable",
        "drop_tracks_behavior": (
            "trackingEvaluation.m invokes dropTracks before saveResults; "
            "printFinalEvaluation.m invokes DETRAC_MOT_EVAL.exe on saved results"
        ),
        "native_pr_execution_prerequisite": native_pr,
    }


def export_native_matrices(
    destination: str | Path,
    *,
    sequence: str,
    frame_count: int,
    observations: tuple[TrackObservation, ...],
    ignored_regions: tuple[tuple[float, float, float, float], ...],
    image_size: tuple[int, int],
    native_ignore_filter_state: str,
    already_applied_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create official LX/LY/W/H matrices, applying legacy filtering at most once."""

    if native_ignore_filter_state not in {"not_applied", "already_applied"}:
        raise NativePrerequisiteError(
            "native_ignore_filter_state must be not_applied or already_applied"
        )
    if native_ignore_filter_state == "already_applied" and not already_applied_provenance:
        raise NativePrerequisiteError(
            "already_applied requires immutable upstream filter provenance"
        )
    if native_ignore_filter_state == "not_applied":
        values, suppressed = filter_ignored_predictions(
            observations,
            ignored_regions=ignored_regions,
            image_size=image_size,
            policy=LEGACY_IGNORE_POLICY,
        )
        responsible = "project_adapter_emulating_trackingEvaluation"
    else:
        values, suppressed, responsible = observations, 0, "upstream_declared_already_applied"
    ids = sorted({item.track_id for item in values})
    columns = {track_id: index for index, track_id in enumerate(ids)}
    matrices = {
        name: [[0.0 for _ in ids] for _ in range(frame_count)] for name in ("LX", "LY", "W", "H")
    }
    for item in values:
        if item.sequence != sequence or item.frame_number > frame_count:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        row, column = item.frame_number - 1, columns[item.track_id]
        matrices["LX"][row][column], matrices["LY"][row][column] = x1, y1
        matrices["W"][row][column], matrices["H"][row][column] = x2 - x1, y2 - y1
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        (root / f"{sequence}_{name}.txt").write_text(
            "\n".join(" ".join(str(value) for value in row) for row in matrix) + "\n",
            encoding="utf-8",
        )
    return {
        "native_ignore_filter_state": native_ignore_filter_state,
        "ignore_filter_responsible": responsible,
        "suppressed_observations": suppressed,
        "track_ids": ids,
        "already_applied_provenance": already_applied_provenance,
    }


def parse_pr_result(path: str | Path) -> dict[str, float]:
    """Parse the official final (threshold -1) PR result row."""
    rows = [
        [float(value) for value in line.split()]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    row = next((row for row in rows if row and row[0] == -1), None)
    if row is None or len(row) != 14:
        raise NativePrerequisiteError("Official PR result requires one 14-column threshold=-1 row")
    names = (
        "pr_recall",
        "pr_precision",
        "pr_far",
        "pr_mt",
        "pr_pt",
        "pr_ml",
        "pr_fp",
        "pr_fn",
        "pr_idsw",
        "pr_fragmentations",
        "pr_mota",
        "pr_motp",
        "pr_motal",
    )
    return dict(zip(names, row[1:], strict=True))


def invoke_native_pr(
    executable: str | Path, arguments: list[str], output_file: str | Path
) -> dict[str, object]:
    """Invoke immutable official executable only in caller-provided disposable staging."""
    completed = subprocess.run(
        [str(executable), *arguments], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise NativePrerequisiteError(
            f"DETRAC_MOT_EVAL.exe failed ({completed.returncode}): {completed.stderr}"
        )
    return {
        "result_family": "ua_detrac_pr",
        "metric_scale": "official_toolkit_scale",
        "metrics": parse_pr_result(output_file),
        "stdout": completed.stdout,
    }
