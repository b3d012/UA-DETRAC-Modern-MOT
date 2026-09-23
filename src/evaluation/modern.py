"""TrackEval MOTChallenge staging and normalized result parsing."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluation.coverage import AnnotatedSegment
from evaluation.schema import TrackObservation
from evaluation.trackeval_adapter import segment_metadata, trackeval_provenance

METRIC_SCALE = "fraction_0_1"
METRICS = ("HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA", "CLR_FP", "CLR_FN", "IDSW")


def _xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def stage_trackeval_mot(
    destination: str | Path,
    *,
    segments: tuple[AnnotatedSegment, ...],
    ground_truth: tuple[TrackObservation, ...],
    predictions: tuple[TrackObservation, ...],
    tracker_name: str = "project",
) -> dict[str, object]:
    """Write isolated MOTChallenge files without changing source data."""

    root = Path(destination)
    gt_root, tracker_root = root / "gt", root / "trackers"
    sequence_ids: list[str] = []
    for segment in segments:
        sequence_ids.append(segment.segment_id)
        mapping = {frame: index for index, frame in enumerate(segment.frame_numbers, start=1)}
        gt_path = gt_root / segment.segment_id / "gt" / "gt.txt"
        result_path = tracker_root / tracker_name / "data" / f"{segment.segment_id}.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with gt_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for item in ground_truth:
                if item.sequence == segment.sequence and item.frame_number in mapping:
                    writer.writerow(
                        (mapping[item.frame_number], item.track_id, *_xywh(item.bbox_xyxy), 1, 1, 1)
                    )
        with result_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for item in predictions:
                if item.sequence == segment.sequence and item.frame_number in mapping:
                    writer.writerow(
                        (
                            mapping[item.frame_number],
                            item.track_id,
                            *_xywh(item.bbox_xyxy),
                            1 if item.score is None else item.score,
                            1,
                            1,
                        )
                    )
        (gt_root / segment.segment_id / "seqinfo.ini").write_text(
            f"[Sequence]\nname={segment.segment_id}\nseqLength={len(mapping)}\nframeRate=1\nimWidth=1\nimHeight=1\nimExt=.jpg\n",
            encoding="utf-8",
        )
    (gt_root / "seqmaps" / "M4.txt").parent.mkdir(parents=True, exist_ok=True)
    (gt_root / "seqmaps" / "M4.txt").write_text(
        "name\n" + "\n".join(sequence_ids) + "\n", encoding="utf-8"
    )
    return {
        "tracker_name": tracker_name,
        "sequence_ids": sequence_ids,
        **segment_metadata(segments),
    }


def parse_trackeval_summary(path: str | Path) -> dict[str, object]:
    """Normalize TrackEval percentage metrics to fractions; retain counts exactly."""

    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=" "))
    row = next((item for item in rows if item.get("HOTA") is not None), None)
    if row is None:
        raise RuntimeError(f"TrackEval summary has no metrics: {path}")
    values: dict[str, float | int] = {}
    for name in METRICS:
        source = "IDs" if name == "IDSW" else name
        if source not in row:
            raise RuntimeError(f"TrackEval summary lacks {source}")
        value = float(row[source])
        values[name] = int(value) if name in {"CLR_FP", "CLR_FN", "IDSW"} else value / 100.0
    return {"result_family": "modern_mot", "metric_scale": METRIC_SCALE, "metrics": values}


def run_trackeval(staging_root: str | Path, *, tracker_name: str = "project") -> dict[str, object]:
    """Invoke the pinned TrackEval library API on previously staged MOT data."""

    provenance = trackeval_provenance()
    root = Path(staging_root)
    # TrackEval at the pinned upstream commit retains the removed NumPy alias.
    # Restore it locally before importing/running its dataset parser; this does
    # not change metric code or inputs.
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(root / "gt"),
            "TRACKERS_FOLDER": str(root / "trackers"),
            "OUTPUT_FOLDER": str(root / "results"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "BENCHMARK": "M4",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQMAP_FILE": str(root / "gt" / "seqmaps" / "M4.txt"),
            "DO_PREPROC": False,
            "PRINT_CONFIG": False,
        }
    )
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    output, messages = evaluator.evaluate(
        [dataset],
        [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()],
    )
    if messages[dataset.get_name()][tracker_name] != "Success":
        raise RuntimeError(f"TrackEval failed: {messages}")
    combined = output[dataset.get_name()][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    hota, clear, identity = combined["HOTA"], combined["CLEAR"], combined["Identity"]
    values = {
        "HOTA": float(hota["HOTA"].mean()),
        "DetA": float(hota["DetA"].mean()),
        "AssA": float(hota["AssA"].mean()),
        "LocA": float(hota["LocA"].mean()),
        "IDF1": float(identity["IDF1"]),
        "MOTA": float(clear["MOTA"]),
        "CLR_FP": int(clear["CLR_FP"]),
        "CLR_FN": int(clear["CLR_FN"]),
        "IDSW": int(clear["IDSW"]),
    }
    return {
        "result_family": "modern_mot",
        "metric_scale": METRIC_SCALE,
        "metrics": values,
        "evaluator": provenance,
    }
