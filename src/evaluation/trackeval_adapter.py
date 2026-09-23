"""TrackEval dependency boundary and deterministic staging metadata."""

from __future__ import annotations

import importlib.metadata

from evaluation.coverage import ANNOTATION_GAP_POLICY, AnnotatedSegment

TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def trackeval_provenance() -> dict[str, str]:
    """Return the pinned evaluator identity; fail explicitly if unavailable."""

    try:
        version = importlib.metadata.version("trackeval")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "TrackEval is not installed; install requirements/evaluation.lock before evaluation"
        ) from exc
    return {"package": "trackeval", "version": version, "commit": TRACKEVAL_COMMIT}


def segment_metadata(segments: tuple[AnnotatedSegment, ...]) -> dict[str, object]:
    return {
        "annotation_gap_policy": ANNOTATION_GAP_POLICY,
        "segments": [
            {
                "segment_id": segment.segment_id,
                "sequence": segment.sequence,
                "original_frame_numbers": list(segment.frame_numbers),
            }
            for segment in segments
        ],
    }
