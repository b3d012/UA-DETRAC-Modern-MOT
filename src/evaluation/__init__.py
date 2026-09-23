"""M4 evaluation package for UA-DETRAC Modern MOT."""

from evaluation.coverage import ANNOTATION_GAP_POLICY, AnnotatedSegment, segment_annotated_frames
from evaluation.schema import EvaluationInputError, TrackObservation, validate_observations

__all__ = [
    "ANNOTATION_GAP_POLICY",
    "AnnotatedSegment",
    "EvaluationInputError",
    "TrackObservation",
    "segment_annotated_frames",
    "validate_observations",
]
