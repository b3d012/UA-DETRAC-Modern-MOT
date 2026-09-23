"""M2 annotation-coverage segmentation for modern MOT evaluation."""

from __future__ import annotations

from dataclasses import dataclass

ANNOTATION_GAP_POLICY = "contiguous_annotated_segments_reset_identity"


@dataclass(frozen=True)
class AnnotatedSegment:
    sequence: str
    ordinal: int
    frame_numbers: tuple[int, ...]

    @property
    def segment_id(self) -> str:
        return (
            f"{self.sequence}__annseg_{self.ordinal:03d}_"
            f"{self.frame_numbers[0]:05d}_{self.frame_numbers[-1]:05d}"
        )


def segment_annotated_frames(
    sequence: str, frame_numbers: tuple[int, ...]
) -> tuple[AnnotatedSegment, ...]:
    """Split XML-present frames at each unknown-coverage gap."""

    values = tuple(sorted(set(frame_numbers)))
    if not values or values[0] < 1:
        raise ValueError("Annotated frame numbers must be nonempty positive integers")
    groups: list[list[int]] = [[values[0]]]
    for number in values[1:]:
        if number == groups[-1][-1] + 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    return tuple(
        AnnotatedSegment(sequence, ordinal, tuple(group))
        for ordinal, group in enumerate(groups, start=1)
    )
