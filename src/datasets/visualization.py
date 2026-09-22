"""Deterministic ground-truth rendering and audit-frame selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from datasets.schema import SequenceAnnotation


@dataclass(frozen=True)
class AuditFrame:
    sequence_name: str
    frame_number: int


def _condition(entry: dict[str, object]) -> tuple[str, str]:
    attributes = entry.get("sequence_attributes", {})
    if not isinstance(attributes, dict):
        return ("", "")
    return (str(attributes.get("sence_weather", "")), str(attributes.get("camera_state", "")))


def select_audit_frames(
    entries: list[dict[str, object]], *, requested_frames: int = 40
) -> tuple[AuditFrame, ...]:
    """Select 30--50 deterministic consecutive official-train audit frames."""

    if not 30 <= requested_frames <= 50:
        raise ValueError("requested_frames must be between 30 and 50")
    eligible = sorted(
        (entry for entry in entries if int(entry.get("frame_count", 0)) > 0),
        key=lambda e: str(e["name"]),
    )
    if not eligible:
        return ()
    representatives: list[dict[str, object]] = []
    seen_conditions: set[tuple[str, str]] = set()
    for entry in sorted(eligible, key=lambda e: (_condition(e), str(e["name"]))):
        if _condition(entry) not in seen_conditions and len(representatives) < 5:
            representatives.append(entry)
            seen_conditions.add(_condition(entry))
    for entry in eligible:
        if len(representatives) >= 5:
            break
        if entry not in representatives:
            representatives.append(entry)

    selected: list[AuditFrame] = []
    base, remainder = divmod(requested_frames, len(representatives))
    for index, entry in enumerate(representatives):
        count = base + (1 if index < remainder else 0)
        available = int(entry["frame_count"])
        count = min(count, available)
        start = max(1, (available - count) // 2 + 1)
        selected.extend(
            AuditFrame(str(entry["name"]), frame) for frame in range(start, start + count)
        )

    if len(selected) < requested_frames:
        used = {(item.sequence_name, item.frame_number) for item in selected}
        for entry in eligible:
            for frame in range(1, int(entry["frame_count"]) + 1):
                candidate = (str(entry["name"]), frame)
                if candidate not in used:
                    selected.append(AuditFrame(*candidate))
                    used.add(candidate)
                    if len(selected) == requested_frames:
                        return tuple(selected)
    return tuple(selected)


def _color(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id + 71) % 200 + 30,
        (67 * track_id + 29) % 200 + 30,
        (97 * track_id + 11) % 200 + 30,
    )


def render_sequence_frames(
    sequence: SequenceAnnotation,
    frame_numbers: list[int],
    output_directory: str | Path,
    *,
    force: bool = False,
) -> tuple[Path, ...]:
    """Render selected frames, refusing to replace existing visual-audit output."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"Visualization output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_names = dict(sequence.image_frames)
    annotations = {frame.number: frame for frame in sequence.frames}
    written = []
    for number in frame_numbers:
        if number not in image_names or number not in annotations:
            raise ValueError(
                f"Frame {number} is not aligned between XML and images for {sequence.name}"
            )
        with Image.open(Path(sequence.image_directory) / image_names[number]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for ignored in sequence.ignored_regions:
            draw.rectangle(ignored.xyxy, outline=(255, 215, 0), width=2)
            draw.text((ignored.xyxy[0], ignored.xyxy[1]), "ignored", fill=(255, 215, 0))
        for target in annotations[number].targets:
            color = _color(target.track_id)
            draw.rectangle(target.box.xyxy, outline=color, width=2)
            label = f"{target.track_id} {target.vehicle_type or 'unknown'}"
            if target.occlusions:
                label += " occ"
            truncation = dict(target.attribute_values).get("truncation_ratio")
            if truncation is not None:
                label += f" trunc={truncation}"
            draw.text((target.box.xyxy[0], max(0, target.box.xyxy[1] - 12)), label, fill=color)
        metadata = dict(sequence.sequence_attributes)
        draw.text(
            (4, 4),
            f"{sequence.name} frame={number} {metadata}",
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        destination = output / f"{sequence.name}_frame_{number:05d}.png"
        image.save(destination)
        written.append(destination)
    return tuple(written)
