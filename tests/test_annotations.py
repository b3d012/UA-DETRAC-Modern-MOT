from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from datasets.ua_detrac import parse_sequence
from datasets.validation import diagnose_frame_alignment, validate_sequence


def _write_sequence(
    root: Path,
    *,
    target_id: str = "7",
    second_frame: bool = True,
    image_size: tuple[int, int] = (100, 80),
) -> tuple[Path, Path]:
    images = root / "MVI_SYNTHETIC"
    images.mkdir(parents=True)
    Image.new("RGB", image_size, "white").save(images / "img00001.jpg")
    if second_frame:
        Image.new("RGB", image_size, "white").save(images / "img00002.jpg")
    frames = f"""
  <frame density="1" num="1"><target_list>
    <target id="{target_id}"><box left="10.5" top="20.25" width="30.75" height="40.5"/>
      <attribute vehicle_type="car" truncation_ratio="0.1" orientation="2" speed="3" trajectory_length="4"/>
      <occlusion><region_overlap left="11" top="21" width="2" height="3" occlusion_id="8" occlusion_status="1"/></occlusion>
    </target>
  </target_list></frame>"""
    if second_frame:
        frames += """
  <frame density="1" num="2"><target_list>
    <target id="7"><box left="11" top="20" width="30" height="40"/>
      <attribute vehicle_type="car" truncation_ratio="0" orientation="2" speed="3" trajectory_length="4"/>
    </target>
  </target_list></frame>"""
    xml = root / "sequence.xml"
    xml.write_text(
        """<sequence name="MVI_SYNTHETIC"><sequence_attribute camera_state="stable" sence_weather="rainy"/>
  <ignored_region><box left="0.5" top="1.5" width="4" height="5"/></ignored_region>"""
        + frames
        + "</sequence>",
        encoding="utf-8",
    )
    return xml, images


def test_parser_preserves_raw_metadata_and_derives_typed_annotation(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)

    sequence = parse_sequence(xml, images, "train")

    target = sequence.frames[0].targets[0]
    assert sequence.raw_attributes == (("name", "MVI_SYNTHETIC"),)
    assert sequence.sequence_attributes == (("camera_state", "stable"), ("sence_weather", "rainy"))
    assert sequence.ignored_region_attributes == ((),)
    assert target.raw_attributes == (("id", "7"),)
    assert target.box.raw_attributes == (
        ("height", "40.5"),
        ("left", "10.5"),
        ("top", "20.25"),
        ("width", "30.75"),
    )
    assert target.box.xyxy == (10.5, 20.25, 41.25, 60.75)
    assert target.occlusions[0].occlusion_id == 8
    assert target.occlusion_attributes == ((),)
    assert sequence.ignored_regions[0].xyxy == (0.5, 1.5, 4.5, 6.5)


def test_parser_rejects_non_finite_box_coordinates(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    xml.write_text(
        xml.read_text(encoding="utf-8").replace('left="10.5"', 'left="nan"'), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-finite"):
        parse_sequence(xml, images, "train")


def test_validator_reports_duplicate_ids_invalid_class_and_out_of_range_box(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    text = xml.read_text(encoding="utf-8")
    text = text.replace('vehicle_type="car"', 'vehicle_type="truck"', 1)
    text = text.replace('width="30.75"', 'width="300"', 1)
    text = text.replace(
        "</target_list></frame>",
        text[text.index('<target id="7"') : text.index("</target_list></frame>")]
        + "</target_list></frame>",
        1,
    )
    xml.write_text(text, encoding="utf-8")

    report = validate_sequence(parse_sequence(xml, images, "train"))

    assert {issue.code for issue in report.issues} >= {
        "duplicate_track_id",
        "unknown_vehicle_type",
        "box_out_of_image_range",
    }


def test_validator_reports_xml_image_sequence_name_mismatch(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    wrong_images = tmp_path / "MVI_WRONG"
    images.rename(wrong_images)

    report = validate_sequence(parse_sequence(xml, wrong_images, "train"))

    assert {issue.code for issue in report.issues} >= {"xml_image_sequence_mismatch"}


def test_frame_alignment_diagnostic_classifies_leading_trailing_and_interior_missing_xml_frames(
    tmp_path: Path,
) -> None:
    xml, images = _write_sequence(tmp_path)
    for number in (3, 4, 5, 6, 7):
        Image.new("RGB", (100, 80), "white").save(images / f"img{number:05d}.jpg")
    xml.write_text(
        xml.read_text(encoding="utf-8")
        .replace('num="2"', 'num="5"', 1)
        .replace('num="1"', 'num="2"', 1),
        encoding="utf-8",
    )

    diagnostic = diagnose_frame_alignment(parse_sequence(xml, images, "train"))

    assert diagnostic.image_frame_count == 7
    assert (diagnostic.image_frame_minimum, diagnostic.image_frame_maximum) == (1, 7)
    assert diagnostic.xml_frame_count == 2
    assert (diagnostic.xml_frame_minimum, diagnostic.xml_frame_maximum) == (2, 5)
    assert diagnostic.image_frames_absent_from_xml == 5
    assert diagnostic.xml_frames_without_images == 0
    assert diagnostic.first_20_image_frames_absent_from_xml == (1, 3, 4, 6, 7)
    assert diagnostic.last_20_image_frames_absent_from_xml == (1, 3, 4, 6, 7)
    assert diagnostic.has_leading_missing_xml_frames
    assert diagnostic.has_trailing_missing_xml_frames
    assert diagnostic.has_interior_missing_xml_gaps
    assert diagnostic.contiguous_missing_xml_blocks == ((1, 1), (3, 4), (6, 7))


def test_interior_xml_coverage_gap_is_explicit_non_fatal_warning(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    for number in (3, 4, 5):
        Image.new("RGB", (100, 80), "white").save(images / f"img{number:05d}.jpg")
    xml.write_text(
        xml.read_text(encoding="utf-8").replace('num="2"', 'num="5"', 1),
        encoding="utf-8",
    )

    sequence = parse_sequence(xml, images, "train")
    report = validate_sequence(sequence)

    assert sequence.annotation_coverage.unannotated_image_frame_numbers == (2, 3, 4)
    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 1
    assert report.issues[0].code == "image_frame_absent_from_xml"
    assert report.issues[0].severity == "warning"


def test_trailing_xml_coverage_gap_is_non_fatal_warning(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path, second_frame=False)
    for number in (2, 3, 4):
        Image.new("RGB", (100, 80), "white").save(images / f"img{number:05d}.jpg")

    sequence = parse_sequence(xml, images, "train")
    report = validate_sequence(sequence)

    assert sequence.annotation_coverage.unannotated_image_frame_numbers == (2, 3, 4)
    assert report.is_valid
    assert report.warning_count == 1


def test_xml_frame_without_image_remains_fatal_integrity_error(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    (images / "img00002.jpg").unlink()

    report = validate_sequence(parse_sequence(xml, images, "train"))

    assert not report.is_valid
    assert report.error_count == 1
    assert report.issues[0].code == "xml_frame_without_image"
    assert report.issues[0].severity == "error"


def test_annotation_cli_exits_zero_when_only_coverage_warnings_exist(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    xml, images = _write_sequence(root, second_frame=False)
    (root / "DETRAC-Images").mkdir()
    target_images = root / "DETRAC-Images" / "MVI_SYNTHETIC"
    images.rename(target_images)
    Image.new("RGB", (100, 80), "white").save(target_images / "img00002.jpg")
    train_annotations = root / "DETRAC-Train-Annotations-XML"
    train_annotations.mkdir()
    xml.rename(train_annotations / "MVI_SYNTHETIC.xml")
    (root / "DETRAC-Test-Annotations-XML").mkdir()
    (root / "DETRAC-Toolkits").mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_annotations.py",
            "--root",
            str(root),
            "--sequence",
            "MVI_SYNTHETIC",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    report = payload["reports"][0]
    assert completed.returncode == 0, completed.stderr
    assert report["error_count"] == 0
    assert report["warning_count"] == 1
    assert report["issues"][0]["code"] == "image_frame_absent_from_xml"


def test_validator_accepts_observed_one_pixel_right_and_bottom_border_extent(
    tmp_path: Path,
) -> None:
    xml, images = _write_sequence(tmp_path, image_size=(960, 540))
    xml.write_text(
        xml.read_text(encoding="utf-8")
        .replace('left="10.5"', 'left="959.5"', 1)
        .replace('top="20.25"', 'top="539.5"', 1)
        .replace('width="30.75"', 'width="1.5"', 1)
        .replace('height="40.5"', 'height="1.5"', 1),
        encoding="utf-8",
    )

    sequence = parse_sequence(xml, images, "train")
    report = validate_sequence(sequence)

    assert sequence.frames[0].targets[0].box.xyxy == (959.5, 539.5, 961.0, 541.0)
    assert "box_out_of_image_range" not in {issue.code for issue in report.issues}


def test_validator_rejects_boxes_beyond_documented_border_tolerance(tmp_path: Path) -> None:
    xml, images = _write_sequence(tmp_path)
    xml.write_text(
        xml.read_text(encoding="utf-8").replace('width="30.75"', 'width="1000"', 1),
        encoding="utf-8",
    )

    report = validate_sequence(parse_sequence(xml, images, "train"))

    assert "box_out_of_image_range" in {issue.code for issue in report.issues}


def test_annotation_cli_rejects_official_test_without_acknowledgement(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_annotations.py",
            "--root",
            str(tmp_path),
            "--partition",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--allow-official-test" in completed.stderr


def test_annotation_cli_resolves_official_train_root_from_environment(raw_dataset_factory) -> None:
    root = raw_dataset_factory()
    environment = {"UA_DETRAC_ROOT": str(root)}
    completed = subprocess.run(
        [sys.executable, "scripts/validate_annotations.py", "--sequence", "TRAIN_001"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 1
    assert "Dataset root is required" not in completed.stderr


def test_visualizer_requires_frame_selection_for_partition_wide_render(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_ground_truth.py",
            "--root",
            str(tmp_path),
            "--all",
            "--output",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--frames" in completed.stderr


def test_audit_selection_falls_back_to_extra_train_sequences_and_stays_within_bounds() -> None:
    from datasets.visualization import select_audit_frames

    entries = [
        {
            "name": f"SEQ_{index:02d}",
            "frame_count": 20,
            "sequence_attributes": {"camera_state": "stable", "sence_weather": "sunny"},
        }
        for index in range(6)
    ]

    selection = select_audit_frames(entries, requested_frames=40)

    assert len(selection) == 40
    assert len({item.sequence_name for item in selection}) == 5
    assert all(1 <= item.frame_number <= 20 for item in selection)


def test_visualizer_renders_overlays_and_refuses_nonempty_output(tmp_path: Path) -> None:
    from datasets.visualization import render_sequence_frames

    xml, images = _write_sequence(tmp_path)
    sequence = parse_sequence(xml, images, "train")
    output = tmp_path / "output"
    written = render_sequence_frames(sequence, [1], output)

    assert written[0].is_file()
    assert Image.open(written[0]).getpixel((10, 20)) != (255, 255, 255)
    with pytest.raises(FileExistsError, match="nonempty"):
        render_sequence_frames(sequence, [1], output)
