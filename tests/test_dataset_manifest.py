from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from datasets.paths import detect_layout


def _manifest_module():
    try:
        return importlib.import_module("datasets.manifest")
    except ModuleNotFoundError as exc:
        pytest.fail(f"datasets.manifest must exist: {exc}")


def test_build_manifest_discovers_sequence_properties(raw_dataset_factory) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory(size=(321, 241), frame_numbers=(1, 2, 3))

    manifest = manifest_module.build_manifest(
        detect_layout(root), expected_train_sequences=1, expected_test_sequences=1
    )

    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "UA-DETRAC"
    assert manifest["summary"] == {
        "test_sequences": 1,
        "total_sequences": 2,
        "train_sequences": 1,
        "sequences_with_missing_frames": 0,
        "sequences_without_annotations": 0,
    }
    train = manifest["partitions"]["train"][0]
    assert train["name"] == "TRAIN_001"
    assert train["frame_count"] == 3
    assert train["frame_naming_pattern"] == "img{frame:05d}.jpg"
    assert train["first_frame"] == "img00001.jpg"
    assert train["last_frame"] == "img00003.jpg"
    assert train["image_dimensions"] == {"height": 241, "width": 321}
    assert train["annotation_file"] == "DETRAC-Train-Annotations-XML/TRAIN_001.xml"
    assert train["image_directory"] == "DETRAC-Images/TRAIN_001"
    assert train["sequence_attributes"] == {
        "camera_state": "stable",
        "sence_weather": "sunny",
    }


def test_manifest_bytes_are_deterministic(raw_dataset_factory, tmp_path: Path) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory(train=("TRAIN_B", "TRAIN_A"), test=("TEST_B", "TEST_A"))
    layout = detect_layout(root)

    first = manifest_module.build_manifest(
        layout, expected_train_sequences=2, expected_test_sequences=2
    )
    second = manifest_module.build_manifest(
        layout, expected_train_sequences=2, expected_test_sequences=2
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    manifest_module.write_manifest(first, first_path, dataset_root=root)
    manifest_module.write_manifest(second, second_path, dataset_root=root)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert [item["name"] for item in first["partitions"]["train"]] == [
        "TRAIN_A",
        "TRAIN_B",
    ]
    assert first_path.read_bytes().endswith(b"\n")


def test_write_manifest_allows_identical_existing_content_but_rejects_changes(
    raw_dataset_factory, tmp_path: Path
) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory()
    manifest = manifest_module.build_manifest(
        detect_layout(root), expected_train_sequences=1, expected_test_sequences=1
    )
    output = tmp_path / "manifest.json"

    manifest_module.write_manifest(manifest, output, dataset_root=root)
    manifest_module.write_manifest(manifest, output, dataset_root=root)
    changed = {**manifest, "dataset": "changed"}

    with pytest.raises(manifest_module.DatasetOutputError, match="differs"):
        manifest_module.write_manifest(changed, output, dataset_root=root)


def test_write_manifest_rejects_destination_inside_raw_dataset(
    raw_dataset_factory,
) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory()
    manifest = manifest_module.build_manifest(
        detect_layout(root), expected_train_sequences=1, expected_test_sequences=1
    )
    raw_destination = root / "DETRAC-Images" / "TRAIN_001" / "manifest.json"

    with pytest.raises(manifest_module.DatasetOutputError, match="raw dataset"):
        manifest_module.write_manifest(manifest, raw_destination, dataset_root=root)

    assert not raw_destination.exists()


def test_write_manifest_rejects_symlinked_destination_inside_raw_dataset(
    raw_dataset_factory, tmp_path: Path
) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory()
    link = tmp_path / "raw-link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    manifest = manifest_module.build_manifest(
        detect_layout(root), expected_train_sequences=1, expected_test_sequences=1
    )

    with pytest.raises(manifest_module.DatasetOutputError, match="raw dataset"):
        manifest_module.write_manifest(manifest, link / "manifest.json", dataset_root=root)


def test_image_header_reads_are_bounded_parallel_and_ordered(monkeypatch) -> None:
    manifest_module = _manifest_module()
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_dimensions(path: Path) -> tuple[int, int]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return (int(path.name), 1)

    monkeypatch.setattr(manifest_module, "_image_dimensions", fake_dimensions, raising=False)
    paths = [Path(str(number)) for number in range(8)]

    try:
        result = manifest_module._read_image_dimensions(paths, max_workers=4)
    except AttributeError as exc:
        pytest.fail(f"parallel image-header reader must exist: {exc}")

    assert result == [(number, 1) for number in range(8)]
    assert maximum_active > 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("gap", "missing frame numbers"),
        ("bad_name", "inconsistent frame naming"),
        ("xml_name", "XML sequence name"),
        ("dimension", "inconsistent image dimensions"),
    ],
)
def test_build_manifest_rejects_invalid_sequence_data(
    raw_dataset_factory, mutation: str, message: str
) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory()
    sequence = root / "DETRAC-Images" / "TRAIN_001"
    if mutation == "gap":
        (sequence / "img00002.jpg").rename(sequence / "img00003.jpg")
    elif mutation == "bad_name":
        (sequence / "img00002.jpg").rename(sequence / "frame2.jpg")
    elif mutation == "xml_name":
        xml = root / "DETRAC-Train-Annotations-XML" / "TRAIN_001.xml"
        xml.write_text(
            xml.read_text(encoding="utf-8").replace('name="TRAIN_001"', 'name="WRONG"'),
            encoding="utf-8",
        )
    elif mutation == "dimension":
        Image.new("RGB", (640, 480)).save(sequence / "img00002.jpg")

    with pytest.raises(manifest_module.DatasetValidationError, match=message):
        manifest_module.build_manifest(
            detect_layout(root), expected_train_sequences=1, expected_test_sequences=1
        )


def test_build_manifest_rejects_partition_count_and_matching_errors(raw_dataset_factory) -> None:
    manifest_module = _manifest_module()
    root = raw_dataset_factory()
    layout = detect_layout(root)

    with pytest.raises(manifest_module.DatasetValidationError, match="expected 60"):
        manifest_module.build_manifest(
            layout, expected_train_sequences=60, expected_test_sequences=40
        )

    (root / "DETRAC-Test-Annotations-XML" / "TEST_001.xml").unlink()
    with pytest.raises(manifest_module.DatasetValidationError, match="without annotations"):
        manifest_module.build_manifest(
            layout, expected_train_sequences=1, expected_test_sequences=0
        )
