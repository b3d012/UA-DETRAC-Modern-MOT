from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def raw_dataset_factory(tmp_path: Path):
    def make(
        *,
        train: tuple[str, ...] = ("TRAIN_001",),
        test: tuple[str, ...] = ("TEST_001",),
        size: tuple[int, int] = (320, 240),
        frame_numbers: tuple[int, ...] = (1, 2),
    ) -> Path:
        root = tmp_path / "ua_detrac"
        images = root / "DETRAC-Images"
        train_xml = root / "DETRAC-Train-Annotations-XML"
        test_xml = root / "DETRAC-Test-Annotations-XML"
        toolkit = root / "DETRAC-Toolkits"
        for directory in (images, train_xml, test_xml, toolkit):
            directory.mkdir(parents=True, exist_ok=True)

        for partition, names in ((train_xml, train), (test_xml, test)):
            for name in names:
                sequence = images / name
                sequence.mkdir()
                for number in frame_numbers:
                    Image.new("RGB", size, color=(number, 0, 0)).save(
                        sequence / f"img{number:05d}.jpg"
                    )
                (partition / f"{name}.xml").write_text(
                    "\n".join(
                        (
                            '<?xml version="1.0" encoding="utf-8"?>',
                            f'<sequence name="{name}">',
                            '  <sequence_attribute camera_state="stable" sence_weather="sunny"/>',
                            '  <frame density="1" num="1">',
                            '    <target_list><target id="1"><box broken="ignored-by-M1"/></target></target_list>',
                            "  </frame>",
                            "</sequence>",
                        )
                    ),
                    encoding="utf-8",
                )
        return root

    return make
