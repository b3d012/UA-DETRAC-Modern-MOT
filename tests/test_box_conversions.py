from __future__ import annotations

import pytest

from datasets.schema import clip_box_to_image, xywh_to_xyxy, xyxy_to_xywh


def test_fractional_xywh_converts_to_half_open_xyxy_without_normalization() -> None:
    box = xywh_to_xyxy(10.5, 20.25, 30.75, 40.5, raw_attributes=(("left", "10.5"),))

    assert box.source_xywh == (10.5, 20.25, 30.75, 40.5)
    assert box.xyxy == (10.5, 20.25, 41.25, 60.75)
    assert box.raw_attributes == (("left", "10.5"),)


def test_xyxy_to_xywh_round_trips_fractional_coordinates() -> None:
    assert xyxy_to_xywh(10.5, 20.25, 41.25, 60.75) == (10.5, 20.25, 30.75, 40.5)


def test_raw_attribute_tuples_cannot_be_mutated() -> None:
    box = xywh_to_xyxy(1, 2, 3, 4, raw_attributes=(("width", "3"),))

    with pytest.raises((AttributeError, TypeError)):
        box.raw_attributes[0] = ("width", "999")  # type: ignore[index]


def test_clipping_is_an_explicit_derived_operation() -> None:
    box = xywh_to_xyxy(959.5, 539.5, 1.5, 1.5)

    assert box.xyxy == (959.5, 539.5, 961.0, 541.0)
    assert clip_box_to_image(box, width=960, height=540) == (959.5, 539.5, 960.0, 540.0)
