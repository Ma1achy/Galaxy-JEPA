"""Unit tests for the Petrosian sampling box (data/bbox.py).

Pins: ``petroRad``→pixel scaling and the ``k`` multiplier; frame clamp; and the
fail-loud contract — a missing / non-finite / non-positive radius logs an explicit
fallback (never a silent default) and is counted in the fallback rate.
"""

import logging
import math

import pytest

from galaxy_jepa.data.bbox import Box, fallback_rate, petrosian_box


def test_box_scales_with_petrorad_and_k():
    # half-width = k * R / scale = 2.5 * 4.0 / 0.396 ≈ 25.25 px
    box = petrosian_box(4.0, 0.396, k=2.5, stamp_px=256, global_half_width_px=90.0)
    assert not box.used_fallback
    assert box.half_width_px == pytest.approx(2.5 * 4.0 / 0.396)


def test_box_clamped_to_frame():
    box = petrosian_box(100.0, 0.396, k=2.5, stamp_px=64, global_half_width_px=22.0)
    assert box.half_width_px == pytest.approx(32.0)  # stamp_px / 2


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -3.0])
def test_missing_radius_falls_back_loudly(bad, caplog):
    with caplog.at_level(logging.WARNING):
        box = petrosian_box(bad, 0.396, stamp_px=64, global_half_width_px=22.0, object_id=42)
    assert box.used_fallback
    assert box.half_width_px == 22.0
    assert any("fall" in rec.message.lower() or "global" in rec.message.lower()
               for rec in caplog.records)


def test_pixel_scale_must_be_positive():
    with pytest.raises(ValueError):
        petrosian_box(4.0, 0.0, stamp_px=64, global_half_width_px=22.0)


def test_fallback_rate():
    boxes = [
        Box(10.0, used_fallback=False),
        Box(22.0, used_fallback=True),
        Box(22.0, used_fallback=True),
        Box(15.0, used_fallback=False),
    ]
    assert fallback_rate(boxes) == pytest.approx(0.5)
    assert fallback_rate([]) == 0.0
    assert math.isclose(fallback_rate([Box(1.0, used_fallback=False)]), 0.0)
