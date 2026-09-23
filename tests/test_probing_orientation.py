"""The pixel-moment position angle recovers a planted ellipse, including across the 180° wrap."""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing.orientation import second_moment_orientation


def _ellipse(theta_deg: float, q: float, *, size: int = 96, sigma: float = 10.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    t = np.radians(theta_deg)
    u = (xx - c) * np.cos(t) + (yy - c) * np.sin(t)
    v = -(xx - c) * np.sin(t) + (yy - c) * np.cos(t)
    return np.exp(-0.5 * ((u / sigma) ** 2 + (v / (q * sigma)) ** 2)) + 0.05  # sky pedestal


def _wrapped(a: float, b: float) -> float:
    """Distance between two axial angles, respecting 0° ≡ 180°."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


@pytest.mark.parametrize("theta", [0.0, 17.0, 45.0, 90.0, 133.0, 179.0])
def test_the_planted_angle_is_recovered(theta: float) -> None:
    got, q = second_moment_orientation(_ellipse(theta, 0.4), radius_px=40)
    assert _wrapped(got, theta) < 1.5
    assert q == pytest.approx(0.4, abs=0.06)


def test_angles_half_a_turn_apart_are_the_same_image() -> None:
    a, _ = second_moment_orientation(_ellipse(10.0, 0.4), radius_px=40)
    b, _ = second_moment_orientation(_ellipse(190.0, 0.4), radius_px=40)
    assert _wrapped(a, b) < 1e-6


def test_channels_are_summed_not_just_the_first() -> None:
    stack = np.stack([np.zeros((96, 96)), _ellipse(60.0, 0.5), np.zeros((96, 96))])
    got, _ = second_moment_orientation(stack, radius_px=40)
    assert _wrapped(got, 60.0) < 1.5


def test_a_round_galaxy_reads_as_round() -> None:
    _, q = second_moment_orientation(_ellipse(30.0, 1.0), radius_px=40)
    assert q > 0.95


def test_an_empty_stamp_is_nan_not_a_number() -> None:
    theta, q = second_moment_orientation(np.zeros((64, 64)), radius_px=20)
    assert np.isnan(theta) and np.isnan(q)
