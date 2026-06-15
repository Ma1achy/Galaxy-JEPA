"""Unit tests for the spatially-separated stretch-sanity metric (T2.stretch-sanity).

Tier: unit (``docs/spec/testing.md`` §1.1) — pure array logic, no network, no fixtures.

The metric is validated against a stamp with a **known planted signal level S above a
known sky noise σ on a non-zero sky pedestal** — asserting it recovers S, σ (as MAD) and
a correctly-signed gap. The pedestal is the discriminator: an earlier incoherent version
measured the annulus relative to the global mean (not the local sky), so it returned
``pedestal + S`` instead of ``S`` — a test without a pedestal (or on un-normalised data)
passes for *both*, which is exactly how the broken version slipped through.
"""

import logging

import numpy as np
import pytest

from galaxy_jepa.data.sanity import galaxy_zone_metrics

PIXEL_SCALE = 0.396  # arcsec/pixel
_MAD_OF_GAUSSIAN = 0.6745  # MAD ≈ 0.6745·σ for N(0, σ)


def _planted_stamp(
    size: int = 64,
    *,
    pedestal: float,
    sigma: float,
    signal: float,
    inner_px: float,
    outer_px: float,
    seed: int = 0,
) -> np.ndarray:
    """(3, H, W): sky = pedestal + N(0,σ) everywhere; +signal in the annulus [inner,outer)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    r = np.hypot(x - c, y - c)
    img = pedestal + rng.normal(0.0, sigma, size=(3, size, size))
    img[:, (r >= inner_px) & (r < outer_px)] += signal
    return img


def test_metric_recovers_planted_signal_and_noise_on_a_pedestal():
    # Non-zero sky pedestal: a metric referenced to the global mean (not local sky) would
    # return pedestal+S here, failing the retention assert — the discriminating case.
    S, sigma, pedestal = 0.5, 0.1, 5.0
    petro_rad_arcsec = 8.0 * PIXEL_SCALE  # inner 8 px, outer 20 px (k=2.5), inside 64 px
    img = _planted_stamp(
        pedestal=pedestal, sigma=sigma, signal=S, inner_px=8.0, outer_px=20.0,
    )
    zm = galaxy_zone_metrics(img, petro_rad_arcsec, PIXEL_SCALE, k=2.5)

    assert zm.faint_valid
    # retention is the annulus level ABOVE local sky → S, NOT pedestal+S.
    assert zm.faint_retention == pytest.approx(S, abs=0.05)
    # sky floor is the corner MAD → 0.6745·σ.
    assert zm.sky_floor == pytest.approx(_MAD_OF_GAUSSIAN * sigma, rel=0.25)
    gap = zm.faint_retention - zm.sky_floor
    assert gap == pytest.approx(S - _MAD_OF_GAUSSIAN * sigma, abs=0.05)
    assert gap > 0  # planted S clearly exceeds the noise → positive margin


def test_gap_goes_negative_when_signal_sits_inside_the_noise():
    # Sign tracking: a faint annulus (S below one MAD) must give a negative gap.
    S, sigma, pedestal = 0.02, 0.2, 3.0
    petro_rad_arcsec = 8.0 * PIXEL_SCALE
    img = _planted_stamp(
        pedestal=pedestal, sigma=sigma, signal=S, inner_px=8.0, outer_px=20.0,
    )
    zm = galaxy_zone_metrics(img, petro_rad_arcsec, PIXEL_SCALE, k=2.5)
    assert zm.faint_retention == pytest.approx(S, abs=0.05)
    assert (zm.faint_retention - zm.sky_floor) < 0


def test_missing_petrorad_excluded_and_logged(caplog):
    img = _planted_stamp(pedestal=1.0, sigma=0.1, signal=0.5, inner_px=8.0, outer_px=20.0)
    with caplog.at_level(logging.WARNING):
        zm = galaxy_zone_metrics(img, None, PIXEL_SCALE)
    assert zm.faint_valid is False
    assert np.isnan(zm.faint_retention)
    assert np.isfinite(zm.sky_floor)  # sky floor is geometry-independent
    assert "excluding from faint-retention" in caplog.text


def test_oversized_petrorad_gives_empty_annulus(caplog):
    img = _planted_stamp(pedestal=1.0, sigma=0.1, signal=0.5, inner_px=8.0, outer_px=20.0)
    # 40″ → 101 px inner radius, well beyond the 32 px stamp half → annulus empty.
    with caplog.at_level(logging.WARNING):
        zm = galaxy_zone_metrics(img, 40.0, PIXEL_SCALE)
    assert zm.faint_valid is False
    assert np.isnan(zm.faint_retention)
    assert "annulus empty" in caplog.text
