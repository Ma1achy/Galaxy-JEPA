"""The pixel-validity detector (Brief E1).

The detector's whole safety argument is that it keys on *exact constancy*, never on value —
the measured pad value sits 0.0 sigma from sky, so a value threshold could not separate them.
These tests pin that argument: real noise must never be flagged, and every constant shape that
actually occurs in the corpus must be.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.data.validity import (
    MIN_REGION_PX,
    RegionKind,
    analyse_validity,
    edge_connected,
    invalid_planes,
    token_invalid_fraction,
    validity_mask,
)

pytestmark = pytest.mark.invariant

# The sky noise measured on 250 pretrain stamps: median 0.0013, sigma 0.114. The pad value is
# 0.0 — inside that noise, which is exactly why the detector cannot use value.
SKY_SIGMA = 0.114
SKY_MEDIAN = 0.0013


def _sky(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(SKY_MEDIAN, SKY_SIGMA, size=(3, h, w))).astype(np.float32)


def test_pure_sky_is_entirely_valid():
    """The null check: noise at the measured sky sigma must produce zero detections."""
    for seed in range(20):
        assert validity_mask(_sky(seed=seed)).all(), f"false positive on pure sky (seed {seed})"


def test_pad_value_is_indistinguishable_from_sky_by_value():
    """Pins the premise: a value threshold cannot do this job, so constancy must."""
    sky = _sky(seed=1)
    assert abs(0.0 - SKY_MEDIAN) / SKY_SIGMA < 0.05
    assert float(sky.min()) < 0.0 < float(sky.max()), "zero sits inside the sky distribution"


def test_edge_padding_is_flagged_whole_including_its_boundary():
    """A padded strip is invalid to its last column, and real sky beyond it is untouched."""
    img = _sky(seed=2)
    img[:, :, :10] = 0.0
    valid = validity_mask(img)
    assert not valid[:, :10].any(), "padding left partly valid"
    assert valid[:, 10:].all(), "detector ate real sky beyond the pad"


def test_interior_dead_column_and_saturated_core_are_classified_apart():
    """The two interior populations must be separable — the sampler treats them differently."""
    col = _sky(seed=3)
    col[:, 10:210, 30:32] = 0.0  # thin and long: a dead column, 400 px
    kinds = {r.kind for r in analyse_validity(col).regions}
    assert kinds == {RegionKind.LINEAR}

    core = _sky(seed=4)
    core[:, 118:138, 118:138] = 7.5  # blobby and central, 400 px, saturation-like value
    regions = analyse_validity(core).regions
    assert [r.kind for r in regions] == [RegionKind.COMPACT]
    assert regions[0].centroid_radius_px < 3.0, "a saturated core sits at the centre"


def test_planes_split_edge_from_interior_and_partition_the_invalid_set():
    img = _sky(seed=5)
    img[:, :, :8] = 0.0  # edge padding
    img[:, 20:220, 40:42] = 0.0  # interior dead column, the same value, 400 px
    edge, interior = invalid_planes(img)
    assert not (edge & interior).any(), "the planes must not overlap"
    assert np.array_equal(edge | interior, ~validity_mask(img))
    assert edge[:, :8].all() and not edge[20:220, 40:42].any()
    assert interior[20:220, 40:42].all()


def test_a_constant_region_touching_the_border_counts_as_edge_even_at_one_corner():
    img = _sky(seed=6)
    img[:, :20, :20] = 0.0  # 400 px, cornered on the border
    edge, interior = invalid_planes(img)
    assert edge[:20, :20].all() and not interior.any()
    assert analyse_validity(img).regions[0].kind == RegionKind.EDGE


def test_edge_connected_leaves_a_detached_region_alone():
    invalid = np.zeros((32, 32), dtype=bool)
    invalid[0, :4] = True  # touches the border
    invalid[20:24, 20:24] = True  # does not
    reached = edge_connected(invalid)
    assert reached[0, :4].all()
    assert not reached[20:24, 20:24].any()


def test_a_region_below_the_size_floor_is_not_flagged():
    """The quantisation guard: SDSS stamps collide bit-exactly ~17 times each by chance."""
    img = _sky(seed=9)
    img[:, 100:104, 100:104] = 0.0  # 16 px, inside the measured 17-255 empty gap
    assert validity_mask(img).all()
    img[:, 100:116, 100:116] = 0.0  # 256 px, exactly one token
    assert not validity_mask(img)[100:116, 100:116].any()
    assert MIN_REGION_PX == 256


def test_token_pooling_reports_the_invalid_fraction_per_token():
    invalid = np.zeros((64, 64), dtype=bool)
    invalid[:, :8] = True  # one full token column of an 8x8 grid (8 px patches)
    frac = token_invalid_fraction(invalid, grid_size=8)
    assert frac.shape == (8, 8)
    assert np.allclose(frac[:, 0], 1.0)
    assert np.allclose(frac[:, 1:], 0.0)


def test_pooling_refuses_a_grid_the_stamp_does_not_divide_into():
    with pytest.raises(ValueError, match="does not divide"):
        token_invalid_fraction(np.zeros((64, 64), dtype=bool), grid_size=7)


def test_detector_survives_the_per_channel_affine_the_pipeline_applies():
    """Stretch + normalise are per-channel affine; constancy within a channel must survive."""
    img = _sky(seed=8)
    img[:, :, :10] = 0.0
    before = validity_mask(img)
    mean = np.array([0.1, -0.3, 0.7]).reshape(-1, 1, 1)
    std = np.array([2.0, 0.5, 1.3]).reshape(-1, 1, 1)
    after = validity_mask((img - mean) / std)
    assert np.array_equal(before, after)
