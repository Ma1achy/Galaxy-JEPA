"""Invariant tests for bbox-biased masking (docs/masking.md §6 — graceful degradation).

The scheme's load-bearing property is that it is a *strict generalisation* of I-JEPA, so
these protect the degradation guarantees plus the headline diagnostic:

* ``β = 0`` ⇒ uniform weights ⇒ standard I-JEPA (every block position equally likely);
* a full-frame box ⇒ uniform weights for *any* β (the bias can't act where all is galaxy);
* sky-waste (target tokens landing on sky) falls as β rises — the scheme does its job.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    box_to_token_mask,
    sky_waste,
    token_weight_map,
)

pytestmark = pytest.mark.invariant


def test_box_projects_to_central_tokens():
    # a box of half-width 8 px on a 32 px / 2x2-token grid covers the whole grid;
    # a tiny box covers only the central tokens.
    full = box_to_token_mask(half_width_px=16, stamp_px=32, grid_size=2)
    assert full.all()
    grid = box_to_token_mask(half_width_px=40, stamp_px=256, grid_size=16)
    assert grid.sum() < grid.size  # not all tokens are in a small central box
    assert grid[8, 8]  # centre is in-box


def test_beta_zero_is_uniform_weights():
    in_box = box_to_token_mask(half_width_px=40, stamp_px=256, grid_size=16)
    w = token_weight_map(in_box, beta=0.0)
    assert np.allclose(w, 1.0)  # β=0 → all weights equal → standard I-JEPA


def test_full_frame_box_is_uniform_for_any_beta():
    in_box = np.ones((16, 16), dtype=bool)  # entire frame is galaxy
    for beta in (0.0, 0.5, 1.0):
        assert np.allclose(token_weight_map(in_box, beta), 1.0)


def test_beta_out_of_range_rejected():
    with pytest.raises(ValueError):
        token_weight_map(np.ones((4, 4), dtype=bool), beta=1.5)
    with pytest.raises(ValueError):
        MaskConfig(beta=-0.1)


def test_masker_shapes_are_rectangular():
    masker = MultiBlockMasker(grid_size=16, config=MaskConfig(beta=0.5))
    in_box = box_to_token_mask(half_width_px=60, stamp_px=256, grid_size=16)
    w = np.stack([token_weight_map(in_box, 0.5)] * 5)  # batch of 5 identical maps
    ctx, tgt = masker.sample(w, seed=0)
    assert ctx.shape[0] == 5 and tgt.shape[0] == 5  # batched
    assert ctx.dim() == 2 and tgt.dim() == 2  # rectangular (min-keep truncation)
    assert ctx.shape[1] >= 1 and tgt.shape[1] >= 1


def test_sky_waste_falls_as_beta_rises():
    # a small central box: most of the frame is sky, so β should pull targets inward.
    in_box = box_to_token_mask(half_width_px=48, stamp_px=256, grid_size=16)
    in_box_maps = np.stack([in_box] * 64)

    def mean_sky_waste(beta: float) -> float:
        w = np.stack([token_weight_map(in_box, beta)] * 64)
        masker = MultiBlockMasker(grid_size=16, config=MaskConfig(beta=beta))
        _, tgt = masker.sample(w, seed=1)
        return sky_waste(in_box_maps, tgt)

    waste0 = mean_sky_waste(0.0)
    waste1 = mean_sky_waste(1.0)
    assert waste1 < waste0  # biasing onto the galaxy reduces wasted sky targets


def test_beta_one_targets_inside_box():
    # β=1 zeroes sky weight, so a block that *fits* inside the box is strongly preferred.
    # The box must be large enough to contain the target blocks (a block wider than the box
    # is forced to straddle — guaranteed sky waste, independent of β), so use a 12-token box.
    in_box = box_to_token_mask(half_width_px=96, stamp_px=256, grid_size=16)
    in_box_maps = np.stack([in_box] * 32)
    w = np.stack([token_weight_map(in_box, 1.0)] * 32)
    masker = MultiBlockMasker(grid_size=16, config=MaskConfig(beta=1.0))
    _, tgt = masker.sample(w, seed=2)
    assert sky_waste(in_box_maps, tgt) < 0.25
