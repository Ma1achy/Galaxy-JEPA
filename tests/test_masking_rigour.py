"""Masking distribution, symmetry, pixel-firewall & cache contract tests (§3, REG-002..005)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    _sample_topleft,
    box_to_token_mask,
    draw_blocks,
    rebuild_masks,
    token_weight_map,
)
from galaxy_jepa.masking.precomputed import (
    SCHEME,
    PrecomputedMasker,
    _cfg_json,
    draw,
    maps_digest,
)
from galaxy_jepa.models.vit import VisionTransformer


@pytest.mark.regression
def test_reg002_initial_context_placement_independent_of_beta_and_bbox() -> None:
    """REG-002: Context top-left corner distribution is uniform and independent of beta / bbox."""
    g = 8
    cfg_lo = MaskConfig(n_target_blocks=2, context_scale=(0.25, 0.25), beta=0.0)
    cfg_hi = MaskConfig(n_target_blocks=2, context_scale=(0.25, 0.25), beta=0.95)

    # Strong central bbox: high beta moves TARGETS to centre, must NOT bias initial CONTEXT.
    in_box = box_to_token_mask(half_width_px=16.0, stamp_px=128, grid_size=g)
    w_lo = token_weight_map(in_box, beta=0.0)[None]
    w_hi = token_weight_map(in_box, beta=0.95)[None]

    n_draws = 1200
    ctx_rows_lo, ctx_cols_lo = [], []
    ctx_rows_hi, ctx_cols_hi = [], []
    tgt_in_lo, tgt_in_hi = [], []
    for s in range(n_draws):
        dims_l, tl_l = draw_blocks(g, cfg_lo, w_lo, seed=s)
        dims_h, tl_h = draw_blocks(g, cfg_hi, w_hi, seed=s)
        # Context top-left is the last entry in tl[0]
        ctx_rows_lo.append(int(tl_l[0, -1, 0]))
        ctx_cols_lo.append(int(tl_l[0, -1, 1]))
        ctx_rows_hi.append(int(tl_h[0, -1, 0]))
        ctx_cols_hi.append(int(tl_h[0, -1, 1]))
        # Check target block 0 falls in-box much more often at beta=0.95 (non-vacuous check)
        r0_l, c0_l = int(tl_l[0, 0, 0]), int(tl_l[0, 0, 1])
        r0_h, c0_h = int(tl_h[0, 0, 0]), int(tl_h[0, 0, 1])
        th, tw = int(dims_l[0, 0]), int(dims_l[0, 1])
        tgt_in_lo.append(float(in_box[r0_l : r0_l + th, c0_l : c0_l + tw].mean()))
        tgt_in_hi.append(float(in_box[r0_h : r0_h + th, c0_h : c0_h + tw].mean()))

    # Non-vacuous: beta=0.95 genuinely concentrates targets onto the galaxy box (>1.5x lift)
    assert np.mean(tgt_in_hi) > 1.5 * np.mean(tgt_in_lo)

    # Initial context top-left is uniform across valid starts [0, g - ctx_h]
    max_r = g - int(dims_l[-1, 0])
    expected_mean = max_r / 2.0
    assert abs(np.mean(ctx_rows_lo) - expected_mean) < 0.25
    assert abs(np.mean(ctx_rows_hi) - expected_mean) < 0.25
    assert abs(np.mean(ctx_cols_lo) - expected_mean) < 0.25
    assert abs(np.mean(ctx_cols_hi) - expected_mean) < 0.25


@pytest.mark.regression
def test_reg003_context_aspect_is_square_and_enumerated_target_law_matches() -> None:
    """REG-003: Context block has square aspect (h == w) and target placement matches exact PMF."""
    g = 6
    cfg = MaskConfig(n_target_blocks=2, context_scale=(0.5, 0.85), beta=0.7)
    w = np.ones((1, g, g), dtype=np.float64)
    for s in range(50):
        dims, _ = draw_blocks(g, cfg, w, seed=s)
        ctx_h, ctx_w = int(dims[-1, 0]), int(dims[-1, 1])
        assert ctx_h == ctx_w

    # Enumerate exact candidate-block probabilities on a 4x4 grid with 2x2 block
    w_small = np.array(
        [
            [1.0, 1.0, 0.2, 0.2],
            [1.0, 1.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    exact_scores = np.array(
        [[w_small[r : r + 2, c : c + 2].mean() for c in range(3)] for r in range(3)],
        dtype=np.float64,
    )
    exact_pmf = (exact_scores / exact_scores.sum()).ravel()
    counts = np.zeros(9, dtype=np.int64)
    rng = np.random.default_rng(123)
    n_samples = 9000
    for _ in range(n_samples):
        r, c = _sample_topleft(w_small, 2, 2, rng)
        counts[r * 3 + c] += 1
    empirical = counts / n_samples
    assert np.allclose(empirical, exact_pmf, atol=0.02)

    # Zero-mass weight map (beta=1.0, empty box) falls back cleanly to uniform over all 9 positions
    zero_w = np.zeros((4, 4), dtype=np.float64)
    counts_z = np.zeros(9, dtype=np.int64)
    for _ in range(4500):
        r, c = _sample_topleft(zero_w, 2, 2, rng)
        counts_z[r * 3 + c] += 1
    assert np.allclose(counts_z / 4500.0, np.full(9, 1.0 / 9.0), atol=0.025)


@pytest.mark.regression
def test_reg004_context_min_keep_subsampling_is_uniform_not_raster_truncated() -> None:
    """REG-004: Context batch truncation keeps spatial positions uniformly without replacement."""
    g = 8
    # Sample 0 has 48 survivors; sample 1 has 44 survivors (batch min = 44).
    # Uniform subsampling gives each survivor in sample 0 equal retention prob 44/48,
    # unlike legacy raster truncation ctx[:44] which drops late spatial positions.
    dims = np.array([[2, 2], [4, 4], [8, 8]], dtype=np.int64)
    tl = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],  # sample 0: 48 survivors (outside [0:4, 0:4])
            [[0, 0], [4, 4], [0, 0]],  # sample 1: 44 survivors (20 tokens excluded)
        ],
        dtype=np.int64,
    )
    hits = np.zeros(g * g, dtype=np.int64)
    n_trials = 1000
    for s in range(n_trials):
        ctx, _ = rebuild_masks(dims, tl, g, seed=s)
        assert ctx.shape == (2, 44)
        assert len(set(ctx[0].tolist())) == 44
        hits[ctx[0].numpy()] += 1

    excluded_0 = {r * g + c for r in range(4) for c in range(4)}
    survivors_0 = sorted(set(range(g * g)) - excluded_0)
    rates = hits[survivors_0] / float(n_trials)
    expected = 44.0 / 48.0
    assert np.all(hits[list(excluded_0)] == 0)
    assert np.allclose(rates, expected, atol=0.05), (
        f"non-uniform retention: min={rates.min():.3f}, max={rates.max():.3f}, want {expected:.3f}"
    )


@pytest.mark.regression
@pytest.mark.invariant
def test_reg005_end_to_end_encoder_pixel_firewall() -> None:
    """REG-005: Perturbing withheld target pixels must NOT change student context representations.

    Runs the actual VisionTransformer patch_embed_tokens -> gather(context) -> run_tokens path
    on fixed masks after normalisation, confirming zero pixel leakage while teacher targets change.
    """
    torch.manual_seed(0)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2).eval()
    masker = MultiBlockMasker(grid_size=4, config=MaskConfig(n_target_blocks=4, beta=0.5))
    w_map = np.ones((2, 4, 4), dtype=np.float64)
    ctx_idx, target_blocks = masker.sample(w_map, seed=7)

    # Verify structural disjointness
    for b in range(2):
        c_set = set(ctx_idx[b].tolist())
        assert len(c_set) > 0
        for tb in target_blocks:
            t_set = set(tb[b].tolist())
            assert len(t_set) > 0
            assert c_set.isdisjoint(t_set)

    img_clean = torch.randn(2, 3, 32, 32)
    img_poisoned = img_clean.clone()
    # Overwrite EVERY non-context patch pixel with +1e4
    for b in range(2):
        c_set = set(ctx_idx[b].tolist())
        for tok in range(16):
            if tok not in c_set:
                r, c = divmod(tok, 4)
                img_poisoned[b, :, r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = 1e4

    with torch.no_grad():
        tok_clean = enc.patch_embed_tokens(img_clean)
        tok_pois = enc.patch_embed_tokens(img_poisoned)
        idx_exp = ctx_idx.unsqueeze(-1).expand(-1, -1, enc.embed_dim)
        z_ctx_clean = enc.run_tokens(torch.gather(tok_clean, 1, idx_exp), norm=True)
        z_ctx_pois = enc.run_tokens(torch.gather(tok_pois, 1, idx_exp), norm=True)
        # Teacher sees full image and MUST change (non-vacuous firewall check)
        z_full_clean = enc.run_tokens(tok_clean, norm=True)
        z_full_pois = enc.run_tokens(tok_pois, norm=True)

    assert torch.equal(z_ctx_clean, z_ctx_pois)
    assert not torch.allclose(z_full_clean, z_full_pois)


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(
    grid=st.sampled_from([4, 6, 8, 16]),
    n_targets=st.integers(min_value=1, max_value=4),
    beta=st.floats(min_value=0.0, max_value=1.0),
    half_w=st.floats(min_value=0.0, max_value=200.0),
    seed=st.integers(min_value=0, max_value=5000),
)
def test_masking_firewall_property_across_grids_boxes_and_betas(
    grid: int, n_targets: int, beta: float, half_w: float, seed: int
) -> None:
    """Property: for every valid grid/box/beta, C & union(T_i) == empty, |C| > 0, |T_i| > 0."""
    cfg = MaskConfig(n_target_blocks=n_targets, beta=beta)
    in_box = box_to_token_mask(half_width_px=half_w, stamp_px=128, grid_size=grid)
    w = token_weight_map(in_box, beta=beta)[None]
    masker = MultiBlockMasker(grid_size=grid, config=cfg)
    ctx, t_blocks = masker.sample(w, seed=seed)
    assert ctx.shape[1] >= 1
    assert len(t_blocks) == n_targets
    c_set = set(ctx[0].tolist())
    for tb in t_blocks:
        assert tb.shape[1] >= 1
        assert c_set.isdisjoint(tb[0].tolist())


@pytest.mark.regression
def test_precomputed_masks_v2_roundtrip_and_negative_refusals(tmp_path: Path) -> None:
    """Verify precomputed-masks-v2 parity with online masker and refusal of stale/tampered files."""
    cfg = MaskConfig(n_target_blocks=4, beta=0.5)
    grid, train_seed = 8, 42
    online = MultiBlockMasker(grid_size=grid, config=cfg)
    w0 = token_weight_map(box_to_token_mask(30.0, 128, grid), beta=0.5)
    w1 = token_weight_map(box_to_token_mask(50.0, 128, grid), beta=0.5)
    batch_w = np.stack([w0, w1], axis=0)

    d0, tl0 = draw(online, batch_w, seed=train_seed)
    d1, tl1 = draw(online, batch_w, seed=train_seed + 1)
    cache_dir = tmp_path / "masks"
    cache_dir.mkdir()
    np.save(cache_dir / "dims.npy", np.stack([d0, d1]))
    np.save(cache_dir / "tl.npy", np.stack([tl0, tl1]))
    np.save(
        cache_dir / "digest.npy",
        np.array([maps_digest(batch_w), maps_digest(batch_w)], dtype=np.uint64),
    )
    meta = {
        "scheme": SCHEME,
        "grid_size": grid,
        "train_seed": train_seed,
        "mask_config": _cfg_json(cfg),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta))

    pm = PrecomputedMasker(cache_dir, grid_size=grid, config=cfg, train_seed=train_seed)
    ctx_p, tbs_p = pm.sample(batch_w, seed=train_seed)
    ctx_o, tbs_o = online.sample(batch_w, seed=train_seed)
    assert torch.equal(ctx_p, ctx_o)
    for tp, to in zip(tbs_p, tbs_o, strict=True):
        assert torch.equal(tp, to)

    # Mismatched batch weight maps (e.g., altered bbox / data order) refused loudly
    with pytest.raises(ValueError, match="weight maps differ"):
        pm.sample(batch_w * 0.5, seed=train_seed)

    # Out-of-range step refused loudly
    with pytest.raises(IndexError, match="outside"):
        pm.sample(batch_w, seed=train_seed + 99)

    # Stale scheme refused loudly
    (cache_dir / "meta.json").write_text(json.dumps({**meta, "scheme": "precomputed-masks-v1"}))
    with pytest.raises(ValueError, match="scheme"):
        PrecomputedMasker(cache_dir, grid_size=grid, config=cfg, train_seed=train_seed)

    # Wrong grid refused loudly
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="grid_size"):
        PrecomputedMasker(cache_dir, grid_size=16, config=cfg, train_seed=train_seed)
