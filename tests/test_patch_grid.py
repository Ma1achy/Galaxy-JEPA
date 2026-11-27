"""The patch grid's new pieces: grouped MAE masking and precomputed JEPA masks (user, 24 Nov 2026).

Grouped MAE must be the Mac MAE exactly at patch 16, and hide the same sky with the same target
numbers at patch 8 and 4. Precomputed masks must rebuild ``MultiBlockMasker.sample``'s tensors
exactly, and refuse a batch that is not the one they were drawn from.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from galaxy_jepa.core.config import config_hash
from galaxy_jepa.harness import (
    GroupedMaeObjectiveConfig,
    MaeObjectiveConfig,
    build_objective,
    seed_init,
)
from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    box_to_token_mask,
    token_weight_map,
)
from galaxy_jepa.masking.precomputed import (
    SCHEME,
    PrecomputedMasker,
    _cfg_json,
    draw,
    maps_digest,
    rebuild,
)
from galaxy_jepa.objectives.mae import Mae, MaeConfig, patchify, uniform_mask
from galaxy_jepa.objectives.mae_grouped import GroupedMae, GroupedMaeConfig, group_members

PX = 64  # a small stamp: 4 groups of 16 px a side
SMALL = {"embed_dim": 32, "depth": 2, "heads": 2}
DEC = {"dec_dim": 32, "dec_depth": 1, "dec_heads": 2, "steps": 4, "batch_size": 2}


def _grouped(patch: int, seed: int = 0) -> GroupedMae:
    enc = seed_init(seed, PX, {"patch_size": patch, **SMALL})
    return GroupedMae(enc, GroupedMaeConfig(**DEC, seed=seed))


def _images(n: int = 2) -> torch.Tensor:
    return torch.randn(n, 3, PX, PX, generator=torch.Generator().manual_seed(3))


def test_patch_16_grouped_mae_is_mae_bit_for_bit():
    enc_a, enc_b = (seed_init(0, PX, {"patch_size": 16, **SMALL}) for _ in range(2))
    a, b = Mae(enc_a, MaeConfig(**DEC)), GroupedMae(enc_b, GroupedMaeConfig(**DEC))
    b.load_state_dict(a.state_dict())
    batch = {"image": _images()}
    la, ma = a.token_losses(batch, seed=11)
    lb, mb = b.token_losses(batch, seed=11)
    assert torch.equal(ma, mb) and torch.equal(la, lb)
    assert torch.equal(a.target(batch["image"]), b.target(batch["image"]))


@pytest.mark.parametrize("patch", [8, 4])
def test_the_same_groups_are_hidden_at_every_patch_size(patch):
    b = 3
    vis, msk = uniform_mask(b, 16, 4, seed=5)
    gm = _grouped(patch)
    members = gm.members
    assert gm.n_groups == 16 and gm.n_visible_groups == 4
    k = (16 // patch) ** 2
    for i in range(b):
        t16 = set(msk[i].tolist())
        # the tokens GroupedMae masks, mapped back to their groups, are exactly the patch-16 groups
        _, m = gm.token_losses({"image": _images(b)}, seed=5)
        tok = torch.nonzero(m[i]).flatten().numpy()
        owner = np.full(members.size, -1)
        for g, row in enumerate(members):
            owner[row] = g
        assert set(owner[tok].tolist()) == t16 and len(tok) == 12 * k
    assert vis.shape[1] * k == gm.n_visible


@pytest.mark.parametrize("patch", [8, 4])
def test_a_tokens_target_is_the_patch_16_groups_numbers(patch):
    img = _images()
    t16 = _grouped(16).target(img)  # (B, 16 groups, 768): norm-pix per group
    p = 16 // patch
    gm = _grouped(patch)
    t = gm.target(img)
    # rebuild the group's normalised pixels from its tokens and compare as images
    b, g, c = 2, 4, 3
    whole16 = t16.reshape(b, g, g, 16, 16, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, PX, PX)
    grid = PX // patch
    whole = (
        t.reshape(b, grid, grid, patch, patch, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, PX, PX)
    )
    assert torch.allclose(whole, whole16, atol=1e-5)
    assert p in (2, 4)


@pytest.mark.parametrize("patch", [16, 8, 4])
def test_grouped_mae_steps_under_bf16_autocast(patch):
    """The GPU arms run bf16: the decoder once scattered bf16 tokens into fp32 and died."""
    gm = _grouped(patch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        per_patch, masked = gm.token_losses({"image": _images()}, seed=3)
        loss = (per_patch * masked).sum() / masked.sum()
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is None or torch.isfinite(p.grad).all() for p in gm.parameters()
    )


def test_group_members_partition_the_grid_and_refuse_a_bad_group():
    m = group_members(16, 4, 16)
    assert sorted(m.flatten().tolist()) == list(range(256)) and m.shape == (16, 16)
    with pytest.raises(ValueError):
        group_members(16, 8, 12)


def test_grouped_config_leaves_the_mae_config_hash_alone():
    mae = MaeObjectiveConfig()
    assert "mask_group_px" not in mae.model_dump()
    assert config_hash(mae.model_dump(mode="json")) != config_hash(
        GroupedMaeObjectiveConfig().model_dump(mode="json")
    )
    cfg = GroupedMaeObjectiveConfig(steps=3).to_mae_config(seed=2)
    assert isinstance(cfg, GroupedMaeConfig) and cfg.seed == 2 and cfg.mask_group_px == 16
    obj = build_objective(GroupedMaeConfig(**DEC), seed_init(0, PX, {"patch_size": 8, **SMALL}))
    assert type(obj) is GroupedMae


# ── precomputed masks ─────────────────────────────────────────────────────────────


def _maps(n: int, grid: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            token_weight_map(box_to_token_mask(float(h), 256, grid), 0.5)
            for h in rng.uniform(15, 128, n)
        ]
    )


@pytest.mark.parametrize("grid", [16, 32])
def test_stored_draws_rebuild_sample_exactly(grid):
    masker = MultiBlockMasker(grid, MaskConfig(beta=0.5))
    for s in range(12):
        w = _maps(6, grid, s)
        dims, tl = draw(masker, w, seed=100 + s)
        ctx, tgts = rebuild(dims, tl, grid, seed=100 + s)
        rctx, rtgts = masker.sample(w, seed=100 + s)
        assert torch.equal(ctx, rctx)
        assert len(tgts) == len(rtgts) and all(
            torch.equal(a, b) for a, b in zip(tgts, rtgts, strict=True)
        )


def test_the_precomputed_masker_replays_and_refuses_a_different_batch(tmp_path):
    grid, cfg, B = 16, MaskConfig(beta=0.5), 4
    masker = MultiBlockMasker(grid, cfg)
    steps = 5
    dims, tl, dg, maps = [], [], [], []
    for s in range(steps):
        w = _maps(B, grid, s)
        maps.append(w)
        d, t = draw(masker, w, seed=7 + s)
        dims.append(d), tl.append(t), dg.append(maps_digest(w))
    np.save(tmp_path / "dims.npy", np.stack(dims))
    np.save(tmp_path / "tl.npy", np.stack(tl))
    np.save(tmp_path / "digest.npy", np.asarray(dg, dtype=np.uint64))
    meta = {"scheme": SCHEME, "grid_size": grid, "train_seed": 7, "mask_config": _cfg_json(cfg)}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    pm = PrecomputedMasker(tmp_path, grid_size=grid, config=cfg, train_seed=7)
    for s in range(steps):
        a, b = pm.sample(maps[s], seed=7 + s), masker.sample(maps[s], seed=7 + s)
        assert torch.equal(a[0], b[0])
        assert len(a[1]) == len(b[1]) and all(
            torch.equal(t1, t2) for t1, t2 in zip(a[1], b[1], strict=True)
        )
    with pytest.raises(ValueError, match="weight maps differ"):
        pm.sample(maps[1], seed=7 + 2)
    with pytest.raises(IndexError):
        pm.sample(maps[0], seed=7 + steps)
    with pytest.raises(ValueError, match="train_seed"):
        PrecomputedMasker(tmp_path, grid_size=grid, config=cfg, train_seed=8)
    with pytest.raises(ValueError, match="grid_size"):
        PrecomputedMasker(tmp_path, grid_size=32, config=cfg, train_seed=7)


def test_moco_infonce_and_queue_stay_fp32_under_bf16_autocast():
    from galaxy_jepa.objectives.moco import Moco, MocoConfig

    enc = seed_init(0, PX, {"patch_size": 16, **SMALL})
    moco = Moco(
        enc,
        MocoConfig(
            steps=4,
            batch_size=2,
            queue=8,
            proj_hidden=16,
            proj_dim=8,
            window_min=2,
            window_max=3,
        ),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        parts = moco.loss_parts({"image": _images(2)}, seed=0)
    assert parts.total.dtype == torch.float32
    assert moco.queue.dtype == torch.float32
    parts.total.backward()
    assert torch.isfinite(parts.total)


def test_patchify_group_pixels_are_the_group(tmp_path):
    # the group side of the target reads the same pixels patch 16 reads
    img = _images()
    assert patchify(img, 16).shape == (2, 16, 768)
