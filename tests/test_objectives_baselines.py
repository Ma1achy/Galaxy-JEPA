"""MoCo and MAE (baselines build, ``artifacts/baselines.md`` §9 items 4–6 and 10).

Each claim the build list makes about the two objectives, pinned: the encoder starts byte-identical
to M's for the same seed; MAE encodes exactly 64 visible tokens, scores only the masked patches, and
normalises its target as He et al.'s code does; MoCo's views are axis-aligned windows of the
unflipped image, its key branch takes no gradient, and its queue cycles; both export an encoder that
``load_frozen_encoder`` reads over all 256 tokens; both resume to the identical trajectory (MoCo's
queue included); and both can fit a tiny fixed set, the planted positive that shows the loss path
learns at all.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.data.cache import TensorCache, bake_cache, fit_normalise, write_scalars
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Pipeline
from galaxy_jepa.harness import (
    HarnessConfig,
    MaeObjectiveConfig,
    MaeRunConfig,
    MocoObjectiveConfig,
    MocoRunConfig,
    build_objective,
    seed_init,
)
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig
from galaxy_jepa.objectives.loop import train_objective
from galaxy_jepa.objectives.mae import Mae, MaeConfig, patchify, uniform_mask
from galaxy_jepa.objectives.moco import Moco, MocoConfig, blur

TINY = {"patch_size": 16, "embed_dim": 32, "depth": 3, "heads": 2}
NORM_HASH = "test-normalisation-hash"


def _mae(steps=20, **kw) -> Mae:
    base = {
        "batch_size": 4,
        "warmup_steps": 2,
        "lr": 1e-3,
        "lr_final": 1e-6,
        "dec_dim": 32,
        "dec_depth": 2,
        "dec_heads": 2,
        "monitor_every": 5,
    }
    return Mae(seed_init(0, 64, TINY), MaeConfig(steps=steps, **{**base, **kw}))


def _moco(steps=20, **kw) -> Moco:
    base = {
        "batch_size": 4,
        "warmup_steps": 2,
        "lr": 1e-3,
        "lr_final": 1e-6,
        "queue": 16,
        "proj_hidden": 32,
        "proj_dim": 8,
        "window_min": 2,
        "window_max": 3,
        "monitor_every": 5,
    }
    return Moco(seed_init(0, 64, TINY), MocoConfig(steps=steps, **{**base, **kw}))


@pytest.fixture(scope="module")
def cache(pretraining_corpus, tmp_path_factory):
    source = DirectorySource(pretraining_corpus)
    stretch = AsinhStretch(q=4.0)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=0).valid
    out = tmp_path_factory.mktemp("baseline_cache")
    baked = bake_cache(source, Pipeline((stretch, norm)), out / "c", normalisation_hash=NORM_HASH)
    write_scalars(
        baked.cache_dir, {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
    )
    return TensorCache(baked.cache_dir)


def _loader(cache):
    ds = StampDataset(cache, {}, list(cache.object_ids), scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=0)
    return DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True), sampler


def _images(seed=0, b=4):
    return torch.randn(b, 3, 64, 64, generator=torch.Generator().manual_seed(seed))


# ── shared ─────────────────────────────────────────────────────────────────────────────────────


def test_a_baselines_encoder_starts_byte_identical_to_ms():
    m = Jepa(seed_init(0, 64, TINY), JepaConfig(pred_dim=16, pred_depth=1, pred_heads=2))
    for obj in (_mae(), _moco()):
        for (k, a), b in zip(
            m.encoder.state_dict().items(), obj.encoder.state_dict().values(), strict=True
        ):
            assert torch.equal(a, b), (obj.name, k)


@pytest.mark.parametrize("make", [_mae, _moco], ids=["mae", "moco"])
def test_the_objective_interface(make):
    obj = make()
    assert obj.soft_rank_floor is False  # F8: reported, not gating
    trained = {id(p) for g in obj.make_optimiser().param_groups for p in g["params"]}
    assert trained >= {id(p) for p in obj.encoder.parameters()}
    for name, mod in obj.checkpoint_modules().items():
        assert isinstance(mod, torch.nn.Module), name


def test_the_optimisers_are_the_settled_ones():
    mae, moco = _mae(), _moco()
    om, oc = mae.make_optimiser(), moco.make_optimiser()
    assert om.defaults["betas"] == (0.9, 0.95)
    assert {g["weight_decay"] for g in om.param_groups} == {0.0, 0.05}
    no_decay = next(g for g in om.param_groups if g["weight_decay"] == 0.0)["params"]
    assert all(p.ndim <= 1 for p in no_decay)  # biases and norms only
    assert oc.defaults["betas"] == (0.9, 0.999) and oc.defaults["weight_decay"] == 0.1
    key = {id(p) for p in [*moco.key_encoder.parameters(), *moco.key_projector.parameters()]}
    assert not key & {id(p) for g in oc.param_groups for p in g["params"]}


# ── MAE ────────────────────────────────────────────────────────────────────────────────────────


def test_mae_masks_exactly_three_quarters_uniformly_and_reproducibly():
    vis, msk = uniform_mask(8, 256, 64, seed=3)
    assert vis.shape == (8, 64) and msk.shape == (8, 192)
    for v, m in zip(vis, msk, strict=True):
        assert sorted([*v, *m]) == list(range(256))
    v2, _ = uniform_mask(8, 256, 64, seed=3)
    assert np.array_equal(vis, v2)
    assert not np.array_equal(vis, uniform_mask(8, 256, 64, seed=4)[0])
    assert _mae().n_visible == 4  # a 64-px tiny stamp: 16 tokens, 4 visible


def test_mae_encodes_only_the_visible_tokens(monkeypatch):
    mae = _mae()
    seen = []
    real = mae.encoder.run_tokens
    monkeypatch.setattr(
        mae.encoder, "run_tokens", lambda t, **k: (seen.append(t.shape), real(t, **k))[1]
    )
    mae.loss_parts({"image": _images()}, seed=0)
    assert seen == [torch.Size([4, 4, 32])]  # 4 visible of 16 tokens


def test_patchify_matches_the_encoders_token_order():
    x = _images()
    enc = VisionTransformer(img_size=64, **TINY)
    w = torch.zeros_like(enc.patch_embed.weight)
    w[0, 1, 2, 3] = 1.0  # token value = pixel (band 1, row 2, col 3) of each patch
    enc.patch_embed.weight.data, enc.patch_embed.bias.data = (
        w,
        torch.zeros_like(enc.patch_embed.bias),
    )
    tok = enc.patch_embed(x).flatten(2).transpose(1, 2)[..., 0]
    p = patchify(x, 16).reshape(4, 16, 16, 16, 3)  # (B, N, row, col, band)
    assert torch.equal(tok, p[:, :, 2, 3, 1])


def test_norm_pix_target_is_he_et_al_s():
    mae = _mae()
    x = _images()
    t = patchify(x, 16)
    want = (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6) ** 0.5
    assert torch.allclose(mae.target(x), want)


def test_mae_loss_reads_only_the_masked_patches(monkeypatch):
    mae = _mae()
    x = _images()
    base = mae.loss_parts({"image": x}, seed=7).total
    vis, _ = uniform_mask(4, 16, 4, seed=7)
    real = mae.target

    def poisoned(images):
        t = real(images).clone()
        for b in range(4):
            t[b, torch.from_numpy(vis[b])] = 1e6  # visible patches' targets: must not matter
        return t

    monkeypatch.setattr(mae, "target", poisoned)
    assert torch.equal(mae.loss_parts({"image": x}, seed=7).total, base)


# ── MoCo ───────────────────────────────────────────────────────────────────────────────────────


def _captured_view(moco, images, seed):
    """The augmented image the query view tokenises (before the window gather)."""
    seen = []
    real = moco.encoder.patch_embed_tokens
    moco.encoder.patch_embed_tokens = lambda x: (seen.append(x), real(x))[1]
    moco.views(images, seed=seed)
    moco.encoder.patch_embed_tokens = real
    return seen[0]


def test_moco_gain_is_global_plus_a_small_per_band_term_and_noise_follows_sky():
    """The settled views (2026-10-13): colour ratios move by at most the per-band term, so colour
    survives the gain; the noise σ follows each band's sky σ."""
    img = torch.rand(8, 3, 64, 64) + 0.5
    ratio = _captured_view(_moco(blur_p=0.0, noise_u_max=0.0), img, 3) / img
    per = ratio[:, :, 0, 0]  # one gain per image and band
    assert torch.allclose(ratio, per[:, :, None, None].expand_as(ratio), atol=1e-5)
    colour = per / per[:, 1:2]  # relative to r: the global term cancels
    assert ((colour > 0.98 / 1.02 - 1e-6) & (colour < 1.02 / 0.98 + 1e-6)).all()
    assert ((per > 0.9 * 0.98 - 1e-6) & (per < 1.1 * 1.02 + 1e-6)).all()
    assert per[:, 1].std() > 0.02  # the global term actually varies brightness
    noisy = _moco(blur_p=0.0, gain_global=0.0, gain_band=0.0, sky_sigma=(0.1, 1.0, 10.0))
    sd = _captured_view(noisy, torch.zeros(64, 3, 64, 64), 4).std(dim=(0, 2, 3))
    assert torch.allclose(sd / sd[1], torch.tensor([0.1, 1.0, 10.0]), rtol=0.05)


def test_moco_views_are_axis_aligned_windows_of_the_unflipped_image():
    """With every photometric augmentation off, a view is exactly a contiguous, axis-aligned
    sub-grid of the image's own tokens: no flip, rotation or transpose can produce that."""
    moco = _moco(gain_global=0.0, gain_band=0.0, noise_u_max=0.0, blur_p=0.0)
    x = _images()
    full = moco.encoder.patch_embed_tokens(x).detach()
    q, k, side = moco.views(x, seed=11)
    grid = moco.encoder.grid_size
    for b in range(4):
        found = False
        for y0 in range(grid - side + 1):
            for x0 in range(grid - side + 1):
                idx = [(y0 + r) * grid + (x0 + c) for r in range(side) for c in range(side)]
                if torch.equal(q[b].detach(), full[b, idx]):
                    found = True
        assert found, b
    q2, _, _ = moco.views(x, seed=11)
    assert torch.equal(q, q2)  # a pure function of the step seed


def test_blur_identity_at_zero_and_symmetric_kernel():
    x = _images()
    assert torch.allclose(blur(x, torch.zeros(4)), x)
    chiral = torch.zeros(1, 3, 64, 64)
    chiral[..., 10, 20] = 1.0
    out = blur(chiral, torch.tensor([1.5]))
    assert torch.allclose(out[..., 10, 20 - 2], out[..., 10, 20 + 2])  # symmetric: no shift


def test_moco_key_branch_takes_no_gradient_and_the_queue_cycles():
    moco = _moco()
    q0 = moco.queue.clone()
    loss = moco.loss_parts({"image": _images()}, seed=0).total
    loss.backward()
    assert all(
        p.grad is None for p in [*moco.key_encoder.parameters(), *moco.key_projector.parameters()]
    )
    assert int(moco.queue_ptr) == 4
    assert not torch.equal(moco.queue[:4], q0[:4]) and torch.equal(moco.queue[4:], q0[4:])
    for s in range(1, 4):
        moco.loss_parts({"image": _images(s)}, seed=s)
    assert int(moco.queue_ptr) == 0  # 16 / 4 steps: wrapped
    assert torch.allclose(moco.queue.norm(dim=1), torch.ones(16))
    assert 0.0 <= moco.last_top1 <= 1.0


def test_moco_momentum_update_is_constant_and_only_moves_the_key_branch():
    moco = _moco()
    with torch.no_grad():
        for p in moco.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in moco.key_encoder.parameters()]
    moco.momentum_update(0)
    for b, k, q in zip(
        before, moco.key_encoder.parameters(), moco.encoder.parameters(), strict=True
    ):
        assert torch.allclose(k, 0.999 * b + 0.001 * q)


# ── through the loop ───────────────────────────────────────────────────────────────────────────


def _train(obj, cache, tmp, *, stop_after=None, ck_dir=None):
    loader, sampler = _loader(cache)
    ck = TrainCheckpointer(
        ck_dir or tmp / "ck",
        every=7,
        config_hash="v2:0123456789abcdef",
        normalisation_hash=NORM_HASH,
        schedule={"steps": obj.config.steps},
    )
    return train_objective(
        obj,
        loader,
        device="cpu",
        checkpoint_path=tmp / "enc.pt",
        checkpointer=ck,
        sampler=sampler,
        stop_after=stop_after,
        monitor_batch=next(iter(_loader(cache)[0])),
    )


@pytest.mark.integration
@pytest.mark.parametrize("make", [_mae, _moco], ids=["mae", "moco"])
def test_resume_is_an_identity(make, cache, tmp_path):
    straight = make()
    rs = _train(straight, cache, tmp_path / "a")
    _train(make(), cache, tmp_path / "b", stop_after=9)
    resumed = make()
    rr = _train(resumed, cache, tmp_path / "b")
    assert rr.steps_completed == rs.steps_completed == 20
    assert rs.losses == rr.losses
    for k, v in straight.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k  # MoCo: the queue and pointer too


@pytest.mark.integration
@pytest.mark.parametrize("make", [_mae, _moco], ids=["mae", "moco"])
def test_the_exported_encoder_is_a_plain_frozen_vit(make, cache, tmp_path):
    obj = make(steps=3)
    _train(obj, cache, tmp_path)
    enc = load_frozen_encoder(tmp_path / "enc.pt")
    assert not any(p.requires_grad for p in enc.parameters())
    x = _images()
    assert enc.encode_tokens(x).shape == (4, 16, 32)  # all tokens, no mask
    assert torch.equal(enc.encode(x), obj.encoder.eval().encode(x))


@pytest.mark.integration
@pytest.mark.parametrize("make", [_mae, _moco], ids=["mae", "moco"])
def test_planted_positive_the_loss_can_be_driven_down(make):
    """A fixed set of 64 separable instances, 300 steps: a loss path that learns must fit them.
    Each instance is a flat colour plus faint noise, so any window of either view identifies it —
    MoCo's instance discrimination is learnable at this scale (on pure-noise images, 2–3-token
    windows of different instances are indistinguishable and the loss rightly stays flat); MAE's
    norm-pix target is the faint noise's per-patch structure around the flat colour."""
    g = torch.Generator().manual_seed(0)
    colours = torch.randn(64, 3, 1, 1, generator=g)
    loader = [
        {
            "image": colours[i * 4 : (i + 1) * 4].expand(-1, -1, 64, 64)
            + 0.1 * torch.randn(4, 3, 64, 64, generator=g)
        }
        for i in range(16)
    ]
    obj = make(steps=300, lr=3e-3, warmup_steps=5)
    res = train_objective(obj, loader, device="cpu")
    first, last = np.mean(res.losses[:20]), np.mean(res.losses[-20:])
    assert last < 0.7 * first, (first, last)


def test_configs_round_trip_settled_defaults():
    mc, ac = MocoConfig(), MaeConfig()
    assert (mc.queue, mc.momentum, mc.tau, mc.lr, mc.weight_decay) == (
        65_536,
        0.999,
        0.2,
        1.875e-5,
        0.1,
    )
    assert (ac.mask_ratio, ac.dec_dim, ac.dec_depth, ac.dec_heads, ac.lr, ac.weight_decay) == (
        0.75,
        512,
        8,
        16,
        1.875e-5,
        0.05,
    )
    assert dataclasses.asdict(ac)["betas"] == (0.9, 0.95)


# ── run configs and the switch point (§9 items 1 and 7) ────────────────────────────────────────

BASELINE_CONFIGS = [("configs/moco.yaml", MocoRunConfig), ("configs/mae.yaml", MaeRunConfig)]


@pytest.mark.parametrize(("path", "cls"), BASELINE_CONFIGS, ids=["moco", "mae"])
def test_a_baseline_config_is_ms_with_one_block_swapped(path, cls):
    """Matched to M (§1): every determining field but the objective is M's, so the corpus, the
    freezes, the backbone, the collapse floor and the probe cannot drift between the arms."""
    yaml = pytest.importorskip("yaml")
    with open("configs/pretrain.yaml") as fh:
        m = HarnessConfig(**yaml.safe_load(fh))
    with open(path) as fh:
        b = cls(**yaml.safe_load(fh))
    md, bd = m.determining_dump(), b.determining_dump()
    assert md.pop("objective") != bd.pop("objective")
    assert md == bd
    assert (b.paths.pretrain_dir, b.paths.probe_dir) == (m.paths.pretrain_dir, m.paths.probe_dir)
    assert b.paths.out_dir != m.paths.out_dir  # never M's directory
    # and the file states the settled recipe: its block is the class defaults, nothing tuned
    assert b.objective == type(b.objective)()
    # the schedule inside `objective` is M's too (§1.1): the block is swapped, the budget is not
    mo, bo = m.objective, b.objective
    for k in ("steps", "batch_size", "warmup_steps", "monitor_every", "checkpoint_every"):
        assert getattr(bo, k) == getattr(mo, k), k
    assert bo.lr_final is not None and mo.lr_final is not None
    assert bo.lr_final / bo.lr == pytest.approx(mo.lr_final / mo.lr)  # M's cosine shape


@pytest.mark.parametrize(
    ("block", "settled"),
    [(MocoObjectiveConfig, MocoConfig), (MaeObjectiveConfig, MaeConfig)],
    ids=["moco", "mae"],
)
def test_the_objective_block_mirrors_its_dataclass_field_for_field(block, settled):
    """The lesson of ``ObjectiveConfig.from_jepa_config``: a field the mirror drops is a
    hyperparameter the stamped config no longer describes. So the sets and defaults are pinned,
    and the conversion carries every field and the run's seed."""
    fields = {f.name: f.default for f in dataclasses.fields(settled) if f.name != "seed"}
    assert set(block.model_fields) - {"name"} == set(fields)
    default = block()
    for k, v in fields.items():
        assert getattr(default, k) == v, k
    converted = getattr(default, f"to_{default.name}_config")(seed=7)
    assert isinstance(converted, settled) and converted.seed == 7
    for k in fields:
        assert getattr(converted, k) == getattr(default, k), k


def test_moco_as_configured_states_the_settled_views():
    yaml = pytest.importorskip("yaml")
    with open("configs/moco.yaml") as fh:
        o = MocoRunConfig(**yaml.safe_load(fh)).objective
    assert (o.gain_global, o.gain_band, o.sky_sigma, o.noise_u_max) == (
        0.1,
        0.02,
        (0.414, 0.501, 0.612),
        1.0,
    )


def test_build_objective_dispatches_on_the_config_type():
    moco = MocoObjectiveConfig(queue=64, batch_size=4)
    built = [
        build_objective(
            JepaConfig(pred_dim=16, pred_depth=1, pred_heads=2), seed_init(0, 64, TINY)
        ),
        build_objective(moco.to_moco_config(seed=0), seed_init(0, 64, TINY)),
        build_objective(
            MaeObjectiveConfig(dec_dim=32, dec_depth=1, dec_heads=2).to_mae_config(seed=0),
            seed_init(0, 64, TINY),
        ),
    ]
    assert [type(o) for o in built] == [Jepa, Moco, Mae]
    with pytest.raises(TypeError):
        build_objective(object(), seed_init(0, 64, TINY))  # type: ignore[call-overload]
