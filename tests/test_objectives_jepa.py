"""Integration smoke for the JEPA objective + collapse monitor (docs/spec/objectives.md).

Not a training run — a wiring check that the whole loop holds together on the fixture
corpus, on CPU, in a couple of steps: finite latent-MSE loss, correct shapes, the EMA
target actually moves, the collapse monitor returns finite signals, and the exported
checkpoint reloads **frozen** (the disk freeze boundary). Plus unit checks on the collapse
signals and the EMA schedule.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from galaxy_jepa.callbacks.collapse import collapse_signals
from galaxy_jepa.core.encoder import assert_frozen
from galaxy_jepa.data.cache import bake_cache, fit_normalise
from galaxy_jepa.data.dataset import StampDataset, rows_by_id
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Pipeline
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, ema_momentum, learning_rate, train_jepa


def _loader(corpus, tmp_path, batch_size=4):
    source = DirectorySource(corpus)
    stretch = AsinhStretch(q=4.0)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=0).valid
    cache = bake_cache(
        source, Pipeline((stretch, norm)), tmp_path / "cache", normalisation_hash="test-norm"
    )
    rows = rows_by_id([r for _, r in source])
    ds = StampDataset(cache, rows, [int(r["object_id"]) for r in rows.values()])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _small_jepa(steps=3):
    encoder = VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=3, heads=2)
    cfg = JepaConfig(
        steps=steps,
        batch_size=4,
        warmup_steps=1,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        mask=MaskConfig(beta=0.5),
        monitor_every=1,
    )
    return Jepa(encoder, cfg)


@pytest.mark.integration
def test_jepa_smoke_step_runs(pretraining_corpus, tmp_path):
    loader = _loader(pretraining_corpus, tmp_path)
    jepa = _small_jepa(steps=3)
    monitor_batch = next(iter(loader))

    result = train_jepa(
        jepa,
        loader,
        device="cpu",
        monitor_batch=monitor_batch,
        checkpoint_path=tmp_path / "ckpt.pt",
    )

    assert not result.halted
    assert len(result.losses) == 3
    assert all(torch.isfinite(torch.tensor(x)) for x in result.losses)  # no NaN/Inf
    assert len(result.collapse_trace["std"]) >= 1
    assert all(s > 0 for s in result.collapse_trace["std"])  # not collapsed in a 3-step smoke


@pytest.mark.integration
def test_ema_target_moves(pretraining_corpus, tmp_path):
    loader = _loader(pretraining_corpus, tmp_path)
    jepa = _small_jepa(steps=2)
    before = next(jepa.target_encoder.parameters()).clone()
    train_jepa(jepa, loader, device="cpu")
    after = next(jepa.target_encoder.parameters())
    assert not torch.equal(before, after)  # EMA pulled the target toward the online encoder


@pytest.mark.integration
def test_checkpoint_reloads_frozen(pretraining_corpus, tmp_path):
    loader = _loader(pretraining_corpus, tmp_path)
    jepa = _small_jepa(steps=2)
    result = train_jepa(jepa, loader, device="cpu", checkpoint_path=tmp_path / "ckpt.pt")
    encoder = load_frozen_encoder(result.checkpoint)
    assert_frozen(encoder)  # must not raise — the freeze boundary through disk
    out = encoder.encode(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 32)


def test_collapse_signals_detect_degeneracy():
    # all-identical embeddings → collapsed: std ~ 0, effective rank ~ 1, cosine ~ 1
    collapsed = collapse_signals(torch.ones(16, 8))
    assert collapsed.std < 1e-6
    assert collapsed.effective_rank < 1.5
    # spread-out embeddings → healthy: rank well above 1
    healthy = collapse_signals(torch.randn(64, 8))
    assert healthy.effective_rank > 2.0
    assert healthy.std > 0.1


def test_ema_momentum_schedule():
    assert ema_momentum(0, 1000, 0.996, 1.0) == pytest.approx(0.996, abs=1e-6)  # starts at start
    assert ema_momentum(1000, 1000, 0.996, 1.0) == pytest.approx(1.0, abs=1e-6)  # ends at end
    mid = ema_momentum(500, 1000, 0.996, 1.0)
    assert 0.996 < mid < 1.0  # monotone ramp in between


class TestTheLearningRateSchedule:
    """D17: warmup, then optional cosine decay. A pure function of ``(step, cfg)``."""

    def test_default_is_byte_identical_to_the_pre_d17_warmup_only_line(self):
        """``lr_final=None`` must reproduce the old schedule exactly, so it lands inert.

        This is what lets the field be added without moving any existing run's ``config_hash``
        or changing what an unchanged config does.
        """
        cfg = JepaConfig(steps=50_000, lr=1e-3, warmup_steps=100)
        assert cfg.lr_final is None
        for step in (0, 1, 50, 99, 100, 101, 1_000, 25_000, 49_999):
            expected = cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)
            assert learning_rate(step, cfg) == expected

    def test_warmup_then_cosine_hits_its_endpoints(self):
        cfg = JepaConfig(steps=50_000, lr=1.25e-4, warmup_steps=1250, lr_final=1.25e-7)
        assert learning_rate(0, cfg) == pytest.approx(cfg.lr / cfg.warmup_steps)
        assert learning_rate(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.lr)
        # the decay only begins after warmup, and ends at the floor
        assert learning_rate(cfg.warmup_steps, cfg) == pytest.approx(cfg.lr, rel=1e-6)
        assert learning_rate(cfg.steps - 1, cfg) == pytest.approx(cfg.lr_final, rel=1e-6)

    def test_the_decay_is_monotone_after_warmup(self):
        cfg = JepaConfig(steps=50_000, lr=1.25e-4, warmup_steps=1250, lr_final=1.25e-7)
        after = [learning_rate(s, cfg) for s in range(1250, 50_000, 500)]
        assert all(b <= a for a, b in zip(after, after[1:], strict=False))

    def test_it_never_runs_past_the_floor_if_a_run_overshoots_its_steps(self):
        """``min(t, 1.0)`` clamps: a step beyond ``cfg.steps`` must not swing back up."""
        cfg = JepaConfig(steps=1_000, lr=1e-3, warmup_steps=100, lr_final=1e-6)
        assert learning_rate(5_000, cfg) == pytest.approx(cfg.lr_final, rel=1e-6)

    def test_the_schedule_is_what_the_loop_applies(self):
        """The loop must use the shared function, not a second copy that could drift from it."""
        import inspect

        from galaxy_jepa.objectives import jepa as mod

        assert "learning_rate(step, cfg)" in inspect.getsource(mod.train_jepa)
