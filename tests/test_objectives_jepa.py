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
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, ema_momentum, train_jepa


def _loader(corpus, tmp_path, batch_size=4):
    source = DirectorySource(corpus)
    stretch = AsinhStretch(q=4.0)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=0)
    cache = bake_cache(source, Pipeline((stretch, norm)), tmp_path / "cache")
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
