"""Integration smoke for the JEPA objective + collapse monitor (docs/spec/objectives.md).

Not a training run — a wiring check that the whole loop holds together on the fixture
corpus, on CPU, in a couple of steps: finite latent-MSE loss, correct shapes, the EMA
target actually moves, the collapse monitor returns finite signals, and the exported
checkpoint reloads **frozen** (the disk freeze boundary). Plus unit checks on the collapse
signals and the EMA schedule.
"""

from __future__ import annotations

import math

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


class TestSigRegLandsInert:
    """Brief I: the penalty is off by default, and off must mean *absent*, not zero.

    The analogue of :class:`TestTheLearningRateSchedule`'s first test. ``lr_final=None`` was
    pinned by reproducing the pre-D17 line; ``sigreg_lambda=0.0`` is pinned by reproducing the
    pre-I1 loss body — so an unchanged config is an unchanged run, and every existing artefact
    keeps describing the thing that produced it.
    """

    @staticmethod
    def _fixture(seed: int = 0):
        torch.manual_seed(seed)
        encoder = VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=4, heads=2)
        jepa = Jepa(encoder, JepaConfig(batch_size=4, mask=MaskConfig(beta=0.5)))
        batch = {
            "image": torch.randn(4, 3, 64, 64, generator=torch.Generator().manual_seed(seed + 1)),
            "petro_rad_arcsec": torch.full((4,), 8.0),
            "pixel_scale": torch.full((4,), 0.396),
        }
        return jepa, batch

    @pytest.mark.invariant
    def test_the_default_loss_is_byte_identical_to_the_pre_i1_body(self):
        """The pre-I1 body, transcribed, must give the same number bit for bit."""
        from galaxy_jepa.objectives.jepa import _gather_tokens

        jepa, batch = self._fixture()
        assert jepa.config.sigreg_lambda == 0.0

        images = batch["image"].float()
        weight_maps = jepa.weight_maps(
            batch["petro_rad_arcsec"].numpy(), batch["pixel_scale"].numpy()
        )
        context_idx, target_idx = jepa.masker.sample(weight_maps, seed=3)
        tokens = jepa.encoder.patch_embed_tokens(images)
        context = jepa.encoder.run_tokens(_gather_tokens(tokens, context_idx))
        with torch.no_grad():
            full = jepa.target_encoder.run_tokens(jepa.target_encoder.patch_embed_tokens(images))
            targets = _gather_tokens(full, target_idx)
        pre_i1 = torch.nn.functional.mse_loss(
            jepa.predictor(context, context_idx, target_idx), targets
        )

        assert torch.equal(jepa.loss_step(batch, seed=3), pre_i1)

    @pytest.mark.invariant
    def test_off_reports_absent_rather_than_a_measured_zero(self):
        jepa, batch = self._fixture()
        parts = jepa.loss_parts(batch, seed=3)
        assert parts.sigreg is None
        assert torch.equal(parts.total, parts.prediction)

    def test_on_it_is_the_papers_convex_combination(self):
        """``(1 − λ)·prediction + λ·sigreg`` — the form algorithm 2 uses."""
        jepa, batch = self._fixture()
        jepa.config.sigreg_lambda = 0.05
        jepa.config.sigreg_slices = 64
        parts = jepa.loss_parts(batch, seed=3)
        assert parts.sigreg is not None
        total, pred, pen = (
            float(v.detach()) for v in (parts.total, parts.prediction, parts.sigreg)
        )
        assert total == pytest.approx(0.95 * pred + 0.05 * pen, rel=1e-6)

    def test_on_it_constrains_the_tensor_the_probe_reads(self):
        """SIGReg is applied at ``DEFAULT_LAYER``, not at the encoder's final output.

        For LeJEPA those coincide; here ``encode()`` reads the penultimate block pre-norm while
        the predictor's context is the final block post-norm. Regularising the wrong one would
        leave the probed representation unconstrained and the null uninterpretable.
        """
        from galaxy_jepa.core.encoder import DEFAULT_LAYER
        from galaxy_jepa.objectives.jepa import _gather_tokens
        from galaxy_jepa.objectives.sigreg import sigreg

        jepa, batch = self._fixture()
        jepa.config.sigreg_lambda, jepa.config.sigreg_slices = 0.5, 64
        parts = jepa.loss_parts(batch, seed=3)

        weight_maps = jepa.weight_maps(
            batch["petro_rad_arcsec"].numpy(), batch["pixel_scale"].numpy()
        )
        context_idx, _ = jepa.masker.sample(weight_maps, seed=3)
        tokens = jepa.encoder.patch_embed_tokens(batch["image"].float())
        layers = jepa.encoder.run_tokens_layers(_gather_tokens(tokens, context_idx))
        expected = sigreg(layers[DEFAULT_LAYER].mean(dim=1), seed=3, slices=64)

        assert parts.sigreg is not None
        assert float(parts.sigreg.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)


class TestTheMpsPoolRelease:
    """Brief I: ``train_jepa`` releases the MPS allocator pool at the monitor interval.

    Measured at batch 32: the pool settles at 6.9 GB while only 0.87 GB is live, so 3.2 GB of an
    18 GB machine is held for nothing, and a second arm was killed for memory after the first had
    finished. Releasing holds it at 3.67 GB and costs nothing (133 s vs 135 s over 200 steps).

    **Bit-identity is an empirical result, not something a CPU test can prove** — the call is
    device-guarded, so it never executes here. It was measured instead: after the change, a
    3,000-step arm reproduced all 3,000 per-step losses of the pre-change run exactly, and ran
    5.5% faster for the reduced pressure (`artifacts/i_findings.md`). What *is* pinned here is
    the structural claim that makes that result inevitable — the release is device-guarded, and
    sits outside the forward, the backward and the optimiser step, where an allocator operation
    cannot reach the arithmetic.
    """

    def test_it_is_guarded_and_outside_the_computation(self):
        import inspect

        source = inspect.getsource(train_jepa)
        assert 'if device.startswith("mps") and step % cfg.monitor_every == 0:' in source
        assert "torch.mps.empty_cache()" in source
        release = source.index("torch.mps.empty_cache()")
        for after in ("loss.backward()", "opt.step()", "jepa.ema_update("):
            assert source.index(after) < release, f"the release must come after {after}"

    @pytest.mark.integration
    def test_a_cpu_run_is_untouched_by_it(self, pretraining_corpus, tmp_path):
        """The guard means a CPU run executes exactly the pre-change loop.

        Integration, like every other `pretraining_corpus` consumer here: it materialises a FITS
        fixture corpus (the `data` extra) and runs real training steps. It was unmarked, so the
        fast gate — which installs `dev` only — tried to build the corpus without astropy.
        """
        loader = _loader(pretraining_corpus, tmp_path)
        jepa = Jepa(
            VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=2, heads=2),
            JepaConfig(steps=2, batch_size=4, monitor_every=1),
        )
        result = train_jepa(jepa, loader, device="cpu")
        assert len(result.losses) == 2
        assert all(math.isfinite(v) for v in result.losses)
