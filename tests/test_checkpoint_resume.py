"""Mid-run checkpointing, and the proof that resuming does not quietly change the run.

``train_jepa`` used to write one checkpoint, at the end. Brief F measured the configured budget
at 10.7 h on the M3 Pro, so a crash at hour eight lost the job. The fix is only worth having if a
resume lands *exactly* where the interruption left off — a resume that runs but diverges is worse
than none, because the run would look continuous and not be.

The load-bearing claim these pin: **the trajectory is a function of ``(seed, step)`` alone.** Three
things make that true — the masker is seeded per step (``loss_step(seed=cfg.seed + step)``), the
encoder has no dropout, and :class:`ResumableShuffle` makes the data order a pure function of the
seed. So the identity test below is not a tolerance check; it is an equality.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader

from galaxy_jepa.callbacks.checkpoint import CHECKPOINT_SCHEMA, TrainCheckpointer
from galaxy_jepa.data.cache import TensorCache, bake_cache, fit_normalise, write_scalars
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Pipeline
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, train_jepa

CONFIG_HASH = "v2:0123456789abcdef"
NORM_HASH = "test-normalisation-hash"
SCHEDULE = {"steps": 20, "lr": 1e-3, "warmup_steps": 1, "ema_start": 0.996, "ema_end": 1.0}


def _cache(corpus, tmp_path):
    source = DirectorySource(corpus)
    stretch = AsinhStretch(q=4.0)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=0).valid
    cache = bake_cache(
        source, Pipeline((stretch, norm)), tmp_path / "cache", normalisation_hash=NORM_HASH
    )
    write_scalars(
        cache.cache_dir, {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
    )
    return TensorCache(cache.cache_dir)


def _jepa(steps: int = 20, seed: int = 0, *, init_seed: int | None = None) -> Jepa:
    # `seed` is the *config* seed — it sets the per-step mask seeds, so changing it changes the
    # trajectory by design. `init_seed` perturbs only the weight init, which is what a resume must
    # overwrite. Conflating the two made this test fail for the right reason at first attempt.
    torch.manual_seed(seed if init_seed is None else init_seed)
    encoder = VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=3, heads=2)
    cfg = JepaConfig(
        steps=steps,
        batch_size=4,
        warmup_steps=1,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        mask=MaskConfig(beta=0.5),
        monitor_every=5,
        seed=seed,
    )
    return Jepa(encoder, cfg)


def _loader(cache, batch_size: int = 4, seed: int = 0):
    ids = list(cache.object_ids)
    ds = StampDataset(cache, {}, ids, scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=seed)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True), sampler


def _entry(ck: TrainCheckpointer) -> dict:
    entry = ck.latest()
    assert entry is not None
    return entry


def _checkpointer(directory, *, every: int, **overrides) -> TrainCheckpointer:
    kwargs = {
        "every": every,
        "config_hash": CONFIG_HASH,
        "normalisation_hash": NORM_HASH,
        "schedule": dict(SCHEDULE),
    }
    kwargs.update(overrides)
    return TrainCheckpointer(directory, **kwargs)


# --- the proof ----------------------------------------------------------------------------


@pytest.mark.integration
def test_a_resumed_run_is_identical_to_an_uninterrupted_one(pretraining_corpus, tmp_path):
    """The whole point. Twenty steps straight through, against ten-stop-resume-ten.

    Equality, not approximation: same seed, same data order, same per-step mask seeds. The
    resumed half starts from a *freshly initialised* model, so anything the checkpoint failed to
    carry — the EMA target weights above all — would show up immediately as a different loss.
    """
    cache = _cache(pretraining_corpus, tmp_path)

    straight, straight_sampler = _loader(cache)
    whole = train_jepa(
        _jepa(),
        straight,
        checkpointer=_checkpointer(tmp_path / "ck_a", every=1000),
        sampler=straight_sampler,
    )

    split_loader, sampler = _loader(cache)
    first = train_jepa(
        _jepa(),
        split_loader,
        checkpointer=_checkpointer(tmp_path / "ck_b", every=1000),
        sampler=sampler,
        stop_after=10,
    )
    assert first.steps_completed == 10

    resumed_loader, resumed_sampler = _loader(cache)
    second = train_jepa(
        _jepa(init_seed=7),  # a *different* init: the checkpoint must overwrite all of it
        resumed_loader,
        checkpointer=_checkpointer(tmp_path / "ck_b", every=1000),
        sampler=resumed_sampler,
    )

    assert second.steps_completed == 20
    assert len(whole.losses) == len(second.losses) == 20
    assert second.losses[:10] == first.losses, "the restored prefix must be the run's own history"
    assert second.losses == whole.losses, (
        "a resumed run diverged from the uninterrupted one — the checkpoint is not carrying "
        "everything the trajectory depends on"
    )


@pytest.mark.integration
def test_the_ema_target_is_restored_not_re_derived(pretraining_corpus, tmp_path):
    """Re-deriving the target from the online encoder would restart the EMA and look fine."""
    cache = _cache(pretraining_corpus, tmp_path)
    loader, sampler = _loader(cache)
    jepa = _jepa()
    ck = _checkpointer(tmp_path / "ck", every=1000)
    train_jepa(jepa, loader, checkpointer=ck, sampler=sampler, stop_after=6)
    saved_target = [p.detach().clone() for p in jepa.target_encoder.parameters()]
    # the target has moved away from the online encoder by now, so the two are distinguishable
    assert any(
        not torch.allclose(t, o)
        for t, o in zip(saved_target, jepa.encoder.parameters(), strict=True)
    )

    fresh = _jepa(init_seed=7)
    opt = torch.optim.AdamW([*fresh.encoder.parameters(), *fresh.predictor.parameters()])
    ck.restore(jepa=fresh, optimiser=opt)
    for restored, original in zip(fresh.target_encoder.parameters(), saved_target, strict=True):
        assert torch.equal(restored, original)


# --- refusals -----------------------------------------------------------------------------


class TestATornOrForeignCheckpointIsRefused:
    """The manifest is the commit point, exactly as ``index.json`` is for the fp16 cache.

    This is ``data.cache._assert_untorn`` applied to a different artefact: an interrupted write
    must fail at load rather than half-load, and a checkpoint from another experiment must not be
    silently continued.
    """

    def _saved(self, tmp_path, **overrides):
        jepa = _jepa()
        opt = torch.optim.AdamW([*jepa.encoder.parameters(), *jepa.predictor.parameters()])
        ck = _checkpointer(tmp_path / "ck", every=1, **overrides)
        ck.save(step=1, jepa=jepa, optimiser=opt, losses=[0.5], collapse_history=[])
        return ck, jepa, opt

    def test_a_payload_the_manifest_does_not_name_is_ignored(self, tmp_path):
        ck, _, _ = self._saved(tmp_path)
        (ck.dir / "ckpt_000000999.pt").write_bytes(b"not a checkpoint")
        latest = ck.latest()
        assert latest is not None and latest["step"] == 1  # the uncommitted file is no candidate

    def test_a_tampered_payload_is_refused(self, tmp_path):
        ck, jepa, opt = self._saved(tmp_path)
        path = ck.dir / str(_entry(ck)["file"])
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(RuntimeError, match="hashes to"):
            ck.restore(jepa=jepa, optimiser=opt)

    def test_a_missing_payload_is_not_silently_replaced_by_an_older_one(self, tmp_path):
        ck, jepa, opt = self._saved(tmp_path)
        (ck.dir / str(_entry(ck)["file"])).unlink()
        with pytest.raises(FileNotFoundError, match="the file is gone"):
            ck.restore(jepa=jepa, optimiser=opt)

    def test_a_checkpoint_from_another_experiment_is_refused(self, tmp_path):
        ck, jepa, opt = self._saved(tmp_path)
        other = _checkpointer(tmp_path / "ck", every=1, config_hash="v2:ffffffffffffffff")
        with pytest.raises(RuntimeError, match="different experiment"):
            other.restore(jepa=jepa, optimiser=opt)

    def test_a_resume_may_not_cross_a_normalisation_freeze(self, tmp_path):
        ck, jepa, opt = self._saved(tmp_path)
        other = _checkpointer(tmp_path / "ck", every=1, normalisation_hash="another-statistic")
        with pytest.raises(RuntimeError, match="may not cross a freeze"):
            other.restore(jepa=jepa, optimiser=opt)

    def test_a_changed_schedule_is_refused(self, tmp_path):
        """``steps`` sets the EMA cosine ramp's length, so changing it moves every momentum."""
        ck, jepa, opt = self._saved(tmp_path)
        other = _checkpointer(tmp_path / "ck", every=1, schedule={**SCHEDULE, "steps": 40})
        with pytest.raises(RuntimeError, match="schedule differs"):
            other.restore(jepa=jepa, optimiser=opt)

    def test_a_foreign_schema_is_refused_not_migrated(self, tmp_path):
        ck, jepa, opt = self._saved(tmp_path)
        path = ck.dir / str(_entry(ck)["file"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["schema"] = CHECKPOINT_SCHEMA + 1
        torch.save(payload, path)
        manifest = json.loads(ck.manifest_path.read_text())
        from galaxy_jepa.callbacks.checkpoint import _sha256_file

        manifest["entries"][0]["sha256"] = _sha256_file(path)
        ck.manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="Refusing to guess"):
            ck.restore(jepa=jepa, optimiser=opt)

    def test_no_checkpoint_at_all_is_not_an_error(self, tmp_path):
        jepa = _jepa()
        opt = torch.optim.AdamW(jepa.encoder.parameters())
        assert _checkpointer(tmp_path / "empty", every=1).restore(jepa=jepa, optimiser=opt) is None


@pytest.mark.integration
def test_only_the_most_recent_checkpoints_are_kept(pretraining_corpus, tmp_path):
    cache = _cache(pretraining_corpus, tmp_path)
    loader, sampler = _loader(cache)
    ck = _checkpointer(tmp_path / "ck", every=2, keep=2)
    train_jepa(_jepa(), loader, checkpointer=ck, sampler=sampler, stop_after=8)
    assert ck.latest() is not None
    files = sorted(p.name for p in ck.dir.glob("ckpt_*.pt"))
    steps = [int(e["step"]) for e in json.loads(ck.manifest_path.read_text())["entries"]]
    assert len(files) == 2 and steps == [6, 8]
    assert not list(ck.dir.glob("*.tmp"))


@pytest.mark.integration
def test_resuming_without_a_resumable_sampler_is_refused(pretraining_corpus, tmp_path):
    """The weights would restore and the data order would not; the run would diverge in silence."""
    cache = _cache(pretraining_corpus, tmp_path)
    loader, sampler = _loader(cache)
    ck = _checkpointer(tmp_path / "ck", every=1000)
    train_jepa(_jepa(), loader, checkpointer=ck, sampler=sampler, stop_after=3)
    plain = DataLoader(
        StampDataset(cache, {}, list(cache.object_ids), scalars=cache.scalars),
        batch_size=4,
        shuffle=True,
    )
    with pytest.raises(ValueError, match="worse than none"):
        train_jepa(_jepa(), plain, checkpointer=ck, sampler=None)
