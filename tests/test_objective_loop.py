"""The generic loop is an identity on JEPA (baselines build, ``artifacts/baselines.md`` §9 item 2).

``train_jepa`` became :func:`~galaxy_jepa.objectives.loop.train_objective` with a :class:`Jepa`, so
MoCo and MAE can share the loop that trained M. The refactor is only safe if it changes nothing, so
these compare it against ``tests/_legacy_jepa_loop.py`` — the pre-refactor loop, verbatim — from the
same init, data order and seeds, and require **equality**: every loss term, the collapse trace,
every weight, the optimiser state and every checkpoint written along the way. On CPU the arithmetic
is deterministic, so a difference of one ulp is a different operation, not noise.

The 500-step identity on M's own config and cache (CPU and MPS) is a separate run against a
worktree at the pre-refactor commit (``artifacts/loop_identity.py``); these are its small,
always-on counterpart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import _legacy_jepa_loop as legacy  # tests/ is on sys.path (rootdir conftest)
from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.data.cache import TensorCache, bake_cache, fit_normalise, write_scalars
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Pipeline
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, train_jepa

NORM_HASH = "test-normalisation-hash"


@pytest.fixture(scope="module")
def cache(pretraining_corpus, tmp_path_factory):
    source = DirectorySource(pretraining_corpus)
    stretch = AsinhStretch(q=4.0)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=0).valid
    out = tmp_path_factory.mktemp("loop_cache")
    baked = bake_cache(
        source, Pipeline((stretch, norm)), out / "cache", normalisation_hash=NORM_HASH
    )
    write_scalars(
        baked.cache_dir, {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
    )
    return TensorCache(baked.cache_dir)


def _jepa(*, steps: int = 20, sigreg_lambda: float = 0.0) -> Jepa:
    torch.manual_seed(0)
    encoder = VisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=3, heads=2)
    cfg = JepaConfig(
        steps=steps,
        batch_size=4,
        warmup_steps=3,
        lr_final=1e-6,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        mask=MaskConfig(beta=0.5),
        monitor_every=5,
        sigreg_lambda=sigreg_lambda,
        seed=0,
    )
    return Jepa(encoder, cfg)


def _loader(cache):
    ds = StampDataset(cache, {}, list(cache.object_ids), scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=0)
    return DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True), sampler


def _checkpointer(directory, steps: int) -> TrainCheckpointer:
    return TrainCheckpointer(
        directory,
        every=7,
        config_hash="v2:0123456789abcdef",
        normalisation_hash=NORM_HASH,
        schedule={
            "steps": steps,
            "lr": 1e-3,
            "warmup_steps": 3,
            "ema_start": 0.996,
            "ema_end": 1.0,
        },
    )


def _run(train, cache, tmp, *, steps=20, sigreg_lambda=0.0, stop_after=None, resume_from=None):
    """One run through `train` (legacy or refactored), returning everything a run leaves behind."""
    jepa = _jepa(steps=steps, sigreg_lambda=sigreg_lambda)
    loader, sampler = _loader(cache)
    monitor = next(iter(_loader(cache)[0]))
    ck = _checkpointer(resume_from or tmp / "ck", steps)
    result = train(
        jepa,
        loader,
        device="cpu",
        monitor_batch=monitor,
        checkpoint_path=tmp / "encoder.pt",
        checkpointer=ck,
        sampler=sampler,
        stop_after=stop_after,
    )
    return jepa, result, ck


def _same_floats(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(
        (math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b, strict=True)
    )


def _assert_identical(old, new):
    (jo, ro, cko), (jn, rn, ckn) = old, new
    assert ro.losses == rn.losses
    assert _same_floats(ro.prediction_losses, rn.prediction_losses)
    assert _same_floats(ro.sigreg_losses, rn.sigreg_losses)
    assert (ro.halted, ro.steps_completed) == (rn.halted, rn.steps_completed)
    assert ro.collapse_trace.keys() == rn.collapse_trace.keys()
    for k in ro.collapse_trace:
        assert _same_floats(ro.collapse_trace[k], rn.collapse_trace[k]), k
    so, sn = jo.state_dict(), jn.state_dict()
    assert so.keys() == sn.keys()
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    # every checkpoint written along the way, byte for byte in its tensors
    eo, en = (
        sorted(p.name for p in cko.dir.glob("*.pt")),
        sorted(p.name for p in ckn.dir.glob("*.pt")),
    )
    assert eo == en and eo
    for name in eo:
        po = torch.load(cko.dir / name, weights_only=False)
        pn = torch.load(ckn.dir / name, weights_only=False)
        for part in ("encoder", "target_encoder", "predictor"):
            for k in po[part]:
                assert torch.equal(po[part][k], pn[part][k]), (name, part, k)
        for so_, sn_ in zip(
            po["optimiser"]["state"].values(), pn["optimiser"]["state"].values(), strict=True
        ):
            for k in so_:
                assert torch.equal(torch.as_tensor(so_[k]), torch.as_tensor(sn_[k])), (name, k)


@pytest.mark.integration
@pytest.mark.parametrize("sigreg_lambda", [0.0, 0.1], ids=["sigreg-off", "sigreg-on"])
def test_refactored_loop_reproduces_the_pre_refactor_loop(cache, tmp_path, sigreg_lambda):
    old = _run(legacy.train_jepa, cache, tmp_path / "old", sigreg_lambda=sigreg_lambda)
    new = _run(train_jepa, cache, tmp_path / "new", sigreg_lambda=sigreg_lambda)
    _assert_identical(old, new)
    assert np.isfinite(new[1].losses).all()


@pytest.mark.integration
def test_refactored_loop_reproduces_stop_and_resume(cache, tmp_path):
    """Stop at 9 (off the checkpoint grid), resume to 20: the restore path is part of the loop."""
    for tag, train in (("old", legacy.train_jepa), ("new", train_jepa)):
        _run(train, cache, tmp_path / tag, stop_after=9)
    old = _run(legacy.train_jepa, cache, tmp_path / "old", resume_from=tmp_path / "old" / "ck")
    new = _run(train_jepa, cache, tmp_path / "new", resume_from=tmp_path / "new" / "ck")
    assert old[1].steps_completed == new[1].steps_completed == 20
    _assert_identical(old, new)


def test_jepa_satisfies_the_objective_interface():
    jepa = _jepa(steps=2)
    assert jepa.name == "jepa"
    assert jepa.soft_rank_floor is True
    assert _jepa(steps=2, sigreg_lambda=0.1).soft_rank_floor is False
    opt = jepa.make_optimiser()
    assert isinstance(opt, torch.optim.AdamW)
    trained = {id(p) for g in opt.param_groups for p in g["params"]}
    assert trained == {id(p) for p in [*jepa.encoder.parameters(), *jepa.predictor.parameters()]}
    assert not trained & {id(p) for p in jepa.target_encoder.parameters()}
