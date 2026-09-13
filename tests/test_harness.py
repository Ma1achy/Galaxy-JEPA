"""End-to-end wiring smoke for the reusable harness (post-slice consolidation).

Pins the consolidation deltas over the slice script: the typed run config round-trips and
stamps; ``run_harness`` produces the headline AUC + bootstrap CI + the web-ready explorer
blobs; and ``evaluate_probe`` reproduces the headline on the same frozen checkpoint without
retraining. Runs offline on micro corpora (the ``eval`` extra: sklearn + matplotlib).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import fit_freeze, make_freeze
from galaxy_jepa.core.config import RunStamp
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.harness import (
    HarnessConfig,
    ModelConfig,
    ObjectiveConfig,
    PathsConfig,
    ProbeConfig,
    RuntimeConfig,
    evaluate_probe,
    run_harness,
)

pytestmark = pytest.mark.integration

_MODEL = ModelConfig(patch_size=16, embed_dim=32, depth=3, heads=2)
_OBJ = ObjectiveConfig(
    steps=3,
    batch_size=4,
    warmup_steps=1,
    pred_dim=16,
    pred_depth=2,
    pred_heads=2,
    beta=0.5,
    monitor_every=1,
)


def _make_corpus(root: Path, *, n: int, base_id: int, labelled: bool, seed: int) -> Path:
    """A tiny corpus: two visually-distinct classes, GZ2 fraction at the confident extremes."""
    from astropy.io import fits  # lazy: integration-only

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        featured = i % 2 == 1
        img = rng.normal(0.0, 0.02, size=(3, 64, 64)).astype(np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        if featured:
            ring = np.exp(-((np.hypot(xx - 32, yy - 32) - 18) ** 2) / 20.0)
            img += ring.astype(np.float32)[None] * 0.5
        else:
            blob = np.exp(-(np.hypot(xx - 32, yy - 32) ** 2) / 80.0)
            img += blob.astype(np.float32)[None] * 0.5
        oid = base_id + i
        fits.PrimaryHDU(data=img).writeto(root / f"{oid}.fits", overwrite=True)
        row = {"object_id": oid, "petroRad_r": 4.0, "pixel_scale": 0.396}
        if labelled:
            row[FEATURED_FRACTION_COL] = 0.95 if featured else 0.05
        rows.append(row)
    with (root / "metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _cfg(pretrain: Path, probe: Path, out: Path, *, fit: bool = True) -> HarnessConfig:
    return HarnessConfig(
        paths=PathsConfig(pretrain_dir=str(pretrain), probe_dir=str(probe), out_dir=str(out)),
        runtime=RuntimeConfig(device="cpu"),
        normalisation=fit_freeze(pretrain)
        if fit
        else make_freeze((0.1, 0.2, 0.3), (1.1, 1.2, 1.3)),
        monitor_frac=0.25,
        objective=_OBJ,
        model=_MODEL,
        probe=ProbeConfig(),
    )


def test_harness_config_roundtrips_and_stamps(tmp_path):
    cfg = _cfg(tmp_path / "pre", tmp_path / "probe", tmp_path / "out", fit=False)
    dumped = cfg.model_dump(mode="json")
    assert HarnessConfig.model_validate(dumped) == cfg  # serialise → load round-trips
    # the objective config builds a JepaConfig carrying the headline β knob
    assert cfg.to_jepa_config().mask.beta == _OBJ.beta
    # it stamps via the existing provenance machinery
    stamp = RunStamp.create(dumped, data_snapshot="manifest:test", seed=cfg.seed)
    assert stamp.config_hash and stamp.seed == cfg.seed


def test_run_harness_end_to_end(tmp_path):
    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=40, base_id=5000, labelled=True, seed=2)
    out = tmp_path / "out"

    report = run_harness(_cfg(pretrain, probe, out))

    assert not report.halted
    assert report.checkpoint is not None and Path(report.checkpoint).exists()
    assert (out / "report.json").exists()
    assert (out / "stamp.json").exists() and (out / "config.json").exists()
    # explorer blobs (numeric, web-ready) are persisted regardless of the AUC outcome
    explorer = out / "explorer"
    assert (explorer / "embeddings.npz").exists()
    assert (explorer / "concept_directions.json").exists()
    assert (explorer / "index.json").exists()
    blob = np.load(explorer / "embeddings.npz")
    assert blob["x"].shape[0] == blob["object_ids"].shape[0]

    if report.auc is not None:  # enough confident extremes landed in both splits
        assert 0.0 <= report.auc <= 1.0
        assert report.auc_lo <= report.auc <= report.auc_hi  # the bootstrap CI brackets it
    assert isinstance(report.go_no_go(), str)


def test_evaluate_probe_reproduces_headline(tmp_path):
    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=40, base_id=5000, labelled=True, seed=2)
    out = tmp_path / "out"
    cfg = _cfg(pretrain, probe, out)

    report = run_harness(cfg)
    if report.auc is None:
        pytest.skip("too few confident extremes for a defined AUC in this micro-run")

    # probe-only re-eval on the existing frozen checkpoint — no retraining — same number
    again = evaluate_probe(cfg)
    assert again.auc == pytest.approx(report.auc)
    assert again.n_test == report.n_test


class TestTheSeedActuallyDeterminesTheEncoder:
    """``RunStamp`` claims a run is determined by ``(config_hash, code_sha, data_snapshot, seed)``.

    It was not. ``seed`` reached the masker (``loss_step(seed=cfg.seed + step)``) and, since Brief
    G1, the data order (``ResumableShuffle``) — but **nothing in the package ever called**
    ``torch.manual_seed``, so the ViT's parameters came from whatever ambient RNG state the process
    held. Two runs with byte-identical stamps produced different encoders. Found while comparing two
    Brief G3 throughput runs whose collapse traces diverged under the same seed.
    """

    def _encoder(self, seed: int):
        import torch

        from galaxy_jepa.harness import seed_init

        torch.manual_seed(999)  # ambient state the fix must override, not inherit
        return seed_init(seed, 64, {"patch_size": 16, "embed_dim": 32, "depth": 2, "heads": 2})

    def test_the_same_seed_gives_byte_identical_weights(self):
        import torch

        a, b = self._encoder(0), self._encoder(0)
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(pa, pb)

    def test_a_different_seed_gives_different_weights(self):
        import torch

        a, b = self._encoder(0), self._encoder(1)
        assert any(
            not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
        )

    def test_the_ambient_rng_state_cannot_leak_in(self):
        """Without the seeding, drawing first from the same generator changed the init."""
        import torch

        from galaxy_jepa.harness import seed_init

        kwargs = {"patch_size": 16, "embed_dim": 32, "depth": 2, "heads": 2}
        torch.manual_seed(7)
        first = seed_init(3, 64, kwargs)
        torch.manual_seed(7)
        _ = torch.randn(1000)  # advance the ambient stream
        second = seed_init(3, 64, kwargs)
        for pa, pb in zip(first.parameters(), second.parameters(), strict=True):
            assert torch.equal(pa, pb)


@pytest.mark.invariant
def test_the_objective_config_round_trip_drops_nothing():
    """Every field the two config classes share must survive ``JepaConfig`` → ``ObjectiveConfig``.

    It did not. ``lr_final`` was added to both in D17 and to ``from_jepa_config`` in neither, so
    a round trip quietly turned the adopted cosine decay back into warmup-only while the config
    still looked clean — and ``ObjectiveConfig`` is what gets *stamped*, so the artefact would
    have described a run that did not happen. Pinned by comparison rather than by hand, so the
    next field added cannot reopen it.
    """
    import dataclasses

    from galaxy_jepa.objectives.jepa import JepaConfig

    shared = {f.name for f in dataclasses.fields(JepaConfig)} & set(ObjectiveConfig.model_fields)
    assert "lr_final" in shared and "sigreg_lambda" in shared  # the two that motivated this

    cfg = JepaConfig(steps=777, lr=1.25e-4, lr_final=1.25e-7, warmup_steps=1250, sigreg_lambda=0.05)
    back = ObjectiveConfig.from_jepa_config(cfg)
    for name in sorted(shared):
        assert getattr(back, name) == getattr(cfg, name), f"{name} was lost in the round trip"


def test_run_harness_keeps_the_whole_checkpoint_trajectory(tmp_path):
    """``keep`` is derived from the schedule, so the trajectory survives to be read back.

    It was not. ``run_harness`` built ``TrainCheckpointer`` with no ``keep`` and took its default
    of 3, so Brief I's arms lost steps 500-1500 *mid-measurement*. The intermediate checkpoints
    are evidence — the label-blind 1C choice reads them, the loss-vs-AUC question is asked of
    them, and a collapse floor is re-derived from them — not a crash backstop that only the
    newest matters for. Pinned by counting what survives on disk and in the manifest rather than
    by reading the argument back, so a refactor cannot satisfy it cosmetically.

    Deliberately not a config field: retention cannot change a number the run produces, and
    ``RunConfig.NON_DETERMINING`` is a top-level-key deny-list, so a nested knob would move
    ``config_hash`` for pure housekeeping. Derived, and bounded by the config either way.
    """
    import json

    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=40, base_id=5000, labelled=True, seed=2)
    out = tmp_path / "out"
    cfg = _cfg(pretrain, probe, out)
    cfg = cfg.model_copy(
        update={"objective": _OBJ.model_copy(update={"steps": 5, "checkpoint_every": 1})}
    )

    run_harness(cfg)

    scheduled = cfg.objective.steps // cfg.objective.checkpoint_every  # 5, and 5 > the old 3
    saved = sorted((out / "checkpoints").glob("*.pt"))
    manifest = json.loads((out / "checkpoints" / "checkpoints.json").read_text())
    assert len(saved) >= scheduled, f"the trajectory was pruned on disk: {[p.name for p in saved]}"
    assert len(manifest["entries"]) >= scheduled, "the manifest forgot part of the trajectory"
    assert {int(e["step"]) for e in manifest["entries"]} >= set(range(1, scheduled + 1))


def test_run_harness_probes_from_the_sidecar_not_the_whole_table(tmp_path, monkeypatch):
    """The in-run probe reads the column sidecar. It was the one call site still re-reading the CSV.

    Brief G2 built the sidecar and Brief I routed ``evaluate_probe`` and
    ``probe_frozen_checkpoint`` through it, but ``run_harness``'s own post-train probe still did
    ``rows_by_id(DirectorySource(probe_dir).rows)`` — 1.49 GB that ``LabelProvider`` copies to
    2.99 GB, built *after* a multi-hour run has already banked its checkpoint. So the path that
    pays for the failure last was the one still carrying it.

    Pinned by making the whole-table path raise for the whole run: a ``run_harness`` that still
    reaches for it cannot finish. ``_probe_rows``' documented fallback is exactly what this closes
    off, and closing it off is the point — ``_prepare`` writes a sidecar here, so the fallback is
    not the route.
    """
    import galaxy_jepa.harness as harness_mod

    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=40, base_id=5000, labelled=True, seed=2)
    out = tmp_path / "out"
    cfg = _cfg(pretrain, probe, out)

    def refuse(rows):
        raise AssertionError("run_harness built the whole metadata table to probe")

    monkeypatch.setattr(harness_mod, "rows_by_id", refuse)
    report = run_harness(cfg)  # must complete without ever reaching for the table

    assert not report.halted
    assert (out / "cache").exists()
    # and the headline is still a number, so the sidecar served the probe rather than starving it
    assert report.auc is None or 0.0 <= report.auc <= 1.0


def test_run_harness_persists_the_numeric_traces(tmp_path):
    """The trajectory is an artefact, not a PNG and a final scalar.

    ``RunReport`` keeps ``final_loss`` and ``_safe_collapse_plot`` renders pixels, and the loss
    decomposition (``prediction_losses`` / ``sigreg_losses``) is deliberately *not* checkpointed
    — diagnostics, not training state. So a multi-hour run's decomposition used to exist only in
    the returned object and die with the process. H5's lesson is that endpoints lie: ``std_final``
    called two arms equivalent where the peak separated them 2.87x.
    """
    import json

    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=40, base_id=5000, labelled=True, seed=2)
    out = tmp_path / "out"
    cfg = _cfg(pretrain, probe, out)
    cfg = cfg.model_copy(
        update={"objective": _OBJ.model_copy(update={"steps": 4, "sigreg_lambda": 0.05})}
    )

    run_harness(cfg)

    traces = json.loads((out / "traces.json").read_text())
    assert len(traces["losses"]) == 4
    # index-aligned with `losses`, so a decomposition can be read step for step
    assert len(traces["prediction_losses"]) == len(traces["losses"])
    assert len(traces["sigreg_losses"]) == len(traces["losses"])
    assert all(v is not None for v in traces["sigreg_losses"])  # the penalty was on
    assert traces["collapse_trace"]["effective_rank"]
    assert traces["steps_completed"] == 4
