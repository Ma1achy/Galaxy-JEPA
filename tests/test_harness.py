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

from galaxy_jepa.core.config import RunStamp
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.harness import (
    HarnessConfig,
    ModelConfig,
    ObjectiveConfig,
    ProbeConfig,
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


def _cfg(pretrain: Path, probe: Path, out: Path) -> HarnessConfig:
    return HarnessConfig(
        pretrain_dir=str(pretrain),
        probe_dir=str(probe),
        out_dir=str(out),
        device="cpu",
        norm_sample=10_000,
        monitor_frac=0.25,
        objective=_OBJ,
        model=_MODEL,
        probe=ProbeConfig(),
    )


def test_harness_config_roundtrips_and_stamps(tmp_path):
    cfg = _cfg(tmp_path / "pre", tmp_path / "probe", tmp_path / "out")
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
