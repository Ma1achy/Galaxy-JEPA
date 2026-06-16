"""End-to-end wiring smoke for the vertical slice (docs/spec, slice plan).

Not a science run — proof the whole sequence holds together offline on micro corpora:
splits (dedup guard exercised) → fp16 cache → JEPA pretrain (collapse monitor) → freeze →
logistic probe AUC on the confident extremes → collapse-trace figure → report. Plus the
calibration pre-flight returning a disk- vs compute-bound verdict.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig
from galaxy_jepa.vertical_slice import _prepare, calibrate, run_slice

pytestmark = pytest.mark.integration

_MODEL = {"patch_size": 16, "embed_dim": 32, "depth": 3, "heads": 2}
_CFG = JepaConfig(
    steps=3,
    batch_size=4,
    warmup_steps=1,
    pred_dim=16,
    pred_depth=2,
    pred_heads=2,
    mask=MaskConfig(beta=0.5),
    monitor_every=1,
)


def _make_corpus(root: Path, *, n: int, base_id: int, labelled: bool, seed: int) -> Path:
    """A tiny corpus: two visually-distinct classes, with the GZ2 fraction at the extremes."""
    from astropy.io import fits  # lazy: integration-only, keep module import dev-light

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        featured = i % 2 == 1
        img = rng.normal(0.0, 0.02, size=(3, 64, 64)).astype(np.float32)
        if featured:  # a bright ring (disk-like) vs a smooth central blob
            yy, xx = np.mgrid[0:64, 0:64]
            ring = np.exp(-((np.hypot(xx - 32, yy - 32) - 18) ** 2) / 20.0)
            img += ring.astype(np.float32)[None] * 0.5
        else:
            yy, xx = np.mgrid[0:64, 0:64]
            blob = np.exp(-(np.hypot(xx - 32, yy - 32) ** 2) / 80.0)
            img += blob.astype(np.float32)[None] * 0.5
        oid = base_id + i
        fits.PrimaryHDU(data=img).writeto(root / f"{oid}.fits", overwrite=True)
        row = {"object_id": oid, "petroRad_r": 4.0, "pixel_scale": 0.396}
        if labelled:
            row[FEATURED_FRACTION_COL] = 0.95 if featured else 0.05  # confident extremes
        rows.append(row)
    with (root / "metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_run_slice_end_to_end(tmp_path):
    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=24, base_id=5000, labelled=True, seed=2)

    report = run_slice(
        pretrain,
        probe,
        tmp_path / "out",
        config=_CFG,
        device="cpu",
        norm_sample=10_000,
        monitor_frac=0.25,
        model_kwargs=_MODEL,
    )

    assert not report.halted
    assert report.checkpoint is not None and Path(report.checkpoint).exists()
    assert report.collapse_png is not None and Path(report.collapse_png).exists()
    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "split_plan.json").exists()
    if report.auc is not None:  # enough confident extremes landed in both splits
        assert 0.0 <= report.auc <= 1.0
    assert isinstance(report.go_no_go(), str)


def test_calibrate_reports_a_bound(tmp_path):
    pretrain = _make_corpus(tmp_path / "pre", n=16, base_id=1000, labelled=False, seed=1)
    probe = _make_corpus(tmp_path / "probe", n=12, base_id=5000, labelled=True, seed=2)
    prep = _prepare(
        pretrain,
        probe,
        tmp_path / "out",
        config=_CFG,
        device="cpu",
        seed=0,
        q=4.0,
        norm_sample=10_000,
        monitor_frac=0.25,
        model_kwargs=_MODEL,
    )
    result = calibrate(prep.jepa, prep.loader, device="cpu", steps=8)
    assert result.bound in ("disk-bound", "compute-bound")
    assert result.it_per_s > 0
    assert "bound" in result.verdict().lower()


def test_jepa_config_carries_mask_beta():
    # the headline knob is reachable + validated (T1.beta-in-range)
    from galaxy_jepa.models.vit import VisionTransformer

    jepa = Jepa(VisionTransformer(img_size=64, **_MODEL), _CFG)
    assert jepa.config.mask.beta == 0.5
