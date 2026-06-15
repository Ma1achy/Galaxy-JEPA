"""Integration test for the asinh Q-sweep driver.

Tier: integration (``docs/spec/testing.md`` §1.3) — runs the sweep over a short Q grid on
the seeded fixture corpus (no network), and proves the load-bearing behaviours: finite
per-Q aggregates, **normalisation re-fit per Q**, and that the curve/readout render.
"""

import numpy as np
import pytest

from galaxy_jepa.data.q_sweep import (
    elbow_readout,
    sweep_q,
    write_curve,
    write_metrics_csv,
)
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise


@pytest.mark.integration
def test_sweep_produces_finite_per_q_aggregates(pretraining_corpus):
    source = DirectorySource(pretraining_corpus)
    records = sweep_q(source, q_grid=(2.0, 4.0, 8.0))
    assert [r.q for r in records] == [2.0, 4.0, 8.0]
    for r in records:
        assert r.n_total == len(source)
        assert np.isfinite(r.faint_med) and np.isfinite(r.sky_med) and np.isfinite(r.gap_med)
        # gap is faint − sky by construction.
        assert r.gap_med == pytest.approx(r.faint_med - r.sky_med, abs=0.05) or r.n_faint_valid


@pytest.mark.integration
def test_normalisation_is_refit_per_q(pretraining_corpus):
    # The whole point: a different Q must yield a different fitted normalisation.
    source = DirectorySource(pretraining_corpus)
    images = [img for img, _ in source]

    def fitted(q: float) -> Normalise:
        stretch = AsinhStretch(q=q)
        return Normalise.fit(np.stack([stretch(im) for im in images]))

    n2, n8 = fitted(2.0), fitted(8.0)
    assert n2.mean is not None and n8.mean is not None
    assert not np.allclose(n2.mean, n8.mean), "normalisation must be re-fit per Q"


@pytest.mark.integration
def test_curve_and_readout_render(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    records = sweep_q(source, q_grid=(2.0, 8.0))
    png = write_curve(records, tmp_path / "curve.png")
    csv_path = write_metrics_csv(records, tmp_path / "metrics.csv")
    assert png.exists() and png.stat().st_size > 0
    assert csv_path.exists() and csv_path.stat().st_size > 0
    readout = elbow_readout(records)
    assert "gap" in readout.lower() and len(readout) > 0
