"""Regression tests for bugs fixed during the November 2026 audit (Phases 3 & 5).

Pure unit tests — fast, deterministic, CPU-only, zero external disk or network dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from galaxy_jepa.callbacks.collapse import collapse_signals
from galaxy_jepa.core.stopping import read_curve
from galaxy_jepa.data.cache import bake_cache, fit_normalise
from galaxy_jepa.data.metadata import assert_radec_agree, photometric_snr
from galaxy_jepa.data.pull import with_derived_columns
from galaxy_jepa.data.sanity import galaxy_zone_metrics
from galaxy_jepa.data.splits import LeakError, assert_uncertainty_firewall, partition_uncertainty
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline
from galaxy_jepa.data.validity import validity_mask
from galaxy_jepa.harness import HarnessConfig, PathsConfig
from galaxy_jepa.objectives.loop import train_objective
from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import ladder as ladder_mod
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import nulls as nz
from galaxy_jepa.probing import stats as stats_mod
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider
from galaxy_jepa.probing.uncertainty import permutation_p, uncertainty_geometry


class _MemorySource:
    """Minimal in-memory _Source for data.cache unit tests."""

    def __init__(self, items: list[tuple[int, np.ndarray]]) -> None:
        self._items = items
        self.rows = [{"object_id": oid} for oid, _ in items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict[str, int]]:
        oid, img = self._items[i]
        return img, {"object_id": oid}


# --- probing/stats.py -------------------------------------------------------------------------


def test_damped_logistic_matches_sklearn_on_balanced_and_rare_targets() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(800, 16))
    w_true = rng.normal(scale=0.5, size=16)
    for bias in (0.0, -4.0):  # ~50% positive and ~2.5% rare positive
        probs = 1.0 / (1.0 + np.exp(-(x @ w_true + bias)))
        y = (rng.random(x.shape[0]) < probs).astype(np.float64)
        assert 5 <= y.sum() <= x.shape[0] - 5
        w, b = stats_mod.damped_logistic(x, y, c=1.0)
        sk = LogisticRegression(C=1.0, max_iter=2000, tol=1e-9).fit(x, y.astype(int))
        logits_ours = x @ w + b
        logits_sk = x @ sk.coef_[0] + sk.intercept_[0]
        assert np.corrcoef(logits_ours, logits_sk)[0, 1] > 0.9999
        assert np.allclose(w, sk.coef_[0], atol=1e-3, rtol=1e-2)


def test_partial_spearman_and_paired_pearson_bootstrap() -> None:
    rng = np.random.default_rng(7)
    n = 600
    nuis = rng.normal(size=(n, 2))
    latent = rng.normal(size=n)
    x = 2.0 * nuis[:, 0] - nuis[:, 1] + 0.8 * latent + rng.normal(scale=0.4, size=n)
    y = 1.5 * nuis[:, 0] + nuis[:, 1] + 0.8 * latent + rng.normal(scale=0.4, size=n)

    rho = stats_mod.partial_spearman(x, y, nuis)
    rx = stats_mod.residualise_ranks(x, nuis)
    ry = stats_mod.residualise_ranks(y, nuis)
    lo, hi = stats_mod.paired_pearson_bootstrap(rx, ry, n_boot=500, seed=0)
    assert 0.5 < rho < 0.95
    assert lo < rho < hi
    assert math.isnan(
        stats_mod.partial_spearman(
            np.array([1.0, np.nan]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
    )


def test_write_json_atomic_sanitises_non_finite_floats(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    stats_mod.write_json_atomic(target, {"ok": 1.25, "nan": float("nan"), "inf": float("inf")})
    raw = target.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"ok": 1.25, "nan": None, "inf": None}


# --- probing/matching.py, ladder.py, config.py ------------------------------------------------


def test_stratified_match_isolates_zero_point_mass() -> None:
    rng = np.random.default_rng(1)
    # 70% literal zeros (common in deep GZ2 vote fractions) + 30% positive tail
    vals = np.concatenate([np.zeros(140), rng.uniform(0.1, 1.0, size=60)])
    labels = np.zeros(200, dtype=int)
    labels[:40] = 1  # 40 positives in zeros
    labels[140:170] = 1  # 30 positives in positive tail
    kept = match.stratified_match(vals, labels, n_strata=5, seed=0)
    assert kept.size > 0
    # Class-balanced overall AND within the zero stratum
    assert int(labels[kept].sum()) * 2 == kept.size
    zero_kept = kept[vals[kept] == 0.0]
    assert int(labels[zero_kept].sum()) * 2 == zero_kept.size


def test_retention_verdict_unresolved_when_unmatched_non_positive() -> None:
    # Ensure n_matched_test >= MIN_MATCHED_TEST (500) AND m_lo > c_m so we actually reach
    # the `if unmatched <= 0:` guard rather than short-circuiting on sample size or COLLAPSES.
    res = match.retention_verdict(
        a=0.50,
        c=0.55,
        m=0.65,
        m_lo=0.60,
        c_m=0.52,
        n_matched_test=600,
        n_test=600,
    )
    assert res.verdict == match.UNRESOLVED
    assert res.retained is None
    assert res.matched_margin is not None and res.matched_margin > 0.0


def test_untrained_split_key_distinguishes_population_and_middle_ids() -> None:
    tr = list(range(100))
    te = list(range(100, 150))
    k_full = ladder_mod.untrained_split_key(tr, te, population="full")
    k_cond = ladder_mod.untrained_split_key(tr, te, population="conditional")
    assert k_full != k_cond
    tr_mod = list(tr)
    tr_mod[50] = 99999
    assert ladder_mod.untrained_split_key(tr_mod, te, population="full") != k_full


def test_load_untrained_bank_refuses_wrong_population(tmp_path: Path) -> None:
    tr, te = [1, 2, 3], [4, 5]
    split_full = ladder_mod.untrained_split_key(tr, te, population="full")
    split_cond = ladder_mod.untrained_split_key(tr, te, population="conditional")
    bank_full = tmp_path / "full.json"
    bank_cond = tmp_path / "cond.json"
    seeds = {str(s): {"f": 0.5 + 0.001 * s} for s in range(10)}
    bank_full.write_text(
        json.dumps({"split_key": split_full, "population": "full", "seeds": seeds})
    )
    bank_cond.write_text(
        json.dumps({"split_key": split_cond, "population": "conditional", "seeds": seeds})
    )
    cfg = ProbingConfig(
        vote_count_min=1,
        existence_method="untrained_z",
        untrained_bank_path=str(bank_full),
        conditional_untrained_bank_path=str(bank_cond),
    )
    loaded = ladder_mod._load_untrained_bank(cfg, ["f"], split_cond, population="conditional")
    assert loaded is not None and len(loaded["f"]) == 10

    cfg_wrong = ProbingConfig(
        vote_count_min=1,
        existence_method="untrained_z",
        untrained_bank_path=str(bank_full),
    )
    with pytest.raises(ValueError, match="wrong population"):
        ladder_mod._load_untrained_bank(cfg_wrong, ["f"], split_cond, population="conditional")


def test_probing_config_warns_on_unknown_escape_hatch() -> None:
    with pytest.warns(UserWarning, match="unrecognised name"):
        ProbingConfig(vote_count_min=1, escape_hatches=("typo_hatch",))


# --- probing/nulls.py, uncertainty.py, controls.py, data/splits.py ----------------------------


def test_nan_guards_in_nulls_and_uncertainty() -> None:
    assert nz.existence_pvalue(float("nan"), np.array([0.5, 0.52])) == 1.0
    assert nz.untrained_z_pvalue(float("nan"), 0.01, np.array([0.5, 0.52])) == 1.0
    with pytest.raises(ValueError, match="finite untrained_aucs"):
        nz.untrained_z_pvalue(0.8, 0.01, np.array([0.5, float("nan")]))
    with pytest.raises(ValueError, match="non-finite p-value"):
        nz.family_significant({"a": 0.0001, "b": float("nan")}, alpha=0.05, n_tests=2)
    with pytest.raises(ValueError, match="n_tests"):
        nz.family_significant({"a": 0.01, "b": 0.02, "c": 0.03}, alpha=0.05, n_tests=2)
    with pytest.raises(ValueError, match="unknown"):
        nz.resolvable_margin(0.01, sd_bar=0.01, alpha=0.05, method="nope", n_tests=5, df=10)
    assert (
        permutation_p(
            np.array([float("nan"), 1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            n_perm=50,
        )
        == 1.0
    )


def test_uncertainty_firewall_rejects_non_finite_fractions() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        partition_uncertainty({1: 0.1, 2: float("nan")})
    with pytest.raises(LeakError, match="non-finite"):
        assert_uncertainty_firewall([0.05, float("inf"), 0.95])


def test_uncertainty_geometry_handles_single_class_extremes_honestly() -> None:
    ids = list(range(20))
    x = np.random.default_rng(0).normal(size=(20, 4))
    rows = {i: {"f_frac": 0.95 if i < 10 else 0.5, "f_n": 20.0} for i in ids}
    labels = LabelProvider(rows, feature_cols={"f": "f_frac"}, nuisance_cols={}, vote_count_min=1)
    res = uncertainty_geometry(
        EmbeddingMatrix(np.asarray(ids, dtype=np.int64), x, encoder_name="test"),
        labels,
        "f",
        ids,
        train_ids=ids[:12],
        test_ids=ids[12:],
        n_perm=100,
    )
    assert math.isnan(res.spearman) and res.pvalue == 1.0


def test_nuisance_panel_uses_train_median_only() -> None:
    from galaxy_jepa.probing.extract import feature_embeddings

    rng = np.random.default_rng(3)
    ids = list(range(80))
    # Train nuisance is around 0..1; test nuisance is shifted to 100..101 (all above train median).
    # Put nuisance directly in x[:, 0]: a buggy pooled median would split test 50/50 and score ~1.0;
    # with train-only median, test is single-class -> chance 0.5 (no test distribution leakage).
    x = rng.normal(scale=0.05, size=(80, 4))
    rows = {}
    for i in ids:
        nuis = float(i) / 40.0 if i < 40 else 100.0 + float(i)
        x[i, 0] = nuis
        rows[i] = {"f_frac": 0.9 if i % 2 else 0.1, "f_n": 30.0, "mag": nuis}
    labels = LabelProvider(
        rows, feature_cols={"f": "f_frac"}, nuisance_cols={"mag": "mag"}, vote_count_min=1
    )
    mat = EmbeddingMatrix(np.asarray(ids, dtype=np.int64), x, encoder_name="test")
    tr_emb = feature_embeddings(mat, labels, "f", ids[:40])
    te_emb = feature_embeddings(mat, labels, "f", ids[40:])
    panel = ctl.nuisance_panel(tr_emb, te_emb, labels, ids[:40], ids[40:], c=1.0)
    assert panel["mag"] == pytest.approx(0.5)


# --- data/ (cache, metadata, pull, validity, sanity) ------------------------------------------


def test_fit_normalise_and_valid_mask_handle_non_finite_pixels() -> None:
    rng = np.random.default_rng(5)
    img = rng.normal(loc=0.5, scale=0.1, size=(3, 16, 16)).astype(np.float32)
    img[0, 3, 4] = np.nan
    img[1, 5, 6] = np.inf
    vm = validity_mask(img)
    assert not vm[3, 4] and not vm[5, 6]

    src = _MemorySource([(i, img.copy()) for i in range(4)])
    fit = fit_normalise(src, AsinhStretch(q=4.0))
    assert all(math.isfinite(m) for m in fit.valid.mean or ())
    assert all(math.isfinite(s) and s > 0 for s in fit.valid.std or ())
    assert all(math.isfinite(m) for m in fit.naive.mean or ())
    assert all(math.isfinite(s) and s > 0 for s in fit.naive.std or ())


def test_bake_cache_noop_preserves_index_file(tmp_path: Path) -> None:
    pipe = Pipeline((AsinhStretch(q=4.0), Normalise(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))))
    img = np.zeros((3, 16, 16), dtype=np.float32)
    src = _MemorySource([(1, img), (2, img)])
    cache_dir = tmp_path / "cache"
    cache = bake_cache(src, pipe, cache_dir, normalisation_hash="testnormhash")
    idx_path = cache.cache_dir / "index.json"
    before = idx_path.read_bytes()
    # Second call with identical IDs does 0 new writes and must leave index.json untouched.
    bake_cache(src, pipe, cache_dir, normalisation_hash="testnormhash")
    assert idx_path.read_bytes() == before


def test_metadata_ra_wrap_and_pull_non_positive_petrorad() -> None:
    assert_radec_agree(
        [{"objID": 1, "gz_ra": 359.9999, "gz_dec": 0.0, "phot_ra": 0.0001, "phot_dec": 0.0}],
        tol_arcsec=2.0,
    )
    with pytest.raises(ValueError, match="at least one row"):
        assert_radec_agree([])
    with pytest.raises(ValueError, match="mismatch"):
        assert_radec_agree(
            [{"objID": 1, "gz_ra": float("nan"), "gz_dec": 0.0, "phot_ra": 0.0, "phot_dec": 0.0}]
        )
    with pytest.raises(ValueError):
        photometric_snr(float("nan"))
    derived = with_derived_columns(
        [
            {"objID": 1, "modelMagErr_r": 0.05, "petroRad_r": -9999.0},
            {"objID": 2, "modelMagErr_r": 0.05, "petroRad_r": 0.0},
            {"objID": 3, "modelMagErr_r": 0.05, "petroRad_r": 4.0},
        ]
    )
    assert [r["petrorad_suspect"] for r in derived] == [1, 1, 0]


def test_sanity_empty_discrete_annulus_returns_invalid() -> None:
    img = np.zeros((3, 32, 32), dtype=np.float64)
    zm = galaxy_zone_metrics(img, petro_rad_arcsec=4.0, pixel_scale=0.4, k=1.0)
    assert zm.faint_valid is False and math.isnan(zm.faint_retention)


# --- callbacks/collapse.py, objectives/loop.py, core/stopping.py, harness.py ------------------


def test_collapse_signals_handles_non_finite_without_crashing() -> None:
    emb = torch.randn(16, 8)
    emb[0, 0] = float("nan")
    sig = collapse_signals(emb)
    assert math.isnan(sig.std) and math.isnan(sig.effective_rank)


def test_fp16_refused_in_harness_and_loop() -> None:
    cfg = HarnessConfig(
        paths=PathsConfig(pretrain_dir="p", probe_dir="q", out_dir="o"),
        autocast="fp16",
    )
    with pytest.raises(ValueError, match="autocast"):
        cfg.autocast_dtype()

    with pytest.raises(ValueError, match="fp16"):
        train_objective(
            None,  # type: ignore[arg-type]
            [],
            device="cpu",
            autocast_dtype=torch.float16,
        )


def test_stopping_guards_order_and_negative_slopes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        read_curve(
            [
                {"epoch": 2.0, "auc": 0.80},
                {"epoch": 2.0, "auc": 0.85},
                {"epoch": 3.0, "auc": 0.90},
            ]
        )
    # Regression / negative gains must stop as DECLINING and never say "climbing"
    verdict = read_curve(
        [
            {"epoch": 1.0, "auc": 0.90},
            {"epoch": 2.0, "auc": 0.88},
            {"epoch": 3.0, "auc": 0.85},
        ],
        flat_delta=0.005,
    )
    assert verdict.stop is True and verdict.label == "DECLINING"
    assert "climbing" not in verdict.reason.lower()
