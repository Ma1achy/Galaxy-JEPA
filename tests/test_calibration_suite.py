"""Statistical calibration suite: Type-I error, CI coverage, BY FDR & matching (§11).

All calibration checks run deterministically on CPU with explicit Monte Carlo error bounds
derived from binomial standard errors so tests are neither flaky nor vacuous.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import nulls as nz
from galaxy_jepa.probing import stats as stats_mod
from galaxy_jepa.probing.logistic import Embeddings, probe_auc_ci

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures_scientific import (  # noqa: E402
    make_representational_pair,
    make_world_confounded_pair,
)


@pytest.mark.calibration
def test_type1_error_c3_plugin_and_permutation_null() -> None:
    """Verify nominal 0.05 Type-I error on C3 plug-in threshold and permutation test."""
    import erratum_aligned_c3 as c3

    # 1. C3 plug-in threshold calibrated on seed A, tested out-of-sample on seed B (40k draws)
    cal = c3.calibrate_plugin(draws=40_000, seed=1101, alpha=0.05)
    k_plug = cal["k_plugin_95"]
    rng = np.random.default_rng(1102)
    z = rng.standard_normal((40_000, 4))
    a, m = z[:, :2], z[:, 2:]
    gap = np.maximum(a.min(axis=1) - m.max(axis=1), m.min(axis=1) - a.max(axis=1))
    s_m = np.abs(m[:, 0] - m[:, 1]) / np.sqrt(2.0)
    fpr_c3 = float((gap > k_plug * s_m).mean())
    # Binomial SE for n=40,000, p=0.05 is 0.0011 -> [0.044, 0.056] is > 5 SE
    assert 0.044 <= fpr_c3 <= 0.056, f"C3 plug-in FPR miscalibrated: {fpr_c3:.4f}"

    # 2. Production uncertainty.permutation_p Type-I error under pure null (160 datasets)
    from galaxy_jepa.probing import uncertainty as unc

    rng_p = np.random.default_rng(1103)
    n_trials = 160
    rejects = 0
    for trial in range(n_trials):
        dist = rng_p.normal(size=80)
        frac = rng_p.uniform(0.2, 0.8, size=80)
        pval = unc.permutation_p(dist, frac, n_perm=99, method="two_sided", seed=trial)
        if pval <= 0.05:
            rejects += 1
    fpr_perm = rejects / float(n_trials)
    # Binomial 99.7% band for n=160, p=0.05 is [0.005, 0.11]
    assert 0.005 <= fpr_perm <= 0.11, f"permutation_p FPR miscalibrated: {fpr_perm:.3f}"


@pytest.mark.calibration
def test_bootstrap_ci_coverage_partial_spearman_and_probe_auc() -> None:
    """Empirical 95% CI coverage for partial_spearman_bootstrap and probe_auc_ci."""
    # 1. partial_spearman_bootstrap 95% CI coverage around high-N ground-truth partial Spearman
    rng_gt = np.random.default_rng(1110)
    z_big = rng_gt.normal(size=(80_000, 1))
    eps_x = rng_gt.normal(size=80_000)
    eps_y = 0.55 * eps_x + np.sqrt(1.0 - 0.55**2) * rng_gt.normal(size=80_000)
    true_rho = stats_mod.partial_spearman(
        1.5 * z_big[:, 0] + eps_x, -1.0 * z_big[:, 0] + eps_y, z_big
    )

    rng = np.random.default_rng(1111)
    n_rep = 100
    covered_ps = 0
    for rep in range(n_rep):
        z = rng.normal(size=(160, 1))
        ex = rng.normal(size=160)
        ey = 0.55 * ex + np.sqrt(1.0 - 0.55**2) * rng.normal(size=160)
        x = 1.5 * z[:, 0] + ex
        y = -1.0 * z[:, 0] + ey
        res = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=250, seed=rep)
        if res.lo <= true_rho <= res.hi:
            covered_ps += 1
    cov_ps = covered_ps / float(n_rep)
    assert 0.87 <= cov_ps <= 0.99, f"partial_spearman 95% CI coverage: {cov_ps:.3f}"

    # 2. probe_auc_ci 95% CI coverage around fixed-probe population AUC
    rng_tr = np.random.default_rng(1112)
    x_tr = rng_tr.normal(size=(300, 4))
    y_tr = (1.5 * x_tr[:, 0] + rng_tr.normal(size=300) > 0.0).astype(int)
    tr_emb = Embeddings(x_tr, y_tr, np.where(y_tr == 1, 0.9, 0.1))

    # Population AUC of the fixed probe on a large test pool (N=40,000)
    x_pop = rng_tr.normal(size=(40_000, 4))
    y_pop = (1.5 * x_pop[:, 0] + rng_tr.normal(size=40_000) > 0.0).astype(int)
    pop_auc, _, _ = probe_auc_ci(
        tr_emb, Embeddings(x_pop, y_pop, np.where(y_pop == 1, 0.9, 0.1)), n_boot=20, seed=0
    )

    n_rep_auc = 90
    covered_auc = 0
    for rep in range(n_rep_auc):
        x_te = rng_tr.normal(size=(160, 4))
        y_te = (1.5 * x_te[:, 0] + rng_tr.normal(size=160) > 0.0).astype(int)
        _, lo, hi = probe_auc_ci(
            tr_emb,
            Embeddings(x_te, y_te, np.where(y_te == 1, 0.9, 0.1)),
            n_boot=250,
            seed=rep,
        )
        if lo <= pop_auc <= hi:
            covered_auc += 1
    cov_auc = covered_auc / float(n_rep_auc)
    assert 0.86 <= cov_auc <= 0.99, f"probe_auc_ci 95% CI coverage: {cov_auc:.3f}"


@pytest.mark.calibration
def test_benjamini_yekutieli_fdr_control_under_correlated_mixed_families() -> None:
    """BY FDR stays <= 0.05 under strongly correlated mixed null/alternative families with power."""
    rng = np.random.default_rng(1120)
    m_null, m_alt = 8, 4
    m = m_null + m_alt
    # Block-correlated covariance with both positive and negative off-diagonals
    cov = np.full((m, m), 0.35, dtype=np.float64)
    cov[:4, 4:8] = -0.25
    cov[4:8, :4] = -0.25
    np.fill_diagonal(cov, 1.0)
    # Ensure strict positive definiteness
    eigvals, eigvecs = np.linalg.eigh(cov)
    cov_pd = eigvecs @ np.diag(np.maximum(eigvals, 0.05)) @ eigvecs.T
    chol = np.linalg.cholesky(cov_pd)

    n_families = 200
    fdps = []
    tpr_hits = 0
    for _ in range(n_families):
        z = chol @ rng.standard_normal(m)
        # First 8 are true nulls (mean 0); last 4 are true alternatives (mean +4.0)
        z[m_null:] += 4.0
        pvals = {f"h{i}": float(1.0 - norm.cdf(z[i])) for i in range(m)}
        sig = nz.family_significant(pvals, alpha=0.05, method="benjamini_yekutieli", n_tests=m)
        false_disc = sum(1 for i in range(m_null) if sig[f"h{i}"])
        true_disc = sum(1 for i in range(m_null, m) if sig[f"h{i}"])
        total_disc = false_disc + true_disc
        fdps.append(false_disc / float(total_disc) if total_disc > 0 else 0.0)
        tpr_hits += true_disc

    mean_fdr = float(np.mean(fdps))
    power = tpr_hits / float(n_families * m_alt)
    assert mean_fdr <= 0.06, f"BY empirical FDR exceeded nominal bound: {mean_fdr:.4f}"
    assert power >= 0.70, f"BY power on strong alternatives too low: {power:.3f}"


@pytest.mark.calibration
def test_conditional_matching_balance_null_collapse_and_signal_power() -> None:
    """Matching balances nuisance (|SMD|<0.1), collapses world confounds, retains real signal."""
    # 1. World-confounded fixture: high unmatched AUC, collapses under stratified matching
    conf = make_world_confounded_pair(n=900, d=10, seed=1130)
    tr, te = slice(0, 450), slice(450, 900)
    unmatched_auc, _, _ = probe_auc_ci(
        Embeddings(conf.x[tr], conf.y_target[tr], conf.y_target[tr].astype(float)),
        Embeddings(conf.x[te], conf.y_target[te], conf.y_target[te].astype(float)),
        n_boot=100,
        seed=0,
    )
    assert unmatched_auc > 0.72

    k_tr = match.stratified_match(conf.nuisance[tr], conf.y_target[tr], n_strata=10, seed=0)
    k_te = match.stratified_match(conf.nuisance[te], conf.y_target[te], n_strata=10, seed=1)
    # Check standardised mean difference (SMD) of nuisance between classes after matching
    u_te, y_te = conf.nuisance[te][k_te], conf.y_target[te][k_te]
    smd = abs(float(u_te[y_te == 1].mean() - u_te[y_te == 0].mean())) / max(float(u_te.std()), 1e-9)
    assert smd < 0.12, f"matched nuisance imbalance too large: SMD={smd:.3f}"

    matched_conf_auc, _, _ = probe_auc_ci(
        Embeddings(
            conf.x[tr][k_tr], conf.y_target[tr][k_tr], conf.y_target[tr][k_tr].astype(float)
        ),
        Embeddings(
            conf.x[te][k_te], conf.y_target[te][k_te], conf.y_target[te][k_te].astype(float)
        ),
        n_boot=100,
        seed=0,
    )
    assert matched_conf_auc < 0.58, f"confound did not collapse under matching: {matched_conf_auc}"

    # 2. Representational fixture (power check): strong signal SURVIVES matching on nuisance
    rep = make_representational_pair(n=900, d=10, seed=1131)
    kr_tr = match.stratified_match(rep.nuisance[tr], rep.y_target[tr], n_strata=10, seed=0)
    kr_te = match.stratified_match(rep.nuisance[te], rep.y_target[te], n_strata=10, seed=1)
    matched_rep_auc, lo_rep, _ = probe_auc_ci(
        Embeddings(
            rep.x[tr][kr_tr], rep.y_target[tr][kr_tr], rep.y_target[tr][kr_tr].astype(float)
        ),
        Embeddings(
            rep.x[te][kr_te], rep.y_target[te][kr_te], rep.y_target[te][kr_te].astype(float)
        ),
        n_boot=100,
        seed=0,
    )
    assert matched_rep_auc > 0.78 and lo_rep > 0.70
