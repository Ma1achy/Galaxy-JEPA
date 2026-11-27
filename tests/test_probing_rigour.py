"""Probing numerical correctness, leakage firewall, bootstrap, mapnull & C3 tests (§4–6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.data.dataset import ResumableShuffle
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig
from galaxy_jepa.objectives.loop import train_objective
from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import ladder as ladder_mod
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import nulls as nz
from galaxy_jepa.probing import stats as stats_mod
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, feature_embeddings
from galaxy_jepa.probing.logistic import probe_direction

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))


# --- REG-017 & REG-006: Unclipped logistic solver derivatives & sklearn parity ------------------


@pytest.mark.regression
@pytest.mark.differential
def test_reg017_logistic_objective_gradient_hessian_exact_beyond_extreme_logits() -> None:
    """REG-017: Objective, gradient, and Hessian must match finite differences even for |z| > 40.

    The historical clipped objective (`clip(z, -35, 35)`) had zero numerical derivative for |z|>35
    while returning non-zero gradient expressions. The unclipped `logaddexp(0, -s*z)` formulation
    is exact across the entire real line.
    """
    rng = np.random.default_rng(17)
    x_aug = np.column_stack([rng.normal(size=(30, 4)), np.ones(30)])
    y = (rng.random(30) > 0.5).astype(np.float64)
    # Deliberately push logits beyond +/- 40 (including misclassified extreme points!)
    beta = np.array([25.0, -22.0, 18.0, -15.0, 5.0], dtype=np.float64)
    z = x_aug @ beta
    assert np.max(np.abs(z)) > 40.0

    reg = 0.5
    loss0, grad, hess = stats_mod.logistic_objective_and_derivatives(x_aug, y, beta, reg)
    assert np.isfinite(loss0) and np.isfinite(grad).all() and np.isfinite(hess).all()

    eps = 1e-6
    fd_grad = np.zeros_like(beta)
    fd_hess = np.zeros_like(hess)
    for j in range(beta.size):
        e = np.zeros_like(beta)
        e[j] = eps
        lp, gp, _ = stats_mod.logistic_objective_and_derivatives(x_aug, y, beta + e, reg)
        lm, gm, _ = stats_mod.logistic_objective_and_derivatives(x_aug, y, beta - e, reg)
        fd_grad[j] = (lp - lm) / (2.0 * eps)
        fd_hess[:, j] = (gp - gm) / (2.0 * eps)

    assert np.allclose(grad, fd_grad, rtol=1e-5, atol=1e-6)
    assert np.allclose(hess, fd_hess, rtol=1e-5, atol=1e-6)


@pytest.mark.regression
@pytest.mark.differential
def test_reg006_damped_logistic_and_axes_h_parity_on_hard_regimes() -> None:
    """REG-006/017: Both probing.stats.damped_logistic and axes_h.newton_logistic match sklearn
    across rare positives, constant columns, ill-conditioning, and near-separability."""
    import axes_h

    rng = np.random.default_rng(6)
    n, d = 600, 10
    x_base = rng.normal(size=(n, d))
    # Add a constant column + strongly collinear column
    x_hard = x_base.copy()
    x_hard[:, 0] = 3.0
    x_hard[:, 1] = x_hard[:, 2] + 1e-3 * rng.normal(size=n)

    for bias, scale in [(0.0, 1.0), (-3.8, 1.0), (0.0, 6.0)]:  # balanced, rare, near-separable
        w_true = rng.normal(scale=scale, size=d)
        w_true[0] = 0.0
        probs = 1.0 / (1.0 + np.exp(-np.clip(x_hard @ w_true + bias, -30, 30)))
        y = (rng.random(n) < probs).astype(np.float64)
        if y.sum() < 5 or (n - y.sum()) < 5:
            continue

        c = 1.0
        w_s, b_s = stats_mod.damped_logistic(x_hard, y, c=c)
        w_h, b_h = axes_h.newton_logistic(x_hard, y, c)
        sk = LogisticRegression(C=c, max_iter=5000, tol=1e-10).fit(x_hard, y.astype(int))

        l_s = x_hard @ w_s + b_s
        l_h = x_hard @ w_h + b_h
        l_sk = x_hard @ sk.coef_[0] + sk.intercept_[0]
        assert np.allclose(l_s, l_sk, atol=1e-3, rtol=1e-3)
        assert np.allclose(l_h, l_sk, atol=1e-3, rtol=1e-3)
        # Constant column weight must be 0 under L2 penalty (no penalty on intercept)
        assert abs(w_s[0]) < 1e-6 and abs(w_h[0]) < 1e-6

    # Iteration exhaustion must fail loudly, not silently return unconverged weights
    with pytest.raises(ValueError, match="failed to converge"):
        stats_mod.damped_logistic(x_hard, y, c=1e6, max_iter=1, tol=1e-12)

    # Large-N convergence check (N=10,000): 1/N-scaled Armijo prevents float64 ULP stall
    x_big = rng.normal(size=(10_000, 6))
    p_big = 1.0 / (1.0 + np.exp(-(x_big[:, 0] - 0.5 * x_big[:, 1])))
    y_big = (rng.random(10_000) < p_big).astype(np.float64)
    w_big_s, b_big_s = stats_mod.damped_logistic(x_big, y_big, c=1.0)
    w_big_h, b_big_h = axes_h.newton_logistic(x_big, y_big, 1.0)
    assert np.allclose(w_big_s, w_big_h, atol=1e-4, rtol=1e-4)
    assert abs(b_big_s - b_big_h) < 1e-4


@pytest.mark.differential
def test_standardised_logistic_raw_coordinate_conversion_and_64bit_ids() -> None:
    """§4.1: Exact w = w_tilde/sigma, b = b_tilde - sum(w_tilde*mu/sigma) and 64-bit IDs > 2^53."""
    rng = np.random.default_rng(41)
    n, d = 400, 6
    x = rng.normal(loc=np.arange(d) * 5.0, scale=np.arange(1, d + 1, dtype=float), size=(n, d))
    y = (x[:, 1] - 0.5 * x[:, 2] + rng.normal(size=n) > 2.0).astype(int)

    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(x), y)
    w_raw = clf.coef_[0] / sc.scale_
    b_raw = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))

    logits_std = sc.transform(x) @ clf.coef_[0] + clf.intercept_[0]
    logits_raw = x @ w_raw + b_raw
    assert np.allclose(logits_raw, logits_std, rtol=1e-12, atol=1e-12)

    # AUC with ties and log_loss parity
    p_hat = 1.0 / (1.0 + np.exp(-logits_raw))
    assert log_loss(y, p_hat) == pytest.approx(
        log_loss(y, clf.predict_proba(sc.transform(x))[:, 1])
    )
    tied_scores = np.round(logits_raw, 1)
    assert 0.5 < roc_auc_score(y, tied_scores) <= 1.0

    # 64-bit SDSS catalogue IDs (> 2^53) must never lose precision in LabelProvider/EmbeddingMatrix
    big_ids = [1237666310706430371 + i for i in range(n)]
    assert all(oid > (1 << 53) for oid in big_ids)
    assert len(set(big_ids)) == n
    mat = EmbeddingMatrix(np.asarray(big_ids, dtype=np.int64), x, encoder_name="bigid")
    rows = {oid: {"f_frac": float(y[i]), "f_n": 20.0} for i, oid in enumerate(big_ids)}
    labels = LabelProvider(rows, feature_cols={"f": "f_frac"}, nuisance_cols={}, vote_count_min=1)
    # Permute IDs and confirm exact alignment of BOTH labels and embedding rows
    perm_ids = list(reversed(big_ids[:50]))
    emb = feature_embeddings(mat, labels, "f", perm_ids)
    assert np.array_equal(emb.y, y[:50][::-1])
    assert np.allclose(emb.x, x[:50][::-1], atol=0.0)


# --- REG-012 & REG-009: Poisoned-partition leakage & cache provenance -------------------------


@pytest.mark.regression
@pytest.mark.invariant
def test_reg012_poisoned_test_partition_does_not_affect_train_fitted_artefacts() -> None:
    """REG-012: Replacing held-out test inputs/labels must leave train-fitted models untouched
    while genuinely changing test metrics (non-vacuous check)."""
    rng = np.random.default_rng(12)
    n_tr, n_te = 120, 80
    ids = list(range(n_tr + n_te))
    x = rng.normal(size=(n_tr + n_te, 5))
    y_bin = (x[:, 0] + rng.normal(scale=0.3, size=n_tr + n_te) > 0.0).astype(float)
    nuis = x[:, 1] + rng.normal(scale=0.2, size=n_tr + n_te)

    rows_clean = {
        i: {"f_frac": 0.9 if y_bin[i] else 0.1, "f_n": 30.0, "mag": float(nuis[i])} for i in ids
    }
    rows_poison = dict(rows_clean)
    x_poison = x.copy()
    # Poison test inputs AND test labels severely
    x_poison[n_tr:] = -1e4
    for i in range(n_tr, n_tr + n_te):
        rows_poison[i] = {"f_frac": 0.1 if y_bin[i] else 0.9, "f_n": 30.0, "mag": 1e6}

    lab_c = LabelProvider(
        rows_clean, feature_cols={"f": "f_frac"}, nuisance_cols={"mag": "mag"}, vote_count_min=1
    )
    lab_p = LabelProvider(
        rows_poison, feature_cols={"f": "f_frac"}, nuisance_cols={"mag": "mag"}, vote_count_min=1
    )
    mat_c = EmbeddingMatrix(np.asarray(ids, dtype=np.int64), x, encoder_name="c")
    mat_p = EmbeddingMatrix(np.asarray(ids, dtype=np.int64), x_poison, encoder_name="p")

    tr_c = feature_embeddings(mat_c, lab_c, "f", ids[:n_tr])
    tr_p = feature_embeddings(mat_p, lab_p, "f", ids[:n_tr])
    dir_c = probe_direction(tr_c, name="f", c=1.0)
    dir_p = probe_direction(tr_p, name="f", c=1.0)
    assert np.allclose(dir_c.w_raw, dir_p.w_raw, atol=0.0)
    assert dir_c.bias == dir_p.bias

    # Non-vacuous: train+test entrypoint (`probe_auc_ci`) receives the full split and test AUC flips
    # when test labels are flipped, while train-fitted weights remain identical.
    from galaxy_jepa.probing.logistic import probe_auc_ci

    te_c = feature_embeddings(mat_c, lab_c, "f", ids[n_tr:])
    te_p = feature_embeddings(mat_p, lab_p, "f", ids[n_tr:])
    auc_clean, _, _ = probe_auc_ci(tr_c, te_c, c=1.0, n_boot=40, seed=0)
    auc_poison, _, _ = probe_auc_ci(tr_p, te_p, c=1.0, n_boot=40, seed=0)
    assert auc_clean > 0.85 and abs(auc_clean - auc_poison) > 0.30


@pytest.mark.regression
def test_reg009_cache_keys_and_bank_provenance_bind_full_ids_and_population(tmp_path: Path) -> None:
    """REG-009: Same prefix/length with different interior IDs or population must differ."""
    tr_a = list(range(100))
    tr_b = list(range(100))
    tr_b[77] = 999999  # identical prefix, suffix, and length — only one interior ID differs
    te = list(range(200, 250))

    assert ladder_mod.untrained_split_key(tr_a, te) != ladder_mod.untrained_split_key(tr_b, te)
    # Order-invariant on sorted IDs
    assert ladder_mod.untrained_split_key(tr_a, te) == ladder_mod.untrained_split_key(
        list(reversed(tr_a)), list(reversed(te))
    )
    # Population-sensitive
    assert ladder_mod.untrained_split_key(
        tr_a, te, population="full"
    ) != ladder_mod.untrained_split_key(tr_a, te, population="conditional")

    # probe_bank.load_bank refuses mismatched checkpoint SHA-1
    import probe_bank

    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_bytes(b"weights-A")
    ckpt_b.write_bytes(b"weights-B")
    bank_path = tmp_path / "bank.npz"
    probe_bank.write_bank(
        bank_path,
        ids=np.arange(4, dtype=np.int64),
        x=np.zeros((4, 3), dtype=np.float32),
        checkpoint=ckpt_a,
    )
    loaded = probe_bank.load_bank(bank_path, expected_checkpoint=ckpt_a)
    assert loaded["ids"].tolist() == [0, 1, 2, 3]
    with pytest.raises(probe_bank.BankProvenanceError):
        probe_bank.load_bank(bank_path, expected_checkpoint=ckpt_b)


# --- REG-007 & REG-008: Partial Spearman bootstrap & Gaussian map-null covariance -------------


@pytest.mark.regression
@pytest.mark.differential
def test_reg007_partial_spearman_vs_reranked_residuals_and_chunked_bootstrap() -> None:
    """REG-007: Pearson-on-rank-residuals != Spearman-on-residuals; chunked bootstrap matches."""
    rng = np.random.default_rng(707)
    n = 400
    z = rng.normal(size=(n, 2))
    # Heavy-tailed residual creates a clear separation between Pearson and Spearman on residuals
    eps_x = np.sign(rng.normal(size=n)) * np.abs(rng.normal(size=n)) ** 3
    eps_y = 0.6 * eps_x + rng.normal(scale=0.5, size=n)
    x = 2.0 * z[:, 0] - z[:, 1] + eps_x
    y = 1.5 * z[:, 0] + z[:, 1] + eps_y

    rx = stats_mod.residualise_ranks(x, z)
    ry = stats_mod.residualise_ranks(y, z)
    pearson_on_resid = stats_mod.partial_spearman(x, y, z)
    spearman_on_resid = float(spearmanr(rx, ry).statistic)
    # Both estimators are finite, and on skewed residuals they are numerically distinct
    assert np.isfinite(pearson_on_resid) and np.isfinite(spearman_on_resid)
    assert abs(pearson_on_resid - spearman_on_resid) > 0.005

    # Full paired-row bootstrap: exact reproducibility across chunk sizes + valid accounting
    b1 = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=300, seed=11, chunk_size=50)
    b2 = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=300, seed=11, chunk_size=128)
    assert b1.lo == pytest.approx(b2.lo, abs=1e-12)
    assert b1.hi == pytest.approx(b2.hi, abs=1e-12)
    assert b1.n_valid == 300 and b1.n_invalid == 0
    assert b1.lo < b1.estimate < b1.hi

    # Constant residual after nuisance projection must return NaN (no cancellation artefacts)
    const_x = 3.0 * z[:, 0] + 1.0
    const_y = 7.0 * z[:, 0] - 2.0
    assert np.isnan(stats_mod.partial_spearman(const_x, y, z))

    # Production Part B path (`artifacts/axes_b.py`): must also guard residual norm (return NaN
    # on nuisance-collinear inputs, NOT 1.0!) and run full paired-row quadratic bootstrap.
    import axes_b

    assert np.isnan(axes_b.partial(const_x, const_y, z[:, :1]))
    ci_b = axes_b.boot_partial_full(x, y, z, seed=11, n=80)
    pt_b = axes_b.partial(x, y, z)
    assert np.isfinite(pt_b) and ci_b[0] < pt_b < ci_b[1]


@pytest.mark.regression
def test_reg008_cholesky_covariance_sampling_matches_target_sigma() -> None:
    """REG-008: axes_b_mapnull.sample_covariance_directions yields Sigma = S S^T."""
    import axes_b_mapnull

    sigma = np.array(
        [
            [2.0, 1.2, -0.5],
            [1.2, 1.5, 0.3],
            [-0.5, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(808)
    draws = axes_b_mapnull.sample_covariance_directions(sigma, k=40_000, rng=rng, jitter=1e-6)
    emp_good = np.cov(draws)
    assert np.allclose(emp_good, sigma, atol=0.05)

    # Contrast against legacy buggy factorisations (inv(S)^T z and Sigma z)
    chol = np.linalg.cholesky(sigma)
    z = rng.standard_normal((3, 40_000))
    assert not np.allclose(np.cov(np.linalg.inv(chol).T @ z), sigma, atol=0.5)
    assert not np.allclose(np.cov(sigma @ z), sigma, atol=0.5)


# --- REG-010, REG-011, REG-013, REG-014, REG-016 & Independent BY Oracle ----------------------


@pytest.mark.regression
def test_reg010_and_reg014_matching_edge_cases_and_retention() -> None:
    """REG-010 & REG-014: All-zero/mixed zero-inflated strata + non-positive retention verdict."""
    # All-zero nuisance
    zeros = np.zeros(40)
    y = np.array([0] * 15 + [1] * 25)
    k_all_zero = match.stratified_match(zeros, y, n_strata=5, seed=0)
    assert k_all_zero.size == 30 and int(y[k_all_zero].sum()) == 15

    # Mixed 60% zeros + 40% positive tail -> zero stratum isolated cleanly
    rng = np.random.default_rng(10)
    vals = np.concatenate([np.zeros(60), rng.uniform(0.2, 1.0, size=40)])
    y_mix = np.array([0] * 40 + [1] * 20 + [0] * 10 + [1] * 30)
    k_mix = match.stratified_match(vals, y_mix, n_strata=5, seed=0)
    assert int(y_mix[k_mix].sum()) * 2 == k_mix.size
    z_kept = k_mix[vals[k_mix] == 0.0]
    assert z_kept.size == 40 and int(y_mix[z_kept].sum()) == 20

    # Non-positive unmatched gain -> UNRESOLVED (never COLLAPSES or SURVIVES), even when
    # m_lo <= c_m or m_lo > c_m with full sample support (n_matched_test >= 500).
    for a_val, c_val, m_val, m_lo_val in [
        (0.50, 0.55, 0.65, 0.60),  # positive matched gain, negative unmatched
        (0.55, 0.55, 0.65, 0.60),  # positive matched gain, zero unmatched
        (0.50, 0.55, 0.56, 0.52),  # m_lo == c_m, negative unmatched -> still UNRESOLVED!
        (0.50, 0.55, 0.51, 0.48),  # negative matched gain, negative unmatched -> UNRESOLVED!
    ]:
        rv = match.retention_verdict(
            a=a_val,
            c=c_val,
            m=m_val,
            m_lo=m_lo_val,
            c_m=0.52,
            n_matched_test=600,
            n_test=600,
        )
        assert rv.verdict == match.UNRESOLVED
        assert rv.retained is None


@pytest.mark.regression
@pytest.mark.differential
def test_independent_benjamini_yekutieli_oracle_with_ties_and_permutations() -> None:
    """Independent reference implementation of Benjamini–Yekutieli step-up FDR control."""
    pvals = {"f1": 0.001, "f2": 0.001, "f3": 0.009, "f4": 0.04, "f5": 0.8}
    m = len(pvals)
    c_m = sum(1.0 / k for k in range(1, m + 1))
    alpha = 0.05

    # Independent textbook BY step-up
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    k_star = 0
    for rank, (_, p) in enumerate(ordered, start=1):
        if p <= (rank / (m * c_m)) * alpha:
            k_star = rank
    expected = {name: (idx < k_star) for idx, (name, _) in enumerate(ordered)}

    got = nz.family_significant(pvals, alpha=alpha, method="benjamini_yekutieli", n_tests=m)
    assert got == expected

    # Permutation invariance over hypothesis ordering
    rev = dict(reversed(list(pvals.items())))
    assert nz.family_significant(rev, alpha=alpha, method="benjamini_yekutieli", n_tests=m) == got
    # Empty family
    assert nz.family_significant({}, alpha=alpha, method="benjamini_yekutieli", n_tests=0) == {}


@pytest.mark.regression
def test_reg011_c3_plugin_independent_out_of_sample_validation() -> None:
    """REG-011: Calibrate k_plugin on seed A, evaluate Type-I error on independent seed B."""
    import erratum_aligned_c3 as c3

    cal = c3.calibrate_plugin(draws=100_000, seed=101, alpha=0.05)
    k_plug = cal["k_plugin_95"]
    assert 5.5 < k_plug < 6.2

    # Independent evaluation draws (never reusing calibration draws!)
    rng = np.random.default_rng(202)
    z = rng.standard_normal((100_000, 4))
    a, m = z[:, :2], z[:, 2:]
    gap = np.maximum(a.min(axis=1) - m.max(axis=1), m.min(axis=1) - a.max(axis=1))
    s_m = np.abs(m[:, 0] - m[:, 1]) / np.sqrt(2.0)

    rate_old = float((gap > c3.K_ORIGINAL * s_m).mean())
    rate_new = float((gap > k_plug * s_m).mean())
    assert 0.155 < rate_old < 0.180
    assert 0.045 < rate_new < 0.055


@pytest.mark.regression
def test_reg016_deterministic_cpu_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """REG-016: Interrupted + resumed CPU training matches an uninterrupted run bit-for-bit."""

    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n: int = 16) -> None:
            g = torch.Generator().manual_seed(99)
            self.imgs = torch.randn(n, 3, 32, 32, generator=g)

        def __len__(self) -> int:
            return len(self.imgs)

        def __getitem__(self, i: int) -> dict[str, object]:
            return {
                "image": self.imgs[i],
                "petro_rad_arcsec": 3.0,
                "pixel_scale": 0.396,
                "object_id": i,
            }

    ds = _TinyDataset(16)
    cfg = JepaConfig(
        steps=6,
        batch_size=4,
        lr=1e-3,
        warmup_steps=2,
        pred_dim=12,
        pred_depth=1,
        pred_heads=2,
        mask=MaskConfig(n_target_blocks=2, beta=0.0),
        monitor_every=3,
        checkpoint_every=3,
        seed=5,
    )

    def _build() -> tuple[Jepa, torch.utils.data.DataLoader, ResumableShuffle]:
        torch.manual_seed(5)
        enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)
        obj = Jepa(enc, cfg)
        sampler = ResumableShuffle(len(ds), seed=5)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler)
        return obj, loader, sampler

    # Uninterrupted 6-step run
    obj_full, loader_full, sampler_full = _build()
    res_full = train_objective(obj_full, loader_full, device="cpu", sampler=sampler_full)

    # Interrupted at step 3, then resumed to step 6
    ckpt_dir = tmp_path / "ckpts"
    ck = TrainCheckpointer(
        ckpt_dir,
        every=3,
        keep=4,
        config_hash="h1",
        normalisation_hash="n1",
        schedule={"steps": cfg.steps, "warmup_steps": cfg.warmup_steps, "lr": cfg.lr},
    )
    obj_a, loader_a, sampler_a = _build()
    res_a = train_objective(
        obj_a, loader_a, device="cpu", checkpointer=ck, sampler=sampler_a, stop_after=3
    )
    assert res_a.steps_completed == 3

    obj_b, loader_b, sampler_b = _build()
    res_b = train_objective(
        obj_b, loader_b, device="cpu", checkpointer=ck, sampler=sampler_b, resume=True
    )
    assert res_b.steps_completed == 6
    assert np.allclose(res_b.losses, res_full.losses, rtol=1e-6, atol=1e-6)
    for (k1, v1), (k2, v2) in zip(
        obj_b.encoder.state_dict().items(), obj_full.encoder.state_dict().items(), strict=True
    ):
        assert k1 == k2 and torch.allclose(v1, v2, rtol=1e-6, atol=1e-6)


@pytest.mark.regression
def test_reg013_entangled_map_symmetric_under_feature_order_swap() -> None:
    """REG-013: Swapping (A, B) <-> (B, A) produces bit-identical non-vacuous verdicts."""
    rng = np.random.default_rng(13)
    # Use n_test >= 900 so matched test set exceeds MIN_MATCHED_TEST (500) and decorrelation runs!
    n = 1600
    n_tr = 700
    ids = list(range(n))
    z = rng.normal(size=(n, 6))
    # Independent strong signals on dims 0 & 1 plus moderate shared component on dim 2
    # so A and B both genuinely SURVIVE matching on the partner's vote fraction.
    fa = 1.0 / (1.0 + np.exp(-(2.2 * z[:, 0] + 0.5 * z[:, 2]) + rng.normal(scale=0.3, size=n)))
    fb = 1.0 / (1.0 + np.exp(-(2.2 * z[:, 1] + 0.5 * z[:, 2]) + rng.normal(scale=0.3, size=n)))
    rows = {i: {"a_f": float(fa[i]), "a_n": 20.0, "b_f": float(fb[i]), "b_n": 20.0} for i in ids}
    labels = LabelProvider(
        rows, feature_cols={"a": "a_f", "b": "b_f"}, nuisance_cols={}, vote_count_min=1
    )
    mat_real = EmbeddingMatrix(np.asarray(ids, dtype=np.int64), z, encoder_name="real")
    u_mats = [
        EmbeddingMatrix(
            np.asarray(ids, dtype=np.int64), rng.normal(size=(n, 6)), encoder_name=f"untr_{k}"
        )
        for k in range(3)
    ]
    controls = ctl.ControlEmbeddings(
        real=mat_real,
        untrained=u_mats[0],
        noise=u_mats[0],
        untrained_extra=(u_mats[1], u_mats[2]),
    )
    cfg = ladder_mod.ProbingConfig(vote_count_min=1, smoke=True, n_boot=80, seed=13)

    from galaxy_jepa.probing import entanglement as ent

    da = probe_direction(feature_embeddings(mat_real, labels, "a", ids[:n_tr]), name="a", c=1.0)
    db = probe_direction(feature_embeddings(mat_real, labels, "b", ids[:n_tr]), name="b", c=1.0)
    geo_ab = ent.EntanglementGeometry(
        names=["a", "b"],
        cosine=np.array([[1.0, 0.9], [0.9, 1.0]]),
        gram_eigenvalues=np.array([1.9, 0.1]),
        gram_effective_rank=1.2,
        embedding_eigenvalues=np.ones(6),
        embedding_effective_rank=6.0,
        mp=ent.MPVerdict(1.9, 1.5, True),
        entangled_pairs=[("a", "b")],
        cav_disagreement={"a": 0.25, "b": 0.25},
    )
    geo_ba = ent.EntanglementGeometry(
        names=["b", "a"],
        cosine=np.array([[1.0, 0.9], [0.9, 1.0]]),
        gram_eigenvalues=np.array([1.9, 0.1]),
        gram_effective_rank=1.2,
        embedding_eigenvalues=np.ones(6),
        embedding_effective_rank=6.0,
        mp=ent.MPVerdict(1.9, 1.5, True),
        entangled_pairs=[("b", "a")],
        cav_disagreement={"a": 0.25, "b": 0.25},
    )
    dirs = {"a": da, "b": db}
    aucs = {"a": 0.90, "b": 0.90}
    bars = {"a": 0.50, "b": 0.50}
    _, pv_ab = ladder_mod._entangled_map(
        geo_ab,
        dirs,
        controls,
        labels,
        ids[:n_tr],
        ids[n_tr:],
        real_aucs=aucs,
        bars=bars,
        established={"a": True, "b": True},
        config=cfg,
    )
    _, pv_ba = ladder_mod._entangled_map(
        geo_ba,
        dirs,
        controls,
        labels,
        ids[:n_tr],
        ids[n_tr:],
        real_aucs=aucs,
        bars=bars,
        established={"a": True, "b": True},
        config=cfg,
    )
    # Non-vacuous: decorrelation actually executed (n_matched_test >= 500, retention SURVIVES)
    assert pv_ab[0].n_matched_test >= match.MIN_MATCHED_TEST
    assert pv_ab[0].retention == match.SURVIVES
    assert pv_ab[0].matched_auc is not None and pv_ab[0].matched_auc > 0.75
    assert pv_ab[0].bar_matched is not None
    # Strictly symmetric under (a, b) <-> (b, a)
    assert pv_ab[0].verdict == pv_ba[0].verdict
    assert pv_ab[0].retention == pv_ba[0].retention
    assert pv_ab[0].matched_auc == pytest.approx(pv_ba[0].matched_auc, abs=1e-12)
    assert pv_ab[0].bar_matched == pytest.approx(pv_ba[0].bar_matched, abs=1e-12)
    assert pv_ab[0].retained == pytest.approx(pv_ba[0].retained, abs=1e-12)


@pytest.mark.regression
def test_reg013_entangled_map_enforces_10pct_coverage_against_unmatched_population() -> None:
    """Enforce D24's 10% coverage gate against each feature's original eligible test set."""
    from galaxy_jepa.probing import entanglement as ent

    rng = np.random.default_rng(131)
    n_tr = 600
    n_te_a = 6000  # 10% threshold = 600
    n_te_joint = 560  # >= 500 absolute floor, yet < 10% of A's 6,000 eligible test rows!
    n = n_tr + n_te_a
    ids = list(range(n))
    z = rng.normal(size=(n, 4))
    fa = 1.0 / (1.0 + np.exp(-(3.0 * z[:, 0]) + rng.normal(scale=0.15, size=n)))
    fb = 1.0 / (1.0 + np.exp(-(3.0 * z[:, 1]) + rng.normal(scale=0.15, size=n)))

    rows: dict[int, dict[str, float]] = {}
    for i in ids:
        # All rows are eligible for A (a_n = 20), while only train + first n_te_joint test rows
        # are eligible for B (b_n = 20); remaining test rows have b_n = 0 (ineligible for B).
        b_elig = i < n_tr + n_te_joint
        rows[i] = {
            "a_f": float(fa[i]),
            "a_n": 20.0,
            "b_f": float(fb[i]) if b_elig else 0.0,
            "b_n": 20.0 if b_elig else 0.0,
        }
    labels = LabelProvider(
        rows, feature_cols={"a": "a_f", "b": "b_f"}, nuisance_cols={}, vote_count_min=1
    )
    mat_real = EmbeddingMatrix(np.asarray(ids, dtype=np.int64), z, encoder_name="real")
    u_mats = [
        EmbeddingMatrix(
            np.asarray(ids, dtype=np.int64), rng.normal(size=(n, 4)), encoder_name=f"untr_{k}"
        )
        for k in range(3)
    ]
    controls = ctl.ControlEmbeddings(
        real=mat_real,
        untrained=u_mats[0],
        noise=u_mats[0],
        untrained_extra=(u_mats[1], u_mats[2]),
    )
    cfg = ladder_mod.ProbingConfig(vote_count_min=1, smoke=True, n_boot=20, seed=131)

    da = probe_direction(feature_embeddings(mat_real, labels, "a", ids[:n_tr]), name="a", c=1.0)
    db = probe_direction(feature_embeddings(mat_real, labels, "b", ids[:n_tr]), name="b", c=1.0)
    geo_ab = ent.EntanglementGeometry(
        names=["a", "b"],
        cosine=np.array([[1.0, 0.9], [0.9, 1.0]]),
        gram_eigenvalues=np.array([1.9, 0.1]),
        gram_effective_rank=1.2,
        embedding_eigenvalues=np.ones(4),
        embedding_effective_rank=4.0,
        mp=ent.MPVerdict(1.9, 1.5, True),
        entangled_pairs=[("a", "b")],
        cav_disagreement={"a": 0.25, "b": 0.25},
    )
    geo_ba = ent.EntanglementGeometry(
        names=["b", "a"],
        cosine=np.array([[1.0, 0.9], [0.9, 1.0]]),
        gram_eigenvalues=np.array([1.9, 0.1]),
        gram_effective_rank=1.2,
        embedding_eigenvalues=np.ones(4),
        embedding_effective_rank=4.0,
        mp=ent.MPVerdict(1.9, 1.5, True),
        entangled_pairs=[("b", "a")],
        cav_disagreement={"a": 0.25, "b": 0.25},
    )
    dirs = {"a": da, "b": db}
    aucs = {"a": 0.95, "b": 0.95}
    bars = {"a": 0.50, "b": 0.50}

    _, pv_ab = ladder_mod._entangled_map(
        geo_ab,
        dirs,
        controls,
        labels,
        ids[:n_tr],
        ids[n_tr:],
        real_aucs=aucs,
        bars=bars,
        established={"a": True, "b": True},
        config=cfg,
    )
    _, pv_ba = ladder_mod._entangled_map(
        geo_ba,
        dirs,
        controls,
        labels,
        ids[:n_tr],
        ids[n_tr:],
        real_aucs=aucs,
        bars=bars,
        established={"a": True, "b": True},
        config=cfg,
    )
    # Absolute 500-row guard MUST be cleared so the 10% coverage gate is genuinely exercised
    assert pv_ab[0].n_matched_test >= match.MIN_MATCHED_TEST
    assert pv_ab[0].n_matched_test < int(0.10 * n_te_a)
    assert pv_ab[0].retention == match.UNRESOLVED
    assert pv_ab[0].verdict == ent.INCONCLUSIVE
    assert pv_ba[0].retention == match.UNRESOLVED
    assert pv_ba[0].verdict == ent.INCONCLUSIVE


@pytest.mark.regression
def test_axes_b_bootstrap_handles_constant_resampled_nuisance_columns() -> None:
    """Sparse/tied nuisances becoming constant inside a resample must not crash lstsq."""
    import axes_b

    n_sparse = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])[:, None]
    a = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 5.0])
    b = np.array([3.0, 1.0, 0.0, 5.0, 2.0, 4.0])

    pt = axes_b.partial(a, b, n_sparse)
    assert np.isfinite(pt)
    det = axes_b.boot_partial_full_detail(a, b, n_sparse, seed=0, n=20)
    assert det["n_valid"] + det["n_invalid"] == 20
    assert det["n_valid"] >= det["min_valid"]
    assert np.isfinite(det["ci"]).all() and det["ci"][0] <= det["ci"][1]
    assert axes_b.boot_partial_full(a, b, n_sparse, seed=0, n=20) == det["ci"]

    # Mixed constant + informative nuisance preserves the quadratic design for the informative col
    rng = np.random.default_rng(77)
    z_info = rng.normal(size=80)
    y_quad = z_info**2 + 0.05 * rng.normal(size=80)
    r_only_info = axes_b.resid_ranks(y_quad, z_info[:, None])
    r_with_const = axes_b.resid_ranks(y_quad, np.column_stack([np.zeros(80), z_info]))
    assert np.allclose(r_only_info, r_with_const, atol=1e-12)

    # When too few usable replicates remain (e.g. collinear nuisance), CI must be [nan, nan]
    z_col = np.linspace(0.0, 1.0, 30)
    det_col = axes_b.boot_partial_full_detail(
        3.0 * z_col + 1.0, 7.0 * z_col - 2.0, z_col[:, None], seed=0, n=20
    )
    assert det_col["n_valid"] == 0 and det_col["n_invalid"] == 20
    assert np.isnan(det_col["ci"]).all()


@pytest.mark.regression
def test_erratum_ladder_v2_e5e6_requires_verified_provenance_for_legacy_bank(
    tmp_path: Path,
) -> None:
    """Refuse legacy prefix-keyed banks unless full split-membership provenance is verified."""
    import json

    import erratum_ladder_v2_e5e6 as err_e56

    orig_tr = [1, 2, 3, 4]
    orig_te = [5, 6, 7, 8]
    changed_tr = [1, 2, 3, 5]
    changed_te = [4, 6, 7, 8]
    # Equal counts and union endpoints -> identical legacy fingerprint, different full-ID hash
    old_key = err_e56._legacy_split_key(orig_tr, orig_te)
    assert err_e56._legacy_split_key(changed_tr, changed_te) == old_key
    orig_new_key = ladder_mod.untrained_split_key(orig_tr, orig_te, population="full")
    changed_new_key = ladder_mod.untrained_split_key(changed_tr, changed_te, population="full")
    assert orig_new_key != changed_new_key

    src_bank = tmp_path / "legacy_bank.json"
    orig_payload = {"split_key": old_key, "seeds": {"0": {"f": 0.55}}}
    src_bank.write_text(json.dumps(orig_payload))
    out_dir = tmp_path / "erratum_out"

    # 1. Without verified split provenance, legacy key alone MUST be refused (rebuild required)
    with pytest.raises(SystemExit, match="rebuild required"):
        err_e56._resolve_full_bank(
            src_bank, out_dir, train_ids=orig_tr, test_ids=orig_te, new_split_key=orig_new_key
        )

    # 2. Trusted provenance for (orig_tr, orig_te) bound to src_bank's SHA-256
    prov_path = tmp_path / "split_provenance.json"
    prov_payload = {
        "population": "full",
        "bank_sha256": err_e56._sha256_file(src_bank),
        "legacy_split_key": old_key,
        "train_ids": orig_tr,
        "test_ids": orig_te,
        "full_split_key": orig_new_key,
    }
    prov_path.write_text(json.dumps(prov_payload))

    # 3. Changed partition within the same legacy equivalence class MUST be refused loudly
    with pytest.raises(SystemExit, match="does not match current train/test partition"):
        err_e56._resolve_full_bank(
            src_bank,
            out_dir,
            train_ids=changed_tr,
            test_ids=changed_te,
            new_split_key=changed_new_key,
            split_provenance=prov_path,
        )

    # 4. Wrong bank_sha256 in provenance MUST be refused
    bad_prov = tmp_path / "bad_prov.json"
    bad_prov.write_text(json.dumps({**prov_payload, "bank_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="bank_sha256"):
        err_e56._resolve_full_bank(
            src_bank,
            out_dir,
            train_ids=orig_tr,
            test_ids=orig_te,
            new_split_key=orig_new_key,
            split_provenance=bad_prov,
        )

    # 5. Exact verified provenance succeeds and writes re-keyed copy without mutating source
    resolved = err_e56._resolve_full_bank(
        src_bank,
        out_dir,
        train_ids=orig_tr,
        test_ids=orig_te,
        new_split_key=orig_new_key,
        split_provenance=prov_path,
    )
    assert resolved != src_bank
    assert json.loads(src_bank.read_text()) == orig_payload
    rekeyed = json.loads(resolved.read_text())
    assert rekeyed["split_key"] == orig_new_key and rekeyed["population"] == "full"
    assert rekeyed["source_bank_sha256"] == prov_payload["bank_sha256"]

    # 6. Already full-keyed bank is returned directly without requiring legacy provenance
    assert (
        err_e56._resolve_full_bank(
            resolved, out_dir, train_ids=orig_tr, test_ids=orig_te, new_split_key=orig_new_key
        )
        == resolved
    )
