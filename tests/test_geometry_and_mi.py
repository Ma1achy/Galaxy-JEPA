"""Geometry, entanglement adjudication, transfer ratio & MI benchmark suite (§7–§8).

Covers:
- Covariance-aware alignment c_Sigma invariance under invertible reparameterisations (§7.1)
- Gram-matrix spectrum, SVD parity, empirical Gram null calibration (§7.2)
- Complete adjudicate_pair truth table including missing-CAV INCONCLUSIVE handling (§7.3)
- Transfer-ratio properties (test-sign bias, denominator floor, non-MI) (§8.1)
- All 10 analytical mutual-information benchmarks + Barber–Agakov variational bound (§8.2)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from galaxy_jepa.probing import entanglement as ent
from galaxy_jepa.probing import information as info
from galaxy_jepa.probing.logistic import ConceptDirection


def _make_dir(name: str, w_raw: np.ndarray) -> ConceptDirection:
    norm = float(np.linalg.norm(w_raw))
    w_unit = w_raw / norm if norm > 0 else np.zeros_like(w_raw)
    return ConceptDirection(
        name=name,
        w_unit=w_unit,
        w_raw=w_raw,
        bias=0.0,
    )


# ── §7.1 Covariance-aware alignment (c_Sigma) ─────────────────────────────────────────────────────


@pytest.mark.scientific
@pytest.mark.differential
def test_covariance_alignment_matches_score_pearson_and_is_gl_invariant() -> None:
    """c_Sigma matches corr(Z w_i, Z w_j) and is invariant under Z' = Z A^T, w' = A^{-1} w."""
    rng = np.random.default_rng(71)
    n, d = 600, 8
    scale = np.diag([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    z = rng.normal(size=(n, d)) @ (q @ scale).T

    w1 = rng.normal(size=d)
    w2 = 0.6 * w1 + rng.normal(size=d)
    d1, d2 = _make_dir("a", w1), _make_dir("b", w2)

    names, c_mat = ent.covariance_alignment([d1, d2], z, use_raw=True)
    assert names == ["a", "b"]
    c_sig = float(c_mat[0, 1])
    pearson = float(np.corrcoef(z @ w1, z @ w2)[0, 1])
    assert c_sig == pytest.approx(pearson, rel=1e-12, abs=1e-12)
    assert c_mat[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert c_mat[1, 1] == pytest.approx(1.0, abs=1e-12)

    # Raw Euclidean cosine differs substantially from c_Sigma under anisotropic Sigma
    _, cos_raw_mat = ent.cosine_matrix([d1, d2])
    cos_raw = float(cos_raw_mat[0, 1])
    assert abs(c_sig - cos_raw) > 0.05

    # Consistent invertible linear reparameterisation: Z' = Z A^T, w' = A^{-1} w
    # Linear scores are preserved: Z' w' = (Z A^T)(A^{-1} w)? Wait: (Z A^T)(A^{-T} w) = Z w.
    a_mat = rng.normal(size=(d, d)) + 2.5 * np.eye(d)
    a_inv_t = np.linalg.inv(a_mat).T
    z_prime = z @ a_mat.T
    d1_p = _make_dir("a", a_inv_t @ w1)
    d2_p = _make_dir("b", a_inv_t @ w2)

    _, c_mat_prime = ent.covariance_alignment([d1_p, d2_p], z_prime, use_raw=True)
    _, cos_raw_prime_mat = ent.cosine_matrix([d1_p, d2_p])
    assert float(c_mat_prime[0, 1]) == pytest.approx(c_sig, rel=1e-11, abs=1e-11)
    assert abs(float(cos_raw_prime_mat[0, 1]) - cos_raw) > 0.02

    # Zero-variance direction produces NaN cleanly in off-diagonals
    d_zero = _make_dir("z0", np.zeros(d))
    _, c_zero = ent.covariance_alignment([d_zero, d2], z, use_raw=True)
    assert math.isnan(float(c_zero[0, 1]))


# ── §7.2 Gram spectrum & empirical Gram null ──────────────────────────────────────────────────────


@pytest.mark.scientific
@pytest.mark.differential
def test_gram_spectrum_svd_parity_and_empirical_gram_null_calibration() -> None:
    """Gram eigenvalues equal squared singular values of W; empirical null calibrates cleanly."""
    rng = np.random.default_rng(72)
    k, d = 6, 24
    w_raw = rng.normal(size=(k, d))
    dirs = [_make_dir(f"f{i}", w_raw[i]) for i in range(k)]
    _, w_unit = ent.stack_directions(dirs)

    eigs, eff_rank = ent.gram_eigenspectrum(w_unit)
    svals = np.linalg.svd(w_unit, compute_uv=False)
    assert np.allclose(np.maximum(eigs, 0.0), svals**2, rtol=1e-11, atol=1e-12)
    assert np.min(eigs) >= -1e-12
    assert 1.0 <= eff_rank <= k

    # Empirical Gram null on isotropic vs strongly anisotropic embeddings
    n = 160
    x_iso = rng.normal(size=(n, d))
    labels = {f"f{i}": rng.integers(0, 2, size=n) for i in range(5)}

    # Uncorrelated observed top eigenvalue (~1.2) does NOT reject null
    null_iso = ent.empirical_gram_null(x_iso, labels, observed_top=1.2, c=1.0, n_draws=30, seed=72)
    assert null_iso.pvalue > 0.10
    assert null_iso.null_tops.shape == (30,)

    # Planted rank-1 collapse (unit-mean top eigenvalue = k = 5.0) MUST reject null
    planted = ent.empirical_gram_null(x_iso, labels, observed_top=4.9, c=1.0, n_draws=30, seed=72)
    assert planted.pvalue == pytest.approx(1.0 / 31.0, abs=1e-12)

    # Low-rank correlated embedding covariance (survives StandardScaler!) raises empirical null
    proj = rng.normal(size=(2, d))
    x_lowrank = rng.normal(size=(n, 2)) @ proj + 0.05 * rng.normal(size=(n, d))
    null_lowrank = ent.empirical_gram_null(
        x_lowrank, labels, observed_top=2.0, c=1.0, n_draws=30, seed=72
    )
    assert float(np.median(null_lowrank.null_tops)) > float(np.median(null_iso.null_tops)) + 0.4

    # Correlated-label null test (X _|_ (Y1..Y5) with identical label columns):
    # Default mode="joint_rows" preserves label equality -> observed_top=5.0 does NOT reject!
    # Stronger mode="independent_cols" destroys label equality -> rejects at p < 0.05.
    rng_cl = np.random.default_rng(2901)
    x_ind = rng_cl.normal(size=(160, 24))
    y_shared = rng_cl.integers(0, 2, size=160)
    dup_labels = {f"c{i}": y_shared for i in range(5)}
    joint_null = ent.empirical_gram_null(
        x_ind, dup_labels, observed_top=5.0, c=1.0, n_draws=25, seed=17, mode="joint_rows"
    )
    indep_null = ent.empirical_gram_null(
        x_ind, dup_labels, observed_top=5.0, c=1.0, n_draws=25, seed=17, mode="independent_cols"
    )
    assert joint_null.pvalue > 0.50
    assert indep_null.pvalue < 0.05


# ── §7.3 Complete adjudicate_pair truth table ─────────────────────────────────────────────────────


@pytest.mark.regression
@pytest.mark.scientific
@pytest.mark.parametrize(
    ("mp_sig", "cosine", "cav_dict", "survived", "expected"),
    [
        # All 4 converge -> REPRESENTATIONAL
        (True, 0.65, {"a": 0.25, "b": 0.02}, True, ent.REPRESENTATIONAL),
        # MP entangled + vanishes under matching -> WORLD_CORRELATION
        (True, 0.65, {"a": 0.25, "b": 0.02}, False, ent.WORLD_CORRELATION),
        (True, 0.10, {}, False, ent.WORLD_CORRELATION),
        # MP clean + low cosine + CAV measured & agrees -> CLEAN
        (False, 0.10, {"a": 0.03, "b": 0.04}, None, ent.CLEAN),
        (False, -0.29, {"a": 0.01, "b": 0.09}, True, ent.CLEAN),
        # Missing or NaN CAV evidence MUST NEVER falsely yield CLEAN or REPRESENTATIONAL
        (False, 0.10, {}, None, ent.INCONCLUSIVE),
        (False, 0.10, {"a": 0.02}, None, ent.INCONCLUSIVE),
        (False, 0.10, {"a": 0.02, "b": float("nan")}, None, ent.INCONCLUSIVE),
        (True, 0.70, {}, True, ent.INCONCLUSIVE),
        (True, 0.70, {"a": float("nan"), "b": 0.30}, True, ent.INCONCLUSIVE),
        # Non-convergent combinations -> INCONCLUSIVE
        (True, 0.70, {"a": 0.02, "b": 0.01}, True, ent.INCONCLUSIVE),
        (True, 0.10, {"a": 0.25, "b": 0.25}, True, ent.INCONCLUSIVE),
        (True, 0.70, {"a": 0.25, "b": 0.25}, None, ent.INCONCLUSIVE),
        (False, 0.50, {"a": 0.02, "b": 0.02}, None, ent.INCONCLUSIVE),
        (False, 0.10, {"a": 0.20, "b": 0.02}, None, ent.INCONCLUSIVE),
        (False, float("nan"), {"a": 0.01, "b": 0.01}, None, ent.INCONCLUSIVE),
    ],
)
def test_adjudicate_pair_complete_truth_table_and_missing_cav_inconclusive(
    mp_sig: bool,
    cosine: float,
    cav_dict: dict[str, float],
    survived: bool | None,
    expected: str,
) -> None:
    """Every adjudication branch is deterministic, order-symmetric, and safe on missing CAV."""
    v_ab = ent.adjudicate_pair(
        "a",
        "b",
        cosine=cosine,
        mp_significant=mp_sig,
        cav_disagreement=cav_dict,
        survived_matching=survived,
    )
    v_ba = ent.adjudicate_pair(
        "b",
        "a",
        cosine=cosine,
        mp_significant=mp_sig,
        cav_disagreement=cav_dict,
        survived_matching=survived,
    )
    assert v_ab.verdict == expected
    assert v_ba.verdict == expected
    assert "unassailable" not in v_ab.reason.lower()


# ── §8.1 Transfer-ratio diagnostic properties (explicitly distinct from MI) ──────────────────────


@pytest.mark.scientific
def test_transfer_ratio_sign_alignment_bias_denominator_floor_and_non_mi() -> None:
    """Characterise Almagest transfer ratio: optimistic null bias, floor necessity, and non-MI."""
    rng = np.random.default_rng(81)

    # 1. Under pure null (independent score and binary label), test-time sign alignment
    #    max(AUC, 1 - AUC) has strictly positive expectation E[max(AUC, 1-AUC) - 0.5] > 0.
    n = 120
    signed_excess = []
    for _ in range(300):
        y = rng.integers(0, 2, size=n)
        s = rng.normal(size=n)
        auc = float(roc_auc_score(y, s))
        signed_excess.append(max(auc, 1.0 - auc) - 0.5)
    assert float(np.mean(signed_excess)) > 0.025

    # 2. Near-chance denominator (AUC_Y - 0.5 ~ 0.005) makes unfloored transfer ratio > 4
    #    even for a noise predictor, proving why RATIO_FLOOR = 0.05 in almagest_axes.py matters.
    auc_y_weak = 0.505
    auc_transfer_noise = 0.53
    raw_ratio = (auc_transfer_noise - 0.5) / (auc_y_weak - 0.5)
    assert raw_ratio > 4.0

    # 3. Transfer ratio is NOT mutual information:
    #    X and Y = X^2 have high MI (> 0.4 nats), yet linear score d_X = X has sign-aligned
    #    AUC ~ 0.50 (transfer ratio ~ 0) for classifying Y > median(Y).
    x = rng.normal(size=4000)
    y_quad = x**2 + 0.1 * rng.normal(size=4000)
    y_bin = (y_quad > np.median(y_quad)).astype(int)
    auc_lin = float(roc_auc_score(y_bin, x))
    sign_aligned_lin = max(auc_lin, 1.0 - auc_lin)
    assert sign_aligned_lin < 0.54
    assert info.ksg_mi(x, y_quad, k=5) > 0.4


# ── §8.2 All 10 Analytical Mutual Information Benchmarks ─────────────────────────────────────────


@pytest.mark.scientific
@pytest.mark.differential
def test_mi_benchmark_01_and_02_scalar_gaussian_analytical() -> None:
    """Benchmarks 1 & 2: independent Gaussian (0) and scalar Gaussian -0.5*ln(1-rho^2)."""
    rng = np.random.default_rng(8201)
    n = 4000
    for rho, tol in [(0.0, 0.04), (0.5, 0.05), (0.8, 0.06), (0.95, 0.09)]:
        x = rng.normal(size=n)
        y = rho * x + math.sqrt(1.0 - rho**2) * rng.normal(size=n)
        exact = -0.5 * math.log(1.0 - rho**2)
        est = info.ksg_mi(x, y, k=5, units="nats")
        assert abs(est - exact) < tol, f"rho={rho}: got {est:.4f}, exact {exact:.4f}"

    # Units conversion parity (nats <-> bits)
    est_nats = info.ksg_mi(x, y, k=5, units="nats")
    est_bits = info.ksg_mi(x, y, k=5, units="bits")
    assert est_bits == pytest.approx(est_nats / math.log(2.0), rel=1e-12)
    assert info.convert_units(math.log(2.0), "bits") == pytest.approx(1.0, rel=1e-12)
    assert info.convert_units(1.0, "nats") == pytest.approx(1.0, rel=1e-12)


@pytest.mark.scientific
@pytest.mark.differential
def test_mi_benchmark_03_multivariate_jointly_gaussian() -> None:
    """Benchmark 3: multivariate Gaussian MI = 0.5*ln(det(Sx)*det(Sy)/det(Sxy))."""
    rng = np.random.default_rng(8203)
    cov = np.array(
        [
            [1.0, 0.3, 0.6, 0.1],
            [0.3, 1.2, 0.2, 0.7],
            [0.6, 0.2, 1.1, 0.25],
            [0.1, 0.7, 0.25, 1.0],
        ],
        dtype=np.float64,
    )
    exact = 0.5 * math.log(
        np.linalg.det(cov[:2, :2]) * np.linalg.det(cov[2:, 2:]) / np.linalg.det(cov)
    )
    xy = rng.multivariate_normal(np.zeros(4), cov, size=6000)
    est = info.ksg_mi(xy[:, :2], xy[:, 2:], k=5, units="nats")
    assert abs(est - exact) < 0.08, f"multivariate Gaussian: got {est:.4f}, exact {exact:.4f}"


@pytest.mark.scientific
@pytest.mark.differential
def test_mi_benchmarks_04_05_06_07_discrete_binary_bsc_and_xor() -> None:
    """Benchmarks 4–7: independent bits, identical bits, BSC(p), and XOR synergy/CMI."""
    x_ind = np.array([0, 0, 1, 1] * 250)
    y_ind = np.array([0, 1, 0, 1] * 250)
    assert info.discrete_mi(x_ind, y_ind, units="nats") == pytest.approx(0.0, abs=1e-12)
    assert info.discrete_mi(x_ind, x_ind, units="nats") == pytest.approx(math.log(2.0), abs=1e-12)
    assert info.discrete_mi(x_ind, x_ind, units="bits") == pytest.approx(1.0, abs=1e-12)

    # Benchmark 6: Binary Symmetric Channel with flip probability p -> ln(2) - H_b(p)
    rng = np.random.default_rng(8206)
    n = 20_000
    for p in (0.0, 0.1, 0.25, 0.5):
        x = rng.integers(0, 2, size=n)
        flip = (rng.random(n) < p).astype(int)
        y = x ^ flip
        h_b = 0.0 if p in (0.0, 1.0) else -(p * math.log(p) + (1 - p) * math.log(1 - p))
        exact = math.log(2.0) - h_b
        assert info.discrete_mi(x, y, units="nats") == pytest.approx(exact, abs=0.015)

    # Benchmark 7: XOR T = X ^ Y (zero pairwise MI, full joint MI and full conditional MI)
    x_xor = np.array([0, 0, 1, 1] * 500)
    y_xor = np.array([0, 1, 0, 1] * 500)
    t_xor = x_xor ^ y_xor
    xy_joint = 2 * x_xor + y_xor
    assert info.discrete_mi(x_xor, t_xor, units="nats") == pytest.approx(0.0, abs=1e-12)
    assert info.discrete_mi(y_xor, t_xor, units="nats") == pytest.approx(0.0, abs=1e-12)
    assert info.discrete_mi(xy_joint, t_xor, units="nats") == pytest.approx(
        math.log(2.0), abs=1e-12
    )
    assert info.discrete_cmi(x_xor, t_xor, y_xor, units="nats") == pytest.approx(
        math.log(2.0), abs=1e-12
    )


@pytest.mark.scientific
@pytest.mark.differential
def test_mi_benchmarks_08_09_10_nonlinear_invariance_dpi_and_variational_bound() -> None:
    """Benchmarks 8–10: Y=X^2+eps, monotone invariance, Markov chain DPI, and Barber–Agakov."""
    rng = np.random.default_rng(8208)
    n = 4000

    # Benchmark 8: Nonlinear dependence Y = X^2 + noise (Pearson ~ 0, MI > 0)
    x8 = rng.normal(size=n)
    y8 = x8**2 + 0.15 * rng.normal(size=n)
    assert abs(float(np.corrcoef(x8, y8)[0, 1])) < 0.06
    assert info.ksg_mi(x8, y8, k=5, units="nats") > 0.4

    # Benchmark 9: Smooth strictly monotone transformation invariance
    x9 = rng.normal(size=n)
    y9 = 0.75 * x9 + math.sqrt(1.0 - 0.75**2) * rng.normal(size=n)
    mi_orig = info.ksg_mi(x9, y9, k=5, units="nats")
    mi_trans = info.ksg_mi(np.sinh(0.5 * x9), y9**3 + 0.5 * y9, k=5, units="nats")
    assert abs(mi_orig - mi_trans) < 0.06

    # Benchmark 10: Data-Processing Inequality on Markov chain X -> Y -> Z
    x10 = rng.normal(size=n)
    y10 = 0.85 * x10 + math.sqrt(1.0 - 0.85**2) * rng.normal(size=n)
    z10 = 0.65 * y10 + math.sqrt(1.0 - 0.65**2) * rng.normal(size=n)
    mi_xy = info.ksg_mi(x10, y10, k=5, units="nats")
    mi_xz = info.ksg_mi(x10, z10, k=5, units="nats")
    assert mi_xz < mi_xy - 0.15

    # Held-out Barber–Agakov variational MI lower bound:
    # (a) On correlated Gaussian -> binary Y, bound is positive and <= H(Y)
    z_sig = rng.normal(size=(1000, 4))
    y_sig = (2.5 * z_sig[:, 0] + rng.normal(size=1000) > 0).astype(int)
    vb_sig = info.variational_mi_lower_bound(
        z_sig[:600], y_sig[:600], z_sig[600:], y_sig[600:], c=1.0, units="nats"
    )
    assert 0.15 < vb_sig.bound < vb_sig.entropy_y
    assert vb_sig.bound == pytest.approx(vb_sig.entropy_y + vb_sig.mean_log_q, abs=1e-12)

    # (b) Under pure null with weak regularisation, unclipped held-out bound CAN be < 0
    z_null = rng.normal(size=(300, 40))
    y_null = rng.integers(0, 2, size=300)
    vb_null = info.variational_mi_lower_bound(
        z_null[:150], y_null[:150], z_null[150:], y_null[150:], c=10.0, units="nats"
    )
    assert vb_null.bound < 0.0

    # Zero k-th neighbour radii (discrete/tied or duplicated samples) MUST raise ValueError
    # rather than silently returning an inflated MI estimate; explicit jitter resolves ties.
    x_tied = np.tile([0.0, 0.0, 1.0, 1.0], 250)
    y_tied = np.tile([0.0, 1.0, 0.0, 1.0], 250)
    with pytest.raises(ValueError, match="zero .* neighbour radius"):
        info.ksg_mi(x_tied, y_tied, k=5, units="nats")
    assert abs(info.ksg_mi(x_tied, y_tied, k=5, units="nats", jitter=1e-4, seed=0)) < 0.05

    # Duplicated continuous samples (6 identical copies > k=5) also caught cleanly
    x_dup = rng.normal(size=200)
    y_dup = rng.normal(size=200)
    x_dup[:6] = x_dup[0]
    y_dup[:6] = y_dup[0]
    with pytest.raises(ValueError, match="zero .* neighbour radius"):
        info.ksg_mi(x_dup, y_dup, k=5, units="nats")
