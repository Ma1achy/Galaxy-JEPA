"""Null-calibrated existence verdict + the multiplicity correction (design 3B / 2B).

The gate's "is this feature real?" bar is **not** a hand-picked constant — it is "exceeds the
negative-control null at p < α" (3B). This module turns the negative-control battery
(``controls.py``) into that verdict. The structural dependency the design insists on holds
here: the null is an *input* to the gate, so the gate cannot fire until this module has
computed it.

**Two of the five grounded statistical decisions live here** (spec §Statistics):

* **(1) existence p-value** — the real value located against a chance null, with the effect
  floor applied as a *separate* second gate (see :func:`existence_verdicts`).
* **(3) multiplicity — Benjamini–Yekutieli.** Chosen over BH knowingly: these tests are
  correlated by construction, and FDR is the right target for a discovery catalogue. The family
  count is **per-scheme**, passed in — never a constant in this module.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np

from galaxy_jepa.probing.controls import FeatureControls

__all__ = [
    "existence_null_samples",
    "existence_pvalue",
    "family_significant",
    "ExistenceVerdict",
    "existence_pvalues",
    "attainable_min_pvalue",
    "assert_null_resolution",
    "family_bar",
    "required_null_draws",
    "BUDGET_FAMILY_SIZE",
]


def existence_null_samples(controls: FeatureControls) -> np.ndarray:
    """Combine the **four chance-calibrated** negative controls into one null: max, per draw.

    Design 3C is explicit that existence is calibrated against the *max (most conservative)*
    across the battery — a real feature must beat the **strongest** null, not an average of
    them. Pooling every sample into one bag (the earlier placeholder) is not that: it lets the
    weak controls dilute the upper tail, which is the wrong direction for a conservative bar.

    So the null is assembled **per draw**: the two resamplable controls contribute a paired draw
    each, the two single-AUC controls are constants that every draw must also clear, and the
    null sample is the elementwise maximum. That keeps a *distribution* (which the empirical
    p-value needs) while making every sample "the best any control managed on this draw".

    **The fifth control (3C-5, sky/noise labels) is deliberately NOT here — D19.** A null has to
    be chance-calibrated: it must answer "what AUC does this machinery reach when the thing being
    measured is absent?". Four of the five break something and therefore do —
    ``shuffled`` destroys the image-label correspondence, ``random_embedding`` replaces the
    representation, ``noise_encoder`` replaces the images, and ``untrained_encoder`` replaces the
    *pretraining* while keeping images and labels real, which is exactly the "the probe, not the
    pretraining, did the work" null. 3C-5 breaks nothing: real images, real encoder, real probe,
    a *different real label*. Its AUC measures how much image-quality content the representation
    holds, which is a **diagnostic**, not a bar — it stays on ``FeatureControls`` and in the
    nuisance panel. Measured at J4 it is bit-identical to ``nuisance_aucs["snr"]``: one
    measurement, entered twice, once as a null and once as a diagnostic. Under the old bar it
    sat at 0.8355–0.8416 and failed *every* feature, featured-ness included. See D19.

    NOTE — the spec fixes "max across the battery" but not whether the max is taken per draw or
    over the controls' means. Per-draw is used because a max-of-means collapses the null to a
    single point and leaves :func:`existence_pvalue` no distribution to locate the real value
    in; changing it is a change to this one function.
    """
    singleton_max = max(
        float(controls.noise_encoder_auc),
        float(controls.untrained_encoder_auc),
    )
    shuffled = np.asarray(controls.shuffled_nulls, dtype=np.float64)
    random_emb = np.asarray(controls.random_embedding_nulls, dtype=np.float64)
    n = min(shuffled.size, random_emb.size)
    if n == 0:  # no resamplable draws — the null degenerates to the strongest singleton
        return np.array([singleton_max], dtype=np.float64)
    return np.maximum(np.maximum(shuffled[:n], random_emb[:n]), singleton_max)


def existence_pvalue(real_auc: float, null_samples: np.ndarray) -> float:
    """One-sided p that ``real_auc`` is no better than the null — decision (1), grounded.

    The real value located in the chance null: the add-one empirical estimator
    ``(1 + #{null ≥ real}) / (1 + N)``. Conservative, never returns exactly 0 (so a finite
    resample count cannot manufacture an infinitely-small p), and assumes nothing about the
    null's shape — which matters because the existence null is a per-draw maximum and is not
    remotely Gaussian.

    OPEN (spec register item 9): tie-handling when ``real_auc`` is ≈1.0. The ``>=`` comparison
    counts exact ties against the real value, which is the conservative direction; whether that
    is the wanted convention at the ceiling is unsettled.
    """
    null = np.asarray(null_samples, dtype=np.float64)
    n = null.size
    if n == 0:
        return 1.0
    return float((1 + int(np.sum(null >= real_auc))) / (1 + n))


def existence_pvalues(controls: Mapping[str, FeatureControls]) -> dict[str, float]:
    """Per-feature existence p-value against that feature's own chance-calibrated nulls."""
    return {
        feat: existence_pvalue(fc.real_auc, existence_null_samples(fc))
        for feat, fc in controls.items()
    }


def family_significant(
    pvalues: Mapping[str, float],
    *,
    alpha: float = 0.05,
    method: str = "benjamini_yekutieli",
    n_tests: int | None = None,
) -> dict[str, bool]:
    """Family-corrected significance over the primary existence tests — decision (3), grounded.

    **Benjamini–Yekutieli** is the choice. The existence tests are correlated by construction
    (bulge levels partition one variable; features co-occur in the sky), so an
    independence-assuming correction is wrong here; and the deliverable is a *discovery
    catalogue*, for which controlling the false-discovery rate is the right target rather than
    the family-wise error rate. BY controls FDR under arbitrary dependence, which is exactly the
    situation. ``bonferroni`` remains selectable as a sensitivity check, not as the default.

    ``n_tests`` is the family size and is **per-scheme** (≈37 for the full tree, ≈10–13 reduced);
    it must be passed by the caller from the active scheme rather than inferred globally. It
    falls back to the number of p-values supplied.
    """
    feats = list(pvalues)
    m = n_tests if n_tests is not None else len(feats)
    if m <= 0:
        return {f: False for f in feats}

    if method == "bonferroni":
        bar = alpha / m
        return {f: pvalues[f] <= bar for f in feats}

    if method == "benjamini_yekutieli":
        # BY step-up: sort ascending, c(m)=Σ 1/i, reject p(k) for the largest k with
        # p(k) ≤ (k / (m·c(m)))·α, and everything ranked below it.
        order = sorted(feats, key=lambda f: pvalues[f])
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
        threshold_rank = 0
        for k, f in enumerate(order, start=1):
            if pvalues[f] <= (k / (m * c_m)) * alpha:
                threshold_rank = k
        passing = set(order[:threshold_rank])
        return {f: f in passing for f in feats}

    raise ValueError(f"unknown multiplicity method {method!r}")


def attainable_min_pvalue(n_null: int) -> float:
    """Smallest p the add-one empirical estimator can return from ``n_null`` draws."""
    return 1.0 / (1.0 + max(int(n_null), 0))


def family_bar(alpha: float, method: str, n_tests: int) -> float:
    """The strictest per-test threshold a family of ``n_tests`` has to clear.

    For BY this is the **rank-1** step-up threshold ``alpha / (m * H_m)`` — the hardest one, so a
    null that clears it clears every rank below.
    """
    if n_tests <= 0:
        raise ValueError("n_tests must be positive")
    if method == "bonferroni":
        return alpha / n_tests
    if method == "benjamini_yekutieli":
        h_m = float(np.sum(1.0 / np.arange(1, n_tests + 1)))
        return (1.0 / (n_tests * h_m)) * alpha
    raise ValueError(f"unknown multiplicity method {method!r}")


def required_null_draws(*, alpha: float, method: str, n_tests: int) -> int:
    """Smallest ``n_null_draws`` whose attainable floor ``1/(n+1)`` clears the family bar.

    The inverse of :func:`assert_null_resolution`, so the budget is *derived* rather than
    guessed. Note this is the bar for "can pass at all", not for "passes stably" — near the
    threshold the empirical tail's own Monte-Carlo error matters, so a real budget sits well
    above this floor.
    """
    return int(np.ceil(1.0 / family_bar(alpha, method, n_tests))) - 1


#: The null budget is sized from the **full tree** (Scheme 1, the larger family) and applied
#: unchanged to both schemes. If Scheme 1 and Scheme 2 disagree about a feature, the candidate
#: explanations must be power and expressibility — "the two runs had different null resolution"
#: must not be on that list. So this is deliberately *not* derived from the active scheme.
BUDGET_FAMILY_SIZE: int = 37


#: The two existence constructions. ``empirical`` is the original add-one estimator over the
#: chance-calibrated null; ``untrained_z`` is D23's two-sided construction.
EXISTENCE_EMPIRICAL: str = "empirical"
EXISTENCE_UNTRAINED_Z: str = "untrained_z"

#: Below this many untrained seeds the bar's own spread is too noisy to stand in the denominator
#: of the z-statistic. The relative standard error of an sd estimate is ``1/sqrt(2(K-1))`` — 16%
#: at K=20, 13% at K=30. Brief P runs at K=30; 20 is the refusal point, not the target.
K_MIN: int = 20


def untrained_z_pvalue(
    real_auc: float, se_real: float, untrained_aucs: Sequence[float] | np.ndarray
) -> float:
    """One-sided p that ``real_auc`` is no better than its untrained bar — D23.

    ``existence_null_samples`` is a **point mass**: the untrained singleton floors every draw
    (see that function), so the add-one p-value can only return ``1/(n+1)`` or ``1.0`` and the
    Benjamini–Yekutieli correction has nothing calibrated to act on. A 37-feature catalogue
    without working multiplicity control is the exact failure BY was chosen to prevent.

    The construction puts an uncertainty on **both** sides and compares them::

        z = (AUC_real - mean(C)) / sqrt( sd(C)^2 + se_real^2 )

    where ``C`` is the untrained-encoder bar measured across K seeds (it has real spread — Brief
    N2 measured per-feature ranges 0.0026–0.0209 over three seeds) and ``se_real`` is the
    bootstrap SE of the real AUC. Continuous in ``real_auc``, so BY applies normally and the
    3,109-draw resolution floor — which is a property of the *add-one estimator*, not of
    existence — does not arise.

    **Student t, not normal, with df = K-1.** The sd in the denominator is *estimated* from K
    samples, and BY's rank-1 bar at family 37 is 3.216e-4 — a ~3.4σ statement. At df=29 the t
    quantile there is 3.70 against the normal's 3.41, so the normal would be optimistic in
    exactly the tail the correction cares about. The conservative direction is the right one for
    a gate.

    The price is a distributional assumption where the empirical estimator had none. That is
    recorded in D23 and tested per feature rather than asserted.
    """
    bar = np.asarray(untrained_aucs, dtype=np.float64)
    k = int(bar.size)
    if k < 2:
        raise ValueError(
            f"untrained_z needs at least 2 untrained seeds to estimate the bar's spread, got {k}"
        )
    mean_bar = float(bar.mean())
    sd_bar = float(bar.std(ddof=1))
    se = max(float(se_real), 0.0)
    denom = math.sqrt(sd_bar**2 + se**2)
    if denom <= 0.0:
        # Both the bar and the real AUC would have to be exactly noiseless. That is not a
        # finding about the feature, it is a broken measurement, so it raises (architecture.md
        # "fail loudly, never silently default") rather than returning a p of 0 or 1.
        raise ValueError(
            f"untrained_z has no scale: sd(bar)={sd_bar} and se_real={se} are both zero, so no "
            f"test statistic can be formed. Check the untrained bank and the bootstrap."
        )
    from scipy.stats import t as student_t

    return float(student_t.sf((float(real_auc) - mean_bar) / denom, df=k - 1))


def assert_untrained_bank_resolution(k: int, *, k_min: int = K_MIN) -> None:
    """Raise if the untrained bank is too small for its spread to be trusted.

    The sibling of :func:`assert_null_resolution`, and it exists for the same reason. Switching
    to ``untrained_z`` removes the 3,109-draw requirement, and a gate that simply stops biting
    once its neighbour is satisfied is not a gate — so the resolution requirement moves rather
    than disappearing. Here it lands on K, because K is what the denominator's ``sd(C)`` is
    estimated from.
    """
    if k < k_min:
        raise ValueError(
            f"untrained bank too small: {k} seeds give a bar whose standard deviation has a "
            f"relative standard error of {1 / math.sqrt(2 * max(k - 1, 1)):.0%}, and that sd is "
            f"the denominator of every existence z-statistic. Build at least {k_min} seeds "
            f"(`uv run python artifacts/p1_untrained_bank.py`)."
        )


def assert_null_resolution(n_null: int, *, alpha: float, method: str, n_tests: int) -> None:
    """Raise if the null is too coarse for any feature to clear the corrected bar.

    The failure this prevents is silent and severe: with too few null draws the smallest
    attainable p exceeds the family-corrected threshold, so **every** feature fails existence
    and the ladder returns an all-R3/R4 catalogue that looks like a scientific null result but
    is really an artefact of the resample count. The bar is a property of the *design*; the
    resolution is a property of the *budget*, and the two must be checked against each other
    before the verdicts are read.
    """
    if n_tests <= 0:
        return
    floor = attainable_min_pvalue(n_null)
    bar = family_bar(alpha, method, n_tests)
    if floor > bar:
        needed = required_null_draws(alpha=alpha, method=method, n_tests=n_tests)
        raise ValueError(
            f"null resolution too coarse: {n_null} draws give a smallest attainable p of "
            f"{floor:.5f}, but the {method} bar at family size {n_tests} (alpha={alpha}) is "
            f"{bar:.5f}. Every feature would fail existence regardless of its signal, and the "
            f"resulting all-R3/R4 catalogue would be an artefact of the resample count, not a "
            f"result. Raise ProbingConfig.n_null_draws to at least {needed}."
        )


#: The margin over its own bar that a feature must be able to RESOLVE before an R4 can be read as
#: "the encoder cannot see this" rather than "this bucket is too small to tell". A **declared**
#: reference, not a derived one — the same posture as O1's 0.5 retention figure. 0.05 sits just
#: above t09-boxy's measured margin (+0.0489), the thinnest real signal the six-feature spread
#: found, so a bucket that cannot resolve 0.05 could not have found the weakest thing yet seen.
UNDERPOWERED_MARGIN: float = 0.05


def resolvable_margin(
    se_real: float,
    *,
    sd_bar: float = 0.0,
    alpha: float = 0.05,
    method: str = "benjamini_yekutieli",
    n_tests: int = 37,
    power: float = 0.80,
    df: int | None = None,
) -> float:
    """The smallest margin over its own bar this feature could show as significant.

    A deep per-bucket feature can fail existence because the encoder cannot read it, or because
    the bucket has 1,198 positives and **no** effect of a plausible size would have cleared a
    family-corrected bar there. Those are different findings and the ladder must not print the
    same rung for both: an underpowered R4 that reads as a scientific null is exactly the failure
    this exists to prevent.

        margin = (t_threshold + t_power) * sqrt( sd_bar^2 + se_real^2 )

    The scale is the **same denominator the test uses**, so the answer is in the units the verdict
    was decided in.

    Stated as a **margin over the bar**, not as an absolute AUC compared to the effect floor. The
    absolute form misfires: where a feature's untrained bar already exceeds the floor — t01's is
    0.7908 against a floor of 0.7267 — ``MDE > floor`` holds by arithmetic however large the
    sample, so the best-powered feature in the catalogue would be labelled underpowered. The
    binding constraint there is the bar, not the floor, and the margin form says so.

    ``t_threshold`` is taken at BY's **most lenient** rank (rank m, ``alpha / H_m``), not its
    strictest. The claim is "this bucket could not have demonstrated a direction *even if one
    existed*", so it must hold under the most favourable bar the feature could face; declaring
    underpowered off the rank-1 bar would condemn features that merely ranked badly.
    """
    scale = math.sqrt(max(float(sd_bar), 0.0) ** 2 + max(float(se_real), 0.0) ** 2)
    if scale <= 0.0:
        return 0.0
    if method == "benjamini_yekutieli":
        h_m = float(np.sum(1.0 / np.arange(1, max(int(n_tests), 1) + 1)))
        threshold = alpha / h_m  # rank-m: the most lenient position in the family
    else:
        threshold = alpha / max(int(n_tests), 1)

    if df is not None and df > 0:
        from scipy.stats import t as student_t

        q_thr, q_pow = float(student_t.isf(threshold, df=df)), float(student_t.ppf(power, df=df))
    else:
        from scipy.stats import norm

        q_thr, q_pow = float(norm.isf(threshold)), float(norm.ppf(power))
    return (q_thr + q_pow) * scale


def is_underpowered(margin: float, *, reference: float = UNDERPOWERED_MARGIN) -> bool:
    """Whether an R4 on this feature means "cannot resolve at this N", not "absent"."""
    return margin > reference


@dataclasses.dataclass(frozen=True)
class ExistenceVerdict:
    """Per-feature existence outcome after the family-wise correction.

    ``exceeds_null`` is the corrected significance (the gate's existence metric, as 0/1);
    ``clean`` additionally clears the effect floor (clean-vs-marginal among the real, 3B).
    """

    feature: str
    real_auc: float
    pvalue: float
    exceeds_null: bool
    clean: bool
    #: Which construction produced ``pvalue`` — the two are not interchangeable and an artefact
    #: read back later must not have to guess which one it carries.
    method: str = EXISTENCE_EMPIRICAL


def existence_verdicts(
    controls: Mapping[str, FeatureControls],
    *,
    alpha: float = 0.05,
    method: str = "benjamini_yekutieli",
    effect_floor: float = 0.65,
    n_tests: int | None = None,
    existence_method: str = EXISTENCE_EMPIRICAL,
    untrained_bank: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    real_se: Mapping[str, float] | None = None,
) -> dict[str, ExistenceVerdict]:
    """The full existence layer: p-values → family correction → effect floor.

    Checks the null's resolution against the corrected bar first (:func:`assert_null_resolution`),
    because a too-coarse null produces a plausible-looking empty catalogue rather than an error.

    **Two gates, both must pass** (decisions (1) and (2)). ``exceeds_null`` is the corrected
    significance and decides real/not-real; ``clean`` additionally requires the effect floor and
    only separates clean from marginal *among the real*. The floor cannot rescue a
    non-significant feature, and significance cannot excuse a trivial effect size.
    """
    if existence_method == EXISTENCE_UNTRAINED_Z:
        if untrained_bank is None or real_se is None:
            raise ValueError(
                "existence_method='untrained_z' needs both `untrained_bank` (per-feature "
                "untrained AUCs across K seeds) and `real_se` (per-feature bootstrap SE). "
                "Build the bank with artifacts/p1_untrained_bank.py."
            )
        missing = sorted(set(controls) - set(untrained_bank))
        if missing:
            raise ValueError(f"untrained bank is missing {len(missing)} feature(s): {missing[:5]}")
        if controls:
            assert_untrained_bank_resolution(
                min(len(np.asarray(untrained_bank[f])) for f in controls)
            )
        pvals = {
            feat: untrained_z_pvalue(fc.real_auc, real_se[feat], untrained_bank[feat])
            for feat, fc in controls.items()
        }
    elif existence_method == EXISTENCE_EMPIRICAL:
        pvals = existence_pvalues(controls)
        if controls:
            n_null = min(existence_null_samples(fc).size for fc in controls.values())
            assert_null_resolution(
                n_null, alpha=alpha, method=method, n_tests=n_tests or len(pvals)
            )
    else:
        raise ValueError(f"unknown existence_method {existence_method!r}")

    significant = family_significant(pvals, alpha=alpha, method=method, n_tests=n_tests)
    return {
        feat: ExistenceVerdict(
            feature=feat,
            real_auc=fc.real_auc,
            pvalue=pvals[feat],
            exceeds_null=significant[feat],
            clean=significant[feat] and fc.real_auc >= effect_floor,
            method=existence_method,
        )
        for feat, fc in controls.items()
    }
