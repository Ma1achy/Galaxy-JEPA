"""Null-calibrated existence verdict + the multiplicity correction (design 3B / 2B).

The gate's "is this feature real?" bar is **not** a hand-picked constant — it is "exceeds the
negative-control null at p < α" (3B). This module turns the five-control battery
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
from collections.abc import Mapping

import numpy as np

from galaxy_jepa.probing.controls import FeatureControls

__all__ = [
    "five_null_samples",
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


def five_null_samples(controls: FeatureControls) -> np.ndarray:
    """Combine the five negative controls into one null: **the strongest control, per draw**.

    Design 3C is explicit that existence is calibrated against the *max (most conservative)
    across all five* — a real feature must beat the **strongest** null, not an average of them.
    Pooling every sample into one bag (the earlier placeholder) is not that: it lets the weak
    controls dilute the upper tail, which is the wrong direction for a conservative bar.

    So the null is assembled **per draw**: the two resamplable controls contribute a paired draw
    each, the three single-AUC controls are constants that every draw must also clear, and the
    null sample is the elementwise maximum. That keeps a *distribution* (which the empirical
    p-value needs) while making every sample "the best any control managed on this draw".

    NOTE — the spec fixes "max across all five" but not whether the max is taken per draw or
    over the controls' means. Per-draw is used because a max-of-means collapses the null to a
    single point and leaves :func:`existence_pvalue` no distribution to locate the real value
    in; changing it is a change to this one function.
    """
    singleton_max = max(
        float(controls.noise_encoder_auc),
        float(controls.untrained_encoder_auc),
        float(controls.sky_noise_auc),
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
    null's shape — which matters because the five-control null is a per-draw maximum and is not
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
    """Per-feature existence p-value against each feature's own five-null battery."""
    return {
        feat: existence_pvalue(fc.real_auc, five_null_samples(fc)) for feat, fc in controls.items()
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


def existence_verdicts(
    controls: Mapping[str, FeatureControls],
    *,
    alpha: float = 0.05,
    method: str = "benjamini_yekutieli",
    effect_floor: float = 0.65,
    n_tests: int | None = None,
) -> dict[str, ExistenceVerdict]:
    """The full existence layer: p-values → family correction → effect floor.

    Checks the null's resolution against the corrected bar first (:func:`assert_null_resolution`),
    because a too-coarse null produces a plausible-looking empty catalogue rather than an error.

    **Two gates, both must pass** (decisions (1) and (2)). ``exceeds_null`` is the corrected
    significance and decides real/not-real; ``clean`` additionally requires the effect floor and
    only separates clean from marginal *among the real*. The floor cannot rescue a
    non-significant feature, and significance cannot excuse a trivial effect size.
    """
    pvals = existence_pvalues(controls)
    if controls:
        n_null = min(five_null_samples(fc).size for fc in controls.values())
        assert_null_resolution(n_null, alpha=alpha, method=method, n_tests=n_tests or len(pvals))
    significant = family_significant(pvals, alpha=alpha, method=method, n_tests=n_tests)
    return {
        feat: ExistenceVerdict(
            feature=feat,
            real_auc=fc.real_auc,
            pvalue=pvals[feat],
            exceeds_null=significant[feat],
            clean=significant[feat] and fc.real_auc >= effect_floor,
        )
        for feat, fc in controls.items()
    }
