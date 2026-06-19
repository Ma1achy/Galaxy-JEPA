"""Null-calibrated existence verdict + the multiplicity correction (design 3B / 2B).

The gate's "is this feature real?" bar is **not** a hand-picked constant — it is "exceeds the
negative-control null at p < α" (3B). This module turns the five-control battery
(``controls.py``) into that verdict. The structural dependency the design insists on holds
here: the null is an *input* to the gate, so the gate cannot fire until this module has
computed it.

**Two FLAGGED decisions live here** (structure built, formula pending the stats grounding):

* **(5) the existence significance + effect floor** — how ``real_auc`` is tested against the
  five-null distribution, and the clean-vs-marginal floor. :func:`existence_pvalue` is the
  flagged placeholder (an empirical one-sided p); the floor is a ``ProbingConfig`` field
  applied as a gate metric.
* **(1) the multiplicity correction** — the family of ~12 existence tests carries the strict
  bar. :func:`family_significant` selects Bonferroni (the defensible placeholder) vs
  Benjamini–Yekutieli (the dependence-robust FDR alternative). Both are implemented; *which
  to use* is the flagged decision (set in ``ProbingConfig.multiplicity``).
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
]


def five_null_samples(controls: FeatureControls) -> np.ndarray:
    """Pool the five negative controls into one conservative null distribution.

    FLAGGED: pending stats grounding — do not finalise. The design calibrates against the
    **most conservative across all five** (3C). The placeholder pools every null sample (the
    two resamplable distributions + the three single-AUC controls) so the existence p is
    computed against the *whole* battery; the conservatism falls out of the upper tail being
    set by the strongest control. The final policy (max-of-means? a per-draw elementwise max?)
    is owned in the grounding session — only this combination rule changes.
    """
    singletons = np.array(
        [controls.noise_encoder_auc, controls.untrained_encoder_auc, controls.sky_noise_auc],
        dtype=np.float64,
    )
    return np.concatenate([controls.shuffled_nulls, controls.random_embedding_nulls, singletons])


def existence_pvalue(real_auc: float, null_samples: np.ndarray) -> float:
    """One-sided p that ``real_auc`` is no better than the null.

    FLAGGED: pending stats grounding — do not finalise. Placeholder: the add-one empirical
    estimator ``(1 + #{null ≥ real}) / (1 + N)`` — conservative, never returns exactly 0, and
    needs no distributional assumption. The grounding session decides whether this, a
    parametric fit, or a different tail definition is the defensible choice *here*.
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
    method: str = "bonferroni",
    n_tests: int | None = None,
) -> dict[str, bool]:
    """Family-wise significance over the primary existence tests (the ~12), corrected.

    FLAGGED: pending stats grounding — do not finalise. The *correction itself* is built (both
    methods below); the flagged decision is **which** to use (``ProbingConfig.multiplicity``):

    * ``bonferroni`` — reject iff ``p ≤ α / n``. Simple, defensible, not crippling at n≈12.
    * ``benjamini_yekutieli`` — the dependence-robust FDR step-up (honest that the tests share
      embeddings / correlated features), higher power.

    ``n_tests`` overrides the family size (the 2C build flag fixes the exact count); by
    default it is the number of p-values supplied.
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
    method: str = "bonferroni",
    effect_floor: float = 0.65,
    n_tests: int | None = None,
) -> dict[str, ExistenceVerdict]:
    """The full family-corrected existence layer: p-values → significance → effect floor."""
    pvals = existence_pvalues(controls)
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
