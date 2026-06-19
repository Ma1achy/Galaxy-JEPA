"""Triggered matched-evaluation — the targeted confound killer (design 3D-ii / 2A).

Fires only when something flags it (bounded cost), not always (too expensive) or never
(leaves confounds unresolved). Two consumers share the one matching machine:

* **Nuisance gate (3D-ii):** when a nuisance-AUC is competitive with the morphology-AUC, the
  feature is re-probed on galaxies *matched* on that nuisance (the nuisance held ~constant
  within the matched set, so it can't be the signal). The feature **survives** (real) or is
  **confounded**.
* **Conditional recoverability (2A):** the surgical cross-check on the eigen-flagged pairs —
  match on feature B, re-probe feature A. What survives matching on B is *representational*
  entanglement; what vanishes was *world-correlation* (astrophysics).

The matching/stratification machinery is built; the **trigger condition** (when matching
fires) is flagged — :func:`nuisance_competitive` carries the placeholder margin.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from galaxy_jepa.probing.logistic import Embeddings, probe_auc

__all__ = [
    "nuisance_competitive",
    "stratified_match",
    "matched_auc",
    "MatchedVerdict",
    "matched_evaluation",
]


def nuisance_competitive(morph_auc: float, nuisance_auc: float, *, margin: float = 0.0) -> bool:
    """Whether a nuisance is competitive enough to trigger matched evaluation (design 3D-ii).

    FLAGGED trigger: pending stats grounding — do not finalise. Placeholder: the nuisance-AUC
    is within ``margin`` of (or above) the morphology-AUC. ``margin=0`` ⇒ fires only when the
    nuisance is at least as decodable as the morphology; a positive margin fires earlier (more
    conservative). The grounding session sets the defensible margin.
    """
    return nuisance_auc >= morph_auc - margin


def stratified_match(
    values: np.ndarray, labels: np.ndarray, *, n_strata: int = 5, seed: int = 0
) -> np.ndarray:
    """Indices of a class-balanced subset within strata of ``values`` (nuisance held constant).

    Bins ``values`` into ``n_strata`` quantile strata; within each stratum keeps an equal
    number of each morphology class (the per-stratum minority count). Across the returned set
    the nuisance distribution is balanced between the classes, so it cannot drive the AUC.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(values)
    idx_all = np.nonzero(finite)[0]
    if idx_all.size == 0:
        return np.array([], dtype=np.int64)
    edges = np.quantile(values[finite], np.linspace(0, 1, n_strata + 1))
    edges[-1] = np.inf  # include the maximum
    kept: list[int] = []
    for s in range(n_strata):
        in_stratum = idx_all[(values[idx_all] >= edges[s]) & (values[idx_all] < edges[s + 1])]
        pos = in_stratum[labels[in_stratum] == 1]
        neg = in_stratum[labels[in_stratum] == 0]
        take = min(len(pos), len(neg))
        if take == 0:
            continue
        kept.extend(rng.choice(pos, take, replace=False).tolist())
        kept.extend(rng.choice(neg, take, replace=False).tolist())
    return np.asarray(sorted(kept), dtype=np.int64)


def matched_auc(
    train: Embeddings,
    test: Embeddings,
    match_train: np.ndarray,
    match_test: np.ndarray,
    *,
    n_strata: int = 5,
    c: float = 1.0,
    seed: int = 0,
) -> float:
    """Re-probe the feature within the matched (nuisance-balanced) train/test subsets."""
    tr = stratified_match(match_train, train.y, n_strata=n_strata, seed=seed)
    te = stratified_match(match_test, test.y, n_strata=n_strata, seed=seed + 1)
    if tr.size == 0 or te.size == 0:
        return 0.5
    train_m = Embeddings(train.x[tr], train.y[tr], train.fraction[tr])
    test_m = Embeddings(test.x[te], test.y[te], test.fraction[te])
    if len(np.unique(train_m.y)) < 2 or len(np.unique(test_m.y)) < 2:
        return 0.5
    return probe_auc(train_m, test_m, c=c)


@dataclasses.dataclass(frozen=True)
class MatchedVerdict:
    """The outcome of a matched evaluation: did the signal survive holding the confound fixed?"""

    matched_auc: float
    survived: bool


def matched_evaluation(
    train: Embeddings,
    test: Embeddings,
    match_train: np.ndarray,
    match_test: np.ndarray,
    *,
    survive_threshold: float,
    n_strata: int = 5,
    c: float = 1.0,
    seed: int = 0,
) -> MatchedVerdict:
    """Matched re-probe → survive (signal real, not the confound) or confounded.

    ``survive_threshold`` is the bar the matched AUC must still clear (the caller passes the
    effect floor); below it the apparent direction was the confound — itself a real finding.
    """
    auc = matched_auc(train, test, match_train, match_test, n_strata=n_strata, c=c, seed=seed)
    return MatchedVerdict(matched_auc=auc, survived=auc >= survive_threshold)
