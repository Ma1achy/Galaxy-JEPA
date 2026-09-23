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
    "matched_indices",
    "RETAIN_FRACTION",
    "MIN_MATCHED_TEST",
    "MIN_MATCHED_SHARE",
    "SURVIVES",
    "PARTIAL",
    "COLLAPSES",
    "UNRESOLVED",
    "RetentionVerdict",
    "retention_verdict",
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
    """Re-probe the feature within the matched (nuisance-balanced) train/test subsets.

    Returns 0.5 on a degenerate match. Callers that need to tell that 0.5 apart from a real
    collapse want :func:`matched_evaluation`, whose verdict carries the survivor counts.
    """
    auc, _tr, _te, _degenerate = _matched_auc_with_counts(
        train, test, match_train, match_test, n_strata=n_strata, c=c, seed=seed
    )
    return auc


def _matched_auc_with_counts(
    train: Embeddings,
    test: Embeddings,
    match_train: np.ndarray,
    match_test: np.ndarray,
    *,
    n_strata: int = 5,
    c: float = 1.0,
    seed: int = 0,
) -> tuple[float, int, int, bool]:
    """``(auc, n_matched_train, n_matched_test, degenerate)`` — the counts the 0.5 needs."""
    tr, te = matched_indices(match_train, train.y, match_test, test.y, n_strata=n_strata, seed=seed)
    if tr.size == 0 or te.size == 0:
        return 0.5, int(tr.size), int(te.size), True
    train_m = Embeddings(train.x[tr], train.y[tr], train.fraction[tr])
    test_m = Embeddings(test.x[te], test.y[te], test.fraction[te])
    if len(np.unique(train_m.y)) < 2 or len(np.unique(test_m.y)) < 2:
        return 0.5, int(tr.size), int(te.size), True
    return probe_auc(train_m, test_m, c=c), int(tr.size), int(te.size), False


@dataclasses.dataclass(frozen=True)
class MatchedVerdict:
    """The outcome of a matched evaluation: did the signal survive holding the confound fixed?

    **The survivor counts are not decoration.** :func:`matched_auc` returns exactly 0.5 when the
    matched set is empty or single-class, which is indistinguishable from "the signal was entirely
    confound" unless the count travels with the number. Brief O1 had to bypass
    :func:`matched_evaluation` altogether and call :func:`stratified_match` itself to report them;
    the second consumer is the full 37-feature ladder, so they live here now.

    ``degenerate`` says plainly which 0.5 this is: a statement about the SAMPLE, never folded into
    a statement about the signal.
    """

    matched_auc: float
    survived: bool
    n_matched_train: int = 0
    n_matched_test: int = 0
    n_train: int = 0
    n_test: int = 0
    degenerate: bool = False
    #: The retention judgement (D24) and what it was computed from; ``None`` on the entanglement
    #: leg, which still judges survival against a threshold (flagged, not changed — see D24).
    retention: RetentionVerdict | None = None
    matched_auc_lo: float | None = None
    bar_unmatched: float | None = None  # C, mean over ``k_bar`` untrained draws
    bar_matched: float | None = None  # C_m, the same draws re-measured on the matched rows
    k_bar: int = 0
    margin_established: bool | None = None  # the unmatched margin passed D23 existence

    @property
    def share_test(self) -> float:
        """Fraction of the unmatched test set that survived matching."""
        return self.n_matched_test / self.n_test if self.n_test else 0.0


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
    auc, tr_n, te_n, degenerate = _matched_auc_with_counts(
        train, test, match_train, match_test, n_strata=n_strata, c=c, seed=seed
    )
    return MatchedVerdict(
        matched_auc=auc,
        survived=auc >= survive_threshold,
        n_matched_train=tr_n,
        n_matched_test=te_n,
        n_train=int(len(train.y)),
        n_test=int(len(test.y)),
        degenerate=degenerate,
    )


def matched_indices(
    match_train: np.ndarray,
    y_train: np.ndarray,
    match_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_strata: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The matched train and test rows — the selection :func:`matched_evaluation` probes on.

    Depends only on the nuisance values, the labels and the seed, never on the embeddings. That is
    what lets a second matrix (the untrained encoder, whose bar is re-measured on the matched rows)
    or a second probe (the MLP) be scored on **exactly** the rows the linear re-probe used: the
    comparison is then between two probes on one question rather than two questions.
    """
    tr = stratified_match(match_train, y_train, n_strata=n_strata, seed=seed)
    te = stratified_match(match_test, y_test, n_strata=n_strata, seed=seed + 1)
    return tr, te


#: O1's pre-registered retention rule, fixed before any number existed and moved here unchanged so
#: every consumer judges survival the same way. ``A`` is the unmatched AUC and ``C`` its
#: untrained-encoder bar; ``M`` is the matched AUC and ``C_m`` the bar **re-measured on the same
#: matched rows**, so the comparison is like-for-like. The 0.5 is a declared choice, not derived.
#:
#: Why this and not the effect floor: the ladder judged "survived matching" as *matched AUC ≥
#: 0.7267*, and a feature already below 0.7267 unmatched fails that whatever matching does. In
#: Brief P that was 18 of the 19 features labelled "confounded by size" — matching moved them by a
#: median of 0.027. Retention asks the question matching is for: how much of the effect is left.
RETAIN_FRACTION: float = 0.5
MIN_MATCHED_TEST: int = 500
MIN_MATCHED_SHARE: float = 0.10
#: Untrained draws averaged into C and C_m (D24) — Brief R0's construction, kept so the ladder
#: reproduces R0 exactly. The margin floor is NOT a further parameter: retention is judged only
#: where the unmatched margin has passed D23's existence test, else UNRESOLVED.
RETENTION_SEEDS: int = 3

SURVIVES = "SURVIVES"
PARTIAL = "PARTIAL"
COLLAPSES = "COLLAPSES"
UNRESOLVED = "UNRESOLVED"


@dataclasses.dataclass(frozen=True)
class RetentionVerdict:
    """Four states, and UNRESOLVED is a statement about the SAMPLE — never folded into COLLAPSES."""

    verdict: str
    retained: float | None
    unmatched_margin: float
    matched_margin: float | None


def retention_verdict(
    a: float,
    c: float,
    m: float | None,
    m_lo: float | None,
    c_m: float | None,
    *,
    n_matched_test: int,
    n_test: int,
) -> RetentionVerdict:
    """O1's rule::

    SURVIVES  : (M − C_m) ≥ 0.5·(A − C)  AND  M's CI lower bound > C_m
    COLLAPSES : (M − C_m) ≤ 0            OR  M's CI contains C_m
    PARTIAL   : between the two — real, but substantially confounded
    UNRESOLVED: < 500 matched test galaxies, or < 10% of the unmatched test set surviving
    """
    unmatched = a - c
    if (
        m is None
        or m_lo is None
        or c_m is None
        or n_matched_test < MIN_MATCHED_TEST
        or n_matched_test < MIN_MATCHED_SHARE * n_test
    ):
        return RetentionVerdict(UNRESOLVED, None, unmatched, None)
    matched = m - c_m
    retained = matched / unmatched if unmatched > 0 else 0.0
    if matched <= 0 or m_lo <= c_m:
        return RetentionVerdict(COLLAPSES, retained, unmatched, matched)
    if matched >= RETAIN_FRACTION * unmatched:
        return RetentionVerdict(SURVIVES, retained, unmatched, matched)
    return RetentionVerdict(PARTIAL, retained, unmatched, matched)
