"""Invariants for the matched-evaluation primitives (probing/matching.py).

This module had **no test file** until Brief O1, and it is now on Paper 1's critical path: 3D-ii's
"targeted, fires only for flagged features" has been measured false on two encoders, so matched
evaluation is what answers "you are not detecting spiral arms, you are detecting that nearby bright
well-resolved galaxies look different". A module carrying that claim needs its edges pinned.

What was uncovered before: `matched_auc` and `MatchedVerdict` entirely, the 0.5 degenerate
fallback, and the train/test seed asymmetry. (`matched_evaluation` and its `survive_threshold` are
gone: D25 moved the last matched leg onto retention, so nothing judges survival by a threshold.) The
one existing balance test (`test_probing_cascade_units.py`) is guarded by `if kept.size:` and so
cannot fail on an empty result — the ungated version is here.
"""

import numpy as np
import pytest

from galaxy_jepa.probing.logistic import Embeddings
from galaxy_jepa.probing.matching import (
    matched_auc,
    stratified_match,
)


def _emb(x: np.ndarray, y: np.ndarray) -> Embeddings:
    return Embeddings(x, y, np.zeros(len(y)))


def _separable(
    n: int = 600, seed: int = 0
) -> tuple[Embeddings, Embeddings, np.ndarray, np.ndarray]:
    """A feature the probe can read, with a nuisance that is *independent* of the label.

    Matching on an independent nuisance must not destroy the signal — that is the control for the
    tests below, where matching on a *predictive* nuisance should.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    x = np.column_stack([y + rng.normal(0, 0.5, n), rng.normal(0, 1, n)])
    nuisance = rng.normal(0, 1, n)  # independent of y by construction
    return _emb(x, y), _emb(x, y), nuisance, nuisance


@pytest.mark.invariant
def test_the_matched_set_is_class_balanced_and_non_empty():
    """The balance property, asserted *ungated* — an empty result must fail, not pass vacuously.

    The existing cascade test wraps this in `if kept.size:`, so a regression that made matching
    return nothing would have been read as a pass. That is the failure mode this replaces.
    """
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, 2000)
    labels = (values > 0.5).astype(int)  # worst case: the nuisance perfectly predicts the label
    kept = stratified_match(values, labels, n_strata=5, seed=0)

    assert kept.size > 0, "matching returned nothing; the balance claim would be vacuous"
    counts = np.bincount(labels[kept], minlength=2)
    assert counts[0] == counts[1]


@pytest.mark.invariant
def test_non_finite_nuisance_rows_are_dropped_not_binned():
    """A NaN magnitude is an absent measurement; it must not land in the first quantile stratum."""
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 1, 500)
    values[:50] = np.nan
    labels = rng.integers(0, 2, 500)
    kept = stratified_match(values, labels, n_strata=5, seed=0)
    assert kept.size > 0
    assert np.all(np.isfinite(values[kept]))


@pytest.mark.invariant
def test_matching_on_an_independent_nuisance_keeps_the_signal():
    """The control. If this fails, a collapse elsewhere says nothing about confounding."""
    train, test, m_tr, m_te = _separable()
    auc = matched_auc(train, test, m_tr, m_te, n_strata=5, seed=0)
    assert auc > 0.8


@pytest.mark.invariant
def test_a_degenerate_match_returns_chance_rather_than_raising():
    """0.5 on an unmatched-away sample is a DOCUMENTED outcome, and an ambiguous one.

    `matched_auc` folds an empty or single-class matched set to chance instead of raising. That is
    deliberate, but it makes "the signal was entirely confound" and "matching ate the sample"
    return the identical number — which is precisely why `artifacts/o1_matched.py` records
    survivor counts and reads a low count as UNRESOLVED rather than as a collapse. Pinned here so
    the ambiguity is a known property of the primitive and not a surprise at the call site.
    """
    y = np.repeat([0, 1], 100)
    x = np.column_stack([y.astype(float), np.zeros(200)])
    # every positive above the split, every negative below: no stratum holds both classes
    nuisance = y.astype(float)
    assert stratified_match(nuisance, y, n_strata=2, seed=0).size == 0
    assert matched_auc(_emb(x, y), _emb(x, y), nuisance, nuisance, n_strata=2, seed=0) == 0.5


@pytest.mark.invariant
def test_train_and_test_are_matched_on_different_draws():
    """`matched_auc` splits at `seed` and `seed + 1`, so the two subsets are not the same draw.

    Sharing one draw would tie which train rows were kept to which test rows were kept. The offset
    is load-bearing and invisible at the call site, so it is pinned rather than trusted.
    """
    rng = np.random.default_rng(2)
    values = rng.uniform(0, 1, 4000)
    labels = rng.integers(0, 2, 4000)
    assert not np.array_equal(
        stratified_match(values, labels, n_strata=5, seed=0),
        stratified_match(values, labels, n_strata=5, seed=1),
    )


@pytest.mark.invariant
def test_matching_is_reproducible_from_its_seed():
    """A verdict that is not reproducible from (config, code, data, seed) is not a verdict."""
    rng = np.random.default_rng(3)
    values, labels = rng.uniform(0, 1, 1000), rng.integers(0, 2, 1000)
    first = stratified_match(values, labels, n_strata=5, seed=7)
    assert np.array_equal(first, stratified_match(values, labels, n_strata=5, seed=7))
