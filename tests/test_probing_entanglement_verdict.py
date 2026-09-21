"""Invariants for 2A's localisation and its pre-registered verdict function.

Three gaps this pins shut, all found in survey before Brief P: the eigen*vectors* — the measure
that says *which* features collapse together — were never computed; the logistic-vs-CAV
cross-check existed but was never called from the ladder; and the mapping from
(eigen, cosine, CAV, conditional) to a verdict did not exist at all, so the ladder decided
entanglement on the matched cross-check alone, which is half the evidence the design specifies.

The discipline being enforced is that the mapping is FIXED BEFORE RESULTS. Many measures with
non-overlapping blind spots are a way to triangulate; a licence to choose a different rule per
feature would make the apparatus unfalsifiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing.entanglement import (
    CLEAN,
    INCONCLUSIVE,
    REPRESENTATIONAL,
    WORLD_CORRELATION,
    adjudicate_pair,
    component_loadings,
    gram_eigenvectors,
)

pytestmark = pytest.mark.invariant


def _pair(**kw):
    base = dict(
        cosine=0.8,
        mp_significant=True,
        cav_disagreement={"a": 0.4, "b": 0.4},
        survived_matching=True,
    )
    return adjudicate_pair("a", "b", **{**base, **kw})


def test_four_agreeing_measures_give_representational_entanglement():
    assert _pair().verdict == REPRESENTATIONAL


def test_vanishing_under_matching_is_world_correlation_not_entanglement():
    """D13's bar+arms shape. A clean finding about the sky, not a mark against the feature."""
    assert _pair(survived_matching=False).verdict == WORLD_CORRELATION


def test_the_mp_null_is_a_gate_input():
    """Surviving matching is NOT sufficient — the spectrum must also clear the MP null."""
    assert _pair(mp_significant=False).verdict != REPRESENTATIONAL


def test_everything_quiet_is_clean():
    v = _pair(cosine=0.05, mp_significant=False, cav_disagreement={"a": 0.0, "b": 0.0})
    assert v.verdict == CLEAN


def test_disagreeing_measures_are_reported_not_averaged():
    """High cosine but a spectrum inside the MP null: the measures do not converge."""
    assert _pair(mp_significant=False, cosine=0.9).verdict == INCONCLUSIVE


def test_an_unrun_conditional_check_is_never_read_as_a_result():
    """A degenerate match confirms neither survival nor collapse."""
    v = _pair(survived_matching=None)
    assert v.verdict not in (REPRESENTATIONAL, WORLD_CORRELATION)


def test_the_verdict_keeps_the_inputs_that_produced_it():
    v = _pair()
    assert (v.a, v.b) == ("a", "b") and v.mp_significant and v.cav_disagree and v.reason


def test_eigenvectors_localise_which_features_share_an_axis():
    """Two identical directions plus one orthogonal: the top component must load on the pair."""
    w = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    evals, evecs = gram_eigenvectors(w)
    assert evals[0] >= evals[1] >= evals[2]  # descending
    top = dict(component_loadings(["bar", "arms", "other"], evecs, component=0, top=2))
    assert set(top) == {"bar", "arms"}
    assert abs(top["bar"]) == pytest.approx(abs(top["arms"]), abs=1e-6)


def test_component_loadings_keep_the_sign():
    """Two features loading with OPPOSITE sign on a shared axis is a different statement."""
    w = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _evals, evecs = gram_eigenvectors(w)
    loads = component_loadings(["a", "b"], evecs, component=0, top=2)
    assert loads[0][1] * loads[1][1] < 0
