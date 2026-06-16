"""Invariant tests for the split-policy structural guards (docs/spec/splits.md).

Tier: invariant / property (``docs/spec/testing.md`` §1.2) — **merge-blocking**. These
protect two methodology guarantees that must be *impossible* to violate, so they are
asserted over Hypothesis-generated inputs, not a hand-picked example:

* cross-corpus dedup — no probe galaxy survives into the pretraining corpus (D6);
* uncertainty-geometry firewall — the ambiguous middle never enters the axis-fit set.

Determinism of the assignment primitive is checked too: the same ``(objID, seed)`` always
lands in the same place, across processes, so a split is reproducible from the seed alone.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from galaxy_jepa.data.splits import (
    LeakError,
    assert_no_cross_corpus_leak,
    assert_uncertainty_firewall,
    assignment_unit,
    exclude_probe_from_pretrain,
    partition_uncertainty,
    to_object_ids,
)

# SDSS objID is a 64-bit integer; draw from a realistic-width space.
_OBJIDS = st.integers(min_value=1, max_value=2**63 - 1)
_ID_SETS = st.sets(_OBJIDS, max_size=50)
_VOTES = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@pytest.mark.invariant
@given(pretrain=_ID_SETS, probe=_ID_SETS)
def test_cross_corpus_dedup_makes_leak_impossible(pretrain, probe):
    # After resolution, the post-condition holds for ANY pair of pulls — the structural
    # guarantee: a probe galaxy cannot survive into the pretraining corpus.
    resolved = exclude_probe_from_pretrain(pretrain, probe)
    assert_no_cross_corpus_leak(resolved, probe)  # must not raise
    assert resolved == to_object_ids(pretrain) - to_object_ids(probe)
    # collision resolution sends shared galaxies to probing: none remain in pretrain.
    assert resolved.isdisjoint(to_object_ids(probe))


@pytest.mark.invariant
@given(shared=st.sets(_OBJIDS, min_size=1, max_size=20), extra=_ID_SETS)
def test_leak_is_detected_when_present(shared, extra):
    # A planted overlap is always caught — the guard cannot pass a real leak through.
    pretrain = shared | extra
    probe = shared
    with pytest.raises(LeakError):
        assert_no_cross_corpus_leak(pretrain, probe)


@pytest.mark.invariant
@given(extremes=st.lists(st.sampled_from([0.0, 0.05, 0.2, 0.8, 0.95, 1.0]), max_size=30))
def test_firewall_passes_on_extremes_only(extremes):
    # Boundary-inclusive extremes (v<=0.2 or v>=0.8) never trip the firewall.
    assert_uncertainty_firewall(extremes)  # must not raise


@pytest.mark.invariant
@given(
    middle=st.lists(
        st.floats(min_value=0.2001, max_value=0.7999), min_size=1, max_size=20
    ),
    extremes=st.lists(st.sampled_from([0.0, 1.0]), max_size=10),
)
def test_firewall_rejects_any_ambiguous_value_in_the_fit_set(middle, extremes):
    # A single ambiguous value anywhere in the fit set is a circularity leak → raise.
    with pytest.raises(LeakError):
        assert_uncertainty_firewall(middle + extremes)


@pytest.mark.invariant
@given(votes=st.dictionaries(_OBJIDS, _VOTES, max_size=40))
def test_partition_is_a_clean_firewall_split(votes):
    # The partition the firewall protects: fit-set is extremes-only, the two are disjoint
    # and exhaustive, and the fit set always survives its own firewall.
    fit, middle = partition_uncertainty(votes)
    assert fit.isdisjoint(middle)
    assert len(fit) + len(middle) == len(votes)
    fit_votes = [v for oid, v in votes.items() if (next(iter(to_object_ids([oid])))) in fit]
    assert_uncertainty_firewall(fit_votes)  # by construction, must not raise


@pytest.mark.invariant
@given(oid=_OBJIDS, seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_assignment_is_deterministic_and_bounded(oid, seed):
    a = assignment_unit(oid, seed)
    b = assignment_unit(oid, seed)
    assert a == b  # stable across calls (and processes — SHA-256, not salted hash())
    assert 0.0 <= a < 1.0
    # A different salt namespaces an independent partition: same id+seed, different coord.
    assert assignment_unit(oid, seed, salt="probe") != a or assignment_unit(
        oid, seed, salt="pretrain-monitor"
    ) != a


def test_object_id_coercion_rejects_ambiguous_types():
    # bool is an int subclass but never a valid objID; non-integral floats are a mistake.
    assert to_object_ids(["123", 123, 123.0]) == frozenset({123})
    with pytest.raises(TypeError):
        to_object_ids([True])
    with pytest.raises(ValueError):
        to_object_ids([1.5])
