"""Invariant + integration tests for the split orchestrator (docs/spec/splits.md).

The orchestrator is where the guards get *used*, so these protect the same guarantees the
guard unit tests do, but on the composed operations:

* ``resolve_corpora`` never lets a probe galaxy survive into pretraining (the dedup guard,
  exercised on the composed path) — and the monitor slice it feeds is disjoint from probe;
* ``assign_three_way`` is deterministic from the seed, disjoint+exhaustive, and hits the
  ratios in the large-n limit;
* the persisted plan round-trips and is stamped.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from galaxy_jepa.data.orchestrate import (
    assign_three_way,
    resolve_corpora,
    split_pretrain,
    write_split_plan,
)
from galaxy_jepa.data.splits import assert_no_cross_corpus_leak

_OBJIDS = st.integers(min_value=1, max_value=2**63 - 1)
_ID_SETS = st.sets(_OBJIDS, max_size=80)


@pytest.mark.invariant
@given(pretrain=_ID_SETS, probe=_ID_SETS)
def test_resolve_corpora_makes_leak_impossible(pretrain, probe):
    deduped = resolve_corpora(pretrain, probe)
    assert_no_cross_corpus_leak(deduped, probe)  # must not raise
    assert deduped.isdisjoint(probe)
    # collision resolution sends shared galaxies to probing; nothing else is dropped.
    assert deduped == frozenset(pretrain) - frozenset(probe)


@pytest.mark.invariant
@given(
    pretrain=st.sets(_OBJIDS, max_size=80),
    probe=st.sets(_OBJIDS, max_size=80),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_monitor_slice_is_disjoint_from_probe(pretrain, probe, seed):
    deduped = resolve_corpora(pretrain, probe)
    split = split_pretrain(deduped, seed=seed, monitor_frac=0.25)
    assert split.monitor.isdisjoint(probe)
    assert split.train.isdisjoint(probe)
    assert split.train | split.monitor == deduped  # exhaustive over the deduped corpus


@pytest.mark.invariant
@given(probe=_ID_SETS, seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_three_way_is_disjoint_exhaustive_and_deterministic(probe, seed):
    a = assign_three_way(probe, seed=seed)
    b = assign_three_way(probe, seed=seed)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)  # seed-reproducible
    assert a.total == len(frozenset(probe))  # exhaustive
    assert a.train.isdisjoint(a.val) and a.train.isdisjoint(a.test) and a.val.isdisjoint(a.test)
    assert (a.train | a.val | a.test) == frozenset(probe)


@settings(max_examples=20)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_three_way_hits_ratios_at_scale(seed):
    # Large n: the uniform hashed coordinate should land near 70/15/15 (loose tolerance).
    ids = range(1, 6001)
    split = assign_three_way(ids, seed=seed)
    n = split.total
    assert abs(len(split.train) / n - 0.70) < 0.05
    assert abs(len(split.val) / n - 0.15) < 0.05
    assert abs(len(split.test) / n - 0.15) < 0.05


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_three_way([1, 2, 3], seed=0, ratios=(0.5, 0.3, 0.1))


def test_write_split_plan_round_trips(tmp_path):
    probe = assign_three_way(range(1, 101), seed=7)
    pretrain = split_pretrain(resolve_corpora(range(200, 401), range(1, 101)), seed=7)
    out = tmp_path / "plan.json"
    ratios = (0.70, 0.15, 0.15)
    snapshot = write_split_plan(out, probe=probe, pretrain=pretrain, seed=7, ratios=ratios)
    plan = json.loads(out.read_text())
    assert plan["data_snapshot"] == snapshot
    assert plan["counts"]["probe-train"] == len(probe.train)
    # the persisted id-sets reconstruct the in-memory split exactly
    assert frozenset(plan["splits"]["probe-test"]) == probe.test
    assert frozenset(plan["splits"]["pretrain-monitor"]) == pretrain.monitor
