"""Invariant tests for the cascade's pure decision logic (no sklearn/torch in the fast gate).

The ceiling / rung / matching-trigger / gate logic is the load-bearing control flow — the part
that must be a *deterministic function of the controls* (3A). These pin that logic directly,
without fitting a probe, so they run in the fast gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.core.gates import MetricGate
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.gates import build_gates
from galaxy_jepa.probing.matching import nuisance_competitive, stratified_match
from galaxy_jepa.probing.mlp import SweepRow, rung_from_sweep, selectivity_ceiling

pytestmark = pytest.mark.invariant


# --- the MLP bounded-capacity ladder verdict (2D) ----------------------------------------


def test_selectivity_ceiling_is_first_control_breakdown():
    rows = [SweepRow(16, 0.7, 0.50), SweepRow(32, 0.8, 0.55), SweepRow(64, 0.85, 0.72)]
    assert selectivity_ceiling(rows, null_threshold=0.70) == 64  # control first exceeds 0.70
    assert selectivity_ceiling(rows, null_threshold=0.99) is None  # control never breaks down


def test_rung_r3_only_below_the_ceiling():
    # decodable (real ≥ 0.65) at width 32, which is below the ceiling 64 → R3
    rows = [SweepRow(16, 0.55, 0.5), SweepRow(32, 0.75, 0.55), SweepRow(64, 0.9, 0.72)]
    assert rung_from_sweep(rows, ceiling=64, decode_threshold=0.65) == "R3"


def test_rung_r4_when_decoding_only_at_or_above_ceiling():
    # real only crosses the bar at width 64 = the ceiling, where the control decodes too → R4
    rows = [SweepRow(16, 0.55, 0.5), SweepRow(32, 0.60, 0.55), SweepRow(64, 0.9, 0.72)]
    assert rung_from_sweep(rows, ceiling=64, decode_threshold=0.65) == "R4"


# --- the nuisance trigger (3D-ii) --------------------------------------------------------


def test_nuisance_competitive_trigger():
    assert nuisance_competitive(0.80, 0.82, margin=0.0) is True  # nuisance ≥ morphology
    assert nuisance_competitive(0.80, 0.70, margin=0.0) is False
    assert nuisance_competitive(0.80, 0.75, margin=0.10) is True  # a margin fires earlier


def test_stratified_match_balances_classes():
    rng = np.random.default_rng(0)
    values = rng.uniform(size=200)
    labels = (values > 0.5).astype(int)  # nuisance perfectly predicts the label — worst case
    kept = stratified_match(values, labels, n_strata=5)
    if kept.size:  # within the matched set the classes are balanced, so the nuisance can't separate
        assert int(np.sum(labels[kept] == 1)) == int(np.sum(labels[kept] == 0))


# --- the gate tree is a deterministic function of the metrics (3A keystone) ----------------


def _metrics(**over):
    base = {
        "auc": 0.9,
        "exceeds_null": 1.0,
        "selectivity": 0.3,
        "nuisance_cleared": 1.0,
        "entangled": 0.0,
        "nuisance_auc_max": 0.5,
    }
    base.update(over)
    return base


def test_rung_gate_is_deterministic():
    gates = build_gates(ProbingConfig())
    from galaxy_jepa.core.gates import all as gate_all

    tree = gate_all(*gates.rung_inputs())
    a = tree.evaluate(_metrics())
    b = tree.evaluate(_metrics())
    assert a.passed == b.passed and a.render() == b.render()  # same metrics → identical verdict


def test_control_cries_wolf_blocks_existence():
    # a probe can predict anything: high AUC must NOT pass existence if it didn't beat the null
    gates = build_gates(ProbingConfig())
    passed_real = gates.existence.evaluate(_metrics(exceeds_null=1.0)).passed
    passed_null = gates.existence.evaluate(_metrics(exceeds_null=0.0)).passed
    assert passed_real is True and passed_null is False


def test_gate_raises_on_a_metric_the_run_never_produced():
    # a structural bug (missing metric) is loud, unlike a failed bar (a finding)
    with pytest.raises(KeyError):
        MetricGate("auc", gte=0.5).evaluate({"selectivity": 0.3})
