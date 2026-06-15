"""Property tests for the Gate types (core/gates.py).

Pins: exactly-one-comparator, empty-composite rejection, nested composition, the
verdict tree renders, and the load-bearing rule that a failed bar is a *finding*
(``passed=False``) not an *error* (no exception) — run status is independent of the
verdict.
"""

import pytest

from galaxy_jepa.core import gates as gate
from galaxy_jepa.core.gates import GateResult, MetricGate


def test_metric_gate_requires_exactly_one_comparator():
    with pytest.raises(ValueError):
        MetricGate("auc")  # zero comparators
    with pytest.raises(ValueError):
        MetricGate("auc", gte=0.8, lte=0.9)  # two comparators


@pytest.mark.parametrize(
    ("kwargs", "value", "expected"),
    [
        ({"gt": 0.8}, 0.81, True),
        ({"gt": 0.8}, 0.80, False),
        ({"gte": 0.8}, 0.80, True),
        ({"lt": 0.65}, 0.60, True),
        ({"lte": 0.65}, 0.65, True),
        ({"lte": 0.65}, 0.66, False),
    ],
)
def test_metric_gate_comparators(kwargs, value, expected):
    result = MetricGate("m", **kwargs).evaluate({"m": value})
    assert result.passed is expected


def test_missing_metric_raises_loudly():
    # A referenced-but-absent metric is a bug, not a finding -> fail loud.
    with pytest.raises(KeyError):
        MetricGate("auc", gte=0.8).evaluate({"something_else": 1.0})


def test_empty_composites_rejected():
    with pytest.raises(ValueError):
        gate.all()
    with pytest.raises(ValueError):
        gate.any()


def test_nested_composition():
    tree = gate.all(
        MetricGate("auc", gte=0.80),
        gate.any(
            MetricGate("selectivity", gte=0.10),
            MetricGate("nuisance_auc_max", lte=0.65),
        ),
    )
    passing = tree.evaluate({"auc": 0.9, "selectivity": 0.0, "nuisance_auc_max": 0.5})
    assert passing.passed is True
    # all() fails when its first child fails, even though the any() branch passes
    failing = tree.evaluate({"auc": 0.5, "selectivity": 0.0, "nuisance_auc_max": 0.5})
    assert failing.passed is False


def test_verdict_is_separate_from_run_status():
    # Every gate failing must NOT raise: a feature failing its controls is a finding.
    tree = gate.all(MetricGate("auc", gte=0.80), MetricGate("selectivity", gte=0.10))
    result = tree.evaluate({"auc": 0.4, "selectivity": -0.2})
    assert isinstance(result, GateResult)
    assert result.passed is False
    assert all(not child.passed for child in result.children)


def test_render_shows_pass_and_fail():
    tree = gate.all(MetricGate("auc", gte=0.80), MetricGate("selectivity", gte=0.10))
    rendered = tree.evaluate({"auc": 0.9, "selectivity": -0.1}).render()
    assert "PASS" in rendered
    assert "FAIL" in rendered
    assert "auc" in rendered
