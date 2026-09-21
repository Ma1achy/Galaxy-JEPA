"""Invariants for the probe-trajectory stopping rule (`core.stopping`).

Written because the rule shipped with a signed comparison — ``d < FLAT_DELTA`` — so a DECLINE
satisfied it and Brief O2's turning-over curve stopped under the label ``FLAT``. The stop decision
was right; the label said the opposite of what happened. These pin the three outcomes apart, and
in particular that a declining curve can never be called flat.
"""

from __future__ import annotations

import pytest

from galaxy_jepa.core.stopping import (
    DECLINING,
    FLAT,
    FLAT_DELTA,
    INSUFFICIENT,
    RISING,
    UNSETTLED,
    read_curve,
)


def _curve(*points: tuple[float, float]) -> list[dict]:
    return [{"epoch": e, "auc": a} for e, a in points]


# M's real trajectory: rises and flattens. O2's: rises, peaks at 2 epochs, turns over.
M_CURVE = _curve((1.0, 0.9631), (2.0, 0.9642), (4.0, 0.9646))
O2_CURVE = _curve((1.0, 0.9663), (2.0, 0.9678), (4.0, 0.9609))


@pytest.mark.invariant
def test_a_declining_curve_is_never_labelled_flat() -> None:
    """The defect, pinned. O2 fell 0.9678 -> 0.9609 and was reported FLAT."""
    v = read_curve(O2_CURVE)
    assert v.label == DECLINING
    assert v.label != FLAT
    assert "FLAT" not in v.reason


@pytest.mark.invariant
def test_a_declining_curve_still_stops_the_run() -> None:
    """DECLINING is a stop condition like FLAT — relabelling must not turn the stop off."""
    assert read_curve(O2_CURVE).stop is True


@pytest.mark.invariant
def test_a_genuinely_flat_curve_is_still_flat() -> None:
    """M's verdict is unaffected by the fix: +0.0011 then +0.0004 is flat in both directions."""
    v = read_curve(M_CURVE)
    assert v.label == FLAT
    assert v.stop is True


@pytest.mark.invariant
def test_a_rising_curve_does_not_stop() -> None:
    v = read_curve(_curve((1.0, 0.90), (2.0, 0.92), (4.0, 0.95)))
    assert v.label == RISING
    assert v.stop is False


@pytest.mark.invariant
@pytest.mark.parametrize("last", [-FLAT_DELTA * 1.01, -0.05])
def test_any_decline_past_the_band_stops_as_declining(last: float) -> None:
    """Whatever the earlier interval did, a decline in the latest one is a decline."""
    v = read_curve(_curve((1.0, 0.90), (2.0, 0.95), (4.0, 0.95 + last)))
    assert v.label == DECLINING and v.stop is True


@pytest.mark.invariant
def test_a_decline_inside_the_band_is_flat_not_declining() -> None:
    """The band is symmetric: a wobble smaller than the threshold is noise, not a turn."""
    v = read_curve(_curve((1.0, 0.9600), (2.0, 0.9610), (4.0, 0.9605)))
    assert v.label == FLAT and v.stop is True


@pytest.mark.invariant
def test_the_rule_needs_three_points() -> None:
    v = read_curve(_curve((1.0, 0.90), (2.0, 0.92)))
    assert v.label == INSUFFICIENT and v.stop is False


@pytest.mark.invariant
def test_it_may_stop_a_run_early_but_never_extend_one() -> None:
    """Every outcome either stops or continues; none of them asks for more than the budget."""
    for curve in (M_CURVE, O2_CURVE, _curve((1.0, 0.90), (2.0, 0.92), (4.0, 0.95))):
        assert isinstance(read_curve(curve).stop, bool)


@pytest.mark.invariant
def test_still_improving_inside_the_band_continues() -> None:
    """Both deltas inside the band but accelerating is not yet a plateau."""
    v = read_curve(_curve((1.0, 0.9600), (2.0, 0.9601), (4.0, 0.9619)))
    assert v.label == UNSETTLED and v.stop is False
