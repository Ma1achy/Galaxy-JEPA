"""Invariant tests for the null-calibrated existence layer + multiplicity (design 3B/2B).

Pure numpy — no sklearn/torch — so these run in the fast gate. They pin the *structure* of
the flagged decisions: the correction is real and monotone, the p-value is bounded and ordered,
and the method/family-size are genuine parameters (the stats grounding sets a value, never
rebuilds).
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing.nulls import existence_pvalue, family_significant

pytestmark = pytest.mark.invariant


def test_existence_pvalue_is_bounded_and_ordered():
    null = np.array([0.5, 0.52, 0.48, 0.55, 0.5])
    p_high = existence_pvalue(0.95, null)  # clearly beats the null
    p_low = existence_pvalue(0.50, null)  # at the null
    assert 0.0 < p_high <= p_low <= 1.0
    assert p_high < 0.2  # the add-one estimator never returns exactly 0


def test_existence_pvalue_empty_null_is_uninformative():
    assert existence_pvalue(0.9, np.array([])) == 1.0


def test_bonferroni_divides_the_threshold():
    pv = {"a": 0.001, "b": 0.04, "c": 0.5}
    out = family_significant(pv, alpha=0.05, method="bonferroni")  # bar = 0.05/3 ≈ 0.0167
    assert out == {"a": True, "b": False, "c": False}


def test_more_tests_is_a_stricter_bar():
    pv = {"a": 0.01}
    # the family-size override (the 2C build flag) makes the bar stricter without a rebuild
    assert family_significant(pv, alpha=0.05, method="bonferroni", n_tests=1)["a"] is True
    assert family_significant(pv, alpha=0.05, method="bonferroni", n_tests=100)["a"] is False


def test_benjamini_yekutieli_is_a_callable_alternative():
    # the flagged decision is *which* method; both are implemented, swappable by a parameter
    pv = {"a": 0.001, "b": 0.04, "c": 0.5}
    by = family_significant(pv, alpha=0.05, method="benjamini_yekutieli")
    assert by["a"] is True and by["c"] is False
    # a strongly-significant feature passes under either correction
    assert family_significant(pv, method="bonferroni")["a"] is by["a"]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        family_significant({"a": 0.01}, method="holm")
