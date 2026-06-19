"""Invariant: nothing debiased (or clean-sample-flagged) ever enters the GZ2 pull or the label.

The debiased GZ2 fractions apply the Willett+2013 redshift correction, which injects z into
the target variable — disqualified for v2's uncertainty geometry and the z-nuisance control
(``CLAUDE.md`` token/parity discipline; ``docs/spec/data.md``). Getting this wrong silently
corrupts the headline result, so the no-debiasing guarantee is pinned structurally here: the
generated SQL, the vote-column list, the slice's label source, and the probing defaults must
carry **zero** ``_debiased`` / ``_flag`` columns, and the generator must *raise* if asked for
one. These are property checks on strings — no network, no GPU.
"""

from __future__ import annotations

import pytest

from galaxy_jepa.data import metadata as m

pytestmark = pytest.mark.invariant

#: Variant suffixes that must never reach the pull or any stored/probed column.
_BANNED = ("debiased", "flag")


def _has_banned(name: str) -> bool:
    return any(b in name for b in _BANNED)


def test_gz2_vote_columns_carry_no_debiased_or_flag():
    cols = m.gz2_vote_columns()
    assert cols, "expected a non-empty GZ2 vote-column list"
    offenders = [c for c in cols if _has_banned(c)]
    assert offenders == [], f"debiased/flag columns leaked into the vote list: {offenders}"
    # every column is exactly one of the three raw variants — nothing else slips through
    assert all(c.endswith(("_fraction", "_weighted_fraction", "_count")) for c in cols)


def test_generated_probe_sql_has_zero_debiased_or_flag_substring():
    sql = m.probe_sql(200)
    assert sql.count("debiased") == 0, "the probe SQL selects a debiased column"
    assert "_flag" not in sql, "the probe SQL selects a clean-sample flag column"


def test_other_sql_templates_carry_no_debiased():
    # the pretraining + join-check queries never touch GZ2 votes; assert it so a future edit
    # that adds one is caught at the gate, not in a pull.
    assert m.pretrain_sql(10).count("debiased") == 0
    assert m.join_check_sql(10).count("debiased") == 0


def test_generator_raises_if_a_disqualified_variant_is_requested(monkeypatch):
    # the structural backstop: if someone adds 'debiased' (or 'flag') to the variant set,
    # generation must raise loudly rather than silently pull it.
    monkeypatch.setattr(m, "GZ2_VOTE_VARIANTS", ("fraction", "debiased"))
    with pytest.raises(ValueError, match="debiased"):
        m.gz2_vote_columns()


def test_slice_label_source_is_the_raw_fraction():
    # the historical pilot AUC was measured on the *debiased* a02 column; the live label
    # source must now be the RAW fraction (the repoint), with no debiasing left in it.
    assert not _has_banned(m.FEATURED_FRACTION_COL)
    assert m.FEATURED_FRACTION_COL.endswith("_fraction")
    assert not m.FEATURED_FRACTION_COL.endswith("_weighted_fraction")  # the headline raw vote


def test_probing_defaults_point_at_no_debiased_column():
    from galaxy_jepa.probing.extract import DEFAULT_FEATURE_COLS, DEFAULT_NUISANCE_COLS

    for col in (*DEFAULT_FEATURE_COLS.values(), *DEFAULT_NUISANCE_COLS.values()):
        assert not _has_banned(col), f"a probing default points at a banned column: {col!r}"
