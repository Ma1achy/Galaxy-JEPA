"""The two-scheme feature experiment (D14) — schemes as configs on one harness.

Covers the three things that must not drift: the schemes' composition and their **per-scheme**
BY family size; the graded existence test staying an explicit unimplemented placeholder rather
than silently defaulting to AUC; and conditional-population probing behaving as a *comparison*
(two providers over the same rows) rather than a hard mask that deletes the off-population
galaxies. Pure — no encoder, no network.
"""

from __future__ import annotations

import pytest

from galaxy_jepa.data.metadata import GZ2_CONDITIONS, GZ2_TREE, vote_column
from galaxy_jepa.probing.extract import LabelProvider
from galaxy_jepa.probing.schemes import (
    GradedExistenceTestUndecided,
    derive_vote_count_min,
    eligible_ids,
    full_tree_scheme,
    get_scheme,
    reduced_scheme,
    scheme_names,
)

pytestmark = pytest.mark.invariant


def test_full_tree_covers_every_answer_once():
    scheme = full_tree_scheme()
    assert len(scheme) == sum(len(a) for a in GZ2_TREE.values()) == 37
    assert all(s.kind == "binary" for s in scheme.specs)


def test_by_family_size_is_per_scheme_not_a_constant():
    """Spec §Statistics (3): the family count comes from the active scheme."""
    assert full_tree_scheme().family_size() == 37
    reduced = reduced_scheme()
    assert reduced.family_size() == len(reduced.primary()) < 37
    # exploratory features are reported, but outside the primary family they would dilute
    assert reduced.family_size() + len(reduced.exploratory()) == len(reduced)


def test_reduced_collapses_graded_questions_to_one_axis_each():
    reduced = reduced_scheme()
    graded = {s.name for s in reduced.graded()}
    assert graded == {"bulge_prominence", "roundedness", "arms_winding", "arms_number"}
    for spec in reduced.graded():
        # the ordered answers are carried; no single fraction column is invented
        assert spec.graded_cols == tuple(
            vote_column(spec.question, a) for a in GZ2_TREE[spec.question]
        )
        assert spec.fraction_col is None


def test_graded_existence_test_refuses_to_default():
    """The AUC-vs-correlation question is open; a graded feature must raise, not pick one."""
    (spec,) = [s for s in reduced_scheme().specs if s.name == "arms_winding"]
    with pytest.raises(GradedExistenceTestUndecided):
        spec.require_testable()


def test_binary_features_keep_auc_regardless():
    (spec,) = [s for s in reduced_scheme().specs if s.name == "bar"]
    assert spec.require_testable() == vote_column("t03_bar", "a06_bar")


def test_scheme_one_has_no_graded_features_so_it_is_unblocked():
    """Order is full-first, so the open graded question blocks nothing now."""
    assert full_tree_scheme().graded() == ()


def test_unknown_scheme_fails_loudly():
    with pytest.raises(KeyError, match="unknown feature scheme"):
        get_scheme("smart")
    assert scheme_names() == ["full_tree", "reduced"]


def test_conditions_follow_the_gz2_tree():
    """Deep questions carry their whole upstream chain, not just the immediate parent."""
    scheme = full_tree_scheme().by_name
    assert scheme["t01_smooth_or_features_a01_smooth"].conditions == ()
    assert scheme["t10_arms_winding_a28_tight"].conditions == GZ2_CONDITIONS["t10_arms_winding"]
    assert len(scheme["t10_arms_winding_a28_tight"].conditions) == 3
    assert scheme["t09_bulge_shape_a26_boxy"].condition_columns() == (
        vote_column("t01_smooth_or_features", "a02_features_or_disk"),
        vote_column("t02_edgeon", "a04_yes"),
    )


# --- the two filters, which are different things ---------------------------------------------

_BAR = "t03_bar_a06_bar"
_FEATURED = vote_column("t01_smooth_or_features", "a02_features_or_disk")
_NOT_EDGEON = vote_column("t02_edgeon", "a05_no")


def _rows():
    return {
        # reaches t03: featured and not edge-on, plenty of votes
        1: {_FEATURED: 0.9, _NOT_EDGEON: 0.9, "t03_bar_a06_bar_count": 30},
        # off-population (smooth) but carries bar votes — the disagreement measurement
        2: {_FEATURED: 0.1, _NOT_EDGEON: 0.9, "t03_bar_a06_bar_count": 30},
        # reaches t03 but almost nobody was asked
        3: {_FEATURED: 0.9, _NOT_EDGEON: 0.9, "t03_bar_a06_bar_count": 2},
    }


def test_vote_count_floor_is_a_hard_filter():
    spec = full_tree_scheme().by_name[_BAR]
    kept = eligible_ids(_rows(), spec, [1, 2, 3], vote_count_min=21, conditional=False)
    assert kept == [1, 2]  # 3 drops: nobody was asked, so there is no measurement


def test_conditional_population_is_opt_in_not_the_default():
    """D14: the off-population galaxy stays in the full run — masking it would be circular."""
    spec = full_tree_scheme().by_name[_BAR]
    full = eligible_ids(_rows(), spec, [1, 2, 3], vote_count_min=21, conditional=False)
    conditional = eligible_ids(_rows(), spec, [1, 2, 3], vote_count_min=21, conditional=True)
    assert 2 in full and 2 not in conditional
    assert set(conditional) < set(full)


def test_consensus_gate_is_a_per_run_knob():
    spec = full_tree_scheme().by_name[_BAR]
    strict = eligible_ids(
        _rows(), spec, [1, 2], vote_count_min=0, conditional=True, consensus_gate=0.95
    )
    loose = eligible_ids(
        _rows(), spec, [1, 2], vote_count_min=0, conditional=True, consensus_gate=0.05
    )
    assert strict == [] and loose == [1, 2]


def test_label_provider_populations_share_everything_but_the_definition():
    scheme = full_tree_scheme()
    full = LabelProvider(_rows(), scheme=scheme, vote_count_min=21)
    cond = full.with_population("conditional")
    assert full.population == "full" and cond.population == "conditional"
    assert cond.feature_cols == full.feature_cols
    assert cond.vote_count_min == full.vote_count_min
    assert full.eligible(_BAR, [1, 2, 3]) == [1, 2]
    assert cond.eligible(_BAR, [1, 2, 3]) == [1]


def test_label_provider_rejects_an_unknown_population():
    with pytest.raises(ValueError, match="population must be"):
        LabelProvider(_rows(), vote_count_min=21, population="conditional-ish")


def test_without_a_scheme_eligibility_is_the_identity():
    """The plumbing path must be unchanged by the scheme machinery existing."""
    labels = LabelProvider(_rows(), vote_count_min=21, feature_cols={"bar": _BAR})
    assert labels.eligible("bar", [1, 2, 3]) == [1, 2, 3]


def test_vote_count_threshold_carries_v1s_method_not_a_magic_number():
    assert derive_vote_count_min([10, 10, 10]) == pytest.approx(10.0)
    assert derive_vote_count_min([0, 10]) == pytest.approx(5.0 + 2 * 5.0)
    with pytest.raises(ValueError, match="unknown vote-count threshold method"):
        derive_vote_count_min([1, 2], method="median")


def test_every_co_indexed_vector_matches_the_feature_rows():
    """The bug Smoke B found: eligibility shortens a feature's rows but not its label vectors.

    An unfiltered id list is *longer* than the embedding matrix it is paired with, which
    surfaces as an sklearn length error at best and as labels quietly attached to the wrong
    galaxies at worst. Every co-indexed vector must come from ``feature_ids``.
    """
    import numpy as np

    from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings, feature_ids

    ids = [1, 2, 3]
    matrix = EmbeddingMatrix(
        object_ids=np.asarray(ids, dtype=np.int64),
        x=np.zeros((3, 4), dtype=np.float64),
        encoder_name="stub",
    )
    labels = LabelProvider(_rows(), scheme=full_tree_scheme(), vote_count_min=21)
    kept = feature_ids(matrix, labels, _BAR, ids)
    assert kept == [1, 2] != ids  # the filter really did shorten it
    emb = feature_embeddings(matrix, labels, _BAR, ids)
    assert len(emb.y) == len(emb.fraction) == emb.x.shape[0] == len(kept)
    # a vector built over `kept` aligns; one built over the raw ids does not
    assert len(labels.nuisance_value("snr", kept)) == emb.x.shape[0]
    assert len(labels.nuisance_value("snr", ids)) != emb.x.shape[0]


# --- t09 bulge shape: one binary contrast, double-conditioned (D14) ---------------------------

_BOXY = vote_column("t09_bulge_shape", "a26_boxy")
_ROUNDED = vote_column("t09_bulge_shape", "a25_rounded")


def test_reduced_family_lands_in_the_specs_band():
    assert reduced_scheme().family_size() == 10  # spec states ~10-13


def test_bulge_shape_is_binary_not_a_graded_axis():
    """Rounded/boxy/no-bulge is a categorical contrast with an absence bolted on, not a scale.

    The four graded axes are all genuinely ordinal; treating bulge shape as one would impose an
    order that does not exist — the failure the graded framing exists to prevent.
    """
    spec = reduced_scheme().by_name["bulge_boxy"]
    assert spec.kind == "binary"
    assert spec.fraction_col == _BOXY
    assert spec.require_testable() == _BOXY  # keeps AUC
    assert "bulge_boxy" not in {g.name for g in reduced_scheme().graded()}


def test_bulge_shape_is_conditioned_on_edge_on_and_bulge_present():
    groups = reduced_scheme().by_name["bulge_boxy"].condition_groups()
    assert groups == (
        (vote_column("t01_smooth_or_features", "a02_features_or_disk"),),
        (vote_column("t02_edgeon", "a04_yes"),),
        (_ROUNDED, _BOXY),  # "a bulge is present" — summed, not a negated no-bulge gate
    )


def test_a_grouped_condition_sums_its_columns():
    """The bulge-present gate must clear on rounded+boxy jointly, not on either alone."""
    spec = reduced_scheme().by_name["bulge_boxy"]
    base = {
        vote_column("t01_smooth_or_features", "a02_features_or_disk"): 0.9,
        vote_column("t02_edgeon", "a04_yes"): 0.9,
        "t09_bulge_shape_a26_boxy_count": 30,
    }
    rows = {
        # split evenly across rounded/boxy: neither alone clears 0.5, together they do
        1: {**base, _ROUNDED: 0.3, _BOXY: 0.3},
        # a bulge is mostly absent: the group does not clear
        2: {**base, _ROUNDED: 0.1, _BOXY: 0.1},
    }
    kept = eligible_ids(rows, spec, [1, 2], vote_count_min=21, conditional=True, consensus_gate=0.5)
    assert kept == [1]


def test_tree_conditions_still_gate_singly():
    """Adding grouped conditions must not change how the plain tree chain behaves."""
    spec = full_tree_scheme().by_name[_BAR]
    assert spec.extra_conditions == ()
    assert spec.condition_groups() == ((_FEATURED,), (_NOT_EDGEON,))


# --- the nuisance validity mask (A3): a flagged radius must not reach its own control --------


def _radius_rows() -> dict[int, dict[str, object]]:
    """Four galaxies, one of them wider than the stamp."""
    return {
        1: {"petroRad_r": 6.4, "petrorad_suspect": 0},
        2: {"petroRad_r": 8.1, "petrorad_suspect": 0},
        3: {"petroRad_r": 12.0, "petrorad_suspect": 0},
        4: {"petroRad_r": 258.4, "petrorad_suspect": 1},
    }


def test_flagged_rows_are_dropped_from_their_own_nuisance_only():
    labels = LabelProvider(_radius_rows(), vote_count_min=21)
    ids = [1, 2, 3, 4]
    assert labels.nuisance_valid("size", ids).tolist() == [True, True, True, False]
    # every other nuisance keeps the galaxy — the flag invalidates one column, not the object
    for other in ("redshift", "magnitude", "snr", "psf"):
        assert labels.nuisance_valid(other, ids).all()


def test_a_corpus_without_the_flag_column_refuses_the_radius_control():
    """A skipped control must raise, not shrug (CLAUDE.md: fail loudly, never silently default)."""
    rows = {k: {"petroRad_r": v["petroRad_r"]} for k, v in _radius_rows().items()}
    labels = LabelProvider(rows, vote_count_min=21)
    with pytest.raises(KeyError, match="backfill_derived"):
        labels.nuisance_valid("size", [1, 2, 3, 4])
    assert labels.nuisance_valid("redshift", [1, 2, 3, 4]).all()


class TestTheReachDenominatorIsTheQuestionTotal:
    """The floor asks "did enough people answer this question", not "did enough pick this answer".

    This pins a defect that shipped. ``eligible_ids`` summed ``spec.count_col`` — the single
    answer's own count — for binary specs, while graded specs already summed every answer. A
    fraction is ``count_answer / total_question``, so the per-answer version filtered on the
    numerator and threw away the **well-defined zeros**: galaxies where the question was answered
    and nobody chose this answer, which is most of the negatives. On the real corpus at a floor of
    1 that discarded 72.6% of t09 boxy's eligible galaxies and 87.2% of t11 >4-arms, leaving probe
    sets of almost nothing but positives. It also disagreed with D8's own published reach table,
    which the fixed version reproduces exactly.
    """

    def test_a_well_defined_zero_survives_the_floor(self):
        """Nobody said "bar", but thirty people were asked — the fraction is 0.0 and it is real."""
        spec = full_tree_scheme().by_name[_BAR]
        rows = {
            1: {
                _FEATURED: 0.9,
                _NOT_EDGEON: 0.9,
                "t03_bar_a06_bar_count": 0,  # the numerator
                "t03_bar_a07_no_bar_count": 30,  # the rest of the denominator
            }
        }
        assert eligible_ids(rows, spec, [1], vote_count_min=1) == [1]

    def test_a_question_nobody_answered_is_excluded_at_the_defined_minimum(self):
        """0/0 is undefined, and GZ2 stores it as a literal 0.0 — the floor is what catches it."""
        spec = full_tree_scheme().by_name[_BAR]
        rows = {
            1: {
                _FEATURED: 0.9,
                _NOT_EDGEON: 0.9,
                "t03_bar_a06_bar_count": 0,
                "t03_bar_a07_no_bar_count": 0,
            }
        }
        assert eligible_ids(rows, spec, [1], vote_count_min=1) == []
        # ...and a floor of 0 would let it straight through, which is why the value is 1
        assert eligible_ids(rows, spec, [1], vote_count_min=0) == [1]

    def test_every_spec_on_a_question_shares_one_denominator(self):
        """Binary and graded specs must agree, or a question's reach depends on which answer."""
        scheme = full_tree_scheme()
        for question, answers in GZ2_TREE.items():
            expected = {vote_column(question, a, "count") for a in answers}
            for spec in scheme.specs:
                if spec.question == question:
                    assert set(spec.reach_count_cols()) == expected, spec.name

    def test_the_numerator_column_is_not_the_reach_filter(self):
        """`count_col` still names this answer's own count; it just is not what the floor reads."""
        spec = full_tree_scheme().by_name[_BAR]
        assert spec.count_col == "t03_bar_a06_bar_count"
        assert spec.count_col in spec.reach_count_cols()
        assert len(spec.reach_count_cols()) == len(GZ2_TREE["t03_bar"])
