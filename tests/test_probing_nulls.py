"""Invariant tests for the null-calibrated existence layer + multiplicity (design 3B/2B).

Pure numpy — no sklearn/torch — so these run in the fast gate. They pin the *structure* of
the flagged decisions: the correction is real and monotone, the p-value is bounded and ordered,
and the method/family-size are genuine parameters (the stats grounding sets a value, never
rebuilds).
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing import nulls as nz
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


# --- the grounded statistical decisions (spec §Statistics) -------------------------------


class TestGroundedStatistics:
    """The five decisions, as behaviour rather than as comments."""

    def test_existence_null_is_the_strongest_control_per_draw(self):
        """3C: a real feature must beat the *strongest* null, not an average of them."""
        import numpy as np

        from galaxy_jepa.probing.controls import FeatureControls
        from galaxy_jepa.probing.nulls import existence_null_samples

        fc = FeatureControls(
            feature="f",
            real_auc=0.9,
            shuffled_nulls=np.array([0.50, 0.60]),
            random_embedding_nulls=np.array([0.55, 0.52]),
            noise_encoder_auc=0.52,
            untrained_encoder_auc=0.70,
            sky_noise_auc=0.51,
            selectivity=0.3,
            nuisance_aucs={},
        )
        null = existence_null_samples(fc)
        # every draw is dominated by the strongest control (untrained, 0.70)
        assert null.tolist() == [0.70, 0.70]
        assert null.min() >= 0.70

    def test_the_sky_noise_control_is_a_diagnostic_and_never_sets_the_bar(self):
        """D19. 3C-5 is not chance-calibrated, so it cannot be part of the existence null.

        The regression this pins is not hypothetical: measured on the 50,000-step encoder
        (J4) it sat at 0.8355-0.8416 on every feature and failed *every* one of them,
        featured-ness included — an all-R3/R4 catalogue that reads like a scientific null and
        is nothing of the kind. Four of the five controls break something and are therefore
        chance-calibrated; 3C-5 keeps images, encoder and probe real and swaps in a different
        real label, so its AUC measures image-quality content. It stays on `FeatureControls`
        and in the nuisance panel; it does not enter the bar.
        """
        import numpy as np

        from galaxy_jepa.probing.controls import FeatureControls
        from galaxy_jepa.probing.nulls import existence_null_samples, existence_verdicts

        def fc(sky: float) -> FeatureControls:
            return FeatureControls(
                feature="f",
                real_auc=0.90,
                shuffled_nulls=np.full(400, 0.50),
                random_embedding_nulls=np.full(400, 0.52),
                noise_encoder_auc=0.51,
                untrained_encoder_auc=0.60,
                sky_noise_auc=sky,
                selectivity=0.4,
                nuisance_aucs={"snr": sky},
            )

        # the J4 value, which under the old bar dominated every draw and sank the feature
        assert existence_null_samples(fc(0.8373)).max() == 0.60
        # moving it cannot move the null, the p-value, or the verdict
        assert np.array_equal(existence_null_samples(fc(0.51)), existence_null_samples(fc(0.99)))
        (low,) = existence_verdicts({"f": fc(0.51)}, n_tests=1, effect_floor=0.65).values()
        (high,) = existence_verdicts({"f": fc(0.99)}, n_tests=1, effect_floor=0.65).values()
        assert (low.pvalue, low.exceeds_null) == (high.pvalue, high.exceeds_null)
        assert high.exceeds_null is True
        # but it is still measured and still carried — removed from the bar, not from the record
        assert fc(0.8373).sky_noise_auc == 0.8373

    def test_multiplicity_defaults_to_benjamini_yekutieli(self):
        from galaxy_jepa.probing.config import ProbingConfig

        assert ProbingConfig(vote_count_min=21).multiplicity == "benjamini_yekutieli"

    def test_permutation_floor_is_enforced_not_documented(self):
        from galaxy_jepa.probing.config import ProbingConfig

        assert ProbingConfig(vote_count_min=21).n_perm >= 10_000
        assert ProbingConfig(vote_count_min=21).permutation_method == "two_sided"
        with pytest.raises(ValueError, match="10,000"):
            ProbingConfig(n_perm=999, vote_count_min=21)
        # a declared deviation is allowed, and is stamped
        assert (
            ProbingConfig(
                n_perm=999, vote_count_min=21, escape_hatches=("reduced_permutations",)
            ).n_perm
            == 999
        )

    def test_mp_edge_must_use_the_actual_matrix_shape(self):
        """Decision (5): a nominal k moves the edge and silently changes the verdict."""
        import numpy as np

        from galaxy_jepa.probing.entanglement import mp_significant

        ev = np.array([3.0, 0.5, 0.5, 0.5])
        assert mp_significant(ev, n_directions=4, n_dims=16).significant is True
        with pytest.raises(ValueError, match="actual matrix shape"):
            mp_significant(ev, n_directions=37, n_dims=16)

    def test_tracy_widom_raises_rather_than_defaulting(self):
        import numpy as np

        from galaxy_jepa.probing.entanglement import mp_significant

        with pytest.raises(NotImplementedError):
            mp_significant(np.array([1.0]), n_directions=1, n_dims=8, method="tracy_widom")

    def test_null_resolution_guard_blocks_an_unattainable_bar(self):
        """A too-coarse null yields an all-fail catalogue that looks like a scientific null."""
        from galaxy_jepa.probing.nulls import assert_null_resolution, attainable_min_pvalue

        assert attainable_min_pvalue(50) == pytest.approx(1 / 51)
        with pytest.raises(ValueError, match="null resolution too coarse"):
            assert_null_resolution(50, alpha=0.05, method="benjamini_yekutieli", n_tests=37)
        assert_null_resolution(5000, alpha=0.05, method="benjamini_yekutieli", n_tests=37)

    def test_effect_floor_is_a_second_gate_not_a_replacement(self):
        """Decision (2): the floor cannot rescue a non-significant feature."""
        import numpy as np

        from galaxy_jepa.probing.controls import FeatureControls
        from galaxy_jepa.probing.nulls import existence_verdicts

        strong_but_null = FeatureControls(
            feature="f",
            real_auc=0.99,  # way over any floor...
            shuffled_nulls=np.full(500, 0.995),  # ...but the null is higher still
            random_embedding_nulls=np.full(500, 0.995),
            noise_encoder_auc=0.5,
            untrained_encoder_auc=0.5,
            sky_noise_auc=0.5,
            selectivity=0.0,
            nuisance_aucs={},
        )
        (v,) = existence_verdicts({"f": strong_but_null}, n_tests=1, effect_floor=0.65).values()
        assert v.exceeds_null is False
        assert v.clean is False


_TINY_VIT = {
    "img_size": 16,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 8,
    "depth": 2,
    "heads": 2,
    "mlp_ratio": 4.0,
}


def test_untrained_encoder_null_is_reproducible_from_the_seed():
    """The strongest chance-calibrated null sets the bar; an unseeded one makes the bar move.

    Unseeded, this manifests as a feature flipping R1↔R3 across reruns of the *identical*
    config — a verdict that is not reproducible from (config_hash, code_sha, data_snapshot,
    seed) is not stamped provenance at all.
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    from galaxy_jepa.probing.controls import untrained_encoder_matrix

    class _Tiny(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, i: int) -> dict:
            g = torch.Generator().manual_seed(i)
            return {"image": torch.randn(3, 16, 16, generator=g), "object_id": i}

    config = _TINY_VIT
    first = untrained_encoder_matrix(config, _Tiny(), seed=0)
    again = untrained_encoder_matrix(config, _Tiny(), seed=0)
    other = untrained_encoder_matrix(config, _Tiny(), seed=1)
    assert np.array_equal(first.x, again.x)
    assert not np.array_equal(first.x, other.x)


def test_the_null_is_order_independent():
    """Seeding must be *scoped* to the construction, not leaked into the surrounding run.

    If the seed leaked, building a seed=1 control first would change the seed=0 control built
    after it — so the existence bar would depend on call order, not on the run stamp. (The
    caller's global RNG is separately perturbed by ``DataLoader``'s own base-seed draw; that is
    pre-existing and not what this pins.)
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    from galaxy_jepa.probing.controls import untrained_encoder_matrix

    class _Tiny(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, i: int) -> dict:
            g = torch.Generator().manual_seed(i)
            return {"image": torch.randn(3, 16, 16, generator=g), "object_id": i}

    alone = untrained_encoder_matrix(_TINY_VIT, _Tiny(), seed=0)
    untrained_encoder_matrix(_TINY_VIT, _Tiny(), seed=1)
    after = untrained_encoder_matrix(_TINY_VIT, _Tiny(), seed=0)
    assert np.array_equal(alone.x, after.x)


def test_add_one_estimator_counts_ties_into_the_tail():
    """DECIDED (spec §Statistics (4)): a permutation p is never zero, and ties are conservative.

    The floor being 1/(n+1) rather than 1/n *is* the add-one convention; ties counted against the
    observed value push p up, never down (Phipson & Smyth). Both are pinned here because the
    null-resolution budget is derived from the 1/(n+1) floor.
    """
    import numpy as np

    from galaxy_jepa.probing.nulls import attainable_min_pvalue, existence_pvalue

    null = np.full(99, 0.5)
    assert existence_pvalue(0.9, null) == pytest.approx(1 / 100)  # never 0
    assert attainable_min_pvalue(99) == pytest.approx(1 / 100)
    # an exact tie counts against the real value
    assert existence_pvalue(0.5, null) == pytest.approx(100 / 100)


def test_the_permutation_p_shares_the_convention():
    # The claim is the add-one convention, but computing it needs Spearman, and scipy rides in
    # the `eval` extra — the fast gate installs `dev` only, deliberately, to stay ~2 min. The
    # scipy-free half of the same convention is pinned above on `existence_pvalue`, so the gate
    # still guards it when this skips.
    pytest.importorskip("scipy.stats")  # the exact module `spearman` imports

    from galaxy_jepa.probing.uncertainty import permutation_p

    x = np.arange(40, dtype=float)
    p = permutation_p(x, x.copy(), n_perm=200, seed=0)  # a perfect rank correlation
    assert p == pytest.approx(1 / 201)  # the floor, not zero


class TestNullBudget:
    """The budget is sized from Scheme 1 and applied to both — never derived per-scheme."""

    def test_required_draws_match_the_by_arithmetic(self):
        from galaxy_jepa.probing.nulls import family_bar, required_null_draws

        # BY rank-1 bar = alpha / (m * H_m); the empirical floor 1/(n+1) must clear it.
        assert family_bar(0.05, "benjamini_yekutieli", 37) == pytest.approx(3.216e-4, rel=1e-3)
        assert required_null_draws(alpha=0.05, method="benjamini_yekutieli", n_tests=37) == 3109
        assert required_null_draws(alpha=0.05, method="benjamini_yekutieli", n_tests=10) == 585

    def test_the_budget_is_sized_from_the_larger_family(self):
        """Scheme 2 needs fewer draws, but must not be *run* with fewer.

        If the two schemes disagree about a feature, the candidate explanations are power and
        expressibility. "The two runs had different null resolution" must not be on that list.
        """
        from galaxy_jepa.probing.nulls import (
            BUDGET_FAMILY_SIZE,
            assert_null_resolution,
            required_null_draws,
        )
        from galaxy_jepa.probing.schemes import full_tree_scheme, reduced_scheme

        assert BUDGET_FAMILY_SIZE == full_tree_scheme().family_size()
        assert BUDGET_FAMILY_SIZE > reduced_scheme().family_size()
        budget = required_null_draws(
            alpha=0.05, method="benjamini_yekutieli", n_tests=BUDGET_FAMILY_SIZE
        )
        # a budget sized from Scheme 1 is automatically sufficient for Scheme 2
        for family in (full_tree_scheme().family_size(), reduced_scheme().family_size()):
            assert_null_resolution(budget, alpha=0.05, method="benjamini_yekutieli", n_tests=family)

    def test_the_draw_count_is_a_config_value_not_a_function_of_the_scheme(self):
        """Nothing may compute n_null_draws from the active scheme's family size."""
        from galaxy_jepa.probing.config import ProbingConfig

        one = ProbingConfig(scheme_name="full_tree", n_null_draws=4000, vote_count_min=21)
        two = ProbingConfig(scheme_name="reduced", n_null_draws=4000, vote_count_min=21)
        assert one.n_null_draws == two.n_null_draws == 4000


# --- D23: the untrained-z existence construction -------------------------------------------
#
# `existence_null_samples` is a point mass (the untrained singleton floors every draw), so the
# add-one p-value returns only 1/(n+1) or 1.0 and BY has nothing calibrated to act on. These pin
# the replacement, and in particular that removing the 3,109-draw requirement did not remove a
# resolution requirement — it moved it onto K.


def _bar(mean: float, sd: float, k: int = 30, seed: int = 0) -> np.ndarray:
    """A synthetic untrained bank with a known mean and spread."""
    draws = np.random.default_rng(seed).standard_normal(k)
    draws = (draws - draws.mean()) / draws.std(ddof=1)  # exact mean/sd, so assertions are tight
    return mean + sd * draws


def test_the_z_pvalue_is_continuous_where_the_add_one_estimator_was_not():
    """The whole point: a p-value BY's rank-1 bar (3.216e-4 at family 37) can actually reach."""
    bar = _bar(0.5359, 0.010)
    strong = nz.untrained_z_pvalue(0.8845, 0.0019, bar)
    assert strong < nz.family_bar(0.05, "benjamini_yekutieli", 37)
    # the add-one estimator at the shipped 50 draws cannot get below 1/51, whatever the signal
    assert nz.attainable_min_pvalue(50) > nz.family_bar(0.05, "benjamini_yekutieli", 37)


def test_a_feature_at_its_bar_does_not_clear():
    bar = _bar(0.5359, 0.010)
    assert nz.untrained_z_pvalue(0.5359, 0.004, bar) == pytest.approx(0.5, abs=0.02)


def test_a_feature_below_its_bar_is_uninformative():
    bar = _bar(0.5359, 0.010)
    assert nz.untrained_z_pvalue(0.5000, 0.004, bar) > 0.9


def test_the_p_value_is_monotone_in_the_real_auc():
    bar = _bar(0.55, 0.010)
    ps = [nz.untrained_z_pvalue(a, 0.004, bar) for a in (0.56, 0.60, 0.65, 0.70)]
    assert ps == sorted(ps, reverse=True)


def test_a_wider_bar_makes_the_same_margin_less_significant():
    """The bar's seed spread is in the denominator — that is the point of the construction."""
    tight = nz.untrained_z_pvalue(0.60, 0.004, _bar(0.55, 0.002))
    wide = nz.untrained_z_pvalue(0.60, 0.004, _bar(0.55, 0.030))
    assert tight < wide


def test_a_wider_real_interval_makes_the_same_margin_less_significant():
    bar = _bar(0.55, 0.010)
    assert nz.untrained_z_pvalue(0.60, 0.002, bar) < nz.untrained_z_pvalue(0.60, 0.040, bar)


def test_student_t_is_used_not_the_normal():
    """df = K-1, so the tail must be heavier than the normal's at the same z."""
    from scipy.stats import norm

    bar = _bar(0.55, 0.010, k=30)
    # z = 3.5 exactly: margin 0.035 against sd 0.010 and se 0
    p = nz.untrained_z_pvalue(0.55 + 3.5 * 0.010, 0.0, bar)
    assert p > float(norm.sf(3.5))


def test_the_bank_resolution_gate_still_bites():
    """Removing the 3,109-draw floor moved the requirement onto K; it did not delete it."""
    with pytest.raises(ValueError, match="untrained bank too small"):
        nz.assert_untrained_bank_resolution(3)
    assert nz.assert_untrained_bank_resolution(nz.K_MIN) is None


def test_the_empirical_resolution_gate_still_bites_under_its_own_method():
    """The old gate must not stop biting just because a new method exists beside it."""
    with pytest.raises(ValueError, match="null resolution too coarse"):
        nz.assert_null_resolution(50, alpha=0.05, method="benjamini_yekutieli", n_tests=37)


def test_a_bank_too_small_to_estimate_spread_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="at least 2 untrained seeds"):
        nz.untrained_z_pvalue(0.9, 0.01, np.array([0.55]))


def test_a_scaleless_comparison_raises_rather_than_returning_zero():
    """Both uncertainties zero is a broken measurement, not a p-value of 0."""
    with pytest.raises(ValueError, match="no scale"):
        nz.untrained_z_pvalue(0.9, 0.0, np.array([0.55, 0.55, 0.55]))


def test_the_verdict_records_which_construction_made_it():
    assert nz.ExistenceVerdict("f", 0.9, 0.01, True, True).method == nz.EXISTENCE_EMPIRICAL
    assert nz.ExistenceVerdict("f", 0.9, 0.01, True, True, method=nz.EXISTENCE_UNTRAINED_Z).method
