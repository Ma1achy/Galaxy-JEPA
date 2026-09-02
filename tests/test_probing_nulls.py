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


# --- the grounded statistical decisions (spec §Statistics) -------------------------------


class TestGroundedStatistics:
    """The five decisions, as behaviour rather than as comments."""

    def test_five_null_is_the_strongest_control_per_draw(self):
        """3C: a real feature must beat the *strongest* null, not an average of the five."""
        import numpy as np

        from galaxy_jepa.probing.controls import FeatureControls
        from galaxy_jepa.probing.nulls import five_null_samples

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
        null = five_null_samples(fc)
        # every draw is dominated by the strongest control (untrained, 0.70)
        assert null.tolist() == [0.70, 0.70]
        assert null.min() >= 0.70

    def test_multiplicity_defaults_to_benjamini_yekutieli(self):
        from galaxy_jepa.probing.config import ProbingConfig

        assert ProbingConfig().multiplicity == "benjamini_yekutieli"

    def test_permutation_floor_is_enforced_not_documented(self):
        from galaxy_jepa.probing.config import ProbingConfig

        assert ProbingConfig().n_perm >= 10_000
        assert ProbingConfig().permutation_method == "two_sided"
        with pytest.raises(ValueError, match="10,000"):
            ProbingConfig(n_perm=999)
        # a declared deviation is allowed, and is stamped
        assert ProbingConfig(n_perm=999, escape_hatches=("reduced_permutations",)).n_perm == 999

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
    """The strongest of the five nulls sets the bar; an unseeded one makes the bar move.

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
    import numpy as np

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

        one = ProbingConfig(scheme_name="full_tree", n_null_draws=4000)
        two = ProbingConfig(scheme_name="reduced", n_null_draws=4000)
        assert one.n_null_draws == two.n_null_draws == 4000
