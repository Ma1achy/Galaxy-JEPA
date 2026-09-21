"""The same-corpus comparison, D13's Hart anchor, and the normality caveat."""

from __future__ import annotations

import numpy as np

from galaxy_jepa.probing import entanglement as ent
from galaxy_jepa.probing import nulls as nz
from galaxy_jepa.probing.extract import LabelProvider


def _provider(rows: dict[int, dict[str, float]], features: dict[str, str]) -> LabelProvider:
    return LabelProvider(rows, feature_cols=features, nuisance_cols={}, vote_count_min=1.0)


def _rows(n: int, *, seed: int = 0) -> tuple[dict[int, dict[str, float]], dict[str, str]]:
    """Three answers: ``b`` tracks ``a``, ``c`` is independent of both."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(size=n)
    b = np.clip(a + rng.normal(scale=0.05, size=n), 0, 1)
    c = rng.uniform(size=n)
    rows = {i: {"fa": float(a[i]), "fb": float(b[i]), "fc": float(c[i])} for i in range(n)}
    return rows, {"a": "fa", "b": "fb", "c": "fc"}


class TestHumanVoteCorrelation:
    def test_recovers_the_planted_structure(self) -> None:
        rows, feats = _rows(600)
        corr, overlap = ent.human_vote_correlation(
            _provider(rows, feats), ["a", "b", "c"], list(rows)
        )
        assert corr[0, 1] > 0.9  # b tracks a
        assert abs(corr[0, 2]) < 0.2  # c does not
        assert corr[0, 1] == corr[1, 0]
        assert np.allclose(np.diag(corr), 1.0)
        assert overlap[0, 1] == 600

    def test_thin_pairs_return_nan_rather_than_a_number(self) -> None:
        """A correlation over 60 galaxies is not the same measurement as one over 60,000."""
        rows, feats = _rows(120)
        corr, overlap = ent.human_vote_correlation(
            _provider(rows, feats), ["a", "b"], list(rows), min_overlap=200
        )
        assert overlap[0, 1] == 120
        assert not np.isfinite(corr[0, 1])

    def test_missing_votes_shrink_the_overlap_not_the_matrix(self) -> None:
        rows, feats = _rows(500)
        for i in range(400, 500):
            rows[i]["fb"] = float("nan")
        corr, overlap = ent.human_vote_correlation(_provider(rows, feats), ["a", "b"], list(rows))
        assert overlap[0, 1] == 400
        assert np.isfinite(corr[0, 1])


class TestCompareToHumanStructure:
    def test_agreement_is_rank_based_and_disagreements_are_signed(self) -> None:
        names = ["a", "b", "c"]
        human = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
        cosine = np.array([[1.0, 0.8, 0.9], [0.8, 1.0, 0.1], [0.9, 0.1, 1.0]])
        agree = ent.compare_to_human_structure(names, cosine, human)
        assert agree.n_pairs == 3
        worst = agree.largest_disagreements[0]
        assert {worst[0], worst[1]} == {"a", "c"}  # encoder ties them, the votes do not
        assert worst[4] > 0  # signed: positive means the encoder places them closer

    def test_nan_pairs_are_dropped_not_zero_filled(self) -> None:
        names = ["a", "b", "c"]
        human = np.full((3, 3), np.nan)
        agree = ent.compare_to_human_structure(names, np.eye(3), human)
        assert agree.n_pairs == 0
        assert not np.isfinite(agree.spearman)
        assert agree.largest_disagreements == []


class TestBarWindingAlignment:
    def _cos(self, names: list[str], values: dict[tuple[str, str], float]) -> np.ndarray:
        m = np.eye(len(names))
        for (a, b), v in values.items():
            i, j = names.index(a), names.index(b)
            m[i, j] = m[j, i] = v
        return m

    def test_leaning_loose_reads_as_physics(self) -> None:
        names = [ent.BAR_FEATURE, *ent.WINDING_ORDER]
        cos = self._cos(
            names,
            {
                (ent.BAR_FEATURE, ent.WINDING_ORDER[0]): 0.20,
                (ent.BAR_FEATURE, ent.WINDING_ORDER[1]): 0.30,
                (ent.BAR_FEATURE, ent.WINDING_ORDER[2]): 0.55,
            },
        )
        assert ent.bar_winding_alignment(names, cos).verdict == ent.PHYSICS

    def test_uniform_alignment_reads_as_bleed(self) -> None:
        """Hart's 4-6 degrees is a *directional* prediction; flat across all three isn't it."""
        names = [ent.BAR_FEATURE, *ent.WINDING_ORDER]
        cos = self._cos(names, {(ent.BAR_FEATURE, w): 0.42 for w in ent.WINDING_ORDER})
        v = ent.bar_winding_alignment(names, cos)
        assert v.verdict == ent.BLEED
        assert v.spread < 0.05

    def test_leaning_tight_is_contrary_not_physics(self) -> None:
        names = [ent.BAR_FEATURE, *ent.WINDING_ORDER]
        cos = self._cos(
            names,
            {
                (ent.BAR_FEATURE, ent.WINDING_ORDER[0]): 0.55,
                (ent.BAR_FEATURE, ent.WINDING_ORDER[1]): 0.30,
                (ent.BAR_FEATURE, ent.WINDING_ORDER[2]): 0.20,
            },
        )
        assert ent.bar_winding_alignment(names, cos).verdict == ent.CONTRARY

    def test_missing_features_are_unavailable_not_a_verdict(self) -> None:
        names = ["t01_smooth_or_features_a01_smooth", ent.BAR_FEATURE]
        v = ent.bar_winding_alignment(names, np.eye(2))
        assert v.verdict == ent.UNAVAILABLE


class TestNormalityReport:
    def test_a_normal_bank_passes_and_carries_no_caveat_weight(self) -> None:
        # Seed 3 draws a genuinely normal bank that Shapiro rejects at p=0.047 — at K=30 the
        # test is noisy enough to fail on real normal data, which is itself part of the caveat.
        rng = np.random.default_rng(11)
        bank = {"a": list(rng.normal(0.55, 0.01, size=30))}
        rep = nz.normality_report(bank)
        assert rep["a"].normal
        assert rep["a"].k == 30

    def test_a_skewed_bank_fails_and_says_the_p_value_is_model_based(self) -> None:
        rng = np.random.default_rng(3)
        bank = {"a": list(0.5 + rng.exponential(0.02, size=30))}
        check = nz.normality_report(bank)["a"]
        assert not check.normal
        assert check.skew > 0.5
        assert "MODEL-BASED" in check.caveat
        assert "30" in check.caveat

    def test_the_report_is_per_feature(self) -> None:
        rng = np.random.default_rng(5)
        bank = {
            "a": list(rng.normal(0.5, 0.01, size=30)),
            "b": list(0.5 + rng.lognormal(sigma=1.2, size=30) / 50),
        }
        rep = nz.normality_report(bank)
        assert set(rep) == {"a", "b"}
        assert rep["a"].normal and not rep["b"].normal

    def test_a_bank_too_small_to_test_never_reads_as_normal(self) -> None:
        """A report, not a gate — but silence must not be mistaken for a pass."""
        check = nz.normality_report({"a": [0.5, 0.51]})["a"]
        assert not check.normal
        assert not np.isfinite(check.shapiro_p)
