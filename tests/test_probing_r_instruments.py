"""Brief R instruments: O1's retention rule, paired bootstrap, cluster control, MLP."""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import mlp
from galaxy_jepa.probing.logistic import Embeddings, paired_auc_bootstrap, weighted_auc


class TestRetentionVerdictReproducesO1:
    """O1's recorded rows, read back through the rule that now lives in the package."""

    @pytest.mark.parametrize(
        ("a", "c", "m", "lo", "c_m", "n", "share", "verdict", "retained"),
        [
            (0.8845, 0.7908, 0.8751, 0.8702, 0.7744, 18038, 0.518, "SURVIVES", 1.08),  # t01 mag
            (0.8845, 0.7908, 0.8021, 0.7952, 0.7137, 14470, 0.415, "SURVIVES", 0.94),  # t01 size
            (0.5889, 0.5474, 0.5819, 0.5738, 0.5410, 18002, 0.889, "SURVIVES", 0.99),  # tight
            (0.5226, 0.5160, 0.5105, 0.5011, 0.5070, 14864, 0.734, "COLLAPSES", 0.53),  # medium
            (0.5226, 0.5160, 0.5150, 0.5056, 0.5089, 14864, 0.734, "COLLAPSES", 0.93),  # medium z
            (0.5226, 0.5160, 0.5030, 0.4939, 0.5024, 14738, 0.728, "COLLAPSES", 0.10),  # med m×s
        ],
    )
    def test_o1_row(self, a, c, m, lo, c_m, n, share, verdict, retained) -> None:
        v = match.retention_verdict(a, c, m, lo, c_m, n_matched_test=n, n_test=round(n / share))
        assert v.verdict == verdict
        assert v.retained == pytest.approx(retained, abs=0.02)

    def test_high_retention_with_a_ci_touching_the_bar_still_collapses(self) -> None:
        """t10-medium on redshift: 0.93 of the margin kept, but the CI reaches C_m."""
        v = match.retention_verdict(
            0.5226, 0.5160, 0.5150, 0.5056, 0.5089, n_matched_test=14864, n_test=20250
        )
        assert v.verdict == match.COLLAPSES
        assert v.retained is not None and v.retained > 0.9

    def test_partial_is_its_own_state(self) -> None:
        v = match.retention_verdict(0.70, 0.55, 0.60, 0.59, 0.54, n_matched_test=5000, n_test=9000)
        assert v.verdict == match.PARTIAL  # 0.06 kept of 0.15: real, substantially confounded

    def test_a_thin_matched_set_is_unresolved_never_collapsed(self) -> None:
        v = match.retention_verdict(0.70, 0.55, 0.50, 0.45, 0.55, n_matched_test=300, n_test=9000)
        assert v.verdict == match.UNRESOLVED
        v = match.retention_verdict(0.70, 0.55, 0.50, 0.45, 0.55, n_matched_test=800, n_test=9000)
        assert v.verdict == match.UNRESOLVED  # 800 < 10% of 9,000

    def test_matched_indices_depend_on_nuisance_and_labels_only(self) -> None:
        rng = np.random.default_rng(0)
        vals, y = rng.normal(size=600), rng.integers(0, 2, 600)
        a = match.matched_indices(vals[:400], y[:400], vals[400:], y[400:], seed=3)
        b = match.matched_indices(vals[:400], y[:400], vals[400:], y[400:], seed=3)
        assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


class TestPairedBootstrap:
    def test_weighted_auc_matches_sklearn_including_ties(self) -> None:
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        s = np.round(rng.normal(size=500) + y, 1)  # rounding forces ties
        assert weighted_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        w = rng.integers(0, 3, 500).astype(float)
        rep = np.repeat(np.arange(500), w.astype(int))
        assert weighted_auc(y, s, w) == pytest.approx(roc_auc_score(y[rep], s[rep]), abs=1e-12)

    def test_identical_scores_give_exactly_zero(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 400)
        s = rng.normal(size=400) + y
        point, lo, hi = paired_auc_bootstrap([y, y], [s, s], [1.0, -1.0], n_boot=200)
        assert point == lo == hi == 0.0

    def test_a_genuinely_better_scorer_excludes_zero(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 3000)
        base = rng.normal(size=3000)
        good, weak = base + 1.2 * y, base + 0.8 * y
        point, lo, hi = paired_auc_bootstrap([y, y], [good, weak], [1.0, -1.0], n_boot=500)
        assert point > 0 and lo > 0

    def test_terms_over_different_galaxies_are_refused(self) -> None:
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="same galaxies"):
            paired_auc_bootstrap([y, y[:3]], [y, y[:3]], [1.0, -1.0])


class TestClusterControl:
    def test_labels_are_constant_within_a_cluster_and_hit_the_base_rate(self) -> None:
        rng = np.random.default_rng(4)
        cluster_of = rng.integers(0, 200, 20_000)
        y = ctl.cluster_control_labels(cluster_of, base_rate=0.3, seed=0)
        for k in range(200):
            assert len(np.unique(y[cluster_of == k])) == 1
        assert y.mean() == pytest.approx(0.3, abs=0.02)

    def test_kmeans_is_fit_on_train_only(self) -> None:
        """Held-out rows far from every train point still get a train centroid, never their own."""
        rng = np.random.default_rng(5)
        train = rng.normal(size=(500, 4))
        far = rng.normal(loc=50.0, size=(20, 4))
        assign = ctl.cluster_assignment(train, np.vstack([train, far]), n_clusters=5, seed=0)
        assert assign.max() < 5  # no sixth cluster conjured for the outliers

    def test_the_control_can_fail_an_mlp_decodes_random_cluster_labels(self) -> None:
        """The control is shown able to fire, not assumed to: arbitrary labels on recurring
        clusters are decodable by an MLP on held-out rows, and much less so by a linear probe."""
        from galaxy_jepa.probing.logistic import probe_auc

        rng = np.random.default_rng(6)
        centres = rng.normal(scale=4.0, size=(40, 8))
        which = rng.integers(0, 40, 4000)
        x = centres[which] + rng.normal(scale=0.5, size=(4000, 8))
        y = ctl.cluster_control_labels(which, base_rate=0.5, seed=1)
        f = np.zeros(4000)
        tr, te = Embeddings(x[:3000], y[:3000], f[:3000]), Embeddings(x[3000:], y[3000:], f[3000:])
        mlp_auc = mlp.mlp_auc(tr, te, width=128, epochs=400, lr=1e-2)
        lin_auc = probe_auc(tr, te)
        assert mlp_auc > 0.9
        assert mlp_auc - lin_auc > 0.15


class TestMLPInstrument:
    def test_mlp_auc_is_unchanged_by_delegating_to_mlp_fit(self) -> None:
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(7)
        x = rng.normal(size=(600, 6))
        y = (x[:, 0] ** 2 > 1).astype(np.int64)
        f = np.zeros(600)
        tr, te = Embeddings(x[:400], y[:400], f[:400]), Embeddings(x[400:], y[400:], f[400:])
        fit = mlp.mlp_fit(tr, te, width=32)
        assert mlp.mlp_auc(tr, te, width=32) == pytest.approx(roc_auc_score(te.y, fit.test_scores))

    def test_memorisation_is_measured_on_train(self) -> None:
        """Permuted labels: held-out AUC stays at chance, train AUC shows what was fitted."""
        rng = np.random.default_rng(8)
        x = rng.normal(size=(300, 16))
        y = rng.integers(0, 2, 300)
        f = np.zeros(300)
        tr, te = Embeddings(x[:200], y[:200], f[:200]), Embeddings(x[200:], y[200:], f[200:])
        fit = mlp.mlp_fit(tr, te, width=512, epochs=400, lr=1e-2)
        assert fit.train_auc > 0.95
        from sklearn.metrics import roc_auc_score

        assert abs(roc_auc_score(te.y, fit.test_scores) - 0.5) < 0.15

    def test_width_selection_cannot_see_test(self) -> None:
        import inspect

        params = inspect.signature(mlp.select_width).parameters
        assert "test" not in params and list(params)[0] == "train"

    def test_width_selection_prefers_capacity_when_the_task_needs_it(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(1500, 4))
        y = (np.sin(2 * x[:, 0]) + x[:, 1] ** 2 > 1).astype(np.int64)
        tr = Embeddings(x, y, np.zeros(1500))
        best, scores = mlp.select_width(tr, widths=(2, 64), epochs=300, lr=1e-2)
        assert best == 64 and scores[64] > scores[2]
