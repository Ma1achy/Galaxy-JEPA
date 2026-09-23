"""Planted-structure tests for the centroid-path geometry — the guarantee it answers its question.

Each synthetic population plants a known conditional-mean path in 64 dimensions and buries it in
isotropic noise at a per-bin occupancy like the real thin buckets. The instrument has to call a
straight path straight at no more than the nominal rate, a bent one curved, a shuffled one
tangled, and a loop a loop — or none of its verdicts on morphology mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_jepa.probing import geometry as geo

D = 64
EDGES = np.linspace(0.0, 1.0, 11)


def _population(path, *, n: int = 4000, noise: float = 1.0, seed: int = 0):
    """``n`` galaxies with values uniform on [0, 1] and embeddings path(value) + noise."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.0, 1.0, n)
    x = np.stack([path(t) for t in v]) + rng.normal(scale=noise, size=(n, D))
    return x, v


def _line(t: float) -> np.ndarray:
    out = np.zeros(D)
    out[0] = 3.0 * t
    return out


def _parabola(t: float) -> np.ndarray:
    out = np.zeros(D)
    out[0] = 3.0 * t
    out[1] = 6.0 * (t - 0.5) ** 2
    return out


def _run(x, v, *, seed: int = 0, top_up_below: float = 0.05):
    stats = geo.path_statistics(x, v, EDGES, seed=seed)
    assert stats is not None
    test = geo.curvature_test(
        x, v, EDGES, stats, top_up_below=top_up_below, b_first=200, b_top=200, seed=seed
    )
    return stats, test


class TestStraightAndCurved:
    def test_a_straight_path_is_nearly_all_on_its_line(self) -> None:
        x, v = _population(_line)
        stats, test = _run(x, v)
        assert test.exists
        assert stats.straightness > 0.9
        assert stats.monotonicity > 0.95

    def test_a_straight_path_is_called_curved_at_no_more_than_the_nominal_rate(self) -> None:
        """The false-positive rate, measured: 20 independent straight worlds at alpha = 0.05."""
        false_curved = 0
        for s in range(20):
            x, v = _population(_line, n=3000, seed=100 + s)
            _, test = _run(x, v, seed=s)
            false_curved += test.p_curved <= 0.05
        assert false_curved <= 3  # nominal 1 of 20; allow binomial slack, not a systematic bias

    def test_a_parabola_is_curved(self) -> None:
        x, v = _population(_parabola)
        stats, test = _run(x, v)
        assert test.exists
        assert test.p_curved <= 0.01
        assert 1.0 - stats.straightness >= geo.CURVED_MIN_BEND
        assert stats.monotonicity > 0.9  # curved AND ordered — a different finding from tangled

    def test_pure_noise_has_no_trajectory_and_curvature_centred_on_zero(self) -> None:
        x, v = _population(lambda t: np.zeros(D))
        stats, test = _run(x, v)
        assert not test.exists
        assert abs(stats.total) < 0.05  # cross-fitting cancels the noise rather than adding it


class TestTangled:
    def test_a_path_visited_out_of_order_is_not_ordered(self) -> None:
        """Bins visit ten well-separated points in a scrambled order: a path, but a tangle."""
        rng = np.random.default_rng(3)
        points = rng.normal(scale=3.0, size=(10, D))
        scramble = rng.permutation(10)

        def tangle(t: float) -> np.ndarray:
            return points[scramble[min(int(t * 10), 9)]]

        x, v = _population(tangle, n=5000, seed=4)
        stats, test = _run(x, v)
        assert test.exists
        assert stats.monotonicity < geo.ORDERED_MIN
        assert stats.effective_dim > 3


class TestCircle:
    def test_a_loop_is_recovered_as_a_circle(self) -> None:
        """R3's premise, tested before it is relied on: 12 bins round one full turn."""

        def loop(t: float) -> np.ndarray:
            out = np.zeros(D)
            out[0], out[1] = 3.0 * np.cos(2 * np.pi * t), 3.0 * np.sin(2 * np.pi * t)
            return out

        edges = np.linspace(0.0, 1.0, 13)
        x, v = _population(loop, n=6000)
        stats = geo.path_statistics(x, v, edges)
        assert stats is not None
        test = geo.curvature_test(x, v, edges, stats, top_up_below=0.05, b_first=200, b_top=200)
        circle = geo.circle_test(stats, curved=test.p_curved <= 0.05, exists=test.exists)
        assert circle.recovered, circle.reason
        assert 1.5 <= stats.effective_dim <= 2.5

    def test_a_line_is_not_a_circle_because_it_does_not_close(self) -> None:
        x, v = _population(_line)
        stats, test = _run(x, v)
        circle = geo.circle_test(stats, curved=True, exists=True)  # grant everything but shape
        assert not circle.closes
        assert not circle.recovered


class TestOccupancyAndMultiplicity:
    def test_too_few_populated_bins_is_uncharacterised_not_straight(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.uniform(0.0, 0.3, 900)  # three bins only
        x = rng.normal(size=(900, D))
        assert geo.path_statistics(x, v, EDGES) is None
        verdicts = geo.curvature_verdicts({"thin": None}, {})
        assert verdicts["thin"]["verdict"] == geo.UNCHARACTERISED

    def test_the_top_up_runs_only_for_candidates(self) -> None:
        x, v = _population(_line, n=3000, seed=9)
        stats = geo.path_statistics(x, v, EDGES)
        assert stats is not None
        cold = geo.curvature_test(x, v, EDGES, stats, top_up_below=1e-9, b_first=50, b_top=400)
        assert cold.n_draws == 50  # cannot pass the loosest bar: keeps its first-pass p
        x2, v2 = _population(_parabola, n=3000, seed=9)
        s2 = geo.path_statistics(x2, v2, EDGES)
        assert s2 is not None
        hot = geo.curvature_test(x2, v2, EDGES, s2, top_up_below=0.05, b_first=50, b_top=400)
        assert hot.n_draws == 400  # a candidate: topped up so the floor does not decide it
        assert hot.p_curved == pytest.approx(1 / 401)

    def test_by_is_applied_across_the_family(self) -> None:
        """Twenty straight worlds and one parabola: BY keeps the parabola, drops the chance hits."""
        stats: dict[str, geo.PathStats | None] = {}
        tests: dict[str, geo.CurvatureTest] = {}
        for s in range(20):
            x, v = _population(_line, n=2500, seed=200 + s)
            st = geo.path_statistics(x, v, EDGES, seed=s)
            assert st is not None
            stats[f"line{s}"] = st
            tests[f"line{s}"] = geo.curvature_test(
                x, v, EDGES, st, top_up_below=0.012, b_first=200, b_top=2000, seed=s
            )
        x, v = _population(_parabola, n=2500, seed=999)
        st = geo.path_statistics(x, v, EDGES)
        assert st is not None
        stats["bent"] = st
        tests["bent"] = geo.curvature_test(
            x, v, EDGES, st, top_up_below=0.012, b_first=200, b_top=2000
        )
        verdicts = geo.curvature_verdicts(stats, tests)
        assert verdicts["bent"]["verdict"] == geo.CURVED
        assert sum(verdicts[f"line{s}"]["verdict"] == geo.CURVED for s in range(20)) == 0
