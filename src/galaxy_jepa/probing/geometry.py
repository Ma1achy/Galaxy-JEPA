"""Concept geometry — the shape of the conditional-mean path E[z | value] (Brief R2 and R3).

A probe AUC measures *decodability*, not shape. For shape: bin galaxies by a value (a vote fraction
for R2, an orientation for R3), take the embedding centroid of each bin, and ask what path the
centroids trace — a line, a bent line, a tangle, or a loop.

**What the object is.** On any population the path is E[z | value]: it carries everything that
co-varies with the value — size, brightness, other answers — so it is the shape of the conditional
mean, not of "the concept" in isolation. It is also exactly what the uncertainty geometry projects
onto a line, which is why its curvature matters there: projecting a curved path onto one direction
attenuates the correlation with the value by construction.

**Noise is the whole difficulty.** A bin centroid in 384 dimensions carries noise of norm roughly
√(384·σ²/n). With a few hundred galaxies per bin that is comparable to the signal, and it spreads
the centroids off any line — curvature manufactured from noise alone. Two defences, both used:

* **Split-half cross-fitting.** Each bin's galaxies are split into halves A and B; the path's
  covariance is formed from the *cross* product of A-centroids and B-centroids. Noise in the two
  halves is independent, so it cancels in expectation instead of adding a positive bias.
* **A shuffled-value null** through the identical procedure. Shuffling the values among the
  galaxies preserves every bin's occupancy exactly, so the null carries the same noise the real
  path does and none of its signal.

**Direction of the residual bias, stated rather than hidden.** A shuffled bin's galaxies are a
random draw from the whole population, so their spread is slightly *larger* than a real bin's
(which excludes the between-bin signal). The null is therefore a little wide: it biases towards
calling a path straight. A "straight" verdict is accordingly weaker evidence than a "curved" one,
and R3's circle is what licenses believing it at all.

Pure numpy (plus scipy's rank correlation); the drivers own the embeddings.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

__all__ = [
    "MIN_OCCUPANCY",
    "MIN_BINS",
    "N_SPLITS",
    "B_FIRST",
    "B_TOP",
    "CURVED_MIN_BEND",
    "ORDERED_MIN",
    "PathStats",
    "path_statistics",
    "CurvatureTest",
    "curvature_test",
    "curvature_verdicts",
    "CircleTest",
    "circle_test",
    "standardise",
]

#: Declared choices, not derived ones.
MIN_OCCUPANCY: int = 100  # per bin, i.e. 50 per half: below it a centroid is mostly noise
MIN_BINS: int = 5  # fewer surviving bins cannot distinguish a bend from an endpoint wobble
N_SPLITS: int = 2  # random half-splits averaged per path (and per null draw — like for like)
B_FIRST: int = 200  # null draws for every feature
B_TOP: int = 4_000  # for candidates only: 1/4001 < BY's rank-1 bar at m = 37 (3.2e-4)
CURVED_MIN_BEND: float = 0.10  # 1 − straightness; a significant 1% bend is not "curved"
ORDERED_MIN: float = 0.9  # |Spearman| of bin order along the best-fit line


def standardise(x: np.ndarray) -> np.ndarray:
    """Z-score each dimension over the rows given. Label-free, so it cannot leak a verdict."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


@dataclasses.dataclass(frozen=True)
class PathStats:
    """The centroid path of one feature, from the cross-fitted spectrum λ₁ ≥ λ₂ ≥ …

    ``straightness`` is λ₁ / tr(S), the share of the path's variance on its best-fit line — the
    line-versus-curve answer. ``curvature`` is tr(S) − λ₁, the off-line variance itself.
    ``monotonicity`` is |Spearman| between bin order and position along that line: a curved but
    ordered path is a different finding from a tangled one. ``effective_dim`` is the participation
    ratio of the positive eigenvalues. ``readout`` is Pearson between bin value and the centroids'
    projection on a supplied direction (the ladder's logistic one) — the attenuation a linear
    uncertainty-geometry readout inherits.
    """

    centres: np.ndarray
    occupancy: np.ndarray
    total: float
    eigenvalues: np.ndarray
    line: np.ndarray
    straightness: float
    curvature: float
    monotonicity: float
    effective_dim: float
    readout: float | None
    centroids: np.ndarray  # pooled (all galaxies) per surviving bin, shape (k, D)
    step_sq: np.ndarray  # cross-fitted squared distance between consecutive bins, and last→first

    @property
    def n_bins(self) -> int:
        return int(self.centres.size)


def _bin_of(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per row, the last edge inclusive; −1 outside the edges or non-finite."""
    values = np.asarray(values, dtype=np.float64)
    b = np.searchsorted(edges, values, side="right") - 1
    b[values == edges[-1]] = len(edges) - 2
    b[(values < edges[0]) | (values > edges[-1]) | ~np.isfinite(values)] = -1
    return b


def _surviving(bin_of: np.ndarray, n_bins: int, min_occupancy: int) -> np.ndarray:
    counts = np.bincount(bin_of[bin_of >= 0], minlength=n_bins)
    return np.nonzero(counts >= min_occupancy)[0]


def _half_centroids(
    x: np.ndarray, group: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Centroids of a random half-split within each of ``k`` groups (``group`` −1 = excluded).

    Vectorised throughout, because the null calls it thousands of times: a lexsort gives each
    row's rank within its group under a random key, alternate ranks form the halves (equal to
    within one galaxy), and one sparse product sums every cell without copying ``x``.
    """
    from scipy import sparse

    rows = np.nonzero(group >= 0)[0]
    g = group[rows]
    order = np.lexsort((rng.random(g.size), g))
    g_sorted = g[order]
    starts = np.searchsorted(g_sorted, np.arange(k))
    half = np.empty(g.size, dtype=np.int64)
    half[order] = (np.arange(g.size) - starts[g_sorted]) % 2
    cell = g * 2 + half
    onehot = sparse.csr_matrix((np.ones(rows.size), (cell, rows)), shape=(2 * k, x.shape[0]))
    sums = np.asarray(onehot @ x)
    counts = np.bincount(cell, minlength=2 * k).astype(np.float64)
    means = sums / counts[:, None]
    return means[0::2], means[1::2]


def _cross_cov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetrised cross-covariance of two centroid sets over bins (noise-unbiased)."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    s = ac.T @ bc
    return (s + s.T) / (2.0 * a.shape[0])


def _group_index(bin_of: np.ndarray, keep_bins: np.ndarray) -> np.ndarray:
    """Map raw bin index → position among the surviving bins; −1 for dropped bins."""
    remap = -np.ones(int(bin_of.max()) + 2 if bin_of.size else 1, dtype=np.int64)
    remap[keep_bins] = np.arange(keep_bins.size)
    out = np.where(bin_of >= 0, remap[np.clip(bin_of, 0, None)], -1)
    return out


def path_statistics(
    x: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
    *,
    direction: np.ndarray | None = None,
    min_occupancy: int = MIN_OCCUPANCY,
    min_bins: int = MIN_BINS,
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> PathStats | None:
    """The cross-fitted centroid path, or ``None`` if fewer than ``min_bins`` bins survive.

    ``x`` should already be standardised (:func:`standardise`); ``values`` is one per row.
    ``None`` is *uncharacterised (occupancy)* — a statement about the sample, never "straight".
    """
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    raw = _bin_of(values, edges)
    keep = _surviving(raw, len(edges) - 1, min_occupancy)
    if keep.size < min_bins:
        return None
    group = _group_index(raw, keep)
    k = keep.size
    rng = np.random.default_rng(seed)

    s = np.zeros((x.shape[1], x.shape[1]))
    step_sq = np.zeros(k)
    for _ in range(n_splits):
        a, b = _half_centroids(x, group, k, rng)
        s += _cross_cov(a, b)
        # Unbiased squared distance between consecutive bins: <a_i − a_j, b_i − b_j>
        nxt = np.roll(np.arange(k), -1)
        step_sq += np.einsum("ij,ij->i", a - a[nxt], b - b[nxt])
    s /= n_splits
    step_sq /= n_splits

    evals, evecs = np.linalg.eigh(s)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    total = float(np.trace(s))
    line = evecs[:, 0]
    lam1 = float(evals[0])

    from scipy import sparse

    sel = np.nonzero(group >= 0)[0]
    onehot = sparse.csr_matrix((np.ones(sel.size), (group[sel], sel)), shape=(k, x.shape[0]))
    occupancy = np.bincount(group[sel], minlength=k)
    pooled = np.asarray(onehot @ x) / occupancy[:, None]
    centres = 0.5 * (edges[keep] + edges[keep + 1])

    pos = (pooled - pooled.mean(axis=0)) @ line
    mono = float(abs(spearmanr(centres, pos).statistic)) if k > 2 else float("nan")
    positive = np.clip(evals, 0.0, None)
    pr = float(positive.sum() ** 2 / (positive**2).sum()) if positive.sum() > 0 else float("nan")
    readout = None
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        readout = float(pearsonr(centres, pooled @ (d / np.linalg.norm(d))).statistic)

    return PathStats(
        centres=centres,
        occupancy=occupancy,
        total=total,
        eigenvalues=evals[: 2 * k],
        line=line,
        straightness=lam1 / total if total > 0 else float("nan"),
        curvature=total - lam1,
        monotonicity=mono,
        effective_dim=pr,
        readout=readout,
        centroids=pooled,
        step_sq=step_sq,
    )


@dataclasses.dataclass(frozen=True)
class CurvatureTest:
    """The shuffled-value null for one path.

    ``exists`` — tr(S) above the null's 95th percentile: there is a path at all. ``p_curved`` is
    the add-one Monte Carlo p-value of the off-line variance against the null's off-line variance
    **along the observed line**, so the two are measured the same way. ``n_draws`` says whether it
    was topped up.
    """

    exists: bool
    total_null_q95: float
    p_curved: float
    n_draws: int
    offline_null_mean: float


def _null_draw(
    x: np.ndarray,
    raw: np.ndarray,
    keep: np.ndarray,
    line: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One shuffle: ``(tr S_null, off-line variance of S_null along the observed line)``.

    Only traces and one quadratic form — no eigendecomposition — so a 4,000-draw top-up stays
    cheap. Shuffling the bin labels among the rows is the same as shuffling the values, and keeps
    every bin's occupancy exactly.
    """
    perm = raw[rng.permutation(raw.size)]
    group = _group_index(perm, keep)
    k = keep.size
    tr = 0.0
    on = 0.0
    for _ in range(n_splits):
        a, b = _half_centroids(x, group, k, rng)
        ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
        tr += float(np.einsum("ij,ij->", ac, bc)) / k
        on += float((ac @ line) @ (bc @ line)) / k
    tr /= n_splits
    on /= n_splits
    return tr, tr - on


def curvature_test(
    x: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
    stats: PathStats,
    *,
    top_up_below: float,
    min_occupancy: int = MIN_OCCUPANCY,
    n_splits: int = N_SPLITS,
    b_first: int = B_FIRST,
    b_top: int = B_TOP,
    seed: int = 0,
) -> CurvatureTest:
    """Shuffled-value null, drawn sequentially: ``b_first`` for all, ``b_top`` for candidates.

    ``top_up_below`` is the **loosest** threshold the multiplicity correction can ever apply — for
    BY, α/H_m, reached only at the highest rank. A path whose p after ``b_first`` draws already
    exceeds it cannot pass at any rank however many draws follow, so it keeps its p. One that does
    not is topped up, because at ``b_first`` its p is floored at 1/(b_first + 1) — far above BY's
    rank-1 bar — and the floor, not the data, would decide it.
    """
    x = np.asarray(x, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    raw = _bin_of(values, edges)
    keep = _surviving(raw, len(edges) - 1, min_occupancy)
    rng = np.random.default_rng(seed + 7919)
    totals: list[float] = []
    offs: list[float] = []

    def draw(n: int) -> None:
        for _ in range(n):
            t, o = _null_draw(x, raw, keep, stats.line, n_splits, rng)
            totals.append(t)
            offs.append(o)

    draw(b_first)
    exceed = int(np.sum(np.asarray(offs) >= stats.curvature))
    p = (1 + exceed) / (b_first + 1)
    if p <= top_up_below:
        draw(b_top - b_first)
        exceed = int(np.sum(np.asarray(offs) >= stats.curvature))
        p = (1 + exceed) / (len(offs) + 1)
    q95 = float(np.quantile(totals[:b_first], 0.95))
    return CurvatureTest(
        exists=stats.total > q95,
        total_null_q95=q95,
        p_curved=p,
        n_draws=len(offs),
        offline_null_mean=float(np.mean(offs)),
    )


CURVED = "curved"
STRAIGHT = "straight_within_noise"
NO_PATH = "no_trajectory"
UNCHARACTERISED = "uncharacterised_occupancy"


def curvature_verdicts(
    stats: Mapping[str, PathStats | None],
    tests: Mapping[str, CurvatureTest],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, object]]:
    """Pre-registered verdicts, BY-corrected over the family of paths that reach the test.

    *curved* iff BY-significant **and** 1 − straightness ≥ :data:`CURVED_MIN_BEND`. *ordered* iff
    monotonicity ≥ :data:`ORDERED_MIN`; curved-and-ordered and tangled are different findings, so
    both flags travel. A path with no trajectory is not tested for curvature at all, and an
    uncharacterised one is never reported as straight.
    """
    from galaxy_jepa.probing.nulls import family_significant

    family = {f: tests[f].p_curved for f, s in stats.items() if s is not None and tests[f].exists}
    significant = family_significant(family, alpha=alpha) if family else {}
    out: dict[str, dict[str, object]] = {}
    for f, s in stats.items():
        if s is None:
            out[f] = {"verdict": UNCHARACTERISED}
            continue
        t = tests[f]
        if not t.exists:
            out[f] = {"verdict": NO_PATH, "ordered": None}
            continue
        bend = 1.0 - s.straightness
        curved = bool(significant.get(f, False)) and bend >= CURVED_MIN_BEND
        out[f] = {
            "verdict": CURVED if curved else STRAIGHT,
            "ordered": bool(s.monotonicity >= ORDERED_MIN),
            "by_significant": bool(significant.get(f, False)),
            "bend": bend,
        }
    return out


@dataclasses.dataclass(frozen=True)
class CircleTest:
    """R3's pre-registered criteria for a loop, on top of R2's own statistics."""

    recovered: bool
    effective_dim: float
    winding_violations: int
    winding_turns: float
    closes: bool
    reason: str


def circle_test(stats: PathStats, *, curved: bool, exists: bool) -> CircleTest:
    """A circle is recovered iff R2's machinery finds a path, calls it curved, finds it at least
    1.5-dimensional, the centroids go round in order, and the path closes.

    *Order:* the angle of each centroid in the plane of the path's top two principal directions
    must advance the same way at every step, with at most one violation. *Closure:* the step from
    the last bin back to the first must be no longer than the longest step between neighbours —
    which a line (whose last→first step is its whole length) fails.
    """
    k = stats.n_bins
    plane = np.linalg.svd(stats.centroids - stats.centroids.mean(axis=0), full_matrices=False)[2][
        :2
    ]
    xy = (stats.centroids - stats.centroids.mean(axis=0)) @ plane.T
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    steps = np.angle(np.exp(1j * (np.roll(phi, -1) - phi)))  # wrapped to (−π, π], incl. last→first
    direction = np.sign(np.sum(steps))
    violations = int(np.sum(np.sign(steps) != direction))
    turns = float(np.sum(steps) / (2 * np.pi))
    step = np.sqrt(np.clip(stats.step_sq, 0.0, None))
    closes = bool(step[k - 1] <= step[: k - 1].max())
    ok = (
        exists
        and curved
        and stats.effective_dim >= 1.5
        and violations <= 1
        and abs(round(turns)) == 1
        and closes
    )
    reason = (
        f"exists={exists} curved={curved} eff_dim={stats.effective_dim:.2f} "
        f"violations={violations} turns={turns:+.2f} closes={closes}"
    )
    return CircleTest(bool(ok), stats.effective_dim, violations, turns, closes, reason)
