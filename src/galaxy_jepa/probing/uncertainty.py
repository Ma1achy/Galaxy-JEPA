"""Uncertainty geometry — the headline (design 4) **[LOCKED]**.

The claim, sharply: not "does the axis classify bars?" (the ladder) but **"does distance
along the unsupervised concept axis reproduce the human vote *fraction* — the graded
uncertainty — for galaxies the axis never saw?"** Fit the axis on the consensus *extremes*
only (binary); project the held-out *ambiguous middle*; test whether projection distance
*ranks* their human vote fractions. If it does, the ambiguity is a real property of the
images, not a labelling artefact — the same observation as v1, the opposite conclusion.

The non-circularity is already structural: the firewall lives in ``data/splits.py``
(``partition_uncertainty`` / ``assert_uncertainty_firewall``) — the axis never sees the
gradient it is later asked to reproduce. This module is the *measurement* on top of it.

* **Spearman (rank) primary** — robust to the projection's arbitrary scale, assumes no linear
  form. **Pearson secondary.**
* **FLAGGED (4): the permutation-test mechanics** — the statistic and the shuffle loop are
  built; :func:`permutation_p` carries the placeholder p-definition.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from galaxy_jepa.data.metadata import featured_label
from galaxy_jepa.data.splits import assert_uncertainty_firewall, partition_uncertainty
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider
from galaxy_jepa.probing.logistic import Embeddings, probe_direction

__all__ = [
    "project",
    "spearman",
    "pearson",
    "permutation_p",
    "UncertaintyGeometry",
    "uncertainty_geometry",
]


def project(x: np.ndarray, w_unit: np.ndarray) -> np.ndarray:
    """Signed projection distance of each embedding onto the unit concept axis."""
    return np.asarray(x, dtype=np.float64) @ np.asarray(w_unit, dtype=np.float64)


def spearman(distances: np.ndarray, fractions: np.ndarray) -> float:
    """Spearman rank correlation between projection distance and vote fraction (primary)."""
    from scipy.stats import spearmanr

    return float(spearmanr(distances, fractions).statistic)


def pearson(distances: np.ndarray, fractions: np.ndarray) -> float:
    """Pearson correlation (secondary)."""
    from scipy.stats import pearsonr

    return float(pearsonr(distances, fractions).statistic)


def permutation_p(
    distances: np.ndarray,
    fractions: np.ndarray,
    *,
    n_perm: int = 10_000,
    method: str = "two_sided",
    seed: int = 0,
) -> float:
    """Permutation-test p for the observed Spearman against a shuffled-fraction null (design 4A).

    FLAGGED: pending stats grounding — do not finalise. Shuffle the vote fractions, recompute
    Spearman ``n_perm`` times, locate the real value in that null. Placeholder: the add-one
    empirical p, two-sided (``|null| ≥ |observed|``) or one-sided-greater per ``method``. The
    grounding session owns *what the resulting p means here* and the tail definition; only this
    function changes.
    """
    rng = np.random.default_rng(seed)
    observed = spearman(distances, fractions)
    frac = np.asarray(fractions, dtype=np.float64)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = spearman(distances, rng.permutation(frac))
    if method == "two_sided":
        hits = int(np.sum(np.abs(null) >= abs(observed)))
    elif method == "greater":
        hits = int(np.sum(null >= observed))
    else:
        raise ValueError(f"unknown permutation method {method!r}")
    return float((1 + hits) / (1 + n_perm))


@dataclasses.dataclass(frozen=True)
class UncertaintyGeometry:
    """The per-feature uncertainty-geometry result on the held-out ambiguous middle."""

    feature: str
    spearman: float
    pearson: float
    pvalue: float
    n_fit: int
    n_middle: int
    distances: np.ndarray
    fractions: np.ndarray


def uncertainty_geometry(
    matrix: EmbeddingMatrix,
    labels: LabelProvider,
    feature: str,
    ids: Sequence[int],
    *,
    low: float = 0.2,
    high: float = 0.8,
    c: float = 1.0,
    n_perm: int = 10_000,
    method: str = "two_sided",
    seed: int = 0,
) -> UncertaintyGeometry:
    """Fit the axis on the extremes, project the middle, rank-correlate with the vote fraction.

    Reuses the ``data/splits.py`` firewall: the fit set is the consensus extremes only and
    ``assert_uncertainty_firewall`` raises if any ambiguous-middle fraction leaked into it —
    so the non-circularity is enforced structurally, not by convention.
    """
    present = [int(o) for o in ids if int(o) in matrix.index]
    fractions: dict[object, float] = {
        o: float(labels.vote_fraction(feature, [o])[0]) for o in present
    }
    fit_ids, middle_ids = partition_uncertainty(fractions, low=low, high=high)

    fit_list = [o for o in present if o in fit_ids]
    middle_list = [o for o in present if o in middle_ids]
    assert_uncertainty_firewall([fractions[o] for o in fit_list], low=low, high=high)

    rows_fit = matrix.rows_for(fit_list)
    y_fit = np.asarray([featured_label(fractions[o]) for o in fit_list], dtype=np.int64)
    fit_emb = Embeddings(matrix.x[rows_fit], y_fit, np.asarray([fractions[o] for o in fit_list]))
    direction = probe_direction(fit_emb, name=feature, c=c)

    rows_mid = matrix.rows_for(middle_list)
    distances = project(matrix.x[rows_mid], direction.w_unit)
    mid_fractions = np.asarray([fractions[o] for o in middle_list], dtype=np.float64)

    if len(middle_list) < 3 or np.unique(mid_fractions).size < 2:
        # too few ambiguous galaxies to rank — an honest null, not an error
        return UncertaintyGeometry(
            feature,
            float("nan"),
            float("nan"),
            1.0,
            len(fit_list),
            len(middle_list),
            distances,
            mid_fractions,
        )

    return UncertaintyGeometry(
        feature=feature,
        spearman=spearman(distances, mid_fractions),
        pearson=pearson(distances, mid_fractions),
        pvalue=permutation_p(distances, mid_fractions, n_perm=n_perm, method=method, seed=seed),
        n_fit=len(fit_list),
        n_middle=len(middle_list),
        distances=distances,
        fractions=mid_fractions,
    )
