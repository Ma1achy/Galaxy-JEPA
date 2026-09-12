"""Entanglement (R1 vs R2) — the eigen-led triangulation (design 2A) **[LOCKED — strong]**.

Several independent measures with *different* failure modes, triangulated: where they agree,
agreement can't be an artefact of any single method's blind spot. This module computes the
buildable spine; the verdict logic (which the ladder applies) is pre-registered.

The eigen spine:

* **Effective rank of the concept-direction Gram** (``WWᵀ``, k features × directions) — the
  global "k named concepts span k_eff effective dimensions". *Reuses the collapse-monitor
  ``effective_rank`` kernel* (now a shared helper).
* **Marchenko–Pastur null** on that spectrum — "significantly more entangled than random
  directions, by random-matrix theory". Grounded (decision 5): past the MP edge computed for
  the **actual** matrix shape.
* **Eigenvectors of ``WWᵀ``** — *which* features collapse onto shared axes (localises it).
* **Embedding-covariance spectrum** — the encoder's intrinsic effective dimensionality, as
  context (the "k concepts occupy 7 of ~40 dims" ratio).

Cross-checks (each covers an eigen blind spot):

* **Cosine matrix** — the ``WWᵀ`` off-diagonals, human-readable; the bridge to v1's confusion
  matrices (Figure 3).
* **Logistic-vs-CAV direction disagreement** — discriminative vs marginal definition of the
  direction; genuinely independent of the Gram analysis.
* **Conditional recoverability (matched)** — fired surgically on the eigen-flagged pairs
  (``matching.py``); the only measure that resolves representation-vs-world.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from galaxy_jepa.callbacks.collapse import effective_rank
from galaxy_jepa.probing.logistic import ConceptDirection, Embeddings

__all__ = [
    "stack_directions",
    "cosine_matrix",
    "gram_eigenspectrum",
    "embedding_covariance_spectrum",
    "cav_direction",
    "logistic_cav_disagreement",
    "MPVerdict",
    "mp_significant",
    "most_entangled_pairs",
    "EntanglementGeometry",
    "entanglement_geometry",
]


def stack_directions(directions: Sequence[ConceptDirection]) -> tuple[list[str], np.ndarray]:
    """Stack concept directions' unit vectors into ``(names, W)`` with ``W`` of shape (k, D)."""
    if not directions:
        raise ValueError("need at least one concept direction to analyse entanglement")
    names = [d.name for d in directions]
    w = np.stack([d.w_unit for d in directions])
    return names, w


def cosine_matrix(directions: Sequence[ConceptDirection]) -> tuple[list[str], np.ndarray]:
    """The k×k cosine matrix between concept directions (``WWᵀ`` for unit rows) — Figure 3."""
    names, w = stack_directions(directions)
    return names, w @ w.T


def gram_eigenspectrum(w: np.ndarray) -> tuple[np.ndarray, float]:
    """``(eigenvalues, effective_rank)`` of the concept-direction Gram, from ``W``'s SVD.

    The Gram ``WWᵀ`` eigenvalues are the squared singular values of ``W``; the effective rank
    reuses the collapse-monitor kernel on the singular values (the same definition the
    pretraining monitor reports), so the two read the spectrum identically.
    """
    svals = torch.linalg.svdvals(torch.as_tensor(w, dtype=torch.float64))
    eigenvalues = (svals**2).numpy()
    return eigenvalues, effective_rank(svals)


def embedding_covariance_spectrum(x: np.ndarray) -> tuple[np.ndarray, float]:
    """``(eigenvalues, effective_rank)`` of the embedding covariance — the encoder's intrinsic
    effective dimensionality (context for the concept-span : total-dimensionality ratio)."""
    xc = torch.as_tensor(x, dtype=torch.float64)
    xc = xc - xc.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(xc)
    return (svals**2).numpy(), effective_rank(svals)


def cav_direction(train: Embeddings) -> np.ndarray:
    """The CAV (concept activation vector): the *marginal* mean-difference direction.

    ``mean(positives) − mean(negatives)``, unit-normed — a different *definition* of the
    concept axis from the logistic probe's discriminative direction. Disagreement between the
    two flags interference from other features (design 2A cross-check).
    """
    pos = train.x[train.y == 1].mean(axis=0)
    neg = train.x[train.y == 0].mean(axis=0)
    diff = pos - neg
    norm = float(np.linalg.norm(diff)) or 1.0
    return diff / norm


def logistic_cav_disagreement(logistic_unit: np.ndarray, cav_unit: np.ndarray) -> float:
    """``1 − |cosine|`` between the discriminative and marginal directions (0 = agree)."""
    cos = float(np.dot(logistic_unit, cav_unit))
    return 1.0 - abs(cos)


@dataclasses.dataclass(frozen=True)
class MPVerdict:
    """Whether the Gram spectrum is more concentrated than a random-direction (MP) null."""

    top_eigenvalue: float
    mp_edge: float
    significant: bool


def mp_significant(
    eigenvalues: np.ndarray, *, n_directions: int, n_dims: int, method: str = "upper_edge"
) -> MPVerdict:
    """Marchenko–Pastur significance on the Gram eigenspectrum — decision (5), grounded.

    An eigenvalue counts as signal if it sits **past the MP edge computed for the actual matrix
    shape**. For ``k`` random unit directions in ``D`` dims the Gram spectrum follows MP with
    aspect ratio ``γ = k/D``, whose bulk upper edge is ``λ₊ = (1 + √γ)²`` after unit-mean
    normalisation; a top eigenvalue above it is concentration beyond chance.

    "Actual, not nominal" is the operative part and is enforced below: ``n_directions`` must
    equal the number of eigenvalues supplied. Passing a nominal k (the scheme's feature count,
    say, rather than the count that actually survived to the Gram) would move the edge and
    silently change the verdict, which is precisely the failure the decision names.

    ``tracy_widom`` stays unimplemented on purpose: it is the "if a reviewer wants a hard
    significance value" upgrade, not the grounded decision, and raises rather than defaulting.
    """
    if method != "upper_edge":
        raise NotImplementedError(
            f"MP method {method!r} is not implemented. 'upper_edge' is the grounded decision "
            "(spec §Statistics (5)); Tracy–Widom is a documented later upgrade."
        )
    ev = np.asarray(eigenvalues, dtype=np.float64)
    if n_directions != ev.size:
        raise ValueError(
            f"MP edge must be computed for the actual matrix shape: got {ev.size} eigenvalues "
            f"but n_directions={n_directions}. Pass the shape of the Gram actually formed."
        )
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}")
    ev = ev / max(float(ev.mean()), 1e-12)  # unit-mean normalisation
    gamma = n_directions / n_dims
    edge = (1.0 + np.sqrt(gamma)) ** 2
    top = float(ev.max())
    # cast: a numpy bool leaks into the stamped JSON summary and compares oddly downstream
    return MPVerdict(top_eigenvalue=top, mp_edge=float(edge), significant=bool(top > edge))


def most_entangled_pairs(
    names: list[str], cosine: np.ndarray, *, quantile: float = 0.90
) -> list[tuple[str, str]]:
    """The feature-pairs whose |cosine| is in the top ``quantile`` — the conditional-test set.

    FLAGGED trigger: pending stats grounding. The surgical conditional recoverability test
    (``matching.py``) fires only on these, so it stays cheap; the quantile is the placeholder
    cutoff for "most entangled".
    """
    k = len(names)
    off = [(i, j, abs(float(cosine[i, j]))) for i in range(k) for j in range(i + 1, k)]
    if not off:
        return []
    cutoff = float(np.quantile([c for *_, c in off], quantile))
    return [(names[i], names[j]) for i, j, c in off if c >= cutoff]


@dataclasses.dataclass(frozen=True)
class EntanglementGeometry:
    """The full eigen spine + cross-checks for the existence-passing features (Figure 3)."""

    names: list[str]
    cosine: np.ndarray
    gram_eigenvalues: np.ndarray
    gram_effective_rank: float
    embedding_eigenvalues: np.ndarray
    embedding_effective_rank: float
    mp: MPVerdict
    entangled_pairs: list[tuple[str, str]]


def entanglement_geometry(
    directions: Sequence[ConceptDirection],
    embedding_x: np.ndarray,
    *,
    mp_method: str = "upper_edge",
    pair_quantile: float = 0.90,
) -> EntanglementGeometry:
    """Assemble the global entanglement geometry over the existence-passing features."""
    names, w = stack_directions(directions)
    cos = w @ w.T
    gram_ev, gram_er = gram_eigenspectrum(w)
    emb_ev, emb_er = embedding_covariance_spectrum(embedding_x)
    mp = mp_significant(gram_ev, n_directions=len(names), n_dims=w.shape[1], method=mp_method)
    pairs = most_entangled_pairs(names, cos, quantile=pair_quantile)
    return EntanglementGeometry(
        names=names,
        cosine=cos,
        gram_eigenvalues=gram_ev,
        gram_effective_rank=gram_er,
        embedding_eigenvalues=emb_ev,
        embedding_effective_rank=emb_er,
        mp=mp,
        entangled_pairs=pairs,
    )
