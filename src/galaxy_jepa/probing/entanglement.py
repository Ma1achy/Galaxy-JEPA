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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from galaxy_jepa.callbacks.collapse import effective_rank
from galaxy_jepa.probing.logistic import ConceptDirection, Embeddings

__all__ = [
    "stack_directions",
    "cosine_matrix",
    "covariance_alignment",
    "gram_eigenspectrum",
    "embedding_covariance_spectrum",
    "cav_direction",
    "logistic_cav_disagreement",
    "MPVerdict",
    "mp_significant",
    "EmpiricalGramNull",
    "empirical_gram_null",
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
    """The k×k raw-coordinate cosine matrix between concept directions (``WWᵀ``) — Figure 3.

    Note: Euclidean cosine is coordinate-specific (not invariant under arbitrary invertible
    reparameterisations ``Z' = A Z, w' = A^{-T} w``). Pair with :func:`covariance_alignment`
    for a reparameterisation-invariant alignment on a shared embedding population.
    """
    names, w = stack_directions(directions)
    return names, w @ w.T


def covariance_alignment(
    directions: Sequence[ConceptDirection],
    embedding_x: np.ndarray,
    *,
    use_raw: bool = True,
    eps: float = 1e-12,
) -> tuple[list[str], np.ndarray]:
    """Covariance-weighted direction alignment ``c_Σ(w_i, w_j)`` on a shared population.

    Computes::

        c_Σ(w_i, w_j) = (w_i^T Σ w_j) / sqrt((w_i^T Σ w_i)(w_j^T Σ w_j))

    where ``Σ`` is the sample covariance of ``embedding_x``. Equals the Pearson correlation of
    the linear scores ``Z w_i`` and ``Z w_j`` on ``embedding_x``, and is invariant under
    consistent invertible coordinate transformations ``Z' = A Z``, ``w' = A^{-T} w``.
    Entries with zero/degenerate score variance (``w_i^T Σ w_i <= eps``) return ``NaN``.
    """
    if not directions:
        raise ValueError("need at least one concept direction to compute covariance alignment")
    x = np.asarray(embedding_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"embedding_x must be 2-D with >= 2 rows, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("covariance_alignment requires finite embedding_x")
    names = [d.name for d in directions]
    w = np.stack(
        [np.asarray(d.w_raw if use_raw else d.w_unit, dtype=np.float64) for d in directions]
    )
    if w.shape[1] != x.shape[1]:
        raise ValueError(f"direction width {w.shape[1]} != embedding width {x.shape[1]}")
    xc = x - x.mean(axis=0, keepdims=True)
    # Sample covariance Σ = X_c^T X_c / (n - 1); scaling cancels in c_Σ
    sigma = (xc.T @ xc) / float(x.shape[0] - 1)
    gram = w @ sigma @ w.T
    diag = np.diag(gram)
    k = len(names)
    out = np.full((k, k), np.nan, dtype=np.float64)
    valid = diag > eps
    if np.any(valid):
        inv_sd = np.zeros(k, dtype=np.float64)
        inv_sd[valid] = 1.0 / np.sqrt(diag[valid])
        corr = gram * np.outer(inv_sd, inv_sd)
        mask2d = np.outer(valid, valid)
        out[mask2d] = np.clip(corr[mask2d], -1.0, 1.0)
        for i in range(k):
            if valid[i]:
                out[i, i] = 1.0
    return names, out


def gram_eigenspectrum(w: np.ndarray) -> tuple[np.ndarray, float]:
    """``(eigenvalues, effective_rank)`` of the concept-direction Gram, from ``W``'s SVD.

    The Gram ``WWᵀ`` eigenvalues are the squared singular values of ``W``; the effective rank
    reuses the collapse-monitor kernel on the singular values (the same definition the
    pretraining monitor reports), so the two read the spectrum identically.
    """
    svals = torch.linalg.svdvals(torch.as_tensor(w, dtype=torch.float64))
    eigenvalues = (svals**2).numpy()
    return eigenvalues, effective_rank(svals)


def gram_eigenvectors(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(eigenvalues, eigenvectors)`` of ``WWᵀ``, descending — spine item 3, the localiser.

    The eigen*values* say "I named k concepts but they span only k_eff dimensions"; they cannot
    say **which** concepts collapsed together. The eigen*vectors* can: a component loading on
    both bar and bulge is the design's own worked example. Columns of the returned matrix are the
    eigenvectors, so ``vectors[:, j]`` is component ``j``'s loading over ``names``.
    """
    gram = torch.as_tensor(w, dtype=torch.float64)
    gram = gram @ gram.T
    evals, evecs = torch.linalg.eigh(gram)  # ascending, orthonormal
    order = torch.argsort(evals, descending=True)
    return evals[order].numpy(), evecs[:, order].numpy()


def component_loadings(
    names: Sequence[str], vectors: np.ndarray, *, component: int = 0, top: int = 4
) -> list[tuple[str, float]]:
    """The features loading most strongly on one component — ``(name, signed loading)``.

    Sorted by |loading|, so a component shared by two concepts reads off directly. The sign is
    kept: two features loading with *opposite* sign on a shared axis is a different statement
    from two loading together.
    """
    col = np.asarray(vectors)[:, component]
    order = np.argsort(-np.abs(col))[:top]
    return [(names[i], float(col[i])) for i in order]


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


@dataclasses.dataclass(frozen=True)
class EmpiricalGramNull:
    """Empirical null distribution for the unit-mean-normalised Gram top eigenvalue.

    Unlike the analytic Marchenko–Pastur reference (which assumes isotropic i.i.d. random
    directions), this runs the actual linear-probe fitting pipeline on label-permuted data
    over the real embedding matrix, capturing anisotropy, regularisation, and sample size.
    """

    observed_top: float
    null_tops: np.ndarray
    pvalue: float
    n_draws: int


def empirical_gram_null(
    embedding_x: np.ndarray,
    labels_by_feature: Mapping[str, np.ndarray],
    *,
    observed_top: float,
    c: float = 1.0,
    n_draws: int = 50,
    seed: int = 0,
) -> EmpiricalGramNull:
    """Shuffled-label empirical null for the unit-mean-normalised Gram top eigenvalue.

    Permutes each feature's binary labels independently (testing the null of no feature
    structure while preserving embedding covariance, prevalence, and L2 regularisation), fits
    standardised logistic directions via ``probe_direction``, and records the top unit-mean
    Gram eigenvalue on each draw, returning the add-one p-value ``(1 + count >= obs) / (B + 1)``.
    """
    from galaxy_jepa.probing.logistic import probe_direction

    x = np.asarray(embedding_x, dtype=np.float64)
    feats = list(labels_by_feature)
    if len(feats) < 2:
        raise ValueError("empirical_gram_null requires at least 2 features")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws!r}")

    rng = np.random.default_rng(seed)
    null_tops = np.empty(n_draws, dtype=np.float64)
    zero_frac = np.zeros(x.shape[0], dtype=np.float64)
    for b in range(n_draws):
        dirs = []
        for f in feats:
            y = np.asarray(labels_by_feature[f], dtype=np.int64)
            yp = rng.permutation(y)
            dirs.append(probe_direction(Embeddings(x, yp, zero_frac), name=f, c=c))
        _, w = stack_directions(dirs)
        ev, _ = gram_eigenspectrum(w)
        ev_norm = ev / max(float(ev.mean()), 1e-12)
        null_tops[b] = float(ev_norm.max())
    pval = float((1 + int(np.sum(null_tops >= observed_top))) / (n_draws + 1))
    return EmpiricalGramNull(
        observed_top=float(observed_top),
        null_tops=null_tops,
        pvalue=pval,
        n_draws=n_draws,
    )


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
    #: Columns are eigenvectors of ``WWᵀ``, descending — ``gram_eigenvectors[:, j]`` is
    #: component ``j``'s loading over ``names``. The measure that localises *which* features
    #: collapse together, as opposed to merely *that* they do.
    gram_eigenvectors: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0, 0)))
    #: ``1 − |cos|`` between each feature's logistic and CAV directions. Genuinely independent of
    #: the Gram: a different *definition* of the concept axis, not a different statistic on the
    #: same one.
    cav_disagreement: Mapping[str, float] = dataclasses.field(default_factory=dict)

    @property
    def span_ratio(self) -> float:
        """Concept span : total embedding dimensionality — "k concepts occupy 7 of ~40 dims".

        The eigen spine's fourth item. Without it the concept effective rank has no scale: 7 is
        crowded in a 40-dimensional representation and roomy in a 700-dimensional one.
        """
        return (
            self.gram_effective_rank / self.embedding_effective_rank
            if self.embedding_effective_rank
            else 0.0
        )


def entanglement_geometry(
    directions: Sequence[ConceptDirection],
    embedding_x: np.ndarray,
    *,
    mp_method: str = "upper_edge",
    pair_quantile: float = 0.90,
    cav_disagreement: Mapping[str, float] | None = None,
) -> EntanglementGeometry:
    """Assemble the global entanglement geometry over the existence-passing features."""
    names, w = stack_directions(directions)
    cos = w @ w.T
    gram_ev, gram_er = gram_eigenspectrum(w)
    _ev, gram_vecs = gram_eigenvectors(w)
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
        gram_eigenvectors=gram_vecs,
        cav_disagreement=dict(cav_disagreement or {}),
    )


#: The 2A verdicts. Fixed as a closed set before any result, because "many measures" is a licence
#: to triangulate and never a licence to pick per feature.
REPRESENTATIONAL = "representational_entanglement"
WORLD_CORRELATION = "world_correlation"
CLEAN = "clean"
INCONCLUSIVE = "inconclusive"


@dataclasses.dataclass(frozen=True)
class PairVerdict:
    """One flagged pair's adjudication, with the four inputs that produced it kept visible."""

    a: str
    b: str
    verdict: str
    cosine: float
    mp_significant: bool
    cav_disagree: bool | None
    survived_matching: bool | None
    reason: str
    #: The conditional leg's D24 retention state and what it was read from (D25); ``None`` on a
    #: verdict built outside the ladder. PARTIAL and UNRESOLVED arrive as survived_matching=None.
    retention: str | None = None
    retained: float | None = None
    matched_auc: float | None = None
    bar_matched: float | None = None
    n_matched_test: int = 0


def adjudicate_pair(
    a: str,
    b: str,
    *,
    cosine: float,
    mp_significant: bool,
    cav_disagreement: Mapping[str, float],
    survived_matching: bool | None,
    cosine_floor: float = 0.30,
    cav_floor: float = 0.10,
) -> PairVerdict:
    """The **one** pre-registered function from (eigen, cosine, CAV, conditional) to a verdict.

    Design 2A, `docs/probing-harness-design.md:294-309`. These four inputs are convergent
    diagnostics (not an identification theorem): fixing the mapping *before* results prevents
    per-feature cherry-picking across measures with different blind spots.

    * **eigen entangled + cosine shows it + logistic/CAV disagree + survives matching** →
      ``representational_entanglement`` (strongest convergent diagnostic branch).
    * **eigen entangled but the conditional test vanishes under matching** →
      ``world_correlation`` (consistent with co-occurrence in the sky rather than shared encoding;
      interpret alongside support, sample size, and label/nuisance noise).
    * **eigen clean + CAV measured and agrees + cosine low** → ``clean``, converging on R1.
    * anything else (including missing CAV evidence or ``survived_matching=None``) →
      ``inconclusive``, reported as such rather than averaged or defaulted.
    """
    raw_a = cav_disagreement.get(a)
    raw_b = cav_disagreement.get(b)
    if (
        raw_a is None
        or raw_b is None
        or not (np.isfinite(float(raw_a)) and np.isfinite(float(raw_b)))
    ):
        disagree: bool | None = None
    else:
        disagree = max(float(raw_a), float(raw_b)) >= cav_floor
    high_cos = bool(np.isfinite(cosine) and abs(cosine) >= cosine_floor)

    if mp_significant and survived_matching is True and high_cos and disagree is True:
        return PairVerdict(
            a,
            b,
            REPRESENTATIONAL,
            cosine,
            mp_significant,
            disagree,
            survived_matching,
            "eigen entangled, cosine shows it, logistic/CAV disagree, and the "
            "association survives matching on the partner — all four diagnostics converge",
        )
    if mp_significant and survived_matching is False:
        return PairVerdict(
            a,
            b,
            WORLD_CORRELATION,
            cosine,
            mp_significant,
            disagree,
            survived_matching,
            "eigen entangled but the association collapses under matching on the partner — "
            "consistent with sky co-occurrence rather than shared encoding "
            "(subject to support and power)",
        )
    if not mp_significant and disagree is False and not high_cos and np.isfinite(cosine):
        return PairVerdict(
            a,
            b,
            CLEAN,
            cosine,
            mp_significant,
            disagree,
            survived_matching,
            "spectrum within the MP null, logistic and CAV agree, cosine low",
        )
    missing_note = "; CAV evidence absent" if disagree is None else ""
    return PairVerdict(
        a,
        b,
        INCONCLUSIVE,
        cosine,
        mp_significant,
        disagree,
        survived_matching,
        f"the diagnostics do not converge{missing_note}; reported rather than averaged",
    )


def human_vote_correlation(
    labels: Any, features: Sequence[str], ids: Sequence[int], *, min_overlap: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """``(correlation, n_overlap)`` between the **human vote fractions** of every feature pair.

    The same-corpus counterpart to the embedding cosine matrix, and the reason Figure 3 does not
    need v1's Figs 18-19. v1's confusion matrix was computed on the PyPI ``galaxy-datasets``
    release, not this pull — the same mismatch that stopped the 21-vote threshold transferring —
    so laying it against an embedding matrix built here would mix **two** differences at once,
    dataset and representation, and neither could be read off the result.

    Computed on the galaxies already in hand, the comparison is clean: same objects, same votes,
    only the representation differs. v1's figure then serves as a *continuity reference*, for
    which the three prose constants are enough.

    **Pairwise-complete, not listwise.** Each answer has its own eligible population (a question
    is only asked of galaxies that reached it), so a pair's correlation is computed over the
    intersection of the two eligible sets. ``n_overlap`` carries that count per pair, because a
    correlation over 200 galaxies and one over 60,000 are not the same measurement, and a matrix
    that hides the difference invites reading the thin cells as hard as the thick ones. Pairs
    below ``min_overlap`` return NaN rather than a number nobody should use.
    """
    fractions: dict[str, dict[int, float]] = {}
    for f in features:
        eligible = labels.eligible(f, ids)
        vals = labels.vote_fraction(f, eligible)
        fractions[f] = {
            int(o): float(v) for o, v in zip(eligible, vals, strict=True) if np.isfinite(v)
        }

    k = len(features)
    corr = np.full((k, k), np.nan, dtype=np.float64)
    overlap = np.zeros((k, k), dtype=np.int64)
    for i, a in enumerate(features):
        corr[i, i], overlap[i, i] = 1.0, len(fractions[a])
        for j in range(i + 1, k):
            b = features[j]
            shared = sorted(fractions[a].keys() & fractions[b].keys())
            overlap[i, j] = overlap[j, i] = len(shared)
            if len(shared) < min_overlap:
                continue
            va = np.array([fractions[a][o] for o in shared])
            vb = np.array([fractions[b][o] for o in shared])
            if va.std() == 0 or vb.std() == 0:  # a constant column has no correlation
                continue
            corr[i, j] = corr[j, i] = float(np.corrcoef(va, vb)[0, 1])
    return corr, overlap


@dataclasses.dataclass(frozen=True)
class MatrixAgreement:
    """How the embedding-recovered structure compares to the human vote structure."""

    spearman: float
    n_pairs: int
    #: ``(a, b, cosine, human_corr, delta)`` — the pairs where the two disagree most, signed.
    largest_disagreements: list[tuple[str, str, float, float, float]]


def compare_to_human_structure(
    names: Sequence[str],
    cosine: np.ndarray,
    human_corr: np.ndarray,
    *,
    top: int = 8,
) -> MatrixAgreement:
    """Rank-correlate the embedding cosine matrix against the human vote-correlation matrix.

    Spearman rather than Pearson: the two matrices are not on a common scale (a cosine between
    unit directions and a correlation between vote fractions measure different things), so what
    transfers is the **ordering** of which pairs are close, not the magnitudes.

    The disagreements are the point, not the agreement. A pair the encoder places together that
    the votes do not is a candidate representational entanglement; a pair the votes place
    together that the encoder separates is the encoder doing better than the labels.
    """
    from scipy.stats import spearmanr

    rows: list[tuple[str, str, float, float, float]] = []
    xs: list[float] = []
    ys: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            h = human_corr[i, j]
            if not np.isfinite(h):
                continue
            c = float(cosine[i, j])
            xs.append(c)
            ys.append(float(h))
            rows.append((names[i], names[j], c, float(h), c - float(h)))
    if len(xs) < 3:
        return MatrixAgreement(float("nan"), len(xs), [])
    rho = float(spearmanr(xs, ys).statistic)
    rows.sort(key=lambda r: -abs(r[4]))
    return MatrixAgreement(rho, len(xs), rows[:top])


#: D13 confound-3's directional anchor. Hart et al. measured spiral arms in strongly barred
#: galaxies as roughly 4-6 degrees LOOSER than in unbarred ones. That makes the bar+arms hard case
#: a *directional* prediction rather than a judgement call: if the encoder's bar direction is
#: tracking physics it should lean towards loose winding specifically, and if it is tracking
#: confident classification it should lean towards every spiral answer about equally.
BAR_FEATURE = "t03_bar_a06_bar"
WINDING_ORDER = (
    "t10_arms_winding_a28_tight",
    "t10_arms_winding_a29_medium",
    "t10_arms_winding_a30_loose",
)

PHYSICS = "physics_consistent"
BLEED = "uniform_bleed"
CONTRARY = "contrary_to_literature"
UNAVAILABLE = "unavailable"


@dataclasses.dataclass(frozen=True)
class BarWindingAlignment:
    """The Hart-anchored read on D13's hard case: is bar+arms physics, or classification bleed?"""

    verdict: str
    cosines: dict[str, float]
    spread: float
    reason: str


def bar_winding_alignment(
    names: Sequence[str], cosine: np.ndarray, *, separation: float = 0.05
) -> BarWindingAlignment:
    """Does the bar direction lean towards LOOSE winding, or towards all winding equally?

    Stage two of D13's adjudication, and only meaningful after stage one (the 2A conditional
    cross-check) has said the bar+arms association is a real correlation in the data rather than
    a representational artefact. This stage asks what *kind* of real correlation it is.

    * loose above tight (and medium between them, when medium is present), by at least
      ``separation`` end to end → ``physics_consistent``: the ordering Hart et al. predict, so
      the direction is tracking a property of the galaxies.
    * all within ``separation`` → ``uniform_bleed``: the bar direction is equally close to every
      spiral answer, which is the confident-classification bleed the hard case warns about — a
      galaxy confidently called barred is a galaxy confidently called everything.
    * ordered the other way → ``contrary_to_literature``, reported as-is rather than explained.

    **Medium winding is not required.** Hart's prediction is a statement about the tight-to-loose
    axis, and the middle answer is the one most likely to fail existence — it is the least
    separable of the three, and did fail in Brief P. Refusing the whole test because the *middle*
    point is missing would discard a reading the two endpoints fully support. When medium is
    absent the verdict is reached on tight-versus-loose alone and the reason says so; when it is
    present it must sit between them for ``physics_consistent``.

    ``separation`` is a **declared** threshold, not a derived one.
    """
    index = {n: i for i, n in enumerate(names)}
    tight_n, medium_n, loose_n = WINDING_ORDER
    missing = [n for n in (BAR_FEATURE, tight_n, loose_n) if n not in index]
    if missing:
        return BarWindingAlignment(
            UNAVAILABLE,
            {},
            0.0,
            f"the hard case needs the bar and both winding endpoints; {', '.join(missing)} did "
            f"not reach the entanglement set (existence-passing features only), so it cannot be "
            f"read on this run",
        )
    present = [w for w in WINDING_ORDER if w in index]
    cos = {w: float(cosine[index[BAR_FEATURE], index[w]]) for w in present}
    tight, loose = cos[tight_n], cos[loose_n]
    medium = cos.get(medium_n)
    spread = max(cos.values()) - min(cos.values())
    note = (
        ""
        if medium is not None
        else " (medium winding failed existence, so this reads the tight-to-loose axis only)"
    )

    if spread < separation:
        return BarWindingAlignment(
            BLEED,
            cos,
            spread,
            f"the bar direction sits within {spread:.3f} of every winding answer available; Hart "
            f"predicts a lean towards loose, and equal alignment is the classification bleed "
            f"D13's hard case warns about{note}",
        )
    if loose > tight and (medium is None or loose > medium > tight):
        middle = "" if medium is None else f"medium {medium:.3f} > "
        return BarWindingAlignment(
            PHYSICS,
            cos,
            spread,
            f"loose {loose:.3f} > {middle}tight {tight:.3f}, spread {spread:.3f} — the ordering "
            f"Hart et al. predict for barred galaxies, so the association is tracking a property "
            f"of the galaxies rather than of the labelling{note}",
        )
    middle = "" if medium is None else f"medium {medium:.3f} / "
    return BarWindingAlignment(
        CONTRARY,
        cos,
        spread,
        f"tight {tight:.3f} / {middle}loose {loose:.3f} — separated but not in the predicted "
        f"order; reported as measured rather than explained{note}",
    )
