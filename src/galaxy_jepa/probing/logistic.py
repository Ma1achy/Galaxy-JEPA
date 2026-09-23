"""Frozen-encoder logistic probe — the one headline number (docs/spec/encoder.md, slice plan).

The canonical "linearly nameable" read-out: an L2-regularised logistic regression on the
**frozen** encoder's mean-pooled penultimate-layer embeddings (``Encoder.encode``). For the
vertical slice this is the whole measurement — train on the ``probe-train`` confident
extremes, report ROC-AUC on the ``probe-test`` confident extremes (the label drowns less in
the ambiguous middle that way; ``data/metadata.is_confident_extreme``).

The probe **standardises** the embedding features (``StandardScaler``) before the logistic
fit: it converges in a few hundred iterations rather than hitting the iteration cap on the
raw, differently-scaled embedding axes, and the converged AUC is the defensible figure
(:func:`probe_auc_ci` attaches a bootstrap confidence interval so the headline is stated
honestly, not as a bare point estimate).

The encoder is asserted **frozen** on entry (``assert_frozen``): a still-trainable encoder
fails loudly rather than letting the probe's gradients bend the representation. This module
consumes a ``models`` encoder + a checkpoint — it never imports ``objectives`` (the freeze
boundary runs through disk; ``docs/spec/objectives.md`` §3).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from galaxy_jepa.core.encoder import Encoder, assert_frozen

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

__all__ = [
    "Embeddings",
    "extract_embeddings",
    "probe_auc",
    "probe_auc_ci",
    "probe_auc_ci_se",
    "probe_scores",
    "weighted_auc",
    "paired_auc_bootstrap",
    "probe_direction",
    "ConceptDirection",
    "ProbeResult",
    "run_probe",
    "DEFAULT_MAX_ITER",
    "EXTREME_LOW",
    "EXTREME_HIGH",
]

# Converged probe defaults — standardised features reach the optimum well inside this cap
# (the slice's lbfgs hit max_iter=2000 unconverged on raw features; ~350 iters converge here).
DEFAULT_MAX_ITER = 20_000
EXTREME_LOW = 0.2
EXTREME_HIGH = 0.8


@dataclasses.dataclass(frozen=True)
class Embeddings:
    """Frozen-encoder features + labels for one split."""

    x: np.ndarray  # (N, D)
    y: np.ndarray  # (N,) binary
    fraction: np.ndarray  # (N,) the GZ2 featured vote fraction (for the extremes filter)


@torch.no_grad()
def extract_embeddings(
    encoder: Encoder,
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> Embeddings:
    """Run the frozen encoder over ``dataset`` → pooled embeddings + labels (requires labels)."""
    assert_frozen(encoder)  # the probing freeze boundary — loud failure if trainable
    cast(torch.nn.Module, encoder).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    xs, ys, fracs = [], [], []
    for batch in loader:
        if "label" not in batch:
            raise ValueError("probe dataset must carry labels (label_fraction_col set)")
        emb = encoder.encode(batch["image"].float().to(device))
        xs.append(emb.cpu().numpy())
        ys.append(np.asarray(batch["label"]))
        fracs.append(np.asarray(batch["featured_fraction"]))
    return Embeddings(np.concatenate(xs), np.concatenate(ys), np.concatenate(fracs))


def _require_two_classes(train: Embeddings, test: Embeddings) -> None:
    if len(np.unique(train.y)) < 2:
        raise ValueError("probe training set has a single class — cannot fit a logistic axis")
    if len(np.unique(test.y)) < 2:
        raise ValueError("probe test set has a single class — AUC undefined")


def _fit(
    train: Embeddings, *, c: float = 1.0, max_iter: int = DEFAULT_MAX_ITER
) -> tuple[StandardScaler, LogisticRegression]:
    """Standardise on the train split, fit the L2-logistic probe. The one fit path."""
    from sklearn.linear_model import LogisticRegression  # lazy: the eval extra, not the gate
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train.x)
    clf = LogisticRegression(C=c, max_iter=max_iter)  # L2 is the default penalty
    clf.fit(scaler.transform(train.x), train.y)
    return scaler, clf


def probe_auc(
    train: Embeddings, test: Embeddings, *, c: float = 1.0, max_iter: int = DEFAULT_MAX_ITER
) -> float:
    """Fit the standardised L2-logistic probe on ``train``, return ROC-AUC on ``test``."""
    _require_two_classes(train, test)
    from sklearn.metrics import roc_auc_score

    scaler, clf = _fit(train, c=c, max_iter=max_iter)
    scores = clf.predict_proba(scaler.transform(test.x))[:, 1]
    return float(roc_auc_score(test.y, scores))


def probe_scores(
    train: Embeddings, test: Embeddings, *, c: float = 1.0, max_iter: int = DEFAULT_MAX_ITER
) -> np.ndarray:
    """The fitted probe's positive-class probability on every ``test`` row.

    What a *paired* comparison needs: two probes are only compared fairly when their scores are
    resampled together, galaxy by galaxy, rather than each AUC carrying its own interval.
    """
    _require_two_classes(train, test)
    scaler, clf = _fit(train, c=c, max_iter=max_iter)
    return np.asarray(clf.predict_proba(scaler.transform(test.x))[:, 1], dtype=np.float64)


class _SortedScores:
    """One score vector sorted once, so every bootstrap draw's AUC is O(n) rather than O(n log n).

    A bootstrap resample is a vector of multinomial counts over the fixed test galaxies, and the
    scores do not change between draws — so the sort, and the grouping of tied scores, can be done
    once and each draw reduced to weighted sums.
    """

    def __init__(self, y: np.ndarray, scores: np.ndarray) -> None:
        order = np.argsort(scores, kind="mergesort")
        self.order = order
        self.pos = np.asarray(y, dtype=np.float64)[order]
        _, self.group = np.unique(np.asarray(scores)[order], return_inverse=True)
        self.n_groups = int(self.group.max()) + 1 if self.group.size else 0

    def auc(self, w: np.ndarray) -> float:
        """Mann–Whitney AUC with each galaxy weighted ``w``; ties score one half."""
        ws = w[self.order]
        wp = ws * self.pos
        wn = ws - wp
        gp = np.bincount(self.group, weights=wp, minlength=self.n_groups)
        gn = np.bincount(self.group, weights=wn, minlength=self.n_groups)
        tp, tn = gp.sum(), gn.sum()
        if tp == 0 or tn == 0:
            return float("nan")
        below = np.cumsum(gn) - gn
        return float((gp * (below + 0.5 * gn)).sum() / (tp * tn))


def weighted_auc(y: np.ndarray, scores: np.ndarray, w: np.ndarray | None = None) -> float:
    """ROC-AUC with per-sample weights (all ones reproduces the unweighted AUC exactly)."""
    y = np.asarray(y)
    return _SortedScores(y, scores).auc(np.ones(len(y)) if w is None else np.asarray(w, float))


def paired_auc_bootstrap(
    labels: list[np.ndarray],
    scores: list[np.ndarray],
    weights: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(point, lo, hi)`` of a linear contrast of AUCs, all measured on the SAME test galaxies.

    Every draw resamples the galaxies once and re-scores every AUC in the contrast on that one
    resample, so shared test-set luck cancels rather than inflating the interval. That is what
    makes a headroom of +0.004 readable against per-AUC standard errors of 0.002::

        headroom            labels=[y, y]          scores=[mlp, lin]            weights=[1, -1]
        selective headroom  labels=[y, y, yc, yc]  scores=[mlp, lin, mlp_c, lin_c]
                                                   weights=[1, -1, -1, 1]

    ``labels`` may differ between terms (a control task relabels the same galaxies) but every
    vector must be over the same galaxies in the same order. A draw that leaves any term with a
    single class is skipped.
    """
    n = len(labels[0])
    if not (len(labels) == len(scores) == len(weights)):
        raise ValueError("labels, scores and weights must have one entry per AUC term")
    if any(len(v) != n for v in (*labels, *scores)):
        raise ValueError("every term must be over the same galaxies — a paired contrast is not")
    terms = [
        _SortedScores(np.asarray(y), np.asarray(s)) for y, s in zip(labels, scores, strict=True)
    ]
    wts = np.asarray(weights, dtype=np.float64)
    ones = np.ones(n)
    point = float(sum(wt * tm.auc(ones) for wt, tm in zip(wts, terms, strict=True)))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        aucs = [tm.auc(w) for tm in terms]
        if any(np.isnan(a) for a in aucs):
            continue
        draws.append(float(np.dot(wts, aucs)))
    if len(draws) < 2:
        return point, point, point
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return point, lo, hi


def probe_auc_ci_se(
    train: Embeddings,
    test: Embeddings,
    *,
    c: float = 1.0,
    max_iter: int = DEFAULT_MAX_ITER,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return ``(auc, lo, hi, se)`` — the point AUC, a bootstrap 95% CI, and the bootstrap SE.

    The probe is fit **once** on ``train``; the interval comes from resampling the scored
    ``test`` set with replacement (degenerate single-class resamples are skipped), so it
    reflects the finite test size — the honest way to state ``n_test`` ≈ a few hundred.

    The **standard error** is the second consumer, and it arrived with Brief P's existence
    test (D23): the ``untrained_z`` construction puts the real AUC's sampling uncertainty on
    one side of the comparison and the untrained bar's seed spread on the other, so it needs a
    scale, not an interval. Taken as the bootstrap SD rather than derived from the percentiles,
    because ``(hi - lo) / (2 * 1.96)`` assumes a symmetry the AUC does not have near the ceiling.
    """
    _require_two_classes(train, test)
    from sklearn.metrics import roc_auc_score

    scaler, clf = _fit(train, c=c, max_iter=max_iter)
    scores = clf.predict_proba(scaler.transform(test.x))[:, 1]
    y = np.asarray(test.y)
    auc = float(roc_auc_score(y, scores))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boots: list[float] = []
    for _ in range(n_boot):
        bi = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[bi])) < 2:  # skip a resample that lost a class
            continue
        boots.append(float(roc_auc_score(y[bi], scores[bi])))
    if len(boots) < 2:  # pathological tiny test set — no informative interval or scale
        return auc, auc, auc, 0.0
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return auc, lo, hi, float(np.std(boots, ddof=1))


def probe_auc_ci(
    train: Embeddings,
    test: Embeddings,
    *,
    c: float = 1.0,
    max_iter: int = DEFAULT_MAX_ITER,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(auc, lo, hi)`` — the point AUC plus a bootstrap 95% CI on the test set.

    The interval-only view of :func:`probe_auc_ci_se`, kept because it is what every existing
    caller wants and a four-tuple would churn them all for a value they do not use.
    """
    auc, lo, hi, _se = probe_auc_ci_se(
        train, test, c=c, max_iter=max_iter, n_boot=n_boot, seed=seed
    )
    return auc, lo, hi


@dataclasses.dataclass(frozen=True)
class ConceptDirection:
    """A concept axis in **embedding space**: the canonical probe's logistic direction.

    ``w_unit`` is the unit direction the explorer's "X-ness score" projects onto
    (``embedding @ w_unit``); ``w_raw``/``bias`` are the affine logit weights mapped back
    through the standardisation, so ``embedding @ w_raw + bias`` reproduces the probe logit.
    """

    name: str
    w_unit: np.ndarray  # (D,) unit-norm direction
    w_raw: np.ndarray  # (D,) logit weights in raw embedding space
    bias: float


def probe_direction(
    train: Embeddings, *, name: str, c: float = 1.0, max_iter: int = DEFAULT_MAX_ITER
) -> ConceptDirection:
    """Fit the canonical probe and return its concept direction in raw embedding space.

    The probe standardises features, so the fitted ``coef_`` lives in standardised space;
    we fold the scaler back in (``w_raw = coef / scale``; ``bias = intercept − Σ coef·μ/σ``)
    so the explorer projects onto a direction in the same space its embeddings live in.
    """
    if len(np.unique(train.y)) < 2:
        raise ValueError("cannot fit a concept direction from a single-class train split")
    scaler, clf = _fit(train, c=c, max_iter=max_iter)
    coef = clf.coef_[0]
    scale = scaler.scale_
    mean = scaler.mean_
    w_raw = coef / scale
    bias = float(clf.intercept_[0] - np.sum(coef * mean / scale))
    norm = float(np.linalg.norm(w_raw)) or 1.0
    return ConceptDirection(name=name, w_unit=w_raw / norm, w_raw=w_raw, bias=bias)


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """The headline read-out: AUC + a bootstrap CI + the (extremes-filtered) split sizes."""

    auc: float
    auc_lo: float
    auc_hi: float
    n_train: int
    n_test: int


def _extremes(
    emb: Embeddings, *, low: float = EXTREME_LOW, high: float = EXTREME_HIGH
) -> Embeddings:
    keep = (emb.fraction <= low) | (emb.fraction >= high)
    return Embeddings(emb.x[keep], emb.y[keep], emb.fraction[keep])


def run_probe(
    encoder: Encoder,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    device: str = "cpu",
    extremes_only: bool = True,
    low: float = EXTREME_LOW,
    high: float = EXTREME_HIGH,
    c: float = 1.0,
) -> ProbeResult:
    """Extract frozen embeddings for both splits and report the headline AUC + CI.

    With ``extremes_only`` (the slice default), train and test are restricted to the
    high-consensus extremes so the number reflects the clean signal, not the ambiguous middle.
    """
    train = extract_embeddings(encoder, train_dataset, device=device)
    test = extract_embeddings(encoder, test_dataset, device=device)
    if extremes_only:
        train, test = _extremes(train, low=low, high=high), _extremes(test, low=low, high=high)
    auc, lo, hi = probe_auc_ci(train, test, c=c)
    return ProbeResult(auc=auc, auc_lo=lo, auc_hi=hi, n_train=len(train.y), n_test=len(test.y))
