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

    The probe is fit **once** on ``train``; the CI comes from resampling the scored ``test``
    set with replacement (degenerate single-class resamples are skipped), so the interval
    reflects the finite test size — the honest way to state ``n_test`` ≈ a few hundred.
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
    if not boots:  # pathological tiny test set — no informative interval
        return auc, auc, auc
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
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
