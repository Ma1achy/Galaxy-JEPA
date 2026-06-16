"""Frozen-encoder logistic probe — the one headline number (docs/spec/encoder.md, slice plan).

The canonical "linearly nameable" read-out: an L2-regularised logistic regression on the
**frozen** encoder's mean-pooled penultimate-layer embeddings (``Encoder.encode``). For the
vertical slice this is the whole measurement — train on the ``probe-train`` confident
extremes, report ROC-AUC on the ``probe-test`` confident extremes (the label drowns less in
the ambiguous middle that way; ``data/metadata.is_confident_extreme``).

The encoder is asserted **frozen** on entry (``assert_frozen``): a still-trainable encoder
fails loudly rather than letting the probe's gradients bend the representation. This module
consumes a ``models`` encoder + a checkpoint — it never imports ``objectives`` (the freeze
boundary runs through disk; ``docs/spec/objectives.md`` §3).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from galaxy_jepa.core.encoder import Encoder, assert_frozen

__all__ = ["Embeddings", "extract_embeddings", "probe_auc", "ProbeResult", "run_probe"]


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


def probe_auc(train: Embeddings, test: Embeddings, *, c: float = 1.0) -> float:
    """Fit an L2-logistic probe on ``train``, return ROC-AUC on ``test``."""
    if len(np.unique(train.y)) < 2:
        raise ValueError("probe training set has a single class — cannot fit a logistic axis")
    if len(np.unique(test.y)) < 2:
        raise ValueError("probe test set has a single class — AUC undefined")
    clf = LogisticRegression(C=c, max_iter=2000)  # L2 is the default penalty
    clf.fit(train.x, train.y)
    scores = clf.predict_proba(test.x)[:, 1]
    return float(roc_auc_score(test.y, scores))


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """The headline read-out: AUC + the (extremes-filtered) split sizes that produced it."""

    auc: float
    n_train: int
    n_test: int


def _extremes(emb: Embeddings, *, low: float = 0.2, high: float = 0.8) -> Embeddings:
    keep = (emb.fraction <= low) | (emb.fraction >= high)
    return Embeddings(emb.x[keep], emb.y[keep], emb.fraction[keep])


def run_probe(
    encoder: Encoder,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    device: str = "cpu",
    extremes_only: bool = True,
) -> ProbeResult:
    """Extract frozen embeddings for both splits and report the headline AUC.

    With ``extremes_only`` (the slice default), train and test are restricted to the
    high-consensus extremes so the number reflects the clean signal, not the ambiguous middle.
    """
    train = extract_embeddings(encoder, train_dataset, device=device)
    test = extract_embeddings(encoder, test_dataset, device=device)
    if extremes_only:
        train, test = _extremes(train), _extremes(test)
    auc = probe_auc(train, test)
    return ProbeResult(auc=auc, n_train=len(train.y), n_test=len(test.y))
