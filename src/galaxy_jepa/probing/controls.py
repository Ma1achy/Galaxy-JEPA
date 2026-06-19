"""The controls battery — the five negative controls + the five nuisances (design 3C/3D).

The controls are the difference between "a probe predicted bars" (worthless) and "bar-ness is
a recoverable direction, and it is not capacity / a nuisance / a label prior" (the paper).
Each control closes a *distinct* "how do I know it's real" attack; the existence verdict
(``nulls.py``) is calibrated against the **most conservative** of the five, so no
"but you didn't control for X" survives.

This module **generates** the controls — the buildable part. How a threshold *reads against*
them (the significance machinery, the multiplicity correction) is flagged and lives in
``nulls.py``; the matched-evaluation a competitive nuisance triggers lives in ``matching.py``.

The five negative controls (3C):

1. **Shuffled vote fractions** — refit the probe with the train labels permuted (image–label
   correspondence destroyed). Kills "the probe exploits label marginals". *Resamplable → the
   primary null distribution.* Also the selectivity baseline (Hewitt–Liang).
2. **Random embeddings** — Gaussian matched to the real embedding covariance, real labels.
   Kills "any high-D vector predicts this". *Resamplable.*
3. **Noise images through the real frozen encoder** — noise stamps matched to the data's
   post-pipeline pixel marginals, one per galaxy, real labels. Kills "the encoder imposes
   structure on anything". *The strong one.*
4. **Untrained-encoder embeddings** — real images through a frozen random-init ViT. Kills
   "the probe, not the pretraining, did the work". *The headline "pretraining mattered" null.*
5. **Sky/noise-level labels** — probe the real embeddings against an image-quality label.
   Kills "the probe reads image depth, not morphology".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from galaxy_jepa.core.encoder import Encoder
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, extract_matrix
from galaxy_jepa.probing.logistic import Embeddings, probe_auc

__all__ = [
    "shuffled_label_nulls",
    "random_embedding_nulls",
    "sky_noise_label_auc",
    "selectivity",
    "untrained_encoder_matrix",
    "noise_through_encoder_matrix",
    "ControlEmbeddings",
    "FeatureControls",
]


def _safe_auc(train: Embeddings, test: Embeddings, *, c: float) -> float:
    """AUC, returning chance (0.5) on a degenerate single-class control rather than raising.

    A negative control collapsing to one class *is* a (chance-level) finding, not a run
    error — so it folds into the null at 0.5 instead of aborting the battery.
    """
    if len(np.unique(train.y)) < 2 or len(np.unique(test.y)) < 2:
        return 0.5
    return probe_auc(train, test, c=c)


def shuffled_label_nulls(
    train: Embeddings, test: Embeddings, *, n_draws: int = 50, c: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Null AUCs from permuting the **train** labels — the primary selectivity null (3C-1).

    The test labels are left intact (so AUC stays defined); only the image–label
    correspondence in the fit is destroyed. ``n_draws`` permutations → a null distribution.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        y_shuf = rng.permutation(train.y)
        out[i] = _safe_auc(Embeddings(train.x, y_shuf, train.fraction), test, c=c)
    return out


def random_embedding_nulls(
    train: Embeddings, test: Embeddings, *, n_draws: int = 50, c: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Null AUCs from Gaussian embeddings matched to the real covariance (3C-2).

    Real labels are kept; the *embeddings* are replaced by draws from a Gaussian with the
    train embeddings' mean and covariance — "does any high-D vector with this covariance
    predict the label?". ``n_draws`` redraws → a null distribution.
    """
    rng = np.random.default_rng(seed)
    mean = train.x.mean(axis=0)
    cov = np.cov(train.x, rowvar=False)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        xr_train = rng.multivariate_normal(mean, cov, size=len(train.y))
        xr_test = rng.multivariate_normal(mean, cov, size=len(test.y))
        out[i] = _safe_auc(
            Embeddings(xr_train, train.y, train.fraction),
            Embeddings(xr_test, test.y, test.fraction),
            c=c,
        )
    return out


def sky_noise_label_auc(
    train: Embeddings,
    test: Embeddings,
    sky_label_train: np.ndarray,
    sky_label_test: np.ndarray,
    *,
    c: float = 1.0,
) -> float:
    """AUC predicting an image-quality (sky/noise-level) label from the real embeddings (3C-5)."""
    return _safe_auc(
        Embeddings(train.x, sky_label_train.astype(np.int64), train.fraction),
        Embeddings(test.x, sky_label_test.astype(np.int64), test.fraction),
        c=c,
    )


def selectivity(real_auc: float, shuffled_nulls: np.ndarray) -> float:
    """Hewitt–Liang selectivity: real-label AUC − the shuffled-label control AUC (the mean)."""
    return float(real_auc - float(np.mean(shuffled_nulls)))


# --- control embedding sources (per-encoder, built once) ----------------------------------


def untrained_encoder_matrix(
    model_config: Mapping[str, Any],
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> EmbeddingMatrix:
    """Real images through a **frozen random-init ViT** → the "pretraining mattered" null (3C-4).

    A fresh ``VisionTransformer`` from the same constructor record, frozen without training,
    so the only difference from the real encoder is the learned weights. Imports ``models``
    (not ``objectives``) — the freeze boundary holds.
    """
    from galaxy_jepa.models.vit import VisionTransformer

    untrained = VisionTransformer(**dict(model_config))
    untrained.eval()
    for p in untrained.parameters():
        p.requires_grad_(False)
    return extract_matrix(untrained, dataset, device=device, batch_size=batch_size)


class _NoiseImageDataset(Dataset):
    """Wraps a base dataset, replacing each image with Gaussian noise matched to the data's
    per-channel post-pipeline marginals — keeping the ``object_id`` so labels still align."""

    def __init__(self, base: Dataset, *, mean: np.ndarray, std: np.ndarray, seed: int = 0):
        self.base = base
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        img = torch.as_tensor(item["image"]).float()
        # per-item deterministic noise (seed + object_id) matched to the channel marginals
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(item.get("object_id", index)))
        noise = torch.randn(img.shape, generator=g)
        c = img.shape[0]
        noise = noise * self.std[:c].view(c, 1, 1) + self.mean[:c].view(c, 1, 1)
        item["image"] = noise
        return item


def _channel_marginals(dataset: Dataset, *, n_sample: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over a sample of the dataset's post-pipeline stamps."""
    imgs: list[np.ndarray] = []
    for i in range(min(n_sample, len(dataset))):  # type: ignore[arg-type]
        imgs.append(np.asarray(torch.as_tensor(dataset[i]["image"]).float()))
    stack = np.stack(imgs)  # (n, C, H, W)
    mean = stack.mean(axis=(0, 2, 3))
    std = stack.std(axis=(0, 2, 3))
    return mean, std


def noise_through_encoder_matrix(
    encoder: Encoder,
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 128,
    seed: int = 0,
) -> EmbeddingMatrix:
    """Noise stamps (matched pixel marginals) through the **real frozen encoder** (3C-3).

    One noise stamp per galaxy, keyed by the same ``object_id`` so the real labels align —
    "noise images through our encoder yield chance-level morphology AUC".
    """
    mean, std = _channel_marginals(dataset)
    noisy = _NoiseImageDataset(dataset, mean=mean, std=std, seed=seed)
    return extract_matrix(encoder, noisy, device=device, batch_size=batch_size)


@dataclasses.dataclass(frozen=True)
class ControlEmbeddings:
    """The per-encoder control embedding sources, built once (3C-3, 3C-4)."""

    real: EmbeddingMatrix
    untrained: EmbeddingMatrix  # real images, random-init encoder
    noise: EmbeddingMatrix  # noise images, real encoder


@dataclasses.dataclass(frozen=True)
class FeatureControls:
    """The full five-null battery for one feature, plus the Hewitt–Liang selectivity.

    Each ``*_nulls`` is an array (a distribution); the single-AUC controls are length-1
    arrays, so :func:`nulls.five_null_max` treats them uniformly. ``nuisance_aucs`` is the
    diagnostic parallel-probe panel (3D-ii); the matched-evaluation it can trigger is in
    ``matching.py``.
    """

    feature: str
    real_auc: float
    shuffled_nulls: np.ndarray
    random_embedding_nulls: np.ndarray
    noise_encoder_auc: float
    untrained_encoder_auc: float
    sky_noise_auc: float
    selectivity: float
    nuisance_aucs: Mapping[str, float]


def build_feature_controls(
    feature: str,
    *,
    real_auc: float,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    controls: ControlEmbeddings,
    labels: LabelProvider,
    sky_label_col: str,
    c: float = 1.0,
    n_draws: int = 50,
    seed: int = 0,
) -> FeatureControls:
    """Assemble the five negative-control nulls + selectivity + the nuisance panel for a feature.

    Reuses the one-shot control embedding sources in ``controls`` (no per-feature re-encode);
    the resamplable nulls (shuffled, random-embedding) are drawn here, the encoder-source
    nulls (noise, untrained) are read from ``controls``.
    """
    from galaxy_jepa.probing.extract import feature_embeddings

    real_train = feature_embeddings(controls.real, labels, feature, train_ids)
    real_test = feature_embeddings(controls.real, labels, feature, test_ids)

    shuffled = shuffled_label_nulls(real_train, real_test, n_draws=n_draws, c=c, seed=seed)
    random_emb = random_embedding_nulls(real_train, real_test, n_draws=n_draws, c=c, seed=seed + 1)

    noise_train = feature_embeddings(controls.noise, labels, feature, train_ids)
    noise_test = feature_embeddings(controls.noise, labels, feature, test_ids)
    noise_auc = _safe_auc(noise_train, noise_test, c=c)

    untrained_train = feature_embeddings(controls.untrained, labels, feature, train_ids)
    untrained_test = feature_embeddings(controls.untrained, labels, feature, test_ids)
    untrained_auc = _safe_auc(untrained_train, untrained_test, c=c)

    sky_train = _binary_column(labels, sky_label_col, train_ids)
    sky_test = _binary_column(labels, sky_label_col, test_ids)
    sky_auc = sky_noise_label_auc(real_train, real_test, sky_train, sky_test, c=c)

    train_present = _present(controls.real, train_ids)
    test_present = _present(controls.real, test_ids)
    nuisance_aucs: dict[str, float] = {}
    for name in labels.nuisances:
        nz_train = Embeddings(
            real_train.x, labels.nuisance_label(name, train_present), real_train.fraction
        )
        nz_test = Embeddings(
            real_test.x, labels.nuisance_label(name, test_present), real_test.fraction
        )
        nuisance_aucs[name] = _safe_auc(nz_train, nz_test, c=c)

    return FeatureControls(
        feature=feature,
        real_auc=real_auc,
        shuffled_nulls=shuffled,
        random_embedding_nulls=random_emb,
        noise_encoder_auc=noise_auc,
        untrained_encoder_auc=untrained_auc,
        sky_noise_auc=sky_auc,
        selectivity=selectivity(real_auc, shuffled),
        nuisance_aucs=nuisance_aucs,
    )


def _present(matrix: EmbeddingMatrix, ids: Sequence[int]) -> list[int]:
    return [int(o) for o in ids if int(o) in matrix.index]


def _binary_column(labels: LabelProvider, col: str, ids: Sequence[int]) -> np.ndarray:
    """Median-split binarisation of an arbitrary metadata column (the sky/noise label)."""
    v = np.asarray([float(labels.rows.get(int(o), {}).get(col, np.nan)) for o in ids])
    return (v >= float(np.nanmedian(v))).astype(np.int64)
