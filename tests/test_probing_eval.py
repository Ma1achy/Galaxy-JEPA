"""Tests for the frozen probe + gut-check figures (docs/spec/encoder.md, slice plan).

The probe is the headline read-out, so these pin: it refuses a trainable encoder (the freeze
boundary), it recovers a separable signal (AUC → 1 on a planted-separable set), the
extremes filter does its job, and the figures render. UMAP is skipped where ``umap-learn``
is absent (an ``eval``-extra dependency).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from galaxy_jepa.eval.embed import plot_collapse_trace
from galaxy_jepa.probing.logistic import Embeddings, extract_embeddings, probe_auc

# Integration tier: the probe needs sklearn and the figures need matplotlib at runtime
# (the `eval` extra), so these run in the integration job, not the dependency-light gate.
pytestmark = pytest.mark.integration

_HAS_UMAP = importlib.util.find_spec("umap") is not None


def _separable(n: int, d: int, seed: int) -> Embeddings:
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(len(y), d)) + y[:, None] * 3.0  # classes shifted apart
    frac = np.where(y == 1, 0.95, 0.05)  # all confident extremes
    return Embeddings(x, y, frac)


def test_probe_recovers_separable_signal():
    auc = probe_auc(_separable(80, 16, 0), _separable(80, 16, 1))
    assert auc > 0.9  # a planted-separable signal must be linearly nameable


def test_probe_rejects_single_class():
    one = Embeddings(np.zeros((4, 4)), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        probe_auc(one, _separable(8, 4, 0))


def test_extract_embeddings_requires_frozen_encoder():
    from galaxy_jepa.models.vit import VisionTransformer

    encoder = VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=2, heads=2)
    # trainable encoder → loud failure before any extraction
    from torch.utils.data import TensorDataset

    with pytest.raises(RuntimeError):
        extract_embeddings(encoder, TensorDataset(torch.randn(2, 3, 32, 32)))


def test_extract_embeddings_runs_on_frozen_encoder():
    from galaxy_jepa.models.vit import VisionTransformer

    encoder = VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=2, heads=2)
    for p in encoder.parameters():
        p.requires_grad_(False)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, i):
            return {
                "image": torch.randn(3, 32, 32),
                "label": i % 2,
                "featured_fraction": 0.9 if i % 2 else 0.1,
            }

    emb = extract_embeddings(encoder, _DS())
    assert emb.x.shape == (6, 16)
    assert emb.y.shape == (6,) and emb.fraction.shape == (6,)


def test_plot_collapse_trace_writes_png(tmp_path):
    trace = {
        "step": [0, 1, 2],
        "std": [1.0, 0.9, 0.8],
        "effective_rank": [10.0, 9.0, 8.5],
        "mean_cosine": [0.1, 0.12, 0.15],
    }
    out = plot_collapse_trace(trace, tmp_path / "collapse.png")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not _HAS_UMAP, reason="umap-learn not installed (eval extra)")
def test_plot_umap_writes_png(tmp_path):
    from galaxy_jepa.eval.embed import plot_umap

    x = np.random.default_rng(0).normal(size=(40, 16))
    labels = np.array([0, 1] * 20)
    out = plot_umap(x, labels, tmp_path / "umap.png")
    assert out.exists() and out.stat().st_size > 0
