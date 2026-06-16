"""Gut-check figures for the slice: a UMAP of the frozen embeddings + the collapse trace.

Two near-free reads the vertical-slice go/no-go leans on (the AUC is the headline; these are
the sanity checks):

* :func:`plot_umap` — a 2-D UMAP of the frozen ``probe-test`` embeddings coloured by the
  binary smooth-vs-featured label. *Any* visible separation is encouraging; a structureless
  blob alongside a flat AUC is the null read.
* :func:`plot_collapse_trace` — the collapse monitor's std / effective-rank / mean-cosine
  versus step, so "did it stay healthy?" is answered by a glance, not a number.

``umap-learn`` is imported lazily (it is an ``eval``-extra dependency), so importing this
module — and the rest of the test suite — does not require it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["umap_2d", "plot_umap", "plot_collapse_trace"]


def umap_2d(x: np.ndarray, *, seed: int = 0, n_neighbors: int = 15) -> np.ndarray:
    """Project ``(N, D)`` embeddings to ``(N, 2)`` with UMAP (lazy import)."""
    import umap  # lazy: eval extra

    n_neighbors = min(n_neighbors, max(2, x.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors)
    return np.asarray(reducer.fit_transform(x))


def plot_umap(x: np.ndarray, labels: np.ndarray, out_path: str | Path, *, seed: int = 0) -> Path:
    """Write a 2-D UMAP scatter of ``x`` coloured by binary ``labels`` to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = umap_2d(x, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.asarray(labels)
    for value, name, colour in ((0, "smooth", "#d95f02"), (1, "featured", "#1b9e77")):
        m = labels == value
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.6, label=name, c=colour)
    ax.set_title("Frozen JEPA embeddings (UMAP) — smooth vs featured")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_collapse_trace(trace: dict[str, list[Any]], out_path: str | Path) -> Path:
    """Plot the collapse monitor's std / effective-rank / mean-cosine versus step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    step = trace.get("step", list(range(len(trace.get("std", [])))))
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    for ax, key, title in (
        (axes[0], "std", "embedding std (→0 = collapse)"),
        (axes[1], "effective_rank", "effective rank (→1 = collapse)"),
        (axes[2], "mean_cosine", "mean pairwise cosine (→1 = collapse)"),
    ):
        ax.plot(step, trace.get(key, []), marker="o", ms=3)
        ax.set_ylabel(title, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle("Collapse monitor — did the representation stay healthy?")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
