"""The headline figures (design 5) — the deliverables sub-systems 2–4 produce.

Not new methodology — specifying the figures so the harness is built to **emit** them, not
reverse-engineer them from raw outputs afterward. ``run_probing`` calls these at Phase 7, so
the figures fall out of the run.

* **Figure 1 — the nameability ladder** (sub-system 2): per-feature rung + AUC + selectivity +
  whether it cleared the controls, ordered as narrative, confused features highlighted.
* **Figure 2 — uncertainty geometry** (sub-system 4): projection distance vs human vote
  fraction on the held-out ambiguous middle, with Spearman. (The v1-vs-v2 panel, 4C, slots in
  here once the v1 confusion data is wired — a later consumer.)
* **Figure 3 — the entanglement geometry** (sub-system 2A): the recovered cosine matrix
  (the bridge to v1's Fig 18/19) + the eigenspectrum (effective rank, the MP comparison).
* **Control figures** (generate generously, triage later): the five-null comparison and the
  nuisance-AUC panel — especially "untrained ~0.5 vs ours ~0.9" and noise-through-encoder.

Follows ``eval/embed.py``: matplotlib (Agg) imported lazily (the ``eval`` extra), returns the
written ``Path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from galaxy_jepa.probing.entanglement import EntanglementGeometry
    from galaxy_jepa.probing.ladder import LadderResult
    from galaxy_jepa.probing.uncertainty import UncertaintyGeometry

__all__ = [
    "figure_ladder",
    "figure_uncertainty",
    "figure_entanglement",
    "figure_controls",
]

_RUNG_COLOUR = {"R1": "#1b9e77", "R2": "#7570b3", "R3": "#d95f02", "R4": "#999999"}


def _axes(figsize: tuple[float, float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, *plt.subplots(figsize=figsize)


def figure_ladder(ladder: LadderResult, out_path: str | Path) -> Path:
    """Figure 1 — per-feature rung + AUC, bars coloured by rung, ordered worst→best."""
    plt, fig, ax = _axes((7, max(3, 0.5 * len(ladder.verdicts) + 1)))
    items = sorted(ladder.verdicts.values(), key=lambda v: (v.rung, -v.metrics.get("auc", 0.0)))
    names = [v.feature for v in items]
    aucs = [v.metrics.get("auc", 0.5) for v in items]
    colours = [_RUNG_COLOUR.get(v.rung, "#000000") for v in items]
    y = np.arange(len(items))
    ax.barh(y, aucs, color=colours)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  [{v.rung}]" for n, v in zip(names, items, strict=True)], fontsize=8)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("linear-probe ROC-AUC")
    ax.set_title("Figure 1 — the nameability ladder")
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in _RUNG_COLOUR.values()]
    ax.legend(handles, list(_RUNG_COLOUR), title="rung", loc="lower right", fontsize=7)
    out = _save(fig, plt, out_path)
    return out


def figure_uncertainty(results: dict[str, UncertaintyGeometry], out_path: str | Path) -> Path:
    """Figure 2 — projection distance vs vote fraction on the ambiguous middle, per feature."""
    plt, fig, ax = _axes((6, 5))
    for feature, u in results.items():
        if u.distances.size and u.fractions.size:
            ax.scatter(
                u.distances,
                u.fractions,
                s=10,
                alpha=0.5,
                label=f"{feature} (ρ={u.spearman:.2f}, p={u.pvalue:.3f})",
            )
    ax.set_xlabel("projection distance along the unsupervised concept axis")
    ax.set_ylabel("human vote fraction (held-out middle)")
    ax.set_title("Figure 2 — uncertainty geometry")
    ax.legend(loc="best", fontsize=7)
    return _save(fig, plt, out_path)


def figure_entanglement(geometry: EntanglementGeometry, out_path: str | Path) -> Path:
    """Figure 3 — recovered cosine matrix + the Gram eigenspectrum (effective rank, MP edge)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    im = ax1.imshow(geometry.cosine, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xticks(range(len(geometry.names)))
    ax1.set_yticks(range(len(geometry.names)))
    ax1.set_xticklabels(geometry.names, rotation=90, fontsize=7)
    ax1.set_yticklabels(geometry.names, fontsize=7)
    ax1.set_title("recovered concept cosine matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ev = np.sort(geometry.gram_eigenvalues)[::-1]
    ax2.plot(range(1, len(ev) + 1), ev, marker="o")
    ax2.axhline(
        geometry.mp.mp_edge, color="r", ls="--", label=f"MP edge ({geometry.mp.mp_edge:.2f})"
    )
    ax2.set_xlabel("component")
    ax2.set_ylabel("Gram eigenvalue")
    ax2.set_title(f"eigenspectrum (erank={geometry.gram_effective_rank:.2f})")
    ax2.legend(fontsize=7)
    fig.suptitle("Figure 3 — the entanglement geometry")
    return _save(fig, plt, out_path)


def figure_controls(ladder: LadderResult, out_path: str | Path) -> Path:
    """Control panel — per-feature real AUC vs the headline nulls (untrained, noise, shuffled)."""
    plt, fig, ax = _axes((7, max(3, 0.5 * len(ladder.feature_controls) + 1)))
    feats = list(ladder.feature_controls)
    y = np.arange(len(feats))
    real = [ladder.feature_controls[f].real_auc for f in feats]
    untrained = [ladder.feature_controls[f].untrained_encoder_auc for f in feats]
    noise = [ladder.feature_controls[f].noise_encoder_auc for f in feats]
    shuffled = [float(np.mean(ladder.feature_controls[f].shuffled_nulls)) for f in feats]
    ax.scatter(real, y, c="#1b9e77", label="ours (real)", zorder=3)
    ax.scatter(untrained, y, c="#d95f02", marker="x", label="untrained encoder")
    ax.scatter(noise, y, c="#7570b3", marker="^", label="noise through encoder")
    ax.scatter(shuffled, y, c="#999999", marker="s", label="shuffled labels")
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Controls — real vs the negative-control nulls")
    ax.legend(loc="best", fontsize=7)
    return _save(fig, plt, out_path)


def _save(fig, plt, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
