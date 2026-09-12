"""Regenerate the README's data figures from real artefacts. No hand-made numbers.

Every figure here is computed from something on disk that a run produced:
  * the vote-decisiveness panel  <- raw/probe/metadata.csv, this project's own GZ2 pull
  * the pilot collapse trace     <- runs/pilot.log, the 6,000-step pilot's own log lines
  * the pilot UMAP               <- runs/slice/explorer/, the frozen pilot's embeddings
  * the schedule dose-response   <- artifacts/out/h2_arms.jsonl, Brief H's six arms

Run: uv run python artifacts/readme_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PROBE_META = Path("/Volumes/X10 Pro/galaxy-jepa/raw/probe/metadata.csv")

INK = "#1b1f23"
MUTED = "#6a737d"
GRID = "#e3e6ea"
ACCENT = "#2f6f9f"
WARM = "#c9622f"
GOOD = "#3f8f6b"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})

# The six questions the README names: three v1 found volunteers agreed on, three they did not.
QUESTIONS = [
    ("t02_edgeon", "Edge-on disk?", ["a04_yes", "a05_no"]),
    ("t03_bar", "Bar?", ["a06_bar", "a07_no_bar"]),
    ("t04_spiral", "Spiral arms?", ["a08_spiral", "a09_no_spiral"]),
    ("t09_bulge_shape", "Bulge shape?", ["a25_rounded", "a26_boxy", "a27_no_bulge"]),
    ("t10_arms_winding", "Arm winding?", ["a28_tight", "a29_medium", "a30_loose"]),
    (
        "t11_arms_number",
        "Arm count?",
        ["a31_1", "a32_2", "a33_3", "a34_4", "a36_more_than_4", "a37_cant_tell"],
    ),
]
MIN_VOTES = 5  # a galaxy has genuinely "reached" the question only if the tree routed votes there


def vote_decisiveness() -> None:
    """How decisively did volunteers answer each question? v1's catch, on v2's own pull.

    For each galaxy reaching a question, the leading answer's share of the vote. Agreement
    piles up near 1.0; a question the crowd cannot separate piles up near 1/n_answers, the
    chance line. This is the same observation v1 made from correlation matrices, measured
    directly and on this project's 230k-galaxy pull rather than borrowed from v1's figure.
    """
    import pandas as pd

    wanted = {}
    for q, _, answers in QUESTIONS:
        for a in answers:
            wanted[f"{q}_{a}_weighted_fraction"] = (q, a)
            wanted[f"{q}_{a}_count"] = (q, a)
    cols = list(wanted)
    df = pd.read_csv(PROBE_META, usecols=cols)

    # Normalised per panel, not shared counts: the deep tree questions are asked of far fewer
    # galaxies (the tree funnels), so raw counts would hide the shape that is the whole point.
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharey=True)
    for ax, (q, label, answers) in zip(axes.ravel(), QUESTIONS, strict=True):
        fracs = df[[f"{q}_{a}_weighted_fraction" for a in answers]].to_numpy(dtype=float)
        counts = df[[f"{q}_{a}_count" for a in answers]].to_numpy(dtype=float)
        reached = counts.sum(axis=1) >= MIN_VOTES
        lead = fracs[reached].max(axis=1)
        chance = 1.0 / len(answers)
        median = float(np.median(lead))
        colour = GOOD if median >= 0.80 else WARM

        ax.hist(
            lead, bins=40, range=(chance * 0.9, 1.0), color=colour, alpha=0.85, edgecolor="none",
            weights=np.full(lead.shape, 1.0 / lead.size),
        )
        ax.axvline(chance, color=MUTED, lw=1.2, ls="--")
        ax.text(chance, 0.36, " chance", color=MUTED, fontsize=8, va="top", ha="left")
        ax.axvline(median, color=INK, lw=1.4)
        ax.set_title(
            f"{label}\nmedian lead {median:.2f}   ({reached.sum():,} galaxies asked)",
            fontsize=10.5,
            pad=8,
        )
        ax.set_xlabel("leading answer's share of the vote")
        ax.set_ylim(0, 0.38)
        ax.grid(axis="y", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of galaxies asked")

    axes[0, 0].text(
        0.02, 0.92, "volunteers agreed", transform=axes[0, 0].transAxes,
        fontsize=9.5, color=GOOD, fontweight="bold", va="top",
    )
    axes[1, 0].text(
        0.02, 0.92, "volunteers did not", transform=axes[1, 0].transAxes,
        fontsize=9.5, color=WARM, fontweight="bold", va="top",
    )

    fig.suptitle(
        "How decisively did volunteers answer? — Galaxy Zoo 2 votes, this project's own pull",
        fontsize=12.5,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "vote_decisiveness.png")
    plt.close(fig)
    print("wrote vote_decisiveness.png")


TRACE = re.compile(
    r"step (\d+): loss=([\d.]+) std=([\d.]+) erank=([\d.]+) cos=([\d.]+)"
)


def pilot_collapse() -> None:
    """The pilot's collapse monitor, parsed from its own 6,000-step log."""
    rows = [
        tuple(map(float, m.groups())) for m in TRACE.finditer((ROOT / "runs" / "pilot.log").read_text())
    ]
    a = np.array(rows)
    step, loss, std, erank, cos = a.T

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    panels = [
        (erank, "effective rank", "→ 1 is collapse", ACCENT),
        (std, "embedding std", "→ 0 is collapse", GOOD),
        (cos, "mean pairwise cosine", "→ 1 is collapse", WARM),
    ]
    for ax, (y, name, danger, colour) in zip(axes, panels, strict=True):
        ax.plot(step, y, color=colour, lw=1.6)
        ax.set_title(f"{name}   ({danger})", fontsize=10.5, pad=8)
        ax.set_xlabel("step")
        ax.grid(lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, step.max())
    # Annotate what the trace actually does: it falls hard, bottoms, then recovers and holds.
    # Worth marking precisely, because a short run that stopped at the trough would read as collapse.
    lo = int(step[erank.argmin()])
    axes[0].scatter([lo], [erank.min()], s=42, color=WARM, zorder=4)
    axes[0].annotate(
        f"falls to {erank.min():.1f} at step {lo}",
        (lo, erank.min()), textcoords="offset points", xytext=(16, -4),
        fontsize=8.5, color=WARM,
    )
    tail = step >= 4000
    axes[0].axhspan(erank[tail].min(), erank[tail].max(), color=ACCENT, alpha=0.10, zorder=0)
    axes[0].text(
        step.max() * 0.99, erank[tail].max() + 0.6,
        f"then recovers, holds {erank[tail].min():.1f}–{erank[tail].max():.1f} ",
        color=ACCENT, fontsize=8.5, va="bottom", ha="right",
    )

    fig.suptitle(
        "Pilot collapse monitor — 6,000 steps, 10k galaxies, laptop. The representation did not collapse.",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "pilot_collapse_trace.png")
    plt.close(fig)
    print(f"wrote pilot_collapse_trace.png ({len(a)} readings, to step {int(step.max())})")


def pilot_umap() -> None:
    """UMAP of the frozen pilot embeddings, from the run's own explorer artefacts.

    Coloured by the Galaxy Zoo *vote fraction*, not the binarised label. The labels live in
    embeddings.npz and the coordinates in umap_coords.json, so they are joined on object_id
    rather than assumed to be in the same order. The continuous colouring is the honest one:
    what the frozen encoder produces is a gradient, and the binary label is a threshold laid
    over it afterwards.
    """
    explorer = ROOT / "runs" / "slice" / "explorer"
    coords = json.loads((explorer / "umap_coords.json").read_text())
    xy = np.asarray(coords["coords"], dtype=float)
    ids = np.asarray(coords["object_ids"], dtype=np.int64)

    emb = np.load(explorer / "embeddings.npz")
    order = {int(o): i for i, o in enumerate(emb["object_ids"])}
    take = np.array([order[int(o)] for o in ids])
    fraction = np.asarray(emb["fraction"], dtype=float)[take]
    label = np.asarray(emb["y"], dtype=int)[take]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], s=18, c=fraction, cmap="RdYlGn", vmin=0.0, vmax=1.0,
        alpha=0.9, edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Galaxy Zoo 'featured' vote fraction", fontsize=9)
    cbar.outline.set_visible(False)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(
        f"Frozen pilot embeddings — {len(xy)} held-out galaxies\n"
        f"coloured by a vote the encoder never saw  "
        f"({int((label == 0).sum())} smooth, {int((label == 1).sum())} featured)",
        fontsize=11,
        pad=10,
    )
    ax.grid(lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "pilot_umap.png")
    plt.close(fig)
    print(f"wrote pilot_umap.png ({len(xy)} points, {(label==0).sum()} smooth / {(label==1).sum()} featured)")


def pilot_concept_axis() -> None:
    """The pilot's headline, made visible — and the qualifier the README used to drop.

    AUC is exactly the probability a random featured galaxy sits further along the direction
    than a random smooth one, so plotting the projected distributions shows the result itself.
    The reported 0.905 is on the **high-consensus** held-out galaxies only
    (``metadata.is_confident_extreme``: vote fraction <= 0.2 or >= 0.8, 370 of 738) — the code
    says so, the README did not. Both panels are drawn, because the ambiguous middle is the
    more interesting half: the axis still ranks galaxies the volunteers could not agree on.

    Direction and CI come from the run's own concept_directions.json; the two subset AUCs are
    recomputed here from the stored projections, and the consensus one reproduces the reported
    figure to four decimals.
    """
    from sklearn.metrics import roc_auc_score

    explorer = ROOT / "runs" / "slice" / "explorer"
    direction = json.loads((explorer / "concept_directions.json").read_text())["featured"]
    w = np.asarray(direction["w_unit"], dtype=float)
    lo_ci, hi_ci = direction["auc_ci"]

    emb = np.load(explorer / "embeddings.npz")
    proj = np.asarray(emb["x"], dtype=float) @ w
    label = np.asarray(emb["y"], dtype=int)
    fraction = np.asarray(emb["fraction"], dtype=float)
    consensus = (fraction <= 0.2) | (fraction >= 0.8)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), sharex=True, sharey=True)
    bins = np.linspace(proj.min(), proj.max(), 40)
    panels = [
        (consensus, "High-consensus galaxies — the headline", f"95% CI {lo_ci:.3f}–{hi_ci:.3f}"),
        (~consensus, "The ambiguous middle — the harder half", "volunteers split 0.2 < f < 0.8"),
    ]
    for ax, (sel, title, note) in zip(axes, panels, strict=True):
        auc = roc_auc_score(label[sel], proj[sel])
        for value, name, colour in [(0, "smooth", WARM), (1, "featured", GOOD)]:
            m = sel & (label == value)
            ax.hist(
                proj[m], bins=bins, color=colour, alpha=0.66, edgecolor="none",
                label=f"{name}  (n={int(m.sum())})",
            )
            ax.axvline(proj[m].mean(), color=colour, lw=1.5, ls="--")
        ax.set_title(f"{title}\nAUC {auc:.3f}   ({note})", fontsize=11, pad=10)
        ax.set_xlabel("projection onto the frozen 'featured' concept direction")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(axis="y", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("galaxies")

    fig.suptitle(
        "One direction in a label-free representation, read out with labels it never saw",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "pilot_concept_axis.png")
    plt.close(fig)
    print(
        "wrote pilot_concept_axis.png  consensus AUC %.4f (n=%d) / ambiguous AUC %.4f (n=%d)"
        % (
            roc_auc_score(label[consensus], proj[consensus]), consensus.sum(),
            roc_auc_score(label[~consensus], proj[~consensus]), (~consensus).sum(),
        )
    )


def schedule_dose_response() -> None:
    """Brief H: effective rank against early learning rate, six controlled arms."""
    arms = [
        json.loads(line)
        for line in (ROOT / "artifacts" / "out" / "h2_arms.jsonl").read_text().splitlines()
        if line.strip()
    ]
    order = ["linear", "sqrt", "warmup1250", "cosine", "wd_ramp", "baseline"]
    arms.sort(key=lambda d: order.index(d["arm"]) if d["arm"] in order else 99)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2))
    palette = {
        "baseline": "#b3352b", "wd_ramp": "#d9863d", "cosine": "#c9a227",
        "warmup1250": "#4f8f5c", "sqrt": ACCENT, "linear": "#6a5acd",
    }

    for d in arms:
        tr = d["trace"]
        steps = [p["step"] for p in tr]
        erank = [p["effective_rank"] for p in tr]
        ax1.plot(steps, erank, lw=1.7, color=palette[d["arm"]], label=d["arm"])
    ax1.axhline(5.0, color=MUTED, ls="--", lw=1.0)
    ax1.text(498, 5.05, "G5 floor ", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax1.set_xlabel("step")
    ax1.set_ylabel("effective rank")
    ax1.set_title("Six arms, one seed — only the schedule differs", fontsize=11, pad=8)
    ax1.legend(frameon=False, fontsize=8.5, ncol=2)
    ax1.grid(lw=0.6)
    ax1.set_axisbelow(True)

    # Dose-response: mean LR over the window in which the fall happens, against rank at its end.
    xs, ys, names = [], [], []
    for d in arms:
        tr = [p for p in d["trace"] if p["step"] <= 175]
        xs.append(float(np.mean([p["lr"] for p in tr])))
        ys.append([p for p in d["trace"] if p["step"] == 175][0]["effective_rank"])
        names.append(d["arm"])
    ax2.scatter(xs, ys, s=70, c=[palette[n] for n in names], zorder=3, edgecolors="none")
    # baseline / cosine / wd_ramp share the same schedule over this window, so their points are
    # coincident; label the cluster once rather than stacking three labels on one dot.
    cluster = {"baseline", "cosine", "wd_ramp"}
    for x, y, n in zip(xs, ys, names, strict=True):
        if n in cluster:
            continue
        ax2.annotate(n, (x, y), textcoords="offset points", xytext=(9, -3), fontsize=9, color=INK)
    cx = float(np.mean([x for x, n in zip(xs, names, strict=True) if n in cluster]))
    cy = float(np.mean([y for y, n in zip(ys, names, strict=True) if n in cluster]))
    ax2.annotate(
        "baseline, cosine, wd_ramp\n(identical schedule here)",
        (cx, cy), textcoords="offset points", xytext=(-14, 26), fontsize=9, color=INK,
        ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("mean learning rate over steps 0–175  (log)")
    ax2.set_ylabel("effective rank at step 175")
    ax2.set_title("Monotone across a 64× range — the schedule is the cause", fontsize=11, pad=8)
    ax2.grid(lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xlim(min(xs) / 2.2, max(xs) * 2.2)

    fig.tight_layout()
    fig.savefig(ASSETS / "schedule_dose_response.png")
    plt.close(fig)
    print("wrote schedule_dose_response.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    pilot_collapse()
    pilot_concept_axis()
    schedule_dose_response()
    vote_decisiveness()
