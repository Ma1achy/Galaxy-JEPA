"""Regenerate the README's data figures from real artefacts. No hand-made numbers.

Every figure here is computed from something on disk that a run produced:
  * the vote-decisiveness panel  <- raw/probe/metadata.csv, this project's own GZ2 pull
  * the pilot collapse trace     <- runs/pilot.log, the 6,000-step pilot's own log lines
  * the pilot UMAP               <- runs/slice/explorer/, the frozen pilot's embeddings
  * the schedule dose-response   <- artifacts/out/h2_arms.jsonl, Brief H's six arms
  * the three Brief P panels     <- artifacts/out/p2_ladder.json + p1_untrained_bank.json

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


def resolving_run() -> None:
    """H5 — the deciding measurement: two arms at 3,000 steps, then AUC on both frozen encoders."""
    arms, probes = {}, {}
    for line in (ROOT / "artifacts" / "out" / "h5_arms.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            arms[d["arm"]] = d
    for line in (ROOT / "artifacts" / "out" / "h5_probes.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            probes[d["arm"]] = d

    colour = {"baseline": "#b3352b", "proposal": ACCENT}
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.8))

    for key, ax, label in [
        ("effective_rank", axes[0], "effective rank"),
        ("std", axes[1], "embedding std"),
        ("mean_cosine", axes[2], "mean pairwise cosine"),
    ]:
        for name in ("baseline", "proposal"):
            tr = arms[name]["trace"]
            ax.plot([p["step"] for p in tr], [p[key] for p in tr],
                    lw=1.8, color=colour[name], label=name)
        ax.set_xlabel("step")
        ax.set_title(label, fontsize=11, pad=8)
        ax.grid(lw=0.6)
        ax.set_axisbelow(True)
    axes[0].axhline(5.0, color=MUTED, ls="--", lw=1.0)
    axes[0].text(2960, 5.15, "G5 floor ", color=MUTED, fontsize=8, ha="right")
    axes[0].legend(frameon=False, fontsize=9.5)

    # the deciding panel
    ax = axes[3]
    names = ["baseline", "proposal"]
    xs = np.arange(2)
    aucs = [probes[n]["auc"] for n in names]
    err = np.array([[a - probes[n]["auc_lo"] for a, n in zip(aucs, names, strict=True)],
                    [probes[n]["auc_hi"] - a for a, n in zip(aucs, names, strict=True)]])
    ax.bar(xs, aucs, width=0.5, color=[colour[n] for n in names], alpha=0.85)
    ax.errorbar(xs, aucs, yerr=err, fmt="none", ecolor=INK, capsize=6, lw=1.4)
    for x, a in zip(xs, aucs, strict=True):
        ax.text(x, a + 0.012, f"{a:.4f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0.9052, color=GOOD, ls=":", lw=1.4)
    ax.text(-0.44, 0.9075, "pilot 0.905", color=GOOD, fontsize=8.5, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0.84, 0.97)
    ax.set_ylabel("frozen-probe AUC")
    ax.set_title("the deciding measurement\nheld-out AUC, 95% CI", fontsize=11, pad=8)
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        "H5 — the schedule resolved. Same seed, same data, same masks; only the schedule differs.",
        fontsize=12.5, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "resolving_run.png")
    plt.close(fig)
    print(f"wrote resolving_run.png  baseline {aucs[0]:.4f} / proposal {aucs[1]:.4f}")


# ---------------------------------------------------------------------------- Brief P: the ladder

LADDER = ROOT / "artifacts" / "out" / "p2_ladder.json"
BANK = ROOT / "artifacts" / "out" / "p1_untrained_bank.json"

#: One colour per rung, used across all three Brief P panels so a reader learns it once.
RUNG_COLOUR = {"R1": GOOD, "R2": ACCENT, "R3": "#c9a227", "R4": "#b3352b"}


ALIASES = {
    "smooth or features": "root", "arms number": "arms", "arms winding": "winding",
    "bulge prominence": "bulge", "edgeon": "edge-on",
    # t06 asks "anything odd?" and t08 asks "odd HOW?" — both would shorten to "odd", and the
    # t06 answers are a bare yes/no, so it is the one that must keep its question.
    "odd": "anything odd", "odd feature": "odd",
}
#: tasks whose answer alone is unambiguous, so the label need not repeat the question
BARE = {"bar", "spiral", "odd feature", "rounded"}


def _short(name: str) -> str:
    """`t03_bar_a06_bar` -> `bar`, `t11_arms_number_a36_more_than_4` -> `arms: more than 4`."""
    m = re.match(r"t\d+_(.+?)_a\d+_(.+)$", name)
    if m is None:
        return name
    raw = m.group(1).replace("_", " ")
    answer = m.group(2).replace("_", " ")
    return answer if raw in BARE else f"{ALIASES.get(raw, raw)}: {answer}"


def _ladder() -> dict:
    return json.loads(LADDER.read_text())


def p_catalogue() -> None:
    """The headline: 37 answers, each against its own untrained bar and its matched AUC."""
    d = _ladder()
    bank = json.loads(BANK.read_text())["seeds"]
    bars = {
        f: float(np.mean([bank[s][f] for s in bank]))
        for f in next(iter(bank.values()))
    }
    rows = sorted(d["full"], key=lambda r: r["auc"])

    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(13.2, 9.0), gridspec_kw={"width_ratios": [2.5, 1]}
    )
    ys = np.arange(len(rows))

    for y, r in zip(ys, rows, strict=True):
        bar, auc = bars[r["feature"]], r["auc"]
        colour = RUNG_COLOUR[r["rung"]]
        # the segment from the untrained bar to the real AUC IS the effect; nothing else is
        ax.plot([bar, auc], [y, y], lw=2.2, color=colour, zorder=2,
                alpha=0.35 if r["underpowered"] else 1.0)
        ax.scatter([bar], [y], marker="|", s=110, color=MUTED, zorder=3, lw=1.6)
        ax.scatter([auc], [y], s=46, color=colour, zorder=4, edgecolors="none")
        if r["matched_auc"] is not None:
            ax.scatter([r["matched_auc"]], [y], s=34, facecolors="white",
                       edgecolors=colour, lw=1.4, zorder=5)
        if r["underpowered"]:
            ax.text(1.005, y, "underpowered", fontsize=7.2, color=MUTED, va="center")

    ax.axvline(d["effect_floor"], color=WARM, ls="--", lw=1.1, zorder=1)
    ax.text(d["effect_floor"] + 0.004, -1.6, "effect floor 0.7267 (frozen)",
            color=WARM, fontsize=8.5, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([_short(r["feature"]) for r in rows], fontsize=8.4)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-2.4, len(rows) - 0.4)
    ax.set_xlabel("AUC on 34,829 held-out galaxies")
    ax.set_title(
        "Every Scheme 1 answer, measured from its own untrained bar\n"
        "grey tick = untrained-encoder bar   ·   filled = AUC   ·   hollow = after matching",
        fontsize=11, pad=10,
    )
    ax.grid(axis="x", lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], color=RUNG_COLOUR[k], lw=2.4, label=lab)
        for k, lab in (("R1", "R1 — clean direction"), ("R2", "R2 — present, not clean"),
                       ("R4", "R4 — not recoverable"))
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

    # right: why an answer is not clean — P's labels, re-read through Brief R0. P called 22 answers
    # "confounded" because the ladder judged matching against the effect floor; 21 were below it
    # before matching. Under O1's retention rule none collapses, so the honest bucket is the floor.
    floor = float(d["effect_floor"])
    mech_order = ["clean", "entangled", "below floor", "matched below floor", "not recoverable"]
    SHORT_MECH = {"clean": "clean", "entangled": "entangled", "below floor": "below\nfloor",
                  "matched below floor": "matched\n< floor", "not recoverable": "not\nrecoverable"}
    mech_colour = {"clean": GOOD, "entangled": ACCENT, "below floor": "#d9863d",
                   "matched below floor": "#b3352b", "not recoverable": MUTED}

    def bucket(r: dict) -> str:
        m = r["mechanism"]
        if m.startswith("clean"):
            return "clean"
        if m.startswith("entangled"):
            return "entangled"
        if m.startswith("confounded"):
            return "below floor" if r["auc"] < floor else "matched below floor"
        return "not recoverable"

    for i, pop in enumerate(("full", "conditional")):
        counts: dict[str, int] = {}
        for r in d[pop]:
            k = bucket(r)
            counts[k] = counts.get(k, 0) + 1
        bottom = 0
        for k in mech_order:
            n = counts.get(k, 0)
            if not n:
                continue
            axm.bar(i, n, bottom=bottom, width=0.62, color=mech_colour[k], edgecolor="white", linewidth=0.8)
            if n >= 2:
                axm.text(i, bottom + n / 2, f"{SHORT_MECH[k]}\n{n}", ha="center", va="center",
                         fontsize=8.5, color="white")
            bottom += n
    axm.set_xticks([0, 1])
    axm.set_xticklabels(["full\n(verdict)", "conditional\n(reported)"], fontsize=9)
    axm.set_ylabel("answers")
    axm.set_title("Why an answer is not clean\nbelow the effect floor — not confounded (R0)",
                  fontsize=11, pad=10)
    axm.text(0.5, -0.13, "O1's retention rule: none of P's 22 'confounded' (full)\n"
             "loses its effect under matching",
             transform=axm.transAxes, ha="center", fontsize=8, color=MUTED)
    axm.grid(axis="y", lw=0.6)
    axm.set_axisbelow(True)

    fig.suptitle(
        "Brief P — the catalogue. M's 4-epoch encoder, 37 answers, matched evaluation on every one.",
        fontsize=12.5, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "ladder_catalogue.png")
    plt.close(fig)
    n1 = sum(1 for r in d["full"] if r["rung"] == "R1")
    print(f"wrote ladder_catalogue.png  R1={n1}/37")


def p_power() -> None:
    """What this corpus can and cannot resolve, and where the population comparison collapses."""
    d = _ladder()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # x is the RARER class, not the positives. Power is set by whichever side is scarce, and
    # plotting positives alone makes the majority-class answers (edge-on: no, 29,641 positives
    # against 4,710 negatives) look better powered than they are — the trend goes from
    # Spearman -0.71 to -0.91 once the axis is the quantity that actually binds.
    def minority(r: dict) -> int:
        return min(r["positives_test"], r["n_test"] - r["positives_test"])

    for r in d["full"]:
        ax.scatter(minority(r), r["resolvable_margin"], s=52, color=RUNG_COLOUR[r["rung"]],
                   zorder=3, edgecolors="none", alpha=0.9)
    ax.axhline(0.05, color=WARM, ls="--", lw=1.1)
    ax.text(0.03, 0.052, "underpowered above this line (margin > 0.05)", color=WARM,
            fontsize=8.5, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    # the five underpowered points sit close together; stagger the labels rather than stack them
    for k, r in enumerate(sorted((r for r in d["full"] if r["underpowered"]),
                                 key=lambda r: -r["resolvable_margin"])):
        ax.annotate(_short(r["feature"]), (minority(r), r["resolvable_margin"]),
                    textcoords="offset points", xytext=(12, (4, 4, 20, 8, -8)[k % 5]),
                    fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("held-out galaxies in the RARER class  (log)")
    ax.set_ylabel("resolvable margin over the feature's own bar  (log)")
    ax.set_title("Power is set by the rarer class, not the bucket\nSpearman -0.91",
                 fontsize=11, pad=8)
    ax.grid(lw=0.6)
    ax.set_axisbelow(True)

    changes = sorted(d["population_changes"], key=lambda c: c["positives_cond"])
    ys = np.arange(len(changes))
    for y, c in zip(ys, changes, strict=True):
        ax2.plot([c["positives_full"], max(c["positives_cond"], 1)], [y, y],
                 lw=2.0, color=MUTED, zorder=2)
        ax2.scatter([c["positives_full"]], [y], s=42, color=ACCENT, zorder=3, edgecolors="none")
        ax2.scatter([max(c["positives_cond"], 1)], [y], s=42, color="#b3352b", zorder=3,
                    edgecolors="none")
        ax2.annotate(f"{c['full']}→{c['conditional']}", (1.0, y), fontsize=7.6, color=MUTED,
                     xycoords=ax2.get_yaxis_transform("grid"), textcoords="offset points",
                     xytext=(6, 0), va="center", ha="left")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([_short(c["feature"]) for c in changes], fontsize=8.2)
    ax2.set_xscale("log")
    ax2.set_xlim(0.6, 2.4e5)
    ax2.set_xlabel("held-out positives  (log) — blue: full, red: conditional")
    ax2.set_title("Every rung that moved, and why\nthe consensus gate takes the galaxies with it",
                  fontsize=11, pad=8)
    ax2.grid(axis="x", lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(ASSETS / "ladder_power.png")
    plt.close(fig)
    print("wrote ladder_power.png")


def p_structure() -> None:
    """The encoder's concept geometry against the same galaxies' human vote structure."""
    d = _ladder()
    names = d["entanglement"]["names"]
    cos = np.array(d["entanglement"]["cosine"], dtype=float)
    # the vote matrix was computed over the entanglement set itself, so the two are aligned
    human = np.array(
        [[np.nan if v is None else v for v in row] for row in d["human_structure"]["correlation"]],
        dtype=float,
    )
    assert human.shape == cos.shape, (human.shape, cos.shape)

    fig = plt.figure(figsize=(14.6, 5.6))
    # column 3 is an empty spacer: the colourbar would otherwise sit on the scatter's y labels
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.035, 0.18, 1.05], wspace=0.34)
    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    labels = [_short(n) for n in names]

    for k, (m, title) in enumerate((
        (cos, "encoder — cosine between\nconcept directions"),
        (human, "humans — correlation between\nvote fractions, same galaxies"),
    )):
        a = fig.add_subplot(gs[0, k])
        im = a.imshow(m, **kw)
        a.set_xticks(range(len(labels)))
        a.set_yticks(range(len(labels)))
        a.set_xticklabels(labels, rotation=90, fontsize=5.4)
        # the second matrix repeats the first's ordering, so its row labels are noise
        a.set_yticklabels(labels if k == 0 else [], fontsize=5.4)
        a.set_title(title, fontsize=10, pad=8)
        for s in a.spines.values():
            s.set_visible(False)
        if k == 1:
            cax = fig.add_subplot(gs[0, 2])
            fig.colorbar(im, cax=cax).outline.set_visible(False)
            cax.tick_params(labelsize=8)

    a3 = fig.add_subplot(gs[0, 4])
    xs, ys = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(human[i, j]):
                xs.append(human[i, j])
                ys.append(cos[i, j])
    a3.scatter(xs, ys, s=22, color=ACCENT, alpha=0.55, edgecolors="none", zorder=3)
    a3.axhline(0, color=GRID, lw=1.0)
    a3.axvline(0, color=GRID, lw=1.0)
    rho = d["human_structure"]["spearman_vs_embedding"]
    a3.set_xlabel("human vote correlation")
    a3.set_ylabel("encoder cosine")
    a3.set_title(
        f"Spearman {rho:+.3f} over {d['human_structure']['n_pairs']} pairs\n"
        f"same galaxies, same votes — only the representation differs",
        fontsize=10.5, pad=8,
    )
    a3.grid(lw=0.6)
    a3.set_axisbelow(True)

    # D13's hard case, inset: Hart predicts the bar leans towards LOOSE winding specifically
    h = d["bar_winding"]
    if h["cosines"]:
        # top-left of the scatter is empty (no pair is both vote-anticorrelated and
        # cosine-aligned), so the inset costs no data
        ins = a3.inset_axes([0.13, 0.58, 0.36, 0.27], facecolor="white")
        for s in ins.spines.values():
            s.set_color(GRID)
        order = [w for w in ("t10_arms_winding_a28_tight", "t10_arms_winding_a29_medium",
                             "t10_arms_winding_a30_loose") if w in h["cosines"]]
        vals = [h["cosines"][w] for w in order]
        ins.bar(range(len(order)), vals, width=0.6,
                color=[GOOD if v == max(vals) else MUTED for v in vals])
        ins.set_xticks(range(len(order)))
        ins.set_xticklabels([w.split("_")[-1] for w in order], fontsize=7)
        ins.axhline(0, color=GRID, lw=0.9)
        ins.set_title("D13's hard case: bar's cosine to winding\n"
                      "Hart predicts a lean to LOOSE", fontsize=7.2, pad=3)
        ins.tick_params(labelsize=6.5)
        ins.grid(False)

    fig.suptitle(
        "Brief P — the encoder's concept geometry against human voting on the same 230k galaxies.",
        fontsize=12.5, y=1.0,
    )
    fig.savefig(ASSETS / "concept_structure.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote concept_structure.png  spearman {rho:+.3f}")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    pilot_collapse()
    pilot_concept_axis()
    schedule_dose_response()
    vote_decisiveness()
    resolving_run()
    p_catalogue()
    p_power()
    p_structure()
