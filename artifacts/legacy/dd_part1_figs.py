"""Brief DD, Part 1 figures (TOOL VALIDATION; from saved maps, no GPU).

  uv run python artifacts/dd_part1_figs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
import dd_part1 as P  # noqa: E402
from dd_step0 import _rgb  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAG = "TOOL VALIDATION on M — no morphology claims"
SHOW = {"t03_bar_a06_bar": "bar", "t04_spiral_a08_spiral": "spiral", "t02_edgeon_a04_yes": "edge-on",
        "band_offset_pc1": "band offset (PC1)", "norm": "concept-free (Δ‖·‖)"}


def _overlay(ax, stamp, m, lim=None, title=""):
    ax.imshow(_rgb(stamp), origin="lower")
    lim = lim or float(np.percentile(np.abs(m), 99)) or 1.0
    ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55, extent=(-0.5, 255.5, -0.5, 255.5),
              interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, name):
    D.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(D.OUT / name, dpi=115)
    plt.close(fig)
    print(D.OUT / name)


def main() -> None:
    ids, st = D.stamps("occl")
    groups = json.loads((D.LOCAL / "occl_groups.json").read_text())
    names = json.loads((P.P1 / "names.json").read_text())
    m1 = np.load(P.P1 / "m_drop1.npy", mmap_mode="r")
    score = json.loads((P.P1 / "score.json").read_text())

    # 1. primary maps on positives
    pick = []
    for a in ("t03_bar_a06_bar", "t04_spiral_a08_spiral", "t02_edgeon_a04_yes"):
        pick += [k for k, o in enumerate(ids) if groups[str(int(o))] == f"{a}:pos"][:2]
    fig, axes = plt.subplots(len(pick), 6, figsize=(13, 2.3 * len(pick)))
    for r, k in enumerate(pick):
        axes[r, 0].imshow(_rgb(st[k]), origin="lower")
        axes[r, 0].set_title(f"{ids[k]}\n{groups[str(int(ids[k]))].replace('_', ' ')}", fontsize=6)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, (n, lab) in enumerate(SHOW.items(), 1):
            _overlay(axes[r, c], st[k], m1[k, names.index(n)], title=lab)
    fig.suptitle(f"DD Part 1: occlusion maps (token drop, single patch; red = the patch supports the read-out). {TAG}",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "dd_part1_maps.png")

    # 2. modes and scales
    sec = [int(k) for k in np.load(P.P1 / "secondary_rows.npy")]
    mode = {t: np.load(P.P1 / f"m_{t}.npy", mmap_mode="r") for t in ("drop2", "noise1", "noise2")}
    sp = [j for j, k in enumerate(sec) if groups[str(int(ids[k]))].endswith(":pos")][:4]
    fig, axes = plt.subplots(len(sp), 5, figsize=(11, 2.3 * len(sp)))
    for r, j in enumerate(sp):
        k = sec[j]
        a = groups[str(int(ids[k]))].rsplit(":", 1)[0]
        ia = names.index(a)
        axes[r, 0].imshow(_rgb(st[k]), origin="lower")
        axes[r, 0].set_title(f"{ids[k]}\nmap: {a.split('_', 2)[-1]}", fontsize=6)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, (lab, m) in enumerate((("drop, 1 patch", m1[k, ia]), ("drop, 2×2", mode["drop2"][j, ia]),
                                      ("sky noise, 1 patch", mode["noise1"][j, ia]),
                                      ("sky noise, 2×2", mode["noise2"][j, ia])), 1):
            _overlay(axes[r, c], st[k], m, title=f"{lab}  ρ vs drop1 {P._rho(m, m1[k, ia]):.2f}")
    fig.suptitle(f"DD Part 1: removal modes and scales, same read-out. {TAG}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "dd_part1_modes.png")

    # 3. cascade
    rows = [int(k) for k in np.load(P.P1 / "v1_rows.npy")]
    lv = [0, 2, 4, 6, 8, 10, 11, 12, 13]
    cas = {v: np.load(P.P1 / f"cascade_L{v:02d}.npy", mmap_mode="r") for v in lv}
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(3, len(lv) + 3)
    ax = fig.add_subplot(gs[:, :3])
    for a, curve in score["V1_curve"].items():
        ax.plot(range(len(curve)), curve, marker="o", label=SHOW.get(a, a))
    pl = json.loads((P.P1 / "plants.json").read_text())
    ax.axhline(0.3, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(14))
    ax.set_xticklabels(["M", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "+pe"], fontsize=7)
    ax.set_xlabel("blocks randomised, top-down (cumulative)")
    ax.set_ylabel("median |Spearman| with M's map")
    ax.set_title("V1 cascade — reads UNREACHABLE: randomised networks' maps\n"
                 f"barely agree with each other (seed-2 vs seed-1: bar {pl['V1 t03_bar_a06_bar']['seed2_vs_seed1']['median']:.2f})",
                 fontsize=8)
    ax.legend(fontsize=7)
    for r, a in enumerate(("t03_bar_a06_bar", "t04_spiral_a08_spiral", "t02_edgeon_a04_yes")):
        j = next(j for j, k in enumerate(rows) if groups[str(int(ids[k]))] == f"{a}:pos")
        for c, v in enumerate(lv):
            axc = fig.add_subplot(gs[r, 3 + c])
            _overlay(axc, st[rows[j]], cas[v][j, names.index(a)], title=("M" if v == 0 else f"L{v}") if r == 0 else "")
            if c == 0:
                axc.set_ylabel(SHOW[a], fontsize=7)
    fig.suptitle(f"DD Part 1: cascading randomisation (blocks 12 → 1, then the patch embedding). {TAG}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "dd_part1_cascade.png")

    # 4. V2
    v2rows = [int(k) for k in np.load(P.P1 / "v2_rows.npy")]
    g = np.load(P.P1 / "v2_gshift.npy")
    ib = names.index("band_offset_pc1")
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(4, 6)
    for r, j in enumerate(range(4)):
        k = v2rows[j]
        be = P.bright_edge(st[k])
        lim = float(np.percentile(np.abs(m1[k, ib]), 99))
        a0 = fig.add_subplot(gs[r, 0])
        a0.imshow(_rgb(st[k]), origin="lower")
        a0.contour(np.kron(be, np.ones((16, 16))), levels=[0.5], colors="y", linewidths=0.6)
        a0.set_title("stamp + bright-edge patches" if r == 0 else "", fontsize=7)
        a0.set_xticks([])
        a0.set_yticks([])
        _overlay(fig.add_subplot(gs[r, 1]), st[k], m1[k, ib], lim, "band map, unshifted" if r == 0 else "")
        _overlay(fig.add_subplot(gs[r, 2]), st[k], g[j, ib], lim, "band map, g rolled 1 px" if r == 0 else "")
        _overlay(fig.add_subplot(gs[r, 3]), st[k], g[j, ib] - m1[k, ib], lim, "difference" if r == 0 else "")
    ax = fig.add_subplot(gs[:, 4:])
    m0 = [P.edge_mass(m1[k, ib], P.bright_edge(st[k])) for k in v2rows]
    ms = [P.edge_mass(g[j, ib], P.bright_edge(st[k])) for j, k in enumerate(v2rows)]
    ax.scatter(m0, ms, s=12)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.axvline(0.1, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("bright-edge mass, unshifted")
    ax.set_ylabel("bright-edge mass, g rolled 1 px")
    v = score["V2"]
    ax.set_title(f"V2 reads FAIL: Δ median {v['delta_g']:.3f}, p {v['p_g']:.2f}.\nThe shift moves pooled PC1 by +0.61 SD "
                 "and rewrites the map\n(Σ|Δ| ≈ 0.6 Σ|map|) on the patches where it already sits (0.54 vs uniform 0.10)",
                 fontsize=8)
    fig.suptitle(f"DD Part 1, V2: the band-offset (AA3a PC1) map. {TAG}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "dd_part1_v2.png")

    # 5. V4
    v4rows = [int(k) for k in np.load(P.P1 / "v4_rows.npy")]
    fig, axes = plt.subplots(3, 7, figsize=(15, 6.8))
    k, j = v4rows[0], 0
    for r, n in enumerate(("t04_spiral_a08_spiral", "t03_bar_a06_bar", "band_offset_pc1")):
        i = names.index(n)
        _overlay(axes[r, 0], st[k], m1[k, i], title=f"{SHOW[n]}: original" if r == 0 else "original")
        for c, op in enumerate(P.V4_OPS, 0):
            fn = P.V4_OPS[op]
            tr = np.load(P.P1 / f"v4_{op}.npy", mmap_mode="r")[j, i]
            img = np.ascontiguousarray(fn(np.asarray(st[k], np.float32)))
            _overlay(axes[r, 1 + 2 * c], img, tr, title=f"map of {op} stamp")
            _overlay(axes[r, 2 + 2 * c], img, fn(m1[k, i]), title=f"{op} of map  ρ {P._rho(tr, fn(m1[k, i])):.2f}")
        axes[r, 0].set_ylabel(SHOW[n], fontsize=8)
    med = {k2: v[0] for k2, v in score["V4"].items()}
    fig.suptitle(f"DD Part 1, V4 (exploratory): equivariance. Median ρ over 100: spiral rot90/180/flip "
                 f"{med['rot90 t04_spiral_a08_spiral']:.2f}/{med['rot180 t04_spiral_a08_spiral']:.2f}/"
                 f"{med['flip t04_spiral_a08_spiral']:.2f}; band offset {med['rot90 band_offset_pc1']:.2f}/"
                 f"{med['rot180 band_offset_pc1']:.2f}/{med['flip band_offset_pc1']:.2f} (sign flips: a directional "
                 f"quantity). {TAG}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "dd_part1_v4.png")

    # 6. answers with < 20 positives: example maps only
    rare = [n for n in names if any(t in n for t in ("a20_lens", "a26_boxy", "a21_disturbed"))]
    fig, axes = plt.subplots(len(rare), 4, figsize=(8.5, 2.3 * len(rare)))
    for r, n in enumerate(rare):
        ks = [k for k, o in enumerate(ids) if groups[str(int(o))] == f"{n}:pos"][:4]
        for c in range(4):
            if c < len(ks):
                _overlay(axes[r, c], st[ks[c]], m1[ks[c], names.index(n)], title=f"{n.split('_', 2)[-1]}  {ids[ks[c]]}")
            else:
                axes[r, c].axis("off")
    fig.suptitle(f"DD Part 1: answers with < 20 positives — example maps only, no statistics. {TAG}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "dd_part1_rare.png")

    # 7. V3: why the pre-registered comparison saturates (labels, footprint, centre prior; no M maps)
    import dd_v3
    lst = json.loads((D.LOCAL / "v3_list.json").read_text())["lists"]
    lab = np.load(D.LOCAL / "v3_patch_labels.npz")
    targets = {t["id"]: t for t in json.loads((D.LOCAL / "v3_targets.json").read_text())}
    show = lst["bar"][:3] + lst["spiral"][:3]
    stv = dd_v3._stamps_for(show)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    cp = P.centre_prior()
    ring = np.floor(-cp / 16).astype(int)
    for c, o in enumerate(show):
        s = "bar" if c < 3 else "spiral"
        dom, lb = lab[f"{o}__footprint"], lab[f"{o}__{s}_3"]
        axes[0, c].imshow(_rgb(stv[c]), origin="lower")
        axes[0, c].contour(np.kron(dom, np.ones((16, 16))), levels=[0.5], colors="w", linewidths=0.6, linestyles=":")
        axes[0, c].contour(np.kron(lb, np.ones((16, 16))), levels=[0.5], colors="cyan" if s == "bar" else "magenta",
                           linewidths=0.9)
        axes[0, c].set_title(f"{o}: {s} patches (≥3 votes, ≥50%)", fontsize=6)
        axes[1, c].imshow(np.where(dom, ring, np.nan), origin="lower", cmap="tab10", interpolation="nearest")
        axes[1, c].contour(lb.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        axes[1, c].set_title("16-px radial rings inside the footprint", fontsize=6)
        for a in axes[:, c]:
            a.set_xticks([])
            a.set_yticks([])
        assert targets[o]
    fig.suptitle("DD V3 — WITHHELD (D28). Inside the GZ3D footprint the positive patches are the central ones: the centre "
                 "prior alone scores median AUC 0.978 (bar) / 0.956 (arms),\nso a noisy oracle cannot beat it. Candidate: "
                 "compare patches only within the same radial ring (centre prior = 0.5 by construction). No M statistic "
                 "computed.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "dd_v3_saturation.png")


if __name__ == "__main__":
    main()
