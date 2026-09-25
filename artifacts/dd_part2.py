"""Brief DD, Part 2: patch parts — PCA of block-11 patch tokens, first 3 components as RGB. EXPLORATORY.

Tool validation only (artifacts/interp_tooling.md). For M and the untrained encoder (seed 0):
  - PCA of the block-11 tokens (the probe layer) over the 2,000-galaxy occlusion sample, the
    covariance accumulated in float64 on batches (no token matrix held);
  - overlays: each galaxy's 256 tokens projected on PC1–3, mapped to RGB (per-component 1st–99th
    percentile over the sample), beside the stamp;
  - does a component track the band offsets? (a) galaxy-mean PC score against v1's recorded per-band
    in-stamp offsets (g−r, i−r; from probe_v2's cut_log, for the occl galaxies re-cut so far);
    (b) the response of each PC to rolling g by 1 px (the V2 manipulation), on bright-edge tokens.

  uv run python artifacts/dd_part2.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
from dd_part1 import _shift_g, bright_edge  # noqa: E402
from dd_step0 import _rgb  # noqa: E402

P2 = D.LOCAL / "part2"
N_PC, N_SHOW, SEED = 10, 12, 20260925
CUT_LOG = D.REPO / "data" / "probe_v2" / "cut_log.csv"


def _tokens(model, imgs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(imgs, np.float32)).to(D.DEVICE)
    return D.block_tokens(model, x, (D.READ_BLOCK,))[D.READ_BLOCK]


def pca(model, st: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, components (D, N_PC), explained-variance fractions) of all 2,000 × 256 tokens."""
    n, s, ss = 0, None, None
    for a in range(0, len(st), batch):
        t = _tokens(model, st[a:a + batch]).reshape(-1, 384).cpu().double()
        s = t.sum(0) if s is None else s + t.sum(0)
        ss = t.T @ t if ss is None else ss + t.T @ t
        n += len(t)
    mu = (s / n).numpy()
    cov = (ss / n).numpy() - np.outer(mu, mu)
    val, vec = np.linalg.eigh(cov)
    return mu, vec[:, ::-1][:, :N_PC], (val[::-1] / val.sum())[:N_PC]


def project(model, st: np.ndarray, mu: np.ndarray, comp: np.ndarray, batch: int = 64) -> np.ndarray:
    """(N, 256, N_PC) token scores."""
    out = []
    for a in range(0, len(st), batch):
        t = _tokens(model, st[a:a + batch]).cpu().double().numpy()
        out.append(((t - mu) @ comp).astype(np.float32))
    return np.concatenate(out)


def offsets(ids: np.ndarray) -> dict[int, np.ndarray]:
    """v1's per-band in-stamp target position minus r's: (g−r x, g−r y, i−r x, i−r y), px."""
    want = {int(i) for i in ids}
    out = {}
    with open(CUT_LOG) as fh:
        for r in csv.DictReader(fh):
            o = int(r["object_id"])
            if o in want:
                v = {k: float(r[k]) for k in ("g_v1_relx", "g_v1_rely", "r_v1_relx", "r_v1_rely",
                                              "i_v1_relx", "i_v1_rely")}
                out[o] = np.array([v["g_v1_relx"] - v["r_v1_relx"], v["g_v1_rely"] - v["r_v1_rely"],
                                   v["i_v1_relx"] - v["r_v1_relx"], v["i_v1_rely"] - v["r_v1_rely"]])
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P2.mkdir(parents=True, exist_ok=True)
    ids, st = D.stamps("occl")
    m = D.m_encoder()
    enc = {"M": m, "untrained": D.untrained_encoder(m.config, seed=0)}
    rec: dict = {}
    scores = {}
    for name, model in enc.items():
        mu, comp, ev = pca(model, st)
        sc = project(model, st, mu, comp)
        np.save(P2 / f"{name}_scores.npy", sc)
        np.savez(P2 / f"{name}_pca.npz", mu=mu, comp=comp, ev=ev)
        scores[name] = (sc, mu, comp)
        rec[name] = {"explained": ev.round(4).tolist()}
        print(name, "explained", ev[:5].round(3), flush=True)

    # (a) galaxy-mean PC score against v1's recorded band offsets
    off = offsets(ids)
    rows = [k for k, o in enumerate(ids) if int(o) in off]
    y = np.stack([off[int(ids[k])] for k in rows])
    labels = ["g-r x", "g-r y", "i-r x", "i-r y"]
    for name, (sc, _, _) in scores.items():
        g = sc[rows].mean(axis=1)  # (n, N_PC)
        rho = np.array([[spearmanr(g[:, p], y[:, j])[0] for j in range(4)] for p in range(N_PC)])
        rec[name]["offset_rho"] = {f"PC{p + 1}": dict(zip(labels, rho[p].round(3).tolist(), strict=True))
                                   for p in range(N_PC)}
        rec[name]["offset_n"] = len(rows)
    # (b) response to rolling g by 1 px, on bright-edge vs other tokens, in units of each PC's token SD
    shift_rows = sorted(int(k) for k in np.random.default_rng(SEED).choice(len(ids), 50, replace=False))
    for name, model in enc.items():
        sc, mu, comp = scores[name]
        sd = sc.reshape(-1, N_PC).std(0)
        sh = project(model, np.stack([_shift_g(st[k]) for k in shift_rows]), mu, comp)
        be = np.stack([bright_edge(st[k]).ravel() for k in shift_rows])
        d = np.abs(sh - sc[shift_rows]) / sd
        rec[name]["gshift_response_edge"] = d[be].mean(0).round(3).tolist()
        rec[name]["gshift_response_other"] = d[~be].mean(0).round(3).tolist()
    (P2 / "part2.json").write_text(json.dumps(rec, indent=1))

    # overlays
    show = [int(k) for k in np.random.default_rng(SEED + 2).choice(len(ids), N_SHOW, replace=False)]
    fig, axes = plt.subplots(3, N_SHOW, figsize=(2.1 * N_SHOW, 7))
    for j, k in enumerate(show):
        axes[0, j].imshow(_rgb(st[k]), origin="lower")
        axes[0, j].set_title(str(ids[k]), fontsize=6)
        for i, name in enumerate(("M", "untrained"), 1):
            sc = scores[name][0]
            lo, hi = np.percentile(sc[..., :3].reshape(-1, 3), [1, 99], axis=0)
            rgb = np.clip((sc[k, :, :3] - lo) / (hi - lo), 0, 1).reshape(D.GRID, D.GRID, 3)
            axes[i, j].imshow(rgb, origin="lower", interpolation="nearest")
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    axes[1, 0].set_ylabel("M: PC1–3 → RGB", fontsize=9)
    axes[2, 0].set_ylabel("untrained: PC1–3 → RGB", fontsize=9)
    ev = {n: rec[n]["explained"][:3] for n in enc}
    fig.suptitle("DD Part 2 — patch parts (EXPLORATORY). Block-11 patch tokens, PCA over 2,000 × 256 tokens; "
                 f"PC1–3 explain M {sum(ev['M']):.0%}, untrained {sum(ev['untrained']):.0%}.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    D.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(D.OUT / "dd_part2_patch_parts.png", dpi=120)
    plt.close(fig)

    # per-component maps for M (PC1–6), so single components can be read off
    fig, axes = plt.subplots(7, 8, figsize=(16, 14))
    sc = scores["M"][0]
    for j, k in enumerate(show[:8]):
        axes[0, j].imshow(_rgb(st[k]), origin="lower")
        for p in range(6):
            v = sc[k, :, p].reshape(D.GRID, D.GRID)
            lim = np.percentile(np.abs(sc[..., p]), 99)
            axes[p + 1, j].imshow(v, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    for p in range(6):
        axes[p + 1, 0].set_ylabel(f"M PC{p + 1}", fontsize=9)
    fig.suptitle("DD Part 2 — M's first six token components, one at a time (EXPLORATORY)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(D.OUT / "dd_part2_components_M.png", dpi=110)
    plt.close(fig)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
