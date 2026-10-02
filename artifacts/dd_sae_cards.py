"""Brief DD, Part 3 feature cards (Stop 3; M, block 11, the chosen dictionary). EXPLORATORY.

Per latent, over the 5,000 held-out sae_eval galaxies:
- density;
- top-activating patches (3×3-patch context crops);
- radial histogram of firing;
- galaxy-level Spearman with the 37 vote fractions (each answer's eligible population);
- galaxy-level Spearman with the nuisance panel (per-band offsets, psfWidth_r, modelMag_r, specz,
  camcol, frame-edge distance, valid fraction);
- brightness: galaxy total r flux, and token-level patch r flux on a 200k-token subsample.
Flag: the strongest |ρ| is nuisance or brightness. Ranking as pre-registered (interp_tooling.md).

  uv run python artifacts/dd_sae_cards.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
import dd_sae as S  # noqa: E402
from dd_sae_score import OUT, chosen_size, galaxy_acts, spearman_cols  # noqa: E402
from dd_step0 import _rgb  # noqa: E402

TOP, DENS = 12, (1e-4, 0.1)
_c = (np.arange(16) + 0.5) * 16 - 128
RADIUS = np.hypot(*np.meshgrid(_c, _c)).ravel()  # per token


def _panel(ids: np.ndarray) -> dict[str, np.ndarray]:
    """Nuisance variables per sae_eval galaxy (NaN where unrecorded)."""
    want = {int(i): k for k, i in enumerate(ids)}
    cols = {n: np.full(len(ids), np.nan) for n in
            ("g-r x", "g-r y", "i-r x", "i-r y", "psfWidth_r", "modelMag_r", "specz", "camcol",
             "edge_dist_r", "valid_frac")}
    with open(D.REPO / "data" / "probe" / "metadata.csv") as fh:
        for r in csv.DictReader(fh):
            k = want.get(int(r["object_id"]))
            if k is not None:
                for n in ("psfWidth_r", "modelMag_r", "specz", "camcol"):
                    cols[n][k] = float(r[n])
    with open(D.REPO / "data" / "probe_v2" / "cut_log.csv") as fh:
        for r in csv.DictReader(fh):
            k = want.get(int(r["object_id"]))
            if k is not None:
                f = {n: float(r[n]) for n in ("g_v1_relx", "g_v1_rely", "r_v1_relx", "r_v1_rely", "i_v1_relx",
                                             "i_v1_rely", "r_edge_dist", "valid_frac")}
                cols["g-r x"][k] = f["g_v1_relx"] - f["r_v1_relx"]
                cols["g-r y"][k] = f["g_v1_rely"] - f["r_v1_rely"]
                cols["i-r x"][k] = f["i_v1_relx"] - f["r_v1_relx"]
                cols["i-r y"][k] = f["i_v1_rely"] - f["r_v1_rely"]
                cols["edge_dist_r"][k] = f["r_edge_dist"]
                cols["valid_frac"][k] = f["valid_frac"]
    return cols


def _corr_masked(acts: np.ndarray, v: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(v)
    if ok.sum() < 50:
        return np.zeros(acts.shape[1])
    return spearman_cols(acts[ok], v[ok][:, None])[:, 0]


@torch.no_grad()
def _token_stats(sae, tok: np.ndarray, flux_tok: np.ndarray, sub: np.ndarray):
    """Density, radial firing histogram, top-TOP token indices, token-level ρ with patch flux."""
    sae = sae.to(D.DEVICE)
    L = sae.W_enc.shape[0]
    n = len(tok)
    fires = np.zeros(L)
    radial = np.zeros((L, 8))
    ring = np.minimum((RADIUS // 16).astype(int), 7)
    top_v = torch.full((TOP, L), -1.0, device=D.DEVICE)
    top_i = torch.zeros((TOP, L), dtype=torch.long, device=D.DEVICE)
    sub_acts = []
    chunk = 65_536
    for a in range(0, n, chunk):
        z = sae.encode(torch.from_numpy(np.asarray(tok[a:a + chunk], np.float32)).to(D.DEVICE))
        f = (z > 0).float()
        fires += f.sum(0).cpu().numpy()
        r = torch.from_numpy(ring[(np.arange(a, a + len(z)) % 256)]).to(D.DEVICE)
        radial += torch.zeros((8, L), device=D.DEVICE).index_add_(0, r, f).T.cpu().numpy()
        v, i = torch.cat([top_v, z]).topk(TOP, dim=0)
        idx = torch.cat([top_i, torch.arange(a, a + len(z), device=D.DEVICE)[:, None].expand(-1, L)])
        top_v, top_i = v, torch.gather(idx, 0, i)
        m = (sub >= a) & (sub < a + len(z))
        if m.any():
            sub_acts.append(z[torch.from_numpy(sub[m] - a).to(D.DEVICE)].cpu().numpy())
    sa = np.concatenate(sub_acts)
    tok_flux_rho = spearman_cols(sa, flux_tok[:, None])[:, 0]
    return fires / n, radial, top_i.cpu().numpy().T, top_v.cpu().numpy().T, tok_flux_rho


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import r_nonlinear as R
    from j4_spread_controls import prepare

    mult = chosen_size(11)
    sae = S.load(D.TAG, 11, mult)
    ids, st = D.stamps("sae_eval")
    tok = np.load(S.TokenStore.path(D.TAG, 11, "sae_eval"), mmap_mode="r")
    rng = np.random.default_rng(20260925)
    sub = np.sort(rng.choice(len(tok), 200_000, replace=False))
    flux_tok = np.array([float(np.asarray(st[t // 256][1], np.float64)[(t % 256) // 16 * 16:(t % 256) // 16 * 16 + 16,
                                                                         (t % 256) % 16 * 16:(t % 256) % 16 * 16 + 16].sum())
                         for t in sub])
    dens, radial, top_i, top_v, tok_flux = _token_stats(sae, tok, flux_tok, sub)
    acts = galaxy_acts(sae, tok)
    setup = prepare(None, R.MAX_TRAIN, label="DD3c", sources=1)
    lab = setup.labels
    votes: dict[str, np.ndarray] = {}
    for f in lab.features:
        el = set(lab.eligible(f, [int(i) for i in ids]))
        v = lab.vote_fraction(f, [int(i) for i in ids]).astype(float)
        v[[k for k, i in enumerate(ids) if int(i) not in el]] = np.nan
        votes[f] = v
    nuis = _panel(ids)
    nuis["brightness (galaxy r flux)"] = np.array([float(np.asarray(s[1], np.float64).sum()) for s in st])
    rv = {f: _corr_masked(acts, v) for f, v in votes.items()}
    rn = {n: _corr_masked(acts, v) for n, v in nuis.items()}
    rn["brightness (token patch flux)"] = tok_flux
    L = acts.shape[1]
    best_v = np.max(np.abs(np.stack(list(rv.values()))), 0)
    best_vn = np.array(list(rv))[np.argmax(np.abs(np.stack(list(rv.values()))), 0)]
    nstack = np.abs(np.stack(list(rn.values())))
    best_n, best_nn = nstack.max(0), np.array(list(rn))[nstack.argmax(0)]
    flagged = best_n > best_v
    live = (dens >= DENS[0]) & (dens <= DENS[1])
    interp = [int(c) for c in np.argsort(-best_v) if live[c] and not flagged[c]][:20]
    nuisance = [int(c) for c in np.argsort(-best_n) if flagged[c]][:20]
    rec = {"chosen": mult, "n_latents": L, "dead_on_eval": float((dens == 0).mean()),
           "flagged_frac_live": float(flagged[live].mean()), "n_live": int(live.sum()),
           "interpretable": [{"latent": c, "density": float(dens[c]), "best_vote": str(best_vn[c]),
                              "rho_vote": float(rv[best_vn[c]][c]), "best_nuisance": str(best_nn[c]),
                              "rho_nuisance": float(rn[best_nn[c]][c])} for c in interp],
           "nuisance": [{"latent": c, "density": float(dens[c]), "best_nuisance": str(best_nn[c]),
                         "rho_nuisance": float(rn[best_nn[c]][c]), "best_vote": str(best_vn[c]),
                         "rho_vote": float(rv[best_vn[c]][c])} for c in nuisance]}
    (OUT / "cards.json").write_text(json.dumps(rec, indent=1))
    for tag, lst in (("interpretable", interp), ("nuisance", nuisance)):
        for part in range(0, len(lst), 10):
            chunk = lst[part:part + 10]
            fig, axes = plt.subplots(len(chunk), 8, figsize=(16, 2.0 * len(chunk)),
                                     gridspec_kw={"width_ratios": [1] * 6 + [1.3, 2.6]})
            for r, c in enumerate(chunk):
                for j in range(6):
                    t = int(top_i[c, j])
                    g, p = t // 256, t % 256
                    py, px = p // 16, p % 16
                    img = _rgb(st[g])
                    y0, x0 = max(0, (py - 1) * 16), max(0, (px - 1) * 16)
                    axes[r, j].imshow(img[y0:y0 + 48, x0:x0 + 48], origin="lower")
                    axes[r, j].add_patch(plt.Rectangle((px * 16 - x0 - 0.5, py * 16 - y0 - 0.5), 16, 16,
                                                       fill=False, ec="y", lw=0.8))
                    axes[r, j].set_xticks([])
                    axes[r, j].set_yticks([])
                axes[r, 6].bar(range(8), radial[c] / max(radial[c].sum(), 1), color="grey")
                axes[r, 6].set_xticks([0, 7])
                axes[r, 6].set_xticklabels(["centre", "edge"], fontsize=6)
                axes[r, 6].tick_params(labelsize=5)
                axes[r, 7].axis("off")
                axes[r, 7].text(0, 0.5, f"latent {c}   density {dens[c]:.2e}\n"
                                        f"vote: {best_vn[c]} ρ {rv[best_vn[c]][c]:+.2f}\n"
                                        f"nuisance: {best_nn[c]} ρ {rn[best_nn[c]][c]:+.2f}"
                                        + ("   [FLAGGED]" if flagged[c] else ""), fontsize=7, va="center")
            fig.suptitle(f"DD Part 3 feature cards — {tag} ({part + 1}–{part + len(chunk)}), M block 11, {mult}× TopK SAE; "
                         "6 top-activating patches (3×3 context), radial firing histogram. TOOL VALIDATION.", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            path = D.OUT / f"dd_sae_cards_{tag}_{part // 10 + 1}.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            print(path)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("interpretable", "nuisance")}, indent=1))


if __name__ == "__main__":
    main()
