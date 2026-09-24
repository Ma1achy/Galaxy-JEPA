"""Brief V3 — the encoder's own directions: salience, subspaces, a held-out label-free axis.

Pre-registered in artifacts/v_findings.md §V3 (hashed before this ran). M's banked embedding (the
ladder's DEFAULT_LAYER = -2 pooled tensor), P2's split: A = train (discovery, alignment), B = test
(held out; read only after the alignment is frozen to disk).

  spectrum  : covariance PCA on A (centred, not rescaled); PR, k* = round(PR); split-half stability
  salience  : CAV energy in the top-k* subspace against 10,000 shuffled-label CAVs (BY, m = gated)
  subspaces : principal angles of the concept and nuisance CAV spans with the top-k* subspace
  held-out  : one PC (of 50) + sign per feature chosen on A, frozen, AUC on B vs the probe's
  discovery : stable top-20 components matching no vote fraction -> correlates + extreme grids

    uv run python artifacts/v3_own_directions.py  -> artifacts/out/v3_own_directions.json (+ PNGs)
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
from t2_other import to_rgb  # noqa: E402
from u2_uncertainty import gated_features  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import probe_direction  # noqa: E402

K_POOL, TOP, N_PERM, N_GROUP_NULL, BATCH = 50, 20, 10_000, 200, 500
MIN_CLASS, STABLE, NAMED, PROXY = 50, 0.9, 0.3, 0.5
KS = (1, 2, 5, 10, 20, 50)
NUISANCES = ("magnitude", "size", "snr", "redshift", "psf")
OUT = R.OUT / "v3_own_directions.json"
ALIGN = R.OUT / "v3_alignment.json"


def sp(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1]) if ok.sum() > 10 else np.nan


def pca(x: np.ndarray, scale: bool = False):
    mu = x.mean(0)
    xc = x - mu
    if scale:
        xc = xc / xc.std(0)
    lam, v = np.linalg.eigh(xc.T @ xc / (x.shape[0] - 1))
    return mu, lam[::-1], v[:, ::-1]


def pr(lam) -> float:
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def energy(v: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Share of each column of ``c``'s squared norm inside span(v[:, :k])."""
    c = c.reshape(c.shape[0], -1)
    return (np.sum((v[:, :k].T @ c) ** 2, axis=0) / np.sum(c ** 2, axis=0)).squeeze()


def null_energy(x: np.ndarray, y: np.ndarray, v: np.ndarray, k: int, rng) -> np.ndarray:
    """E_k of 10,000 CAVs of permuted labels (same class sizes), batched as X^T W."""
    n1, n0 = int(y.sum()), int((1 - y).sum())
    xf = x.astype(np.float32)
    out = []
    for s in range(0, N_PERM, BATCH):
        b = min(BATCH, N_PERM - s)
        yy = rng.permuted(np.broadcast_to(y, (b, y.size)).copy(), axis=1).T.astype(np.float32)
        w = yy / n1 - (1 - yy) / n0
        out.append(energy(v, (xf.T @ w).astype(np.float64), k))
    return np.concatenate(out)


def cav(x, y) -> np.ndarray:
    return x[y == 1].mean(0) - x[y == 0].mean(0)


def auc_perm(score: np.ndarray, y: np.ndarray, rng) -> tuple[float, float]:
    r = rankdata(score)
    n1, n0 = int(y.sum()), int(y.size - y.sum())
    obs = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    hits = 0
    for s in range(0, N_PERM, BATCH):
        b = min(BATCH, N_PERM - s)
        pr_ = rng.permuted(np.broadcast_to(r, (b, r.size)).copy(), axis=1)[:, :n1].sum(1)
        a = (pr_ - n1 * (n1 + 1) / 2) / (n1 * n0)
        hits += int(np.sum(np.abs(a - 0.5) >= abs(obs - 0.5) - 1e-12))
    return float(obs), (1 + hits) / (1 + N_PERM)


def auc(score, y) -> float:
    r = rankdata(score)
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (y.size - n1)))


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="V3", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    lab, union = setup.labels, ctx.real.object_ids
    pos = {int(o): i for i, o in enumerate(union)}
    A = np.array([pos[int(o)] for o in setup.train_ids])
    B = np.array([pos[int(o)] for o in setup.test_ids])
    feats = list(lab.features)
    gated = gated_features()["full"]
    rng = np.random.default_rng(setup.pc.seed)
    rec: dict = {"gated": gated}

    # label-free covariates over the union (A used for naming, B for the grids)
    bank = np.load(R.OUT / "r3_orientation.npz", allow_pickle=False)
    assert np.array_equal(bank["ids"], union)
    ab, th, q = bank["ab"], bank["theta"], bank["q"]
    elong = (ab <= R.R3_MAX_AB) & (q <= R.R3_MAX_Q) & np.isfinite(th)
    cov = {n: np.where(np.asarray(lab.nuisance_valid(n, union.tolist()), bool),
                       lab.nuisance_value(n, union.tolist()), np.nan) for n in NUISANCES}
    cov["expAB_r"] = ab
    cov["cos2theta"] = np.where(elong, np.cos(np.radians(2 * th)), np.nan)
    cov["sin2theta"] = np.where(elong, np.sin(np.radians(2 * th)), np.nan)
    votes = {}
    for f in feats:
        ids = feature_ids(ctx.real, lab, f, setup.train_ids)
        votes[f] = (np.array([pos[int(o)] for o in ids]), lab.vote_fraction(f, ids))

    def spectrum_of(m, name):
        x = m.x[A].astype(np.float64)
        mu, lam, v = pca(x)
        k_star = int(round(pr(lam)))
        half = rng.permutation(A.size) % 2 == 0
        v1, v2 = pca(x[half])[2], pca(x[~half])[2]
        stab = np.max(np.abs(v1[:, :K_POOL].T @ v2[:, :2 * K_POOL]), axis=1)
        return {"x": x, "mu": mu, "lam": lam, "v": v, "k_star": k_star, "stability": stab,
                "name": name}

    specs = [spectrum_of(m, m.encoder_name) for m in [ctx.real, *ctx.untrained]]
    S = specs[0]
    lam = S["lam"]
    rec["spectrum"] = {"participation_ratio": pr(lam), "k_star": S["k_star"],
                       "top50_share": (lam[:50] / lam.sum()).tolist(),
                       "cum_share": {k: float(lam[:k].sum() / lam.sum()) for k in KS},
                       "stability_top50": S["stability"].tolist(),
                       "untrained_pr": [pr(s["lam"]) for s in specs[1:]],
                       "correlation_pca_pr": pr(pca(S["x"], scale=True)[1])}
    k_star = S["k_star"]
    print(f"V3 spectrum: PR {pr(lam):.1f} -> k* {k_star}; top-k* share "
          f"{lam[:k_star].sum() / lam.sum():.3f}", file=sys.stderr)

    # naming the top-20 components on A
    scores_a = (S["x"] - S["mu"]) @ S["v"][:, :K_POOL]
    naming = []
    for j in range(TOP):
        sa = scores_a[:, j]
        nz = {n: sp(sa, cov[n][A]) for n in cov}
        rows_pos = {p: i for i, p in enumerate(A)}
        vt = {}
        for f, (rows, fr) in votes.items():
            idx = np.array([rows_pos[p] for p in rows])
            vt[f] = sp(sa[idx], fr)
        best_v = max(vt, key=lambda f: abs(vt[f]) if np.isfinite(vt[f]) else -1)
        naming.append({"pc": j + 1, "share": float(lam[j] / lam.sum()),
                       "stable": bool(S["stability"][j] >= STABLE),
                       "stability": float(S["stability"][j]), "nuisance": nz,
                       "best_vote": best_v, "best_vote_rho": vt[best_v], "votes": vt})
    rec["naming"] = naming

    # salience: CAV energy in the top-k* subspace against shuffled-label CAVs
    sal = {}
    nulls_mod.assert_null_resolution(N_PERM, alpha=0.05, method="benjamini_yekutieli",
                                     n_tests=len(gated))
    for f in feats:
        emb = feature_embeddings(ctx.real, lab, f, setup.train_ids)
        y = emb.y.astype(int)
        n1, n0 = int(y.sum()), int(y.size - y.sum())
        r = {"n1": n1, "n0": n0}
        if min(n1, n0) < MIN_CLASS:
            r["verdict"] = "INSUFFICIENT"
            sal[f] = r
            continue
        xc = emb.x - S["mu"]
        c = cav(xc, y)
        e = float(energy(S["v"], c, k_star))
        nul = null_energy(xc, y, S["v"], k_star, rng)
        med = float(np.median(nul))
        p = (1 + np.sum(np.abs(nul - med) >= abs(e - med) - 1e-12)) / (1 + N_PERM)
        w = probe_direction(emb, name=f, c=setup.pc.c).w_unit
        r.update(energy=e, null_median=med, null_q01_q99=np.percentile(nul, [1, 99]).tolist(),
                 ratio=e / med, p=float(p), iso=k_star / 384,
                 sweep={k: float(energy(S["v"], c, k)) for k in KS},
                 logistic_energy=float(energy(S["v"], w, k_star)),
                 untrained=[float(energy(s["v"], cav(m.x[np.array(
                     [pos[int(o)] for o in feature_ids(ctx.real, lab, f, setup.train_ids)])]
                     - s["mu"], y), s["k_star"])) for s, m in zip(specs[1:], ctx.untrained,
                                                                 strict=True)])
        sal[f] = r
    sig = nulls_mod.family_significant({f: sal[f]["p"] for f in gated if "p" in sal[f]},
                                       alpha=0.05, method="benjamini_yekutieli",
                                       n_tests=len(gated))
    for f in gated:
        r = sal[f]
        if "p" not in r:
            continue
        v = ("SALIENT" if sig[f] and r["energy"] > r["null_median"] else
             "SUBMERGED" if sig[f] else "UNREMARKABLE")
        if sig[f] and 0.9 <= r["ratio"] <= 1.1:
            v += " (negligible)"
        r["verdict"] = v
    # nuisance positive controls
    nui = {}
    for n in NUISANCES:
        val = cov[n][A]
        ok = np.isfinite(val)
        y = (val[ok] > np.median(val[ok])).astype(int)
        xc = S["x"][ok] - S["mu"]
        c = cav(xc, y)
        nul = null_energy(xc, y, S["v"], k_star, rng)
        med = float(np.median(nul))
        e = float(energy(S["v"], c, k_star))
        nui[n] = {"energy": e, "null_median": med, "ratio": e / med,
                  "p": float((1 + np.sum(np.abs(nul - med) >= abs(e - med) - 1e-12))
                             / (1 + N_PERM)),
                  "sweep": {k: float(energy(S["v"], c, k)) for k in KS}}
    rec["salience"], rec["nuisance_controls"] = sal, nui
    R._save(OUT, rec)
    print("V3 salience: " + ", ".join(f"{f.split('_', 1)[1][:22]} {sal[f].get('verdict')}"
                                      for f in gated), file=sys.stderr)

    # subspaces: principal angles with the top-k* subspace
    def span(cs):
        return np.linalg.qr(np.column_stack(cs))[0]

    def angles(qg):
        s = np.linalg.svd(qg.T @ S["v"][:, :k_star], compute_uv=False)
        return s.tolist(), float(np.sum(s ** 2) / min(qg.shape[1], k_star))

    concept_c, shuffled = [], []
    for f in gated:
        emb = feature_embeddings(ctx.real, lab, f, setup.train_ids)
        concept_c.append(cav(emb.x - S["mu"], emb.y.astype(int)))
    nuis_c = []
    for n in NUISANCES:
        val = cov[n][A]
        ok = np.isfinite(val)
        nuis_c.append(cav(S["x"][ok] - S["mu"], (val[ok] > np.median(val[ok])).astype(int)))
    null_o = []
    embs = [feature_embeddings(ctx.real, lab, f, setup.train_ids) for f in gated]
    for _ in range(N_GROUP_NULL):
        shuffled = [cav(e.x - S["mu"], rng.permutation(e.y.astype(int))) for e in embs]
        null_o.append(angles(span(shuffled))[1])
    ca, co = angles(span(concept_c))
    na, no = angles(span(nuis_c))
    rec["subspaces"] = {"concept": {"dim": len(concept_c), "cosines": ca, "overlap": co},
                        "nuisance": {"dim": len(nuis_c), "cosines": na, "overlap": no},
                        "iso_concept": max(len(concept_c), k_star) / 384,
                        "iso_nuisance": max(len(nuis_c), k_star) / 384,
                        "shuffled_concept_overlap_q": np.percentile(null_o, [1, 50, 99]).tolist()}
    R._save(OUT, rec)

    # held-out: align one component per feature on A, freeze, then test on B
    align = {}
    for s in specs:
        sa = (s["x"] - s["mu"]) @ s["v"][:, :K_POOL]
        rows_pos = {p: i for i, p in enumerate(A)}
        al = {}
        for f in gated:
            rows, fr = votes[f]
            idx = np.array([rows_pos[p] for p in rows])
            rho = np.array([sp(sa[idx, j], fr) for j in range(K_POOL)])
            j = int(np.nanargmax(np.abs(rho)))
            al[f] = {"pc": j + 1, "sign": int(np.sign(rho[j])), "rho_A": float(rho[j])}
        align[s["name"]] = al
    ALIGN.write_text(json.dumps({"k_pool": K_POOL, "alignment": align}, indent=1))  # frozen
    print(f"V3 alignment frozen -> {ALIGN}", file=sys.stderr)

    held = {}
    for f in gated:
        r: dict = {}
        te = feature_embeddings(ctx.real, lab, f, setup.test_ids)
        y = te.y.astype(int)
        r["n1"], r["n0"] = int(y.sum()), int(y.size - y.sum())
        if min(r["n1"], r["n0"]) < MIN_CLASS:
            r["verdict"] = "INSUFFICIENT"
            held[f] = r
            continue
        te_rows = np.array([pos[int(o)] for o in feature_ids(ctx.real, lab, f, setup.test_ids)])
        tr = feature_embeddings(ctx.real, lab, f, setup.train_ids)
        a = align[S["name"]][f]
        score = (te.x - S["mu"]) @ S["v"][:, a["pc"] - 1] * a["sign"]
        r["auc_pc"], r["p"] = auc_perm(score, y, rng)
        d = probe_direction(tr, name=f, c=setup.pc.c)
        r["auc_probe"] = auc(te.x @ d.w_raw, y)
        r["recovery"] = (r["auc_pc"] - 0.5) / (r["auc_probe"] - 0.5)
        r["pc"], r["sign"], r["rho_A"] = a["pc"], a["sign"], a["rho_A"]
        j = a["pc"] - 1
        r["pc_nuisance"] = {n: sp(scores_a[:, j], cov[n][A]) for n in NUISANCES}
        rec_u = []
        for s, m in zip(specs[1:], ctx.untrained, strict=True):
            au = align[s["name"]][f]
            xs = m.x[te_rows]
            sc = (xs - s["mu"]) @ s["v"][:, au["pc"] - 1] * au["sign"]
            trm = feature_embeddings(m, lab, f, setup.train_ids)
            du = probe_direction(trm, name=f, c=setup.pc.c)
            ap, apr = auc(sc, y), auc(xs @ du.w_raw, y)
            rec_u.append({"pc": au["pc"], "auc_pc": ap, "auc_probe": apr,
                          "recovery": (ap - 0.5) / (apr - 0.5)})
        r["untrained"] = rec_u
        held[f] = r
    sig = nulls_mod.family_significant({f: held[f]["p"] for f in held if "p" in held[f]},
                                       alpha=0.05, method="benjamini_yekutieli",
                                       n_tests=len(gated))
    for f, r in held.items():
        if "p" not in r:
            continue
        umax = max(u["recovery"] for u in r["untrained"])
        if sig[f] and r["auc_pc"] < 0.5:
            v = "INVERTED"
        elif not sig[f]:
            v = "NO AXIS"
        elif r["recovery"] <= umax:
            v = "NOT ABOVE UNTRAINED"
        elif r["recovery"] >= 0.8:
            v = "ENCODER AXIS"
        elif r["recovery"] >= 0.5:
            v = "PARTIAL AXIS"
        else:
            v = "WEAK AXIS"
        if max(abs(x) for x in r["pc_nuisance"].values() if np.isfinite(x)) >= PROXY:
            v += " — NUISANCE PROXY"
        r["verdict"] = v
    rec["held_out"] = held
    R._save(OUT, rec)

    # discovery channel: stable, unnamed top-20 components
    disc = []
    sb = (ctx.real.x[B] - S["mu"]) @ S["v"][:, :TOP]
    for c in naming:
        if abs(c["best_vote_rho"]) >= NAMED:
            continue
        j = c["pc"] - 1
        nz = c["nuisance"]
        top_n = max(nz, key=lambda n: abs(nz[n]) if np.isfinite(nz[n]) else -1)
        orient = float(np.nanmax(np.abs([nz["cos2theta"], nz["sin2theta"]])))
        kind = ("ORIENTATION" if orient >= NAMED else
                "NUISANCE" if abs(nz[top_n]) >= NAMED else "UNEXPLAINED")
        entry = {"pc": c["pc"], "stable": c["stable"], "kind": kind, "top_covariate": top_n,
                 "top_rho": nz[top_n], "orientation_rho": orient, "best_vote": c["best_vote"],
                 "best_vote_rho": c["best_vote_rho"]}
        if not c["stable"]:
            entry["note"] = "unstable: near-degenerate, basis-arbitrary; not interpreted singly"
            disc.append(entry)
            continue
        order = np.argsort(sb[:, j])
        for side, idx in (("low", order[:24]), ("high", order[-24:][::-1])):
            path = R.OUT / f"v3_pc{c['pc']}_{side}.png"
            grid(path, [B[i] for i in idx], [float(sb[i, j]) for i in idx], setup,
                 f"PC{c['pc']} {side} end (B galaxies); stable={c['stable']}; {kind}")
            entry[f"grid_{side}"] = str(path)
        disc.append(entry)
    rec["discovery"] = disc or "NO UNNAMED COMPONENT"
    R._save(OUT, rec)
    print(f"V3 discovery: {[(d['pc'], d['kind'], d['top_covariate']) for d in disc]}",
          file=sys.stderr)


def grid(path, rows, scores, setup, title) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 6, figsize=(13, 9.4))
    for ax, i, s in zip(axes.flat, rows, scores, strict=False):
        ax.imshow(to_rgb(setup.ds[int(i)]["image"].float().numpy()), origin="lower",
                  interpolation="nearest")
        ax.set_title(f"{s:+.1f}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
