"""Brief V1 — the suppression check (V1.3) and the escape-hatch test (V1.5).

Pre-registered in artifacts/v_findings.md §V1 (hashed before this ran).

  suppression : U2's headline rows for merger and bulge "obvious" (full, floor 21) — the same bend
                coordinate and ambiguity — partial Spearman controlling ONE composite visibility
                index (PC1 of the four rank-z-scored covariates) instead of the four together.
  escape      : votes against photometry, no embedding. Arm count's plurality "can't tell" vs the
                ordered categories on U3's visibility index (primary); bulge "none" vs the rest on
                seeing and SNR within size × magnitude strata (secondary, exploratory).

    uv run python artifacts/v1_loose_ends.py  -> artifacts/out/v1_loose_ends.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402
from s1_spiral_inclination import bend_direction  # noqa: E402
from u3_graded import auc, visibility_index  # noqa: E402

from galaxy_jepa.data.metadata import GZ2_TREE, vote_column  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import feature_ids  # noqa: E402

SUPPRESSION = ("t08_odd_feature_a24_merger", "t05_bulge_prominence_a12_obvious")
FLOOR, FALLBACK, MIN_GROUP = 21, 10, 100
D02 = 0.556  # AUC at Cohen's d = 0.2
N_PERM = 10_000
OUT = R.OUT / "v1_loose_ends.json"


# ------------------------------------------------------------------ V1.3 suppression


def composite(vis: np.ndarray) -> tuple[np.ndarray, dict]:
    z = np.column_stack([rankdata(c) for c in vis.T])
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    corr = np.corrcoef(z, rowvar=False)
    evals, evecs = np.linalg.eigh(corr)
    load = evecs[:, -1] * np.sign(evecs[0, -1])  # + = fainter (magnitude is column 0)
    return z @ load, {"loadings": dict(zip(U2.VISIBILITY, load.tolist(), strict=True)),
                      "pc1_share": float(evals[-1] / evals.sum()),
                      "shares": (evals[::-1] / evals.sum()).tolist(),
                      "spearman": corr.tolist()}


def rows_for(ctx, f: str, m):
    """U2's measure(): the same test rows, bend coordinate, ambiguity and visibility."""
    s, lab = ctx.setup, U2.with_floor(ctx.labels["full"], FLOOR)
    ids_tr = feature_ids(ctx.real, lab, f, s.train_ids)
    ids_te = feature_ids(ctx.real, lab, f, s.test_ids)
    f_tr, f_te = lab.vote_fraction(f, ids_tr), lab.vote_fraction(f, ids_te)
    amb = 1.0 - 2.0 * np.abs(f_te - 0.5)
    vis = np.column_stack([np.where(np.asarray(lab.nuisance_valid(n, ids_te), bool),
                                    lab.nuisance_value(n, ids_te), np.nan) for n in U2.VISIBILITY])
    vok = np.all(np.isfinite(vis), axis=1)
    z = ctx.z(m)
    bend = z.x[z.rows_for(ids_te)] @ bend_direction(z.x[z.rows_for(ids_tr)], f_tr)
    return bend[vok], amb[vok], vis[vok]


def suppression(ctx) -> dict:
    out = {}
    for i, f in enumerate(SUPPRESSION):
        bend, amb, vis = rows_for(ctx, f, ctx.real)
        pc1, info = composite(vis)
        raw = float(np.corrcoef(rankdata(bend), rankdata(amb))[0, 1])
        rc, p = U2.partial_perm(bend, amb, pc1[:, None], seed=ctx.pc.seed + i)
        four = U2.partial_perm(bend, amb, vis, seed=0, perm=False)[0]
        sp = lambda a, b: float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])  # noqa: E731
        untr = []
        for m in ctx.untrained:
            b_u, a_u, v_u = rows_for(ctx, f, m)
            untr.append(U2.partial_perm(b_u, a_u, composite(v_u)[0][:, None], seed=0,
                                        perm=False)[0])
        out[f] = {"n": int(bend.size), "raw": raw, "four_way_partial": four,
                  "composite_partial": rc, "p": p, "composite": info,
                  "bend_vs": {n: sp(bend, vis[:, j]) for j, n in enumerate(U2.VISIBILITY)},
                  "amb_vs": {n: sp(amb, vis[:, j]) for j, n in enumerate(U2.VISIBILITY)},
                  "bend_vs_pc1": sp(bend, pc1), "amb_vs_pc1": sp(amb, pc1),
                  "untrained_composite_partial": untr}
    sig = nulls_mod.family_significant({f: out[f]["p"] for f in out}, alpha=0.05,
                                       method="benjamini_yekutieli", n_tests=len(out))
    for f, r in out.items():
        rc, half = r["composite_partial"], 0.5 * abs(r["raw"])
        if rc < 0 and sig[f]:
            v = "REVERSAL SURVIVES" + (" (small)" if abs(rc) <= half else "")
        elif rc < 0 and abs(rc) > half:
            v = "REVERSAL, UNRESOLVED"
        elif abs(rc) <= half:
            v = "SUPPRESSION; VISIBILITY ACCOUNTS"
        elif sig[f]:
            v = "SUPPRESSION; ASSOCIATION SURVIVES"
        else:
            v = "SUPPRESSION, UNRESOLVED"
        if r["composite"]["pc1_share"] < 0.5:
            v += " — COMPOSITE INADEQUATE"
        r["significant"], r["verdict"] = bool(sig[f]), v
    return out


# ------------------------------------------------------------------ V1.5 escape hatch


def plurality(lab, question: str, ids: list[int], floor: int):
    answers = GZ2_TREE[question]
    frac = np.column_stack([lab._column(ids, vote_column(question, a)) for a in answers])
    reach = sum(lab._column(ids, vote_column(question, a, "count")) for a in answers)
    ok = np.isfinite(frac).all(axis=1) & (reach >= floor)
    tied = (frac == frac.max(axis=1, keepdims=True)).sum(axis=1) > 1
    keep = ok & ~tied
    top = np.array([answers[t] for t in np.argmax(np.nan_to_num(frac, nan=-1.0), axis=1)])
    return np.asarray(ids)[keep], top[keep], frac[keep]


def perm_auc(x: np.ndarray, g: np.ndarray, strata: np.ndarray | None, seed: int):
    """Stratified AUC P(x[g] > x[~g]), weights n1·n0; one-sided p both ways, labels permuted
    within strata (10,000, add-one)."""
    strata = np.zeros(x.size, int) if strata is None else strata
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    cells = [c for c in cells if min(g[c].sum(), (~g[c]).sum()) >= 10]
    ranks = [rankdata(x[c]) for c in cells]

    def stat(gg):
        num = den = 0.0
        for c, r in zip(cells, ranks, strict=True):
            n1, n0 = int(gg[c].sum()), int((~gg[c]).sum())
            u = r[gg[c]].sum() - n1 * (n1 + 1) / 2
            num, den = num + u, den + n1 * n0
        return num / den

    obs = stat(g)
    rng, hi, lo = np.random.default_rng(seed), 0, 0
    for _ in range(N_PERM):
        gg = g.copy()
        for c in cells:
            gg[c] = rng.permutation(g[c])
        s = stat(gg)
        hi += s >= obs - 1e-12
        lo += s <= obs + 1e-12
    return {"auc": float(obs), "p_greater": (1 + hi) / (1 + N_PERM),
            "p_less": (1 + lo) / (1 + N_PERM), "cells": len(cells),
            "n1": int(sum(g[c].sum() for c in cells)), "n0": int(sum((~g[c]).sum() for c in cells))}


def read(r: dict, fallback: bool) -> str:
    if r["p_greater"] < 0.05:
        v = "SUPPORTED" if r["auc"] >= D02 else "SIGNIFICANT BUT NEGLIGIBLE"
    elif r["p_less"] < 0.05:
        v = "CONTRARY" + (" (negligible)" if r["auc"] > 1 - D02 else "")
    else:
        v = "NOT SUPPORTED"
    return v + (" [fallback: reach ≥ 10]" if fallback else "")


def quintile(v: np.ndarray) -> np.ndarray:
    return np.minimum((rankdata(v) - 1) * 5 // v.size, 4).astype(int)


def column(lab, name: str, ids) -> np.ndarray:
    return np.where(np.asarray(lab.nuisance_valid(name, ids), bool),
                    lab.nuisance_value(name, ids), np.nan)


def escape(setup) -> dict:
    lab, union = setup.labels, list(setup.union)
    bank = np.load(R.OUT / "r3_orientation.npz", allow_pickle=False)  # R3's expAB_r over the union
    if not np.array_equal(bank["ids"], np.asarray(setup.union)):
        raise SystemExit("V1: R3's orientation bank is not over P2's union")
    ab_all = dict(zip(bank["ids"].tolist(), bank["ab"].tolist(), strict=True))
    out: dict = {}
    # primary: arm count
    floor = FLOOR
    ids, top, frac = plurality(lab, "t11_arms_number", union, floor)
    if (top == "a37_cant_tell").sum() < MIN_GROUP:
        floor = FALLBACK
        ids, top, frac = plurality(lab, "t11_arms_number", union, floor)
    n_ct = int((top == "a37_cant_tell").sum())
    prim: dict = {"floor": floor, "n_cant_tell": n_ct, "n_total": int(ids.size)}
    if n_ct < MIN_GROUP:
        prim["verdict"] = "INSUFFICIENT"
        out["arms_number"] = prim
        return out
    V = visibility_index(lab, ids.tolist())
    ok = np.isfinite(V)
    prim["dropped_missing_covariate"] = int((~ok).sum())
    ids, top, frac, V = ids[ok], top[ok], frac[ok], V[ok]
    g = top == "a37_cant_tell"
    prim["visibility"] = perm_auc(V, g, None, seed=1)
    prim["verdict"] = read(prim["visibility"], floor != FLOOR)
    ab = np.array([ab_all[int(o)] for o in ids])
    aok = np.isfinite(ab)
    prim["inclination"] = perm_auc(1 - ab[aok], g[aok], None, seed=2)
    prim["visibility_within_ab_quintiles"] = perm_auc(V[aok], g[aok], quintile(ab[aok]), seed=3)
    incl, strat = prim["inclination"]["auc"], prim["visibility_within_ab_quintiles"]["auc"]
    if incl >= D02 and prim["visibility"]["auc"] >= D02 and strat < D02:
        prim["verdict"] += " — INCLINATION, NOT VISIBILITY"
    ordered = ("a31_1", "a32_2", "a33_3", "a34_4", "a36_more_than_4")
    prim["mean_V"] = {a: float(V[top == a].mean()) for a in (*ordered, "a37_cant_tell")}
    prim["n_per"] = {a: int((top == a).sum()) for a in (*ordered, "a37_cant_tell")}
    prim["auc_vs_each"] = {a: auc(V[top == a], V[g]) for a in ordered}
    ct = frac[:, GZ2_TREE["t11_arms_number"].index("a37_cant_tell")]
    prim["spearman_V_cant_tell_fraction"] = float(np.corrcoef(rankdata(V), rankdata(ct))[0, 1])
    psf = column(lab, "psf", ids.tolist())
    pk = np.isfinite(psf)
    prim["seeing"] = perm_auc(psf[pk], g[pk], None, seed=4)
    out["arms_number"] = prim

    # secondary: bulge "none", within size × magnitude strata (exploratory)
    ids, top, _ = plurality(lab, "t05_bulge_prominence", union, FLOOR)
    size, mag = column(lab, "size", ids.tolist()), column(lab, "magnitude", ids.tolist())
    psf, snr = column(lab, "psf", ids.tolist()), column(lab, "snr", ids.tolist())
    ok = np.isfinite(size) & np.isfinite(mag) & np.isfinite(psf) & np.isfinite(snr)
    top, size, mag, psf, snr = top[ok], size[ok], mag[ok], psf[ok], snr[ok]
    g = top == "a10_no_bulge"
    strata = quintile(size) * 5 + quintile(mag)
    sec = {"n": int(ok.sum()), "n_none": int(g.sum()), "dropped": int((~ok).sum()),
           "seeing": perm_auc(psf, g, strata, seed=5),
           "low_snr": perm_auc(-snr, g, strata, seed=6)}
    sec["seeing"]["verdict"] = read(sec["seeing"], False) + " (exploratory)"
    sec["low_snr"]["verdict"] = read(sec["low_snr"], False) + " (exploratory)"
    a, b = (sec[k]["verdict"].split(" (")[0].split(" [")[0] for k in ("seeing", "low_snr"))
    sec["verdict"] = (a if a == b else f"SPLIT (seeing {a}; SNR {b})") + " (exploratory)"
    sec["unstratified"] = {"seeing": auc(psf[~g], psf[g]), "low_snr": auc(-snr[~g], -snr[g])}
    out["bulge_none"] = sec
    return out


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="V1", sources=1)
    rec = {"escape": escape(setup)}
    R._save(OUT, rec)
    print(json.dumps(rec["escape"], indent=1, default=float)[:3000], file=sys.stderr)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    rec["suppression"] = suppression(ctx)
    R._save(OUT, rec)
    for f, r in rec["suppression"].items():
        print(f"V1.3 {f}: raw {r['raw']:+.3f} four-way {r['four_way_partial']:+.3f} composite "
              f"{r['composite_partial']:+.3f} (p {r['p']:.1e}; PC1 {r['composite']['pc1_share']:.2f})"
              f" -> {r['verdict']}", file=sys.stderr)


if __name__ == "__main__":
    main()
