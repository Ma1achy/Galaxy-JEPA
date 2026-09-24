"""Brief U3 — Scheme 2: graded existence for the ordered questions (the D26 test).

Pre-registered in artifacts/u_findings.md §U3 (hashed before this ran). M's 4-epoch checkpoint,
P2's split. A galaxy's category is its plurality answer; its question reach (all answers' counts
summed) must be ≥ 21. Arm count drops galaxies whose plurality is "can't tell"; exact ties drop.

  A  existence: logistic axis on TRAIN endpoint categories only; project TEST galaxies of every
     category; Jonckheere–Terpstra ordering as Kendall's τ-b between category index and projection
     (τ-b is JT's S rescaled for a fixed tie structure), one-sided permutation of category labels
     (10,000, add-one). Adjacent-pair AUCs (does each middle land between its neighbours?).
     Untrained encoder: the same τ-b on R's three draws.
  B  continuous: Masters et al. (2019) Eq. 1 w_avg = 0.5 p_medium + 1.0 p_tight; Eq. 3
     B_avg = 0.2 p_just noticeable + 0.8 p_obvious + 1.0 p_dominant. Spearman vs projection.
  C  where the middle sits: cross-fitted position along, and distance off, the endpoint-centroid
     line (z-scored embedding); the off-line offset's cosine with the visibility PATTERN direction
     against a shuffled-category null; and the visibility index itself, middle vs interpolation.

    uv run python artifacts/u3_graded.py  -> artifacts/out/u3_graded.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import kendalltau, norm, rankdata

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.data.metadata import GZ2_TREE, vote_column  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_direction  # noqa: E402

AXES = {
    "arms_winding": ("t10_arms_winding", ("a28_tight", "a29_medium", "a30_loose")),
    "bulge_prominence": ("t05_bulge_prominence",
                         ("a10_no_bulge", "a11_just_noticeable", "a12_obvious", "a13_dominant")),
    "roundedness": ("t07_rounded", ("a16_completely_round", "a17_in_between", "a18_cigar_shaped")),
    "arms_number": ("t11_arms_number", ("a31_1", "a32_2", "a33_3", "a34_4", "a36_more_than_4")),
}
FLOOR = 21
N_PERM = 10_000
N_NULL_C = 200
N_BOOT = 1_000
ALPHA = 0.05
VISIBILITY = (("magnitude", +1), ("snr", -1), ("size", -1), ("redshift", +1))  # + = less visible
OUT = R.OUT / "u3_graded.json"


def categories(lab, question: str, ordered: tuple[str, ...], ids: list[int]):
    """(kept ids, category index) — plurality answer, reach ≥ FLOOR, ties and non-ordered dropped."""
    answers = GZ2_TREE[question]
    frac = np.column_stack([lab._column(ids, vote_column(question, a)) for a in answers])
    reach = sum(lab._column(ids, vote_column(question, a, "count")) for a in answers)
    ok = np.isfinite(frac).all(axis=1) & (reach >= FLOOR)
    top = np.argmax(frac, axis=1)
    tied = (frac == frac.max(axis=1, keepdims=True)).sum(axis=1) > 1
    idx = np.array([ordered.index(answers[t]) if answers[t] in ordered else -1 for t in top])
    keep = ok & ~tied & (idx >= 0)
    return [o for o, k in zip(ids, keep, strict=True) if k], idx[keep]


def auc(lo: np.ndarray, hi: np.ndarray) -> float:
    r = rankdata(np.concatenate([lo, hi]))
    return float((r[lo.size:].sum() - hi.size * (hi.size + 1) / 2) / (lo.size * hi.size))


def tau_perm(g: np.ndarray, x: np.ndarray, seed: int) -> tuple[float, float]:
    obs = float(kendalltau(g, x).statistic)
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(N_PERM):
        if kendalltau(rng.permutation(g), x).statistic >= obs - 1e-12:
            hits += 1
    return obs, (1 + hits) / (1 + N_PERM)


def spearman_perm(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rx = rankdata(x) - (x.size + 1) / 2
    ry = rankdata(y) - (y.size + 1) / 2
    rx, ry = rx / np.linalg.norm(rx), ry / np.linalg.norm(ry)
    obs = float(rx @ ry)
    rng, hits = np.random.default_rng(seed), 0
    for s in range(0, N_PERM, 250):
        b = min(250, N_PERM - s)
        p = rng.permuted(np.broadcast_to(ry, (b, ry.size)).copy(), axis=1)
        hits += int(np.sum(np.abs(p @ rx) >= abs(obs) - 1e-12))
    return obs, (1 + hits) / (1 + N_PERM)


def visibility_index(lab, ids: list[int]) -> np.ndarray:
    cols = []
    for name, sign in VISIBILITY:
        v = np.where(np.asarray(lab.nuisance_valid(name, ids), bool),
                     lab.nuisance_value(name, ids), np.nan)
        ok = np.isfinite(v)
        q = np.full(v.size, np.nan)
        q[ok] = norm.ppf((rankdata(v[ok]) - 0.5) / ok.sum())  # rank-normal, robust to scale
        cols.append(sign * q)
    return np.mean(np.column_stack(cols), axis=1)  # NaN where any is missing


def middle_geometry(z: np.ndarray, g: np.ndarray, k: int, rng) -> dict:
    """Cross-fitted position t along, and signed distance off, the endpoint line, per middle."""
    e0, e1 = z[g == 0].mean(axis=0), z[g == k - 1].mean(axis=0)
    u = e1 - e0
    L = float(np.linalg.norm(u))
    u /= L
    out = {"endpoint_separation": L}
    for j in range(1, k - 1):
        rows = np.flatnonzero(g == j)
        half = rng.permutation(rows.size) % 2 == 0
        res = []
        ts = []
        for h in (half, ~half):
            m = z[rows[h]].mean(axis=0) - e0
            t = float(m @ u) / L
            res.append(m - (m @ u) * u)
            ts.append(t)
        off2 = float(res[0] @ res[1])
        off = float(np.sign(off2) * np.sqrt(abs(off2)))
        out[str(j)] = {"t": float(np.mean(ts)), "t_expected_equal_spacing": j / (k - 1),
                       "offline": off, "offline_share": off / L,
                       "residual": (0.5 * (res[0] + res[1])).tolist(), "n": int(rows.size)}
    return out


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="U3", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    lab, c, seed = ctx.labels["full"], ctx.pc.c, ctx.pc.seed
    t1 = {r["feature"]: r for r in json.loads((R.OUT / "t1_ladder.json").read_text())["full"]}

    zr = ctx.z(ctx.real)
    tr_all = [o for o in setup.train_ids if o in zr.index]
    vis_tr = visibility_index(lab, tr_all)
    okv = np.isfinite(vis_tr)
    ztr_all = zr.x[zr.rows_for(tr_all)][okv]
    vt = (vis_tr[okv] - vis_tr[okv].mean()) / vis_tr[okv].std()
    pattern = ztr_all.T @ vt / vt.size  # the visibility PATTERN (how centroids move), label-free
    pattern /= np.linalg.norm(pattern)

    rec: dict = {"floor": FLOOR, "axes": {}}
    for name, (q, ordered) in AXES.items():
        k = len(ordered)
        tr_ids, g_tr = categories(lab, q, ordered, setup.train_ids)
        te_ids, g_te = categories(lab, q, ordered, setup.test_ids)
        a: dict = {"question": q, "order": list(ordered),
                   "n_train": np.bincount(g_tr, minlength=k).tolist(),
                   "n_test": np.bincount(g_te, minlength=k).tolist(), "matrices": {}}
        for m in [ctx.real, *ctx.untrained]:
            is_real = m is ctx.real
            ends = (g_tr == 0) | (g_tr == k - 1)
            xtr = m.x[m.rows_for(tr_ids)][ends]
            w = probe_direction(Embeddings(xtr, (g_tr[ends] == k - 1).astype(np.int64),
                                           g_tr[ends].astype(float)), name=name, c=c).w_unit
            proj = m.x[m.rows_for(te_ids)] @ w
            tau = tau_perm(g_te, proj, seed) if is_real else (
                float(kendalltau(g_te, proj).statistic), float("nan"))
            adj = [auc(proj[g_te == j], proj[g_te == j + 1]) for j in range(k - 1)]
            r = {"tau_b": tau[0], "p_one_sided": tau[1], "adjacent_auc": adj,
                 "endpoint_auc": auc(proj[g_te == 0], proj[g_te == k - 1])}
            if is_real:
                # B — Masters et al. (2019), their coefficients, from the paper
                if name in ("arms_winding", "bulge_prominence"):
                    f = {a_: lab._column(te_ids, vote_column(q, a_)) for a_ in GZ2_TREE[q]}
                    if name == "arms_winding":
                        score = 0.5 * f["a29_medium"] + 1.0 * f["a28_tight"]  # Eq. 1, tight = 1
                    else:
                        score = (0.2 * f["a11_just_noticeable"] + 0.8 * f["a12_obvious"]
                                 + 1.0 * f["a13_dominant"])  # Eq. 3
                    rho, p = spearman_perm(proj, score, seed + 1)
                    r["B"] = {"score": "w_avg" if name == "arms_winding" else "B_avg",
                              "rho": rho, "p": p}
                # C — where the middle sits (z-scored embedding, all galaxies at the floor)
                all_ids = tr_ids + te_ids
                g_all = np.concatenate([g_tr, g_te])
                z = zr.x[zr.rows_for(all_ids)]
                rng = np.random.default_rng(seed)
                geom = middle_geometry(z, g_all, k, rng)
                e0, e1 = z[g_all == 0].mean(axis=0), z[g_all == k - 1].mean(axis=0)
                u = (e1 - e0) / np.linalg.norm(e1 - e0)
                vp = pattern - (pattern @ u) * u
                vp /= np.linalg.norm(vp)
                null = []
                for _ in range(N_NULL_C):
                    gg = middle_geometry(z, rng.permutation(g_all), k, rng)
                    null.append([float(np.asarray(gg[str(j)]["residual"]) @ vp
                                       / max(np.linalg.norm(gg[str(j)]["residual"]), 1e-12))
                                 for j in range(1, k - 1)])
                null = np.asarray(null)
                V = visibility_index(lab, all_ids)
                for j in range(1, k - 1):
                    gj = geom[str(j)]
                    res = np.asarray(gj.pop("residual"))
                    cos = float(res @ vp / max(np.linalg.norm(res), 1e-12))
                    gj["cos_visibility"] = cos
                    gj["null_q99"] = float(np.quantile(null[:, j - 1], 0.99))
                    gj["null_q01"] = float(np.quantile(null[:, j - 1], 0.01))
                    # the visibility index itself: middle vs linear interpolation of the ends
                    fin = np.isfinite(V)
                    v0 = V[fin & (g_all == 0)]
                    v1 = V[fin & (g_all == k - 1)]
                    vj = V[fin & (g_all == j)]
                    t = gj["t"]
                    brng = np.random.default_rng(seed + j)
                    boots = [np.mean(brng.choice(vj, vj.size)) - (1 - t) * np.mean(
                        brng.choice(v0, v0.size)) - t * np.mean(brng.choice(v1, v1.size))
                        for _ in range(N_BOOT)]
                    gj["delta_visibility"] = float(vj.mean() - (1 - t) * v0.mean() - t * v1.mean())
                    gj["delta_visibility_ci"] = [float(np.quantile(boots, 0.025)),
                                                 float(np.quantile(boots, 0.975))]
                r["C"] = geom
                r["scheme1_auc"] = {a_: t1.get(f"{q}_{a_}", {}).get("auc") for a_ in ordered}
            a["matrices"]["real" if is_real else m.encoder_name] = r
        rec["axes"][name] = a
        R._save(OUT, rec)
        print(f"U3 {name:<17s} n_test {a['n_test']}  tau {a['matrices']['real']['tau_b']:+.3f}",
              file=sys.stderr)
    p = {n: rec["axes"][n]["matrices"]["real"]["p_one_sided"] for n in AXES}
    sig = nulls_mod.family_significant(p, alpha=ALPHA, method="benjamini_yekutieli", n_tests=4)
    for n, a in rec["axes"].items():
        real = a["matrices"]["real"]
        untr = [v["tau_b"] for k_, v in a["matrices"].items() if k_ != "real"]
        between = all(x > 0.5 for x in real["adjacent_auc"])
        above = real["tau_b"] > max(untr)
        a["verdict"] = ("ORDERED" if sig[n] and between and above else
                        "PARTLY ORDERED" if sig[n] and above else
                        "NOT ABOVE UNTRAINED" if sig[n] else "NOT ORDERED")
    R._save(OUT, rec)


if __name__ == "__main__":
    main()
