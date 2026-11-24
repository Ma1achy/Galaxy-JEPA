"""Part F, F3–F5 (user, 24 Nov 2026). Frozen embeddings; Part A's fits imported unchanged. Descriptive.

    .venv/bin/python artifacts/axes_f345.py f3   # ordinal direction fitted on clean labels only
    .venv/bin/python artifacts/axes_f345.py f4   # Part A with GZ2 user-weighted fractions
    .venv/bin/python artifacts/axes_f345.py f5   # PCA / ICA components of cleaned A1 against measurements and ordinal scores, and strips

Fixed before any number: F3's clean filter = votes on the ordered answers ≥ the train 75th percentile of the question, and top category
fraction ≥ 0.6 (both splits use the train threshold); directions are compared in embedding units (w / SD). F4 uses the `_weighted_fraction`
columns with Part A's own rows and raw counts as n. Debiased columns stay forbidden. F5: PCA and FastICA on the train-standardised cleaned
union train rows; the component sign is arbitrary and is fixed to a positive skew on train; |ρ| over TEST rows.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

import axes_a_ordinal as A
import axes_f as F

SEED = A.SEED + 300
OUTF = A.OUT / "f"


def fit_dir(X, ct, p, n, trm, ev, seed):
    """Part A's two models on rows `trm`, evaluated on each boolean mask in `ev` (with the SAME models); returns the record and w in embedding units."""
    mu, sd = X[trm].mean(0), X[trm].std(0)
    Z = (X - mu) / sd
    c = (ct - ct[trm].mean()) / (ct[trm].std() or 1.0)
    m, rec = {}, {}
    for kind in ("ord", "free"):
        m[kind], rec[kind] = A.cv_fit(kind, Z[trm], c[trm], p[trm], n[trm], np.random.default_rng(seed))
    marg = n[trm] @ p[trm] / n[trm].sum()
    out = {"lambda": {k: rec[k]["lambda"] for k in rec}}
    s = Z @ m["ord"]["w"]
    ec = p @ np.arange(1, p.shape[1] + 1)
    for name, sel in ev.items():
        lls = {**{k: (p[sel] * A.logp(k, m[k], Z[sel], c[sel])).sum(1) for k in ("ord", "free")}, "base": p[sel] @ np.log(np.clip(marg, 1e-12, None))}
        r = A.reading(lls, n[sel], np.random.default_rng(SEED + 1))
        out[name] = {k: r[k] for k in ("gap_free_base", "gap_ci95", "gap_ord_base", "share", "share_ci95", "reading")}
        out[name]["n"] = int(sel.sum())
        out[name]["tau"], out[name]["tau_ci"] = F.tau_ci(s[sel], ec[sel], np.random.default_rng(SEED + 2))
        if "can't tell" in out:
            pass
    return out, m["ord"]["w"] / sd, s


def step_f3():
    AXS, d = A.load()
    out = {"questions": {}}
    for qi, name in enumerate(A.QUESTIONS):
        r = A.question_rows(AXS, d, name)
        rows, p, n, ct, tr, te = (r[k] for k in ("rows", "p", "n", "ct", "tr", "te"))
        thr = float(np.percentile(n[tr], 75))
        cl = (n >= thr) & (p.max(1) >= 0.6)
        rec = {"vote_threshold_p75": thr, "n_train_clean": int((tr & cl).sum()), "n_train_all": int(tr.sum()), "n_test_clean": int((te & cl).sum()),
               "n_test_all": int(te.sum()), "top_category_mix_train_clean": [float((p[tr & cl].argmax(1) == k).mean()) for k in range(p.shape[1])], "enc": {}}
        if min(rec["n_train_clean"], rec["n_test_clean"]) < 200:
            rec["skipped"] = "fewer than 200 clean rows in a split"
        else:
            for kind in ("cleaned", "raw"):
                rec["enc"][kind] = {}
                for enc in F.ENC:
                    t0 = time.perf_counter()
                    X = (d.clean if kind == "cleaned" else d.mats)[enc][rows]
                    ev = {"test_clean": te & cl, "test_all": te}
                    a, wa, sa = fit_dir(X, ct, p, n, tr & cl, ev, SEED + qi)
                    b, wb, sb = fit_dir(X, ct, p, n, tr, ev, SEED + qi)
                    rec["enc"][kind][enc] = {"clean_fit": a, "all_fit": b, "cos_clean_vs_all": AXS.cosv(wa, wb),
                                             "score_pearson_test_clean_vs_all": float(np.corrcoef(sa[te], sb[te])[0, 1])}
                    A.log(f"F3 {name} {kind} {enc} {time.perf_counter() - t0:.0f}s cos {rec['enc'][kind][enc]['cos_clean_vs_all']:.3f} τ clean-fit clean/all "
                          f"{a['test_clean']['tau']:.3f}/{a['test_all']['tau']:.3f}, all-fit {b['test_clean']['tau']:.3f}/{b['test_all']['tau']:.3f}")
        out["questions"][name] = rec
        (A.DOC / "f3.json").write_text(json.dumps(out, indent=1, default=float))


def weighted(d, ids, cols):
    """The user-weighted fractions, read from the probe catalogue itself: the compact label sidecar drops them (its columns come back NaN)."""
    import pandas as pd
    from pathlib import Path
    t = pd.read_csv(Path(d.setup.cfg.paths.probe_dir) / "metadata.csv", usecols=["object_id", *cols], dtype={"object_id": str}).set_index("object_id")
    return t.reindex([str(i) for i in ids])[cols].to_numpy(float)


def step_f4():
    AXS, d = A.load()
    out = {"questions": {}}
    for qi, name in enumerate(A.QUESTIONS):
        Q = A.QUESTIONS[name]
        r = A.question_rows(AXS, d, name)
        rows, n, ct, tr, te = (r[k] for k in ("rows", "n", "ct", "tr", "te"))
        ids = d.union[rows]
        cols = [f"{Q['q']}_{c}_weighted_fraction" for c in Q["cats"]]
        W = weighted(d, ids, cols)
        ok = np.isfinite(W).all(1) & (np.nansum(W, 1) > 0)
        assert ok.mean() > 0.95, f"{name}: weighted fractions missing for {100 * (1 - ok.mean()):.1f}% of the rows"
        pw = W[ok] / W[ok].sum(1, keepdims=True)
        rr = {k: r[k][ok] for k in ("n", "ct", "tr", "te")}
        rows_ = rows[ok]
        pr = r["p"][ok]
        ec_r, ec_w = pr @ np.arange(1, pr.shape[1] + 1), pw @ np.arange(1, pw.shape[1] + 1)
        rec = {"n_train": int(rr["tr"].sum()), "n_test": int(rr["te"].sum()), "dropped": int((~ok).sum()),
               "spearman_ecat_raw_vs_weighted_test": float(spearmanr(ec_r[rr["te"]], ec_w[rr["te"]])[0]),
               "mean_abs_fraction_difference": float(np.abs(pr - pw).mean()), "enc": {}}
        for kind in ("cleaned", "raw"):
            rec["enc"][kind] = {}
            for enc in F.ENC:
                X = (d.clean if kind == "cleaned" else d.mats)[enc][rows_]
                e = {}
                for t, p in (("raw_votes", pr), ("weighted_votes", pw)):
                    a, (Z, m, _, _) = F.fit_test(X, rr["ct"], p, rr["n"], rr["tr"], rr["te"], SEED + 50 + qi, Q["cov"] is not None)
                    s = (Z @ m["ord"]["w"])[rr["te"]]
                    ec = p @ np.arange(1, p.shape[1] + 1)
                    e[t] = {**F.slim(a), "tau": float(kendalltau(s, ec[rr["te"]])[0])}
                rec["enc"][kind][enc] = e
                A.log(f"F4 {name} {kind} {enc} share raw {e['raw_votes']['share']:.3f} weighted {e['weighted_votes']['share']:.3f}; gap {e['raw_votes']['gap_free_base']:.4f}/{e['weighted_votes']['gap_free_base']:.4f} "
                      f"τ {e['raw_votes']['tau']:.3f}/{e['weighted_votes']['tau']:.3f}")
        out["questions"][name] = rec
        (A.DOC / "f4.json").write_text(json.dumps(out, indent=1, default=float))


def measurements(d):
    """Columns aligned to d.union: whatever Part B data exists (SDSS photometry, MPA-JHU, Simard table 1 from the earlier pull); NaN where missing."""
    import pandas as pd
    df = pd.read_csv(A.REPO / "runs/almagest/axes_phys/b/sdss.csv", dtype={"objID": str}).set_index("objID")
    df = df.apply(pd.to_numeric, errors="coerce")
    u = [str(i) for i in d.union]
    t = df.reindex(u)
    bad = lambda v: v.where(v > -9000)  # noqa: E731
    out = {"concentration_R90/R50": bad(t.petroR90_r) / bad(t.petroR50_r), "expAB_r": bad(t.expAB_r), "deVAB_r": bad(t.deVAB_r),
           "fracDeV_r": bad(t.fracDeV_r), "petroR50_r": bad(t.petroR50_r), "g_minus_r_observed_model": bad(t.modelMag_g) - bad(t.modelMag_r),
           "log_mass_mpa": bad(t.lgm_tot_p50), "log_ssfr_mpa": bad(t.specsfr_tot_p50), "redshift_spec": t.spec_z.where(t.spec_z > 0)}
    import v2_independent as V2
    V2.SIMARD = A.REPO / "artifacts/out/ext/simard2011_table1.tsv"  # Almagest mode moves R.OUT; the catalogue stays where it was pulled
    md = V2.metadata(d.union)
    sm = V2.simard(md["dr7objid"])
    out["simard_BT_r"] = pd.Series(sm["bt"])
    return {k: np.asarray(v, float) for k, v in out.items()}


def step_f5():
    from PIL import Image, ImageDraw, ImageFont
    from sklearn.decomposition import PCA, FastICA
    AXS, d = A.load()
    OUTF.mkdir(parents=True, exist_ok=True)
    X = d.clean["A1"]
    tr_ids = np.isin(d.union, np.asarray(d.setup.train_ids)) & d.ok
    te_ids = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
    mu, sd = X[tr_ids].mean(0), X[tr_ids].std(0)
    Z = (X - mu) / sd
    S = {}
    p = PCA(20, random_state=SEED).fit(Z[tr_ids])
    S["pca"] = (Z @ p.components_.T - p.mean_ @ p.components_.T)
    ica = FastICA(20, random_state=SEED, max_iter=2000, tol=1e-4, whiten="unit-variance").fit(Z[tr_ids])
    S["ica"] = ica.transform(Z)
    meas = measurements(d)
    ords = {}
    for q, f in (("arm count", "t11_arms_number"), ("winding", "t10_arms_winding"), ("bulge prominence", "t05_bulge_prominence"), ("roundness", "t07_rounded")):
        z = np.load(A.OUT / f"a_{f}.npz")
        ords[q] = ((X - z["mu"]) / z["sd"]) @ z["w_cleaned"]
    tgt = {**{f"measure: {k}": v for k, v in meas.items()}, **{f"ordinal: {k}": v for k, v in ords.items()}}
    out = {"n_train": int(tr_ids.sum()), "n_test": int(te_ids.sum()), "explained_variance_ratio_pca": p.explained_variance_ratio_.tolist(), "ica_iterations": int(ica.n_iter_),
           "measurements_available": list(meas), "note": "Simard disc parameters, Sersic n and statmorph are not in this run (Part B); ordinals are cleaned scores", "components": {}}
    rng = np.random.default_rng(SEED + 5)
    for kind, s in S.items():
        for k in range(20):
            v = s[:, k]
            if ((v[tr_ids] - v[tr_ids].mean()) ** 3).mean() < 0:
                v = -v
            s[:, k] = v
            row = {}
            for n, y in tgt.items():
                o = te_ids & np.isfinite(y)
                row[n] = {"rho": float(spearmanr(v[o], y[o])[0]), "n": int(o.sum())}
            top = sorted(row, key=lambda n: -abs(row[n]["rho"]))[:3]
            out["components"][f"{kind}{k + 1}"] = {"targets": row, "top3": [(n, round(row[n]["rho"], 3)) for n in top]}
    (A.DOC / "f5.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez(OUTF / "f5_scores.npz", ids=d.union, **{f"{k}": v.astype(np.float32) for k, v in S.items()})
    import j4_spread_controls as J
    _, cache = J.check(verbose=False)
    f = lambda n: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", n)  # noqa: E731
    T, NC = 56, 9
    te = np.flatnonzero(te_ids)
    for kind, s in S.items():
        rowh = 3 * T + 20
        im = Image.new("RGB", (NC * T + 180, 20 * rowh + 8), (250, 249, 246))
        dr = ImageDraw.Draw(im)
        for k in range(20):
            v = s[te, k]
            qs = np.percentile(v, np.linspace(3, 97, NC))
            c = out["components"][f"{kind}{k + 1}"]
            y0 = k * rowh + 4
            dr.text((6, y0 + 10), f"{kind.upper()} {k + 1}", fill=(29, 27, 25), font=f(13))
            for t, (n, rho) in enumerate(c["top3"]):
                dr.text((6, y0 + 32 + 16 * t), f"{n.split(': ')[1][:18]} {rho:+.2f}", fill=(90, 85, 80), font=f(10))
            for j, q in enumerate(qs):
                near = te[np.argsort(np.abs(v - q))[:60]]
                for i, g in enumerate(rng.choice(near, 3, replace=False)):
                    x = np.asarray(cache.get(int(d.union[g])), dtype=np.float32)
                    rgb = np.stack([x[2], x[1], x[0]], -1)
                    lo, hi = np.percentile(rgb, [1, 99.7])
                    a = (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)[64:192, 64:192] * 255).astype(np.uint8)
                    im.paste(Image.fromarray(a).resize((T, T), Image.LANCZOS), (180 + j * T, y0 + i * T))
        im.save(A.DOC / f"f5_{kind}_strips.png", optimize=True)
    for k, v in out["components"].items():
        print(k, v["top3"])


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
