"""Part B (brief, 24 Nov 2026): which physical quantities does A1 store as linear directions? Frozen embeddings, catalogues and statmorph. Descriptive.

    .venv/bin/python artifacts/axes_b.py catalogue    # measurements aligned to the union, with sources, SHA-1s and match counts
    .venv/bin/python artifacts/axes_b.py selftest     # planted targets: a linear one reads STRONG, noise reads NOISE, a nuisance vanishes under the partial
    .venv/bin/python artifacts/axes_b.py fit [--only NAME ...]
    .venv/bin/python artifacts/axes_b.py extras       # ordinal scores vs Simard / photometry; SDSS photo flags as binary targets
    .venv/bin/python artifacts/axes_b.py map          # score correlation with the 30 STRONG/SEPARATE GZ2 axes, and the Plate copy

Fixed before any B fit (25 Nov 2026):
- Rows: the ladder's union, its train (39,997) and test (34,828) rows with complete nuisances and a finite target. statmorph exists for 20,000 test
  galaxies only, so its targets are fitted on a fixed random half of those and tested on the other half (marked `half`).
- Target transforms: log10 for Re, Rd (kpc, as listed) and petroRad_r; mass and sSFR are already log. All targets winsorised at the train 0.5 / 99.5 percentiles.
- Simard: n = 4 bulge + disc model (table 1) for B/T, inclination, Re, Rd, ellipticity; the free-n model (table 2) for n_b. Bulge parameters (Re, e, n_b) only
  where that table's B/T ≥ 0.1, disc parameters (Rd, i) only where B/T ≤ 0.9. No cut on P_pS; the fraction with P_pS ≤ 0.32 is reported.
- g−r: modelMag, dereddened, k-corrected to z = 0 (kcorrect 5.1.9 on g, r, i; spectroscopic z > 0.002, zWarning 0, class GALAXY; 2% error floor).
- statmorph: fits with flag > 0 dropped; n also drops flag_sersic > 0.
- Ridge on train-standardised features, α from GCV over 10^-2…10^4 (13). Reported on test: R², Spearman with a 500-resample CI, and the partial Spearman given
  the five nuisances (both sides rank-transformed and residualised on the nuisance ranks and their squares).
- Split-half: two random halves of train, cosine of the two w's (embedding units), against 39 shuffled targets on the same rows, halves and α (the 37-axis run
  used 5, which calls pure noise stable 1 time in 6: seen in the self-test, so tightened before any fit).
- Label, on the cleaned embeddings (as the 37 axes): NOISE if cos ≤ the shuffles' max; else STRONG if A1's test Spearman exceeds every untrained seed's; else WEAK.
  Targets that are nuisances (size, magnitude): raw only, marked, no label.
- Flags: binary, union rows, logistic probe (the ladder's C); fewer than 100 positives (or negatives) in train or in test → counts only, no AUC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
import axes_a_ordinal as A  # noqa: E402
import axes_f as F  # noqa: E402

B, DOC = REPO / "runs/almagest/axes_phys/b", REPO / "docs/axes"
SEED, NBOOT, NSH, NNULL = 20261125, 500, 39, 5
ALPHAS = np.logspace(-2, 4, 13)
ENC = F.ENC
SM = ("asymmetry", "gini", "m20", "smoothness", "sersic_n")
NUISANCE = ("petroRad_r (size)", "modelMag_r (magnitude)")
MIN_POS = 100


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    (DOC / name).write_text(json.dumps(obj, indent=1, default=float))


# --- catalogue ---------------------------------------------------------------------------------------------------------
def step_catalogue():
    import pandas as pd
    import galaxy_jepa  # noqa: F401
    ids = np.load(REPO / "runs/almagest/a1/a1_union_ids.npy")
    u = [str(i) for i in ids]
    rec_s = json.loads((B / "sdss_record.json").read_text())
    assert sha1(B / "sdss.csv") == rec_s["sha1"]
    t = pd.read_csv(B / "sdss.csv", dtype={"objID": str, "specObjID": str, "spec_class": str}).set_index("objID").reindex(u)
    num = lambda c: pd.to_numeric(t[c], errors="coerce").where(lambda v: v > -9000)  # noqa: E731
    m, src = {}, {}
    phot = "SDSS DR17 PhotoObjAll (public SkyServer)"
    m["concentration R90/R50"] = num("petroR90_r") / num("petroR50_r")
    for c in ("expAB_r", "deVAB_r", "fracDeV_r"):
        m[c] = num(c)
    m["petroRad_r (size)"] = np.log10(num("petroRad_r").where(lambda v: v > 0))
    m["modelMag_r (magnitude)"] = num("modelMag_r")
    for k in list(m):
        src[k] = phot
    # k-corrected g − r
    z, ok = num("spec_z"), (num("zWarning") == 0) & (t.spec_class == "GALAXY") & (num("spec_z") > 0.002)
    inp = pd.DataFrame({"tag": u, "z0": z.to_numpy()})
    good = ok.to_numpy().copy()
    for b in "gri":
        mag, err = (num(f"modelMag_{b}") - num(f"extinction_{b}")).to_numpy(), np.hypot(num(f"modelMagErr_{b}").to_numpy(), 0.02)
        mg = 10 ** (-0.4 * mag)
        inp[f"maggies_{b}"], inp[f"ivar_{b}"] = mg, 1 / (mg * 0.4 * np.log(10) * err) ** 2
        good &= np.isfinite(mg) & np.isfinite(err)
    inp = inp[good]
    inp.to_csv(B / "kc_in.csv", index=False)
    subprocess.run([str(REPO / "runs/almagest/a1/visibility/kc_venv/bin/python"), str(REPO / "artifacts/b2b_kcorrect.py"), str(B / "kc_in.csv"), str(B / "kc_out.csv"), "0.0"], check=True)
    kc = pd.read_csv(B / "kc_out.csv", dtype={"tag": str}).set_index("tag").reindex(u)
    gr = (num("modelMag_g") - num("extinction_g")) - (num("modelMag_r") - num("extinction_r"))
    m["g-r, k-corrected z=0"] = gr + (kc["dm_g_0.00"] - kc["dm_r_0.00"])
    src["g-r, k-corrected z=0"] = phot + " + kcorrect 5.1.9"
    m["log M* (MPA-JHU)"], m["log sSFR (MPA-JHU)"] = num("lgm_tot_p50"), num("specsfr_tot_p50")
    for k in ("log M* (MPA-JHU)", "log sSFR (MPA-JHU)"):
        src[k] = "MPA-JHU galSpecExtra via SDSS DR17"
        m[k] = m[k].where(num("mpa_reliable") == 1)
    # Simard 2011 by DR7 objID
    dr7 = pd.read_csv(REPO / "data/probe_v2/metadata.csv", usecols=["object_id", "dr7objid"], dtype=str).set_index("object_id").reindex(u).dr7objid
    S = {}
    for n in (1, 2):
        d = pd.read_csv(B / f"simard2011_table{n}.tsv", sep="\t", comment="#", skip_blank_lines=True, dtype=str)
        d = d[d.objID.str.strip().str.fullmatch(r"\d{18}", na=False)]
        d = d.assign(objID=d.objID.str.strip())
        dup = int(d.objID.duplicated().sum())
        d = d.drop_duplicates("objID").set_index("objID").reindex(dr7.to_numpy())
        S[n] = d.apply(pd.to_numeric, errors="coerce").set_axis(t.index)
        src[f"simard table {n}"] = {"rows_duplicated": dup, "sha1": sha1(B / f"simard2011_table{n}.tsv"), "matched": int(S[n]["(B/T)r"].notna().sum())}
    s1, s2 = S[1], S[2]
    bt1, bt2 = s1["(B/T)r"], s2["(B/T)r"]
    b1, b2, d1 = bt1 >= 0.1, bt2 >= 0.1, bt1 <= 0.9
    m["Simard B/T r"] = bt1
    m["Simard disc inclination"] = s1["i"].where(d1)
    m["Simard bulge n (free n)"] = s2["nb"].where(b2)
    m["Simard bulge Re (log kpc)"] = np.log10(s1["Re"].where(b1 & (s1["Re"] > 0)))
    m["Simard disc Rd (log kpc)"] = np.log10(s1["Rd"].where(d1 & (s1["Rd"] > 0)))
    m["Simard bulge ellipticity"] = s1["e"].where(b1)
    for k in m:
        src.setdefault(k, "Simard et al. 2011, VizieR J/ApJS/196/11 (Harvard mirror)" if k.startswith("Simard") else "")
    out = {k: np.asarray(v, float) for k, v in m.items()}
    # statmorph
    f = B / "statmorph.jsonl"
    rec_m = None
    if f.exists():
        q = pd.DataFrame([json.loads(x) for x in f.read_text().splitlines()]).drop_duplicates("object_id").set_index("object_id")
        err = q.error.notna() if "error" in q else pd.Series(False, index=q.index)
        good = ~err & (q.flag == 0)
        rec_m = {"n": len(q), "errors": q.error.value_counts().to_dict() if "error" in q else {}, "flag": q.flag.value_counts().to_dict(),
                 "flag_sersic": q.flag_sersic.value_counts().to_dict(), "kept": int(good.sum()), "kept_sersic": int((good & (q.flag_sersic == 0)).sum()),
                 "sn_per_pixel_p5_50": np.nanpercentile(q.sn_per_pixel, [5, 50]).tolist(), "sha1": sha1(f)}
        for k in SM:
            v = q[k].where(good & ((q.flag_sersic == 0) if k == "sersic_n" else True)).reindex(u)
            out[f"statmorph {k}"] = v.to_numpy(float)
            src[f"statmorph {k}"] = "statmorph 0.7.3 on our r-band stamps"
    np.savez(B / "measures.npz", ids=ids, **out)
    rec = {"counts": {k: int(np.isfinite(v).sum()) for k, v in out.items()}, "sources": src, "statmorph": rec_m,
           "sdss": {k: v for k, v in rec_s.items() if k != "query"}, "kcorrect_n": int(len(inp)),
           "simard_PpS_le_0.32": {str(n): float((S[n]["PpS"] <= 0.32).mean()) for n in S},
           "licences": {"SDSS": "public domain with acknowledgement", "VizieR J/ApJS/196/11": "VizieR licence (rights_uri in the TSV header); cite Simard et al. 2011",
                        "MPA-JHU": "SDSS data policy", "statmorph": "BSD-3", "kcorrect": "BSD-3"}}
    save("b_catalogue.json", rec)
    print(json.dumps(rec["counts"], indent=1), json.dumps(rec_m, default=str))


# --- ridge -------------------------------------------------------------------------------------------------------------
class Gram:
    """Ridge with GCV for many targets on one design: one eigendecomposition of the standardised train Gram matrix."""

    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.Z = (X - self.mu) / self.sd
        self.ev, self.V = np.linalg.eigh(self.Z.T @ self.Z)
        self.ev = np.maximum(self.ev, 0)

    def fit(self, y, alpha=None):
        ym = y.mean()
        r = self.V.T @ (self.Z.T @ (y - ym))
        n, yy = len(y), float(((y - ym) ** 2).sum())
        if alpha is None:
            best = None
            for a in ALPHAS:
                d = self.ev + a
                rss = yy - 2 * float((r * r / d).sum()) + float((r * r * self.ev / d**2).sum())
                g = rss / n / (1 - (float((self.ev / d).sum()) + 1) / n) ** 2
                if best is None or g < best[0]:
                    best = (g, a)
            alpha = best[1]
        w = self.V @ (r / (self.ev + alpha))
        return {"we": w / self.sd, "b": ym - float((self.mu / self.sd) @ w), "alpha": float(alpha)}


def spear(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def resid_ranks(v, N):
    R = np.column_stack([rankdata(N[:, j]) for j in range(N.shape[1])])
    R = (R - R.mean(0)) / R.std(0)
    D = np.column_stack([np.ones(len(v)), R, R**2])
    r = rankdata(v)
    return r - D @ np.linalg.lstsq(D, r, rcond=None)[0]


def partial(a, b, N):
    return float(np.corrcoef(resid_ranks(a, N), resid_ranks(b, N))[0, 1])


def boot(a, b, seed, n=NBOOT):
    rng = np.random.default_rng(seed)
    v = [spear(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def boot_pearson(a, b, seed, n=NBOOT):
    """Paired bootstrap of Pearson r (used on nuisance-residualised ranks for partial Spearman CI)."""
    rng = np.random.default_rng(seed)
    v = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def rows_for(d, y, name):
    trm, tem = (np.isin(d.union, np.asarray(x)) & d.ok & np.isfinite(y) for x in (d.setup.train_ids, d.setup.test_ids))
    if name.startswith("statmorph"):
        te = np.flatnonzero(tem)
        h = np.random.default_rng(SEED + 1).permutation(len(te)) < len(te) // 2
        return te[h], te[~h]
    return np.flatnonzero(trm), np.flatnonzero(tem)


def fit_target(d, AXS, name, y, j, null_scores=True):
    tr, te = rows_for(d, y, name)
    lo, hi = np.percentile(y[tr], [0.5, 99.5])
    yc = np.clip(y, lo, hi)
    nuis = name in NUISANCE
    rec = {"n_train": len(tr), "n_test": len(te), "winsor": [float(lo), float(hi)], "nuisance": nuis, "half": name.startswith("statmorph")}
    arr = {}
    for kind in ("raw",) if nuis else ("raw", "cleaned"):
        k = {"enc": {}}
        for e in ENC:
            X = (d.clean if kind == "cleaned" else d.mats)[e]
            g = Gram(X[tr])
            m = g.fit(yc[tr])
            s = X[te] @ m["we"] + m["b"]
            x = {"alpha": m["alpha"], "r2": float(1 - ((yc[te] - s) ** 2).sum() / ((yc[te] - yc[te].mean()) ** 2).sum()), "spearman": spear(s, yc[te]), "partial": partial(s, yc[te], d.N[te])}
            if e == "A1":
                x["spearman_ci"] = boot(s, yc[te], SEED + j)
                rs, ry = resid_ranks(s, d.N[te]), resid_ranks(yc[te], d.N[te])
                x["partial_ci"] = boot_pearson(rs, ry, SEED + j + 500)
                rng = np.random.default_rng(SEED + 1000 + j)
                half = rng.permutation(len(tr)) < len(tr) // 2
                ga, gb = Gram(X[tr[half]]), Gram(X[tr[~half]])
                wa, wb = (gg.fit(yc[tr[h]], m["alpha"])["we"] for gg, h in ((ga, half), (gb, ~half)))
                sh = []
                for _ in range(NSH):
                    p = rng.permutation(yc[tr])
                    sh.append(AXS.cosv(ga.fit(p[half], m["alpha"])["we"], gb.fit(p[~half], m["alpha"])["we"]))
                x["split_half_cos_w"], x["shuffled_cos_w"] = AXS.cosv(wa, wb), sh
                arr[f"{kind}_we"], arr[f"{kind}_b"] = m["we"], m["b"]
                arr[f"{kind}_sigma"] = np.cov(X[tr].T) @ m["we"]
                if kind == "cleaned" and null_scores:
                    arr["null_we"] = np.array([g.fit(rng.permutation(yc[tr]), m["alpha"])["we"] for _ in range(NNULL)])
            k["enc"][e] = x
        un = [k["enc"][e]["spearman"] for e in ENC[1:]]
        k["untrained_spearman_range"], k["untrained_partial_range"] = [min(un), max(un)], [min(k["enc"][e]["partial"] for e in ENC[1:]), max(k["enc"][e]["partial"] for e in ENC[1:])]
        a = k["enc"]["A1"]
        k["stable"] = a["split_half_cos_w"] > max(a["shuffled_cos_w"])
        k["label"] = None if nuis else "NOISE" if not k["stable"] else "STRONG" if a["spearman"] > max(un) else "WEAK"
        rec[kind] = k
    rec["label"] = None if nuis else rec["cleaned"]["label"]
    return rec, arr


def step_fit(only=None):
    AXS, d = A.load()
    z = np.load(B / "measures.npz")
    assert np.array_equal(z["ids"], d.union)
    names = [k for k in z.files if k != "ids"]
    f = DOC / "b_fit.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    (B / "fits").mkdir(exist_ok=True)
    for j, n in enumerate(names):
        if (only and n not in only) or (n in out and not only):
            continue
        t0 = time.perf_counter()
        rec, arr = fit_target(d, AXS, n, z[n], j)
        out[n] = rec
        np.savez(B / "fits" / (n.replace("/", "_").replace(" ", "_") + ".npz"), name=n, **arr)
        save("b_fit.json", out)
        k = rec.get("cleaned", rec["raw"])
        A.log(f"B {n} {time.perf_counter() - t0:.0f}s label {rec['label']} spearman A1 {k['enc']['A1']['spearman']:.3f} untr {k['untrained_spearman_range'][1]:.3f} partial {k['enc']['A1']['partial']:.3f}")


def step_selftest():
    AXS, d = A.load()
    rng = np.random.default_rng(0)
    Z = d.clean["A1"]
    v = np.full(len(Z), np.nan)
    ok = d.ok
    w = rng.normal(size=384)
    s = (Z[ok] / Z[ok].std(0)) @ w
    v[ok] = s / s.std() + rng.normal(size=ok.sum())
    r1, _ = fit_target(d, AXS, "planted linear", v, 0, False)
    r2, _ = fit_target(d, AXS, "planted noise", np.where(ok, rng.normal(size=len(Z)), np.nan), 1, False)
    r3, _ = fit_target(d, AXS, "planted magnitude", np.where(ok, d.N[:, 1], np.nan), 2, False)
    show = lambda r: {k: (r[k]["label"], round(r[k]["enc"]["A1"]["spearman"], 3), round(r[k]["enc"]["A1"]["partial"], 3), round(r[k]["enc"]["A1"]["split_half_cos_w"], 3), round(max(r[k]["enc"]["A1"]["shuffled_cos_w"]), 3)) for k in ("raw", "cleaned")}  # noqa: E731
    # closed-form GCV ridge against sklearn's
    from sklearn.linear_model import RidgeCV
    tr = np.flatnonzero(ok)[:8000]
    g = Gram(Z[tr]).fit(v[tr])
    sk = RidgeCV(alphas=ALPHAS).fit((Z[tr] - Z[tr].mean(0)) / Z[tr].std(0), v[tr])
    out = {"linear": show(r1), "noise": show(r2), "magnitude": show(r3), "alpha_ours_sklearn": [g["alpha"], float(sk.alpha_)],
           "max_abs_w_diff": float(np.abs(g["we"] * Z[tr].std(0) - sk.coef_).max())}
    print(json.dumps(out, indent=1))
    # the false-stable rate of the rule on 40 noise targets (A1 cleaned, halves re-drawn): expected 1/40
    tr = np.flatnonzero(ok & np.isin(d.union, np.asarray(d.setup.train_ids)))
    st = 0
    for k in range(4):
        h = rng.permutation(len(tr)) < len(tr) // 2
        ga, gb = Gram(Z[tr[h]]), Gram(Z[tr[~h]])
        c = []
        for _ in range(400):
            y = rng.normal(size=len(tr))
            c.append(AXS.cosv(ga.fit(y[h], 1e4)["we"], gb.fit(y[~h], 1e4)["we"]))
        c = np.array(c).reshape(10, 40)
        st += int((c[:, 0] > c[:, 1:].max(1)).sum())
    out["false_stable_of_40_noise_targets"] = st
    print("false stable:", st, "of 40")
    assert st <= 5
    assert r1["label"] == "STRONG" and r2["label"] != "STRONG" and g["alpha"] == sk.alpha_ and out["max_abs_w_diff"] < 1e-6
    assert abs(r3["raw"]["enc"]["A1"]["partial"]) < 0.5 * r3["raw"]["enc"]["A1"]["spearman"]
    save("b_selftest.json", out)


# --- additions: ordinal scores; photo flags ----------------------------------------------------------------------------
def step_extras():
    import pandas as pd
    AXS, d = A.load()
    z = np.load(B / "measures.npz")
    te = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
    out = {"ordinal_vs_measure": {}, "flags": {}}
    ords = {}
    for q, f in (("roundness", "t07_rounded"), ("bulge prominence", "t05_bulge_prominence")):
        a = np.load(A.OUT / f"a_{f}.npz")
        for kind in ("cleaned", "raw"):
            X = (d.clean if kind == "cleaned" else d.mats)["A1"]
            ords[q, kind] = (X - a["mu"]) / a["sd"] @ a["w_cleaned"] if kind == "cleaned" else None
    # raw and cleaned ordinal directions refitted by Part A's code (its npz keeps μ, σ for the cleaned fit only), with the free model's dominant − obvious column
    for q in ("roundness", "bulge prominence"):
        r = A.question_rows(AXS, d, q)
        for kind in ("cleaned", "raw"):
            X = (d.clean if kind == "cleaned" else d.mats)["A1"]
            rec, (Z, m, _, _) = F.fit_test(X[r["rows"]], r["ct"], r["p"], r["n"], r["tr"], r["te"], SEED, False)
            mu, sd = X[r["rows"]][r["tr"]].mean(0), X[r["rows"]][r["tr"]].std(0)
            s = (X - mu) / sd @ m["ord"]["w"]
            if kind == "cleaned":
                o = np.isfinite(s) & np.isfinite(ords[q, kind])
                out["ordinal_vs_measure"][f"{q} refit vs Part A, Pearson"] = float(np.corrcoef(s[o], ords[q, kind][o])[0, 1])
            ords[q, kind] = s
            if q == "bulge prominence":
                W = m["free"]["W"]
                ords["dominant - obvious", kind] = (X - mu) / sd @ (W[:, 3] - W[:, 2])
    pairs = {"roundness": ("Simard disc inclination", "expAB_r"), "bulge prominence": ("Simard B/T r", "Simard bulge n (free n)", "concentration R90/R50"),
             "dominant - obvious": ("Simard B/T r", "Simard bulge n (free n)", "concentration R90/R50")}
    for (q, kind), s in ords.items():
        for k in pairs[q]:
            y = z[k]
            o = te & np.isfinite(y) & np.isfinite(s)
            out["ordinal_vs_measure"][f"{q} | {kind} | {k}"] = {"n": int(o.sum()), "spearman": spear(s[o], y[o]), "ci": boot(s[o], y[o], SEED), "partial": partial(s[o], y[o], d.N[o])}
    save("b_extras.json", out)
    for k, v in out["ordinal_vs_measure"].items():
        A.log(k, v if isinstance(v, float) else {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    # flags
    const = json.loads((A.OUT / "c/photoflags.json").read_text())
    used = json.loads((DOC / "c_star.json").read_text())["flags_used_single"]
    names = ["SATURATED", "EDGE", "BLENDED", "DEBLENDED_AS_MOVING", *[x for x in used if x not in ("SATURATED", "EDGE", "BLENDED")]]
    t = pd.read_csv(B / "sdss.csv", dtype={"objID": str, "flags": str}).set_index("objID").reindex([str(i) for i in d.union])
    bits = np.array([int(x) for x in t["flags"]])
    trm = np.isin(d.union, np.asarray(d.setup.train_ids)) & d.ok
    for n in names:
        y = (bits & const[n] != 0).astype(int)
        r = {"train_pos": int(y[trm].sum()), "train_n": int(trm.sum()), "test_pos": int(y[te].sum()), "test_n": int(te.sum())}
        if min(r["train_pos"], r["train_n"] - r["train_pos"], r["test_pos"], r["test_n"] - r["test_pos"]) < MIN_POS:
            r["rare"] = f"under {MIN_POS} of a class in train or test: counts only"
        else:
            for kind in ("raw", "cleaned"):
                sc = {}
                for e in ENC:
                    X = (d.clean if kind == "cleaned" else d.mats)[e]
                    sc[e] = AXS.Probe(X[trm], y[trm], d.c).d(X[te])
                a = {e: AXS.fast_auc(y[te], s) if hasattr(AXS, "fast_auc") else AXS.S.auc(y[te], s) for e, s in sc.items()}
                best = max(ENC[1:], key=lambda e: a[e])
                rng = np.random.default_rng(SEED)
                v, dv = [], []
                for _ in range(NBOOT):
                    i = rng.integers(0, te.sum(), te.sum())
                    yy = y[te][i]
                    x1, x2 = AXS.fast_auc(yy, sc["A1"][i]), AXS.fast_auc(yy, sc[best][i])
                    v.append(x1)
                    dv.append(x1 - x2)
                r[kind] = {"auc": a["A1"], "ci": np.percentile(v, [2.5, 97.5]).tolist(), "untrained": [a[e] for e in ENC[1:]], "gain_over_best_untrained": a["A1"] - a[best], "gain_ci": np.percentile(dv, [2.5, 97.5]).tolist()}
        out["flags"][n] = r
        save("b_extras.json", out)
        A.log(n, r.get("rare") or {k: round(r[k]["auc"], 3) for k in ("raw", "cleaned")}, r["test_pos"])


# --- the map and the Plate -----------------------------------------------------------------------------------------------
def step_map():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    AXS, d = A.load()
    import plate_export as PE  # after A.load: it reads f0_preconditions, whose freeze the ladder environment sets
    fit = json.loads((DOC / "b_fit.json").read_text())
    ax_ = json.loads((REPO / "docs/almagest/a1_axes.json").read_text())
    gz = [f for f in ax_["order"] if ax_["labels"][f] in ("STRONG", "SEPARATE")]
    phys = [n for n in fit if fit[n]["label"] in ("STRONG", "WEAK")]
    te = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
    X = d.clean["A1"]
    score, elig, null = {}, {}, {}
    for f in gz:
        a = np.load(AXS.AX / f"{f}.npz")
        score[f] = X @ a["we"] + float(a["be"])
        e = np.zeros(len(X), bool)
        e[d.rows(f)[1]] = True
        elig[f] = e
    for n in phys:
        a = np.load(B / "fits" / (n.replace("/", "_").replace(" ", "_") + ".npz"))
        score[n], elig[n] = X @ a["cleaned_we"] + float(a["cleaned_b"]), te
        null[n] = X @ a["null_we"].T
    names = gz + phys
    R, NN = np.eye(len(names)), np.zeros((len(names),) * 2, int)
    for i, p in enumerate(names):
        for k in range(i + 1, len(names)):
            o = elig[p] & elig[names[k]]
            NN[i, k] = NN[k, i] = int(o.sum())
            R[i, k] = R[k, i] = np.corrcoef(score[p][o], score[names[k]][o])[0, 1] if o.sum() >= 100 else np.nan
    nmax = {n: {f: float(max(abs(np.corrcoef(null[n][elig[f] & te, s], score[f][elig[f] & te])[0, 1]) for s in range(NNULL))) for f in gz} for n in phys}
    D = 1 - np.abs(np.nan_to_num(R))
    np.fill_diagonal(D, 0)
    lk = linkage(squareform((D + D.T) / 2, checks=False), "average")
    leaf = leaves_list(lk)
    cl = fcluster(lk, 0.5, "distance")
    P = [n for n in phys]
    tops = {f: sorted(((abs(R[names.index(f), names.index(n)]), n, R[names.index(f), names.index(n)]) for n in P), reverse=True)[:3] for f in gz}
    # how much of each human axis's score the physical scores explain together (test galaxies eligible for it; OLS on the STRONG physical scores, half fit, half tested)
    expl = {}
    strong = [n for n in phys if fit[n]["label"] == "STRONG"]
    rng = np.random.default_rng(SEED)
    for f in gz:
        o = np.flatnonzero(elig[f] & te)
        h = rng.permutation(len(o)) < len(o) // 2
        M = np.column_stack([np.ones(len(o))] + [score[n][o] for n in strong])
        M[:, 1:] = (M[:, 1:] - M[h, 1:].mean(0)) / M[h, 1:].std(0)
        c = np.linalg.solve(M[h].T @ M[h] + 1e-3 * np.eye(M.shape[1]), M[h].T @ score[f][o][h])
        y = score[f][o][~h]
        expl[f] = float(1 - ((y - M[~h] @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    save("b_map.json", {"names": names, "n_gz": len(gz), "r": np.round(R, 4).tolist(), "n_both": NN.tolist(), "null_max_abs_r": nmax, "leaf": [names[i] for i in leaf],
                        "cluster_at_0.5": {n: int(c) for n, c in zip(names, cl)}, "top3_physical": {f: [(n, r) for _, n, r in v] for f, v in tops.items()},
                        "r2_from_all_strong_physical_scores": expl, "strong_physical": strong})
    short = [n.split("_", 2)[-1][:26] if n in gz else n[:26].upper() for n in names]
    lab = [f"{short[i]}" for i in leaf]
    fig, ax = plt.subplots(figsize=(17, 15))
    im = ax.imshow(R[np.ix_(leaf, leaf)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    for i, k in enumerate(leaf):
        if names[k] in phys:
            ax.get_yticklabels()[i].set_color("#a03c14")
            ax.get_xticklabels()[i].set_color("#a03c14")
    fig.colorbar(im, ax=ax, shrink=0.6, label="correlation of cleaned A1 scores (test galaxies eligible for both)")
    ax.set_title("Physical axes (CAPITALS) and the 30 STRONG/SEPARATE Galaxy Zoo 2 axes, clustered on 1 − |r|")
    fig.tight_layout()
    fig.savefig(DOC / "b_map.png", dpi=110)
    # the Plate: a copy with additive fields; plate_axes.bin itself is not touched
    h, Pl = PE.read_plate(AXS.PLATE_OUT)
    w80, p80, hd = [], [], []
    for n in strong:
        a = np.load(B / "fits" / (n.replace("/", "_").replace(" ", "_") + ".npz"))
        w_st, p_st = a["cleaned_we"] * Pl["std_scale"], a["cleaned_sigma"] / Pl["std_scale"]
        wr, pr = Pl["basis"] @ w_st, Pl["basis"] @ p_st
        w80.append((wr / np.linalg.norm(wr)).astype(np.float32))
        p80.append((pr / np.linalg.norm(pr)).astype(np.float32))
        hd.append({"id": n, "label": "STRONG", "spearman_test": fit[n]["cleaned"]["enc"]["A1"]["spearman"], "n_train": fit[n]["n_train"],
                   "retention": {"filter_w": float(np.linalg.norm(wr) / np.linalg.norm(w_st)), "pattern_sigma_w": float(np.linalg.norm(pr) / np.linalg.norm(p_st))}})
    secs = [(s["name"], Pl[s["name"]], s["meaning"]) for s in h["sections"]]
    if strong:
        secs += [("phys_filter_w", np.stack(w80), f"[{len(strong)}, 80] unit filter w of each STRONG physical axis (cleaned), READ"), ("phys_pattern", np.stack(p80), f"[{len(strong)}, 80] unit pattern Σw, WALK")]
    h2 = {k: v for k, v in h.items() if k != "sections"}
    h2["phys_axes"] = hd
    h2["phys_axes_note"] = "Part B (docs/axes/b_report.md): physical quantities from catalogues and statmorph, ridge on cleaned A1, descriptive"
    dst = AXS.PLATE_OUT.with_name("plate_phys.bin")
    PE.write_plate(dst, h2, secs)
    h3, P3 = PE.read_plate(dst)
    assert all(np.array_equal(P3[k], Pl[k]) for k in Pl)
    A.log(f"map: {len(phys)} physical ({len(strong)} STRONG), plate {dst.name} {dst.stat().st_size / 1e6:.1f} MB, sections unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["catalogue", "selftest", "fit", "extras", "map"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    step_fit(a.only) if a.step == "fit" else globals()["step_" + a.step]()
