"""Part G (kickoff, 24 Nov 2026): expert gold sets. EFIGI (Baillard et al. 2011, J/A+A/532/A74, v1.6.2) and Nair & Abraham 2010 (J/ApJS/186/427 table 2).

    .venv/bin/python artifacts/axes_g.py match    # G1: 3″ matches, chance control, counts per resolution decile → docs/axes/g1.json
    .venv/bin/python artifacts/axes_g.py embed    # the three untrained seeds on the EFIGI matches outside the union (A1 is banked for train + test)
    .venv/bin/python artifacts/axes_g.py g2       # ordinal fits per EFIGI attribute, correlation with the GZ2 ordinal axes → docs/axes/g2.json
    .venv/bin/python artifacts/axes_g.py g3       # label sources against the expert grades → docs/axes/g3.json

Fixed before any G number was computed (descriptive, not hashed):
- Match: nearest of our galaxies within 3″, one per galaxy; a chance control shifts the catalogue by +60″ in Dec. The DR7 objID EFIGI carries is
  a cross-check on the position match, never a filter. Deciles = petroRad_r / psfWidth_r, edges from the union's test galaxies.
- G2 rows: train = the probe corpus' train partition (all 160,849, not only the 39,997 probe rows), test = its test partition; complete nuisances; val
  rows and pretrain matches are not used. Targets are EFIGI's value (0, .25, .5, .75, 1) as a one-hot over the ordered categories with weight 1; a category with
  < 20 train rows merges into its neighbour nearer the middle. Attributes with ≥ 500 (train + test) are fitted: Part A's cumulative logit and free multinomial
  through its own cv_fit / reading. Expert uncertainty (the confidence interval) is reported, not modelled. Cleaner = the union's.
- τ = Kendall τ-b of w·x with the attribute on test (200-resample CI for A1); per decile where ≥ 30 test rows, counts otherwise.
- Transfer: Part A's GZ2-trained ordinal score (cleaned, stored w) against the EFIGI attribute on the same test rows; and the Pearson r
  of the EFIGI axis' score with the GZ2 axis' score over the union's test rows eligible for the GZ2 question. Arm Curvature is 1 = tight and GZ2 winding is
  tight → loose, so agreement is NEGATIVE there; signs are reported as they come.

Investigation code: terse, excluded from lint/CI. Large outputs in runs/almagest/axes_phys/g/.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time

import numpy as np
from scipy.stats import kendalltau

import axes_a_ordinal as A

G = A.OUT / "g"
SEED = A.SEED + 80
RADIUS = 3.0
ENC = ("A1", "untrained-s0", "untrained-s1", "untrained-s2")
ATTR = {"armC": "Arm Curvature", "armS": "Arm Strength", "B/T": "B/T", "floc": "Flocculence", "inc": "Inclination/Elongation", "Pert": "Perturbation",
        "vDust": "Visible Dust", "pDust": "Dust Dispersion", "spot": "Hot Spots", "armR": "Arm Rotation", "barL": "Bar Length", "iRing": "Inner Ring",
        "oRing": "Outer Ring", "pRing": "Pseudo-Ring", "cont": "Contamination", "mult": "Multiplicity"}
COUNTERPART = {"armC": "t10_arms_winding", "armS": "t10_arms_winding", "B/T": "t05_bulge_prominence", "inc": "t07_rounded"}  # armS beside armC: reported, not a pair
CATS = {"efigi": ("efigi.tsv", "RAJ2000", "DEJ2000"), "na10": ("nair2010.tsv", "_RA", "_DE")}


def vizier(name):
    import pandas as pd
    L = [x for x in open(G / name) if not x.startswith("#") and x.strip()]
    return pd.read_csv(io.StringIO(L[0] + "".join(L[3:])), sep="\t", dtype=str).apply(lambda s: s.str.strip())


def ours(d, corpus="probe"):
    import pandas as pd
    p = A.REPO / str(d.setup.cfg.paths.probe_dir if corpus == "probe" else d.setup.cfg.paths.pretrain_dir) / "metadata.csv"
    cols = ["object_id", "ra", "dec", "petroRad_r"] + (["psfWidth_r", "dr7objid"] if corpus == "probe" else [])
    return pd.read_csv(p, usecols=cols, dtype={"object_id": str, "dr7objid": str}, low_memory=False)


def nearest(cat, ra, de, t, shift=0.0):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import pandas as pd
    c = SkyCoord(cat[ra].astype(float).to_numpy() * u.deg, (cat[de].astype(float).to_numpy() + shift / 3600) * u.deg)
    i, s, _ = c.match_to_catalog_sky(SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg))
    m = pd.DataFrame({"row": i, "sep": s.arcsec, "src": np.arange(len(cat))})
    return m[m.sep < RADIUS].sort_values("sep").drop_duplicates("row").sort_values("src")


def edges(d):
    size, psf = (np.asarray(d.lab.nuisance_value(k, d.union), float) for k in ("size", "psf"))
    te = np.isin(d.union, np.asarray(d.setup.test_ids))
    return np.quantile((size / psf)[te & np.isfinite(size / psf)], np.linspace(0, 1, 11))


def partitions(d):
    import y_pitch as Y
    return {k: {"A": "train", "B": "test"}.get(v, v) for k, v in Y.partitions(d.setup).items()}


def step_match():
    AXS, d = A.load()
    ed = edges(d)
    part = partitions(d)
    out = {"radius_arcsec": RADIUS, "files": {f.name: {"sha1": hashlib.sha1(f.read_bytes()).hexdigest(), "rows": len(vizier(f.name))} for f in sorted(G.glob("*.tsv"))},
           "source": "VizieR (Harvard mirror): J/A+A/532/A74 efigi + sdss; J/ApJS/186/427 table2; VizieR usage terms, cite the papers; no explicit open licence stated",
           "decile_edges_petroRad_over_psf": ed.tolist(), "catalogues": {}}
    for name, (f, ra, de) in CATS.items():
        c = vizier(f)
        rec = {"rows": len(c)}
        for corpus in ("probe", "pretrain"):
            t = ours(d, corpus)
            m = nearest(c, ra, de, t)
            ids = t.object_id.to_numpy()[m.row].astype(np.int64)
            if corpus == "probe":
                r = (t.petroRad_r.astype(float) / t.psfWidth_r.astype(float)).to_numpy()[m.row]
                dec = np.clip(np.searchsorted(ed, r, side="right") - 1, 0, 9)
            e = {"corpus_rows": len(t), "matched": len(m), "chance_shift_60arcsec": len(nearest(c, ra, de, t, 60.0)), "sep_q50_q95": np.percentile(m.sep, [50, 95]).tolist(),
                 "per_decile": np.bincount(dec, minlength=10).tolist() if corpus == "probe" else "no psfWidth_r in the pretrain metadata", "median_petroRad": float(np.median(t.petroRad_r.astype(float).to_numpy()[m.row]))}
            if corpus == "probe":
                e["partition"] = {k: int(sum(part.get(int(i)) == k for i in ids)) for k in sorted({str(v) for v in part.values()})}
                e["in_union"] = int(np.isin(ids, d.union).sum())
                if name == "efigi":
                    s = vizier("efigi_sdss.tsv").set_index("PGC")["objID"]
                    e["dr7_objid_agrees"] = int((s.reindex(c.PGC.iloc[m.src]).to_numpy() == t.dr7objid.to_numpy()[m.row]).sum())
                    e["dr7_objid_known"] = int((s.reindex(c.PGC.iloc[m.src]).fillna("").str.len() > 10).sum())
                np.savez(G / f"match_{name}.npz", ids=ids, src=m.src.to_numpy(), sep=m.sep.to_numpy(), dec=dec, part=np.array([str(part.get(int(i), "absent")) for i in ids]))
            rec[corpus] = e
        if name == "efigi":
            z = np.load(G / "match_efigi.npz", allow_pickle=False)
            rec["attributes"] = {k: {"finite_matched": int(np.isfinite(vals(c, k)[z["src"]]).sum()), "distribution": np.unique(vals(c, k)[z["src"]][np.isfinite(vals(c, k)[z["src"]])], return_counts=True)[1].tolist()} for k in ATTR}
        out["catalogues"][name] = rec
    (A.DOC / "g1.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["catalogues"], indent=1, default=float))


def vals(c, k):
    import pandas as pd
    return pd.to_numeric(c[k], errors="coerce").to_numpy(float)


def efigi_rows(d):
    """(ids, catalogue rows, train mask, test mask) of the EFIGI matches in the train / test partitions."""
    z = np.load(G / "match_efigi.npz", allow_pickle=False)
    tr, te = z["part"] == "train", z["part"] == "test"
    te_u = set(map(int, d.setup.test_ids))
    assert all(int(i) in te_u for i in z["ids"][te]), "the test partition is not the union's"
    assert not any(int(i) in te_u for i in z["ids"][tr])
    k = tr | te
    return z["ids"][k], z["src"][k], tr[k], te[k], z["dec"][k]


def step_embed():
    AXS, d = A.load()
    ids = efigi_rows(d)[0]
    todo = np.sort(ids[~np.isin(ids, d.union)])
    import j4_spread_controls as J
    from galaxy_jepa.probing.controls import untrained_encoder_matrix
    from galaxy_jepa.probing.extract import extract_matrix
    st = d.setup
    fz = AXS.load_frozen_encoder(st.ckpt)
    chk = np.sort(ids[np.isin(ids, d.union)])[:200]  # 200 union galaxies re-embedded: the code path must reproduce the bank
    allids = np.sort(np.concatenate([todo, chk]))
    ds = type(st.ds)(J.check(verbose=False)[1], st.rows, [int(i) for i in allids])
    out = {"ids": allids, "A1": extract_matrix(fz, ds, device=st.device, batch_size=32).x.astype(np.float32)}
    for s in range(3):
        m = untrained_encoder_matrix(fz.config, ds, device=st.device, seed=s, batch_size=32)
        assert np.array_equal(np.asarray(m.object_ids), allids)
        out[f"untrained-s{s}"] = m.x.astype(np.float32)
    bank = np.load(A.REPO / "runs/ladder_v2/a1/c2_a1_embeddings.npz")
    pos = {int(o): k for k, o in enumerate(bank["ids"])}
    c = {"A1 vs bank": float(np.abs(out["A1"] - bank["x"][[pos[int(i)] for i in allids]]).max())}
    inu = np.isin(allids, d.union)
    for n in ENC:
        c[f"{n} vs union ({int(inu.sum())})"] = float(np.abs(out[n][inu] - d.mats[n][[d.row[int(i)] for i in allids[inu]]]).max())
    print(c)
    assert max(c.values()) < 5e-2, c
    np.savez(G / "efigi_embeddings.npz", **out, check=json.dumps(c))


def matrices(d, ids):
    z = np.load(G / "efigi_embeddings.npz", allow_pickle=False)
    pos = {int(o): k for k, o in enumerate(z["ids"])}
    out = {"raw": {}, "cleaned": {}}
    import almagest_axes as AXS
    N = np.column_stack([d.lab.nuisance_value(n, ids) for n in AXS.S.NUIS])
    ok = np.isfinite(N).all(1)
    for n in ENC:
        x = np.array([d.mats[n][d.row[int(i)]] if int(i) in d.row else z[n][pos[int(i)]] for i in ids], dtype=np.float64)
        c = np.full_like(x, np.nan)
        c[ok] = d.cleaner[n](x[ok], N[ok])
        out["raw"][n], out["cleaned"][n] = x, c
    return out, ok


def onehot(v, tr):
    """EFIGI value → one-hot over the ordered categories; a category under 20 train rows merges towards the middle."""
    cats = [[float(x)] for x in np.unique(v)]
    cnt = lambda g: int(np.isin(v[tr], g).sum())  # noqa: E731
    merged = []
    while len(cats) > 2:
        small = [k for k in range(len(cats)) if cnt(cats[k]) < 20]
        if not small:
            break
        k = small[0] if small[0] < len(cats) / 2 else small[-1]
        j = k + 1 if k < (len(cats) - 1) / 2 else k - 1
        lo, hi = sorted((k, j))
        merged.append([cats[lo], cats[hi]])
        cats[lo:hi + 1] = [cats[lo] + cats[hi]]
    P = np.column_stack([np.isin(v, g) for g in cats]).astype(float)
    return P, cats, merged


def tau_ci(s, e, rng):
    bs = [kendalltau(s[i], e[i])[0] for i in (rng.integers(0, len(s), len(s)) for _ in range(200))]
    return float(kendalltau(s, e)[0]), [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]


def step_g2():
    AXS, d = A.load()
    cat = vizier("efigi.tsv")
    ids, src, tr0, te0, dec0 = efigi_rows(d)
    M, ok = matrices(d, ids)
    path = A.DOC / "g2.json"
    out = {"n_rows": int(len(ids)), "n_complete_nuisances": int(ok.sum()), "embed_check": json.loads(str(np.load(G / "efigi_embeddings.npz")["check"])), "attributes": {}}
    te_u = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
    # the GZ2 ordinal axes (cleaned; stored by Part A)
    gz = {q: np.load(A.OUT / f"a_{q}.npz") for q in set(COUNTERPART.values())}
    gzs = lambda q, X: ((X - gz[q]["mu"]) / gz[q]["sd"]) @ gz[q]["w_cleaned"]  # noqa: E731
    elig = {q: A.question_rows(AXS, d, n) for n, q in ((n, A.QUESTIONS[n]["q"]) for n in A.QUESTIONS) if q in gz}
    for ai, (k, label) in enumerate(ATTR.items()):
        v = vals(cat, k)[src]
        lo, hi = vals(cat, "b_" + k)[src], vals(cat, "B_" + k)[src]
        m = ok & np.isfinite(v)
        rec = {"label": label, "n_train": int((m & tr0).sum()), "n_test": int((m & te0).sum()), "median_ci_width": float(np.nanmedian((hi - lo)[m])),
               "test_per_decile": np.bincount(dec0[m & te0], minlength=10).tolist()}
        out["attributes"][k] = rec
        if rec["n_train"] + rec["n_test"] < 500:
            rec["skipped"] = "fewer than 500 matched"
            continue
        y, tr, te, dec = v[m], tr0[m], te0[m], dec0[m]
        P, cats, merged = onehot(y, tr)
        ecat = P @ np.arange(1, P.shape[1] + 1)
        rec.update(categories=cats, merged=merged, counts_train=P[tr].sum(0).tolist(), counts_test=P[te].sum(0).tolist(), enc={})
        n = np.ones(len(y))
        for kind in ("cleaned", "raw"):
            rec["enc"][kind] = {}
            for enc in ENC:
                X = M[kind][enc][m]
                t0 = time.perf_counter()
                try:
                    r, ex = A.analyse(X, np.zeros(len(y)), P, n, tr, te, SEED + ai, want_models=True)
                except RuntimeError as err:
                    rec["enc"][kind][enc] = {"failed": str(err)}
                    A.log(f"G2 {k} {kind} {enc} FAILED {err}")
                    continue
                s = ex["Zte"] @ ex["mods"]["ord"]["w"]
                e = {q: r[q] for q in ("LL", "gap_free_base", "gap_ci95", "gap_ord_base", "share", "share_ci95", "reading")}
                e["lambda"] = {q: r[q]["lambda"] for q in ("ord", "free")}
                e["tau"], e["tau_ci"] = tau_ci(s, y[te], np.random.default_rng(SEED)) if enc == "A1" else (float(kendalltau(s, y[te])[0]), None)
                e["tau_per_decile"] = [float(kendalltau(s[b], y[te][b])[0]) if (b := dec[te] == j).sum() >= 30 else None for j in range(10)]
                if k in COUNTERPART and kind == "cleaned" and enc == "A1":
                    q = COUNTERPART[k]
                    g = gzs(q, X[te])
                    rng = np.random.default_rng(SEED)
                    t, ci = tau_ci(g, y[te], rng)
                    r_ = elig[q]
                    Xu = d.clean["A1"][r_["rows"][r_["te"]]]
                    e["gz2_axis"] = {"question": q, "tau_gz2_axis_vs_attribute": t, "ci": ci, "r_scores_on_efigi_test": float(np.corrcoef(s, g)[0, 1]),
                                     "r_scores_on_union_test_eligible": float(np.corrcoef((Xu - ex["mu"]) / ex["sd"] @ ex["mods"]["ord"]["w"], gzs(q, Xu))[0, 1]), "n_union_test_eligible": len(Xu)}
                rec["enc"][kind][enc] = e
                A.log(f"G2 {k} {kind} {enc} n={tr.sum()}/{te.sum()} gap {e['gap_free_base']:.4f} {np.round(e['gap_ci95'], 3)} share {e['share']:.2f} τ {e['tau']:.3f} {r['reading']} {time.perf_counter() - t0:.0f}s")
                if enc == "A1":
                    np.savez(G / f"efigi_{k.replace('/', '')}_{kind}.npz", w=ex["mods"]["ord"]["w"], th=ex["mods"]["ord"]["th"], mu=ex["mu"], sd=ex["sd"])
            path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
