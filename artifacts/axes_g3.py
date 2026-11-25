"""Part G3 (kickoff, 24 Nov 2026): label sources scored against expert grades. No encoder, no Claude API.

    .venv/bin/python artifacts/axes_g3.py      # → docs/axes/g3.json

Fixed before any τ was computed (descriptive, not hashed):
- Galaxies: every probe-corpus match (all partitions; no embedding is read). Sources: GZ2 votes (our metadata); DECaLS votes in GZD-5 else GZD-8 (one tree);
  GZD-1/2 separately (its own answer sets); Galaxy Zoo DESI = Zoobot predictions v1.0.1 (Zenodo 8360385, 'friendly': questions blank unless predicted to be
  reached by ≥ half the volunteers), nearest within 3″ of our position.
- A vote source counts only where it has ≥ 5 votes on the question's ordered answers; ≥ 1 is the sensitivity arm. For ring, dust lane and disturbance GZ2's value is P(odd = yes) × share of the
  relevant odd answers, which exists for every galaxy with ≥ 5 votes on "odd".
- Scale per quantity: expected category rescaled to [0, 1] (bulge: each tree's own answers); bar = (strong + ½ weak) for GZD-5/8 and DESI, P(bar) for GZ2, P(yes) for GZD-1/2;
  winding is reported as tightness (1 − rescaled category), so the experts' curvature agrees with it positively.
- Metric: Kendall τ-b with a 1,000-resample bootstrap CI over galaxies, on (a) each source's own coverage, (b) the galaxies every named source covers: GZ2 ∩ DESI, and
  GZ2 ∩ GZD-5/8 ∩ DESI. Paired bootstrap differences against GZ2 on those sets. n < 100 is reported as a count only. Binary N&A grades also get AUC.
- DESI and the DECaLS votes are not independent of each other (Zoobot was trained on those votes), and neither is blind to the galaxies here; experts read SDSS colour images, GZ DESI and
  DECaLS deeper ones. A higher τ can therefore be the better image, not the better labeller.

Investigation code: terse, excluded from lint/CI.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import kendalltau

import axes_a_ordinal as A
import axes_g as G0

NB, MINN = 1000, 100
PARQUET = G0.G / "gz_desi_friendly.parquet"
ARMS = {"armC": "winding", "armS": "spiral", "B/T": "bulge", "inc": "roundness", "inc ": "edge_on", "barL": "bar", "Pert": "disturbed", "vDust": "dust_lane", "ring": "ring", "T": "featured", "T ": "bulge"}
NA = {"TT": "featured", "TT ": "bulge", "TT  ": "winding", "bar_ord": "bar", "bar_any": "bar", "ring_any": "ring", "interaction": "disturbed"}
BINARY = {"bar_any", "ring_any", "interaction"}


def ecat(cols):
    """Expected category in [0, 1] and the votes, from counts (or fractions, votes NaN) in order."""
    M = np.column_stack(cols).astype(float)
    t = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return M @ np.linspace(0, 1, M.shape[1]) / t, t


def two(yes, no):
    return ecat([no, yes])


def gz2(d, ids):
    L = d.lab
    n = lambda q: np.nan_to_num(np.asarray(L._column(ids, q + "_count"), float))  # noqa: E731
    out = {}
    c = lambda q, a: n(f"{q}_{a}")  # noqa: E731
    out["featured"] = two(c("t01_smooth_or_features", "a02_features_or_disk"), c("t01_smooth_or_features", "a01_smooth"))
    out["edge_on"] = two(c("t02_edgeon", "a04_yes"), c("t02_edgeon", "a05_no"))
    out["spiral"] = two(c("t04_spiral", "a08_spiral"), c("t04_spiral", "a09_no_spiral"))
    out["bar"] = two(c("t03_bar", "a06_bar"), c("t03_bar", "a07_no_bar"))
    out["bulge"] = ecat([c("t05_bulge_prominence", a) for a in A.QUESTIONS["bulge prominence"]["cats"]])
    out["roundness"] = ecat([c("t07_rounded", a) for a in A.QUESTIONS["roundness"]["cats"]])
    e, t = ecat([c("t10_arms_winding", a) for a in A.QUESTIONS["winding"]["cats"]])
    out["winding"] = (1 - e, t)
    odd, no = two(c("t06_odd", "a14_yes"), c("t06_odd", "a15_no"))

    F = {a: n(f"t08_odd_feature_{a}") for a in ("a19_ring", "a20_lens_or_arc", "a21_disturbed", "a22_irregular", "a23_other", "a24_merger", "a38_dust_lane")}
    tot = sum(F.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        share = lambda ks: np.where(tot > 0, sum(F[k] for k in ks) / np.where(tot > 0, tot, 1), 0.0)  # noqa: E731
    out["disturbed"] = (odd * share(["a21_disturbed", "a22_irregular", "a24_merger"]), no)
    out["ring"] = (odd * share(["a19_ring"]), no)
    out["dust_lane"] = (odd * share(["a38_dust_lane"]), no)
    return out


def from_counts(df, tree):
    """Counts (DECaLS) or fractions (DESI), per quantity. tree: 'g5' (GZD-5/8/DESI answers) or 'g12'."""
    g = lambda c: df[c].astype(float).fillna(0).to_numpy()  # noqa: E731
    o = {"featured": two(g("smooth-or-featured_featured-or-disk"), g("smooth-or-featured_smooth")), "edge_on": two(g("disk-edge-on_yes"), g("disk-edge-on_no")),
         "spiral": two(g("has-spiral-arms_yes"), g("has-spiral-arms_no"))}
    e, t = ecat([g(f"spiral-winding_{a}") for a in ("tight", "medium", "loose")])
    o["winding"] = (1 - e, t)
    o["roundness"] = ecat([g(f"how-rounded_{a}") for a in (["completely"] if tree == "g12" else ["round"])[:1] + ["in-between", "cigar-shaped"]])
    if tree == "g12":
        o["bulge"] = ecat([g(f"bulge-size_{a}") for a in ("none", "obvious", "dominant")])
        o["bar"] = two(g("bar_yes"), g("bar_no"))
        o["disturbed"] = (1 - g("merging_neither") / np.where((t2 := sum(g(f"merging_{a}") for a in ("merger", "tidal-debris", "both", "neither"))) > 0, t2, np.nan), t2)
    else:
        o["bulge"] = ecat([g(f"bulge-size_{a}") for a in ("none", "small", "moderate", "large", "dominant")])
        o["bar"] = ecat([g("bar_no"), g("bar_weak"), g("bar_strong")])
        t2 = sum(g(f"merging_{a}") for a in ("none", "minor-disturbance", "major-disturbance", "merger"))
        with np.errstate(invalid="ignore", divide="ignore"):
            o["disturbed"] = (1 - g("merging_none") / np.where(t2 > 0, t2, np.nan), t2)
    return o


def desi(ra, dec):
    import pyarrow.parquet as pq
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    f = pq.ParquetFile(PARQUET)
    pos = f.read(columns=["ra", "dec"])
    c = SkyCoord(pos["ra"].to_numpy() * u.deg, pos["dec"].to_numpy() * u.deg)
    i, s, _ = SkyCoord(ra * u.deg, dec * u.deg).match_to_catalog_sky(c)
    ok = s.arcsec < G0.RADIUS
    import pandas as pd
    cols = [x for x in f.schema.names if x.endswith("_fraction")]
    df = pd.DataFrame({x[:-9]: f.read(columns=[x])[x].to_numpy()[i] for x in cols})
    df[~ok] = np.nan
    o = from_counts(df, "g5")
    nan = {"featured": "smooth-or-featured_smooth", "edge_on": "disk-edge-on_yes", "spiral": "has-spiral-arms_yes", "winding": "spiral-winding_tight", "roundness": "how-rounded_round", "bulge": "bulge-size_none", "bar": "bar_no", "disturbed": "merging_none"}
    for k, c_ in nan.items():  # a blank question is "not asked", not a zero
        e, t = o[k]
        e = np.where(df[c_].isna().to_numpy(), np.nan, e)
        o[k] = (e, np.where(np.isfinite(e), 99, 0))
    return o, int(ok.sum())


def boot(x, y, idx):
    return np.array([kendalltau(x[b], y[b])[0] for b in idx])


def auc(y, s):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    AXS, d = A.load()
    import axes_f

    raw = axes_f.decals_counts(d.setup)
    cats = {"efigi": G0.vizier("efigi.tsv"), "na10": G0.vizier("nair2010.tsv")}
    t = G0.ours(d, "probe")
    t.index = t.object_id.astype(np.int64)
    out = {"min_votes": 5, "rule": "see docstring", "references": {}}
    for ref, cat in cats.items():
        z = np.load(G0.G / f"match_{ref}.npz", allow_pickle=False)
        ids, src = z["ids"], z["src"]
        ex = {}
        if ref == "efigi":
            for k in ("armC", "armS", "B/T", "inc", "barL", "Pert", "vDust"):
                ex[k] = G0.vals(cat, k)[src]
            ex["ring"] = np.nanmax(np.column_stack([G0.vals(cat, k)[src] for k in ("iRing", "oRing", "pRing")]), 1)
            ex["T"] = G0.vals(cat, "T")[src]
            ex["inc "], ex["T "] = ex["inc"], ex["T"]
            pairs = [(k, ARMS[k]) for k in ARMS]
        else:
            g = lambda k: G0.vals(cat, k)[src]  # noqa: E731
            ex["TT"] = ex["TT "] = ex["TT  "] = np.where(g("TT") > 11, np.nan, g("TT"))
            b = g("Bar").astype(int)
            ex["bar_ord"] = np.where(b & 2, 3, np.where(b & 4, 2, np.where(b & 8, 1, 0))).astype(float)
            ex["bar_any"] = ((b & 14) > 0).astype(float)
            ex["ring_any"] = ((g("Ring").astype(int) & 14) > 0).astype(float)
            ex["interaction"] = ((g("Int").astype(int) & ~2) > 0).astype(float)
            pairs = [(k, NA[k]) for k in NA]
        sub = t.loc[ids]
        S = {"GZ2": gz2(d, ids)}
        for name, cs, tree in (("GZD-5/8", ("gzd5", "gzd8"), "g5"), ("GZD-1/2", ("gzd12",), "g12")):
            parts = {}
            frames = [(raw[c][0], raw[c][1].assign(_r=0)) for c in cs]
            o = None
            for oid, df in frames:
                q = from_counts(df, tree)
                idx = {int(a): k for k, a in enumerate(oid)}
                take = np.array([idx.get(int(i), -1) for i in ids])
                for k, (v, n_) in q.items():
                    vv = np.where(take >= 0, np.asarray(v, float)[np.maximum(take, 0)], np.nan)
                    nn = np.where(take >= 0, np.asarray(n_, float)[np.maximum(take, 0)], 0.0)
                    if k in parts:
                        use = (parts[k][1] <= 0) & (nn > 0)
                        parts[k] = (np.where(use, vv, parts[k][0]), np.where(use, nn, parts[k][1]))
                    else:
                        parts[k] = (vv, nn)
            S[name] = parts
        S["GZ DESI"], nm = desi(sub.ra.to_numpy(float), sub.dec.to_numpy(float))
        rec = {"n_matched": len(ids), "n_desi_matched": nm, "n_with_any_votes": {k: int((v["featured"][1] > 0).sum()) for k, v in S.items()}, "pairs": {}}
        for key, q in pairs:
            y = ex[key]
            e = {"expert": key.strip(), "label": q, "min1": {}, "min5": {}}
            for tag, mv in (("min5", 5), ("min1", 1)):
                cov = {s: np.isfinite(y) & np.isfinite(S[s][q][0]) & (S[s][q][1] >= mv) for s in S if q in S[s]}
                blk = {"own_coverage": {}, "common": {}}
                rng = np.random.default_rng(A.SEED + 90)
                for s, m in cov.items():
                    n = int(m.sum())
                    if n < MINN:
                        blk["own_coverage"][s] = {"n": n}
                        continue
                    ys, xs = y[m], S[s][q][0][m]
                    idx = rng.integers(0, n, (NB, n))
                    bs = boot(xs, ys, idx)
                    r = {"n": n, "tau": float(kendalltau(xs, ys)[0]), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
                    if key in BINARY:
                        r["auc"] = auc(ys.astype(int), xs)
                        r["n_positive"] = int(ys.sum())
                    blk["own_coverage"][s] = r
                for lab, need in (("GZ2+DESI", ("GZ2", "GZ DESI")), ("GZ2+GZD-5/8+DESI", ("GZ2", "GZD-5/8", "GZ DESI"))):
                    if not all(s in cov for s in need):
                        continue
                    m = np.logical_and.reduce([cov[s] for s in need])
                    n = int(m.sum())
                    r = {"n": n}
                    if n >= MINN:
                        idx = rng.integers(0, n, (NB, n))
                        ys = y[m]
                        B = {s: boot(S[s][q][0][m], ys, idx) for s in need}
                        for s in need:
                            r[s] = {"tau": float(kendalltau(S[s][q][0][m], ys)[0]), "ci": np.percentile(B[s], [2.5, 97.5]).tolist()}
                            if key in BINARY:
                                r[s]["auc"] = auc(ys.astype(int), S[s][q][0][m])
                        for s in need[1:]:
                            r[f"{s} − GZ2"] = {"delta": r[s]["tau"] - r["GZ2"]["tau"], "ci": np.percentile(B[s] - B["GZ2"], [2.5, 97.5]).tolist()}
                    blk["common"][lab] = r
                e[tag] = blk
            rec["pairs"][f"{key.strip()} ~ {q}" + (" (2)" if f"{key.strip()} ~ {q}" in rec["pairs"] else "")] = e
            b5 = e["min5"]["common"].get("GZ2+DESI", {})
            A.log(f"G3 {ref} {key.strip()} ~ {q}: " + "; ".join(f"{s} n={v['n']} τ={v['tau']:.3f}" if "tau" in v else f"{s} n={v['n']}" for s, v in e["min5"]["own_coverage"].items())
                  + (f" | common n={b5['n']} " + " ".join(f"{k} {b5[k]['tau']:.3f}" for k in ("GZ2", "GZ DESI") if k in b5) if b5 else ""))
        out["references"][ref] = rec
        (A.DOC / "g3.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
