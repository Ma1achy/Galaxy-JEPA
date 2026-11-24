"""Part C (brief, 24 Nov 2026): star/artifact — is the label lumped, or is A1 blind to the subtypes? Descriptive; frozen embeddings + SDSS flags.

    .venv/bin/python artifacts/axes_c_star.py pull    # public SkyServer SQL, no token: flags, types, spectra (pinned by SHA-1)
    .venv/bin/python artifacts/axes_c_star.py embed   # three untrained seeds (and A1, as a check) on the sample
    .venv/bin/python artifacts/axes_c_star.py run     # k-means, silhouette, ARI against the flag groups
    .venv/bin/python artifacts/axes_c_star.py sheet   # 16 galaxies per cluster

Investigation code: terse, excluded from lint/CI. Writes docs/axes/ and runs/almagest/axes_phys/c/.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
DOC, OUT = REPO / "docs/axes", REPO / "runs/almagest/axes_phys/c"
BANK = REPO / "runs/ladder_v2/a1/c2_a1_embeddings.npz"
FRAC = "t01_smooth_or_features_a03_star_or_artifact_fraction"
SEED, BATCH = 20261124, 100
CSV, REC, CONST = OUT / "flags.csv", OUT / "flags_record.json", OUT / "photoflags.json"
SQL = ("SELECT CAST(p.objID AS varchar(20)) AS objID, CAST(p.flags AS varchar(24)) AS flags, "
       "CAST(p.flags_g AS varchar(24)) AS flags_g, CAST(p.flags_r AS varchar(24)) AS flags_r, CAST(p.flags_i AS varchar(24)) AS flags_i, "
       "p.type, p.type_g, p.type_r, p.type_i, p.clean, p.mode, p.nChild, CAST(p.parentID AS varchar(20)) AS parentID, p.insideMask, "
       "p.psfMag_r, p.modelMag_r, p.modelMag_g, p.modelMag_i, p.petroR50_r, p.petroR90_r, p.petroRad_r, p.probPSF_r, "
       "s.class AS spec_class, s.subClass AS spec_subclass, s.z AS spec_z, s.zWarning AS spec_zwarning "
       "FROM PhotoObjAll p LEFT JOIN SpecObj s ON s.bestObjID = p.objID WHERE p.objID IN ({})")


def sample() -> tuple[np.ndarray, np.ndarray, dict]:
    """(star ids, control ids, n record): bank galaxies at ≥ 0.5, or ≥ 0.3 if that leaves < 200; controls = an equal-size
    random draw of bank galaxies with fraction < 0.1 (base rates only, never clustered)."""
    import pandas as pd
    m = pd.read_csv(REPO / "data/probe_v2/metadata.csv", usecols=["object_id", FRAC]).set_index("object_id")[FRAC]
    ids = np.load(BANK)["ids"].astype(np.int64)
    v = m.reindex(ids).to_numpy()
    n = {f">={t}": {"bank": int((v >= t).sum()), "corpus": int((m >= t).sum())} for t in (0.3, 0.5)}
    thr = 0.5 if n[">=0.5"]["bank"] >= 200 else 0.3
    star = np.sort(ids[v >= thr])
    ctrl = np.sort(np.random.default_rng(SEED).choice(ids[v < 0.1], len(star), replace=False))
    return star, ctrl, {"counts": n, "threshold": thr, "n": len(star), "bank": len(ids), "corpus": len(m)}


def retry(sql, tries=12):
    from galaxy_jepa.data.metadata import run_sql
    for k in range(tries):
        try:
            return run_sql(sql, timeout=120)
        except Exception as exc:  # noqa: BLE001 — transient
            if k == tries - 1:
                raise
            print(f"  SkyServer {type(exc).__name__}: {str(exc)[:120]}; retry {k + 1}", file=sys.stderr)
            time.sleep(min(5 * 2 ** k, 60))


def step_pull():
    import pandas as pd
    if REC.exists():
        raise SystemExit("pulled already; pinned")
    star, ctrl, rec = sample()
    const = {r["name"].strip(): int(r["value"], 16) if str(r["value"]).lower().startswith("0x") else int(r["value"])
             for r in retry("SELECT name, CAST(CAST(value AS bigint) AS varchar(24)) AS value FROM PhotoFlags")}
    CONST.write_text(json.dumps(const, indent=1))
    ids = [str(i) for i in np.concatenate([star, ctrl])]
    rows = []
    for k in range(0, len(ids), BATCH):
        rows += retry(SQL.format(",".join(ids[k:k + BATCH])))
        time.sleep(1)
    df = pd.DataFrame(rows)
    dup = int(df.objID.duplicated().sum())
    df = df.drop_duplicates("objID").sort_values("objID")
    df.to_csv(CSV, index=False)
    r = {"when": time.strftime("%Y-%m-%d %H:%M"), "service": "public SkyServer SQL, DR17, no token", "query": SQL, "sample": rec,
         "requested": len(ids), "rows": len(df), "dups": dup, "missing": len(set(ids) - set(df.objID)), "n_flag_names": len(const),
         "sha1": hashlib.sha1(CSV.read_bytes()).hexdigest()}
    REC.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "query"}, indent=1))


def table():
    import pandas as pd
    r = json.loads(REC.read_text())
    assert hashlib.sha1(CSV.read_bytes()).hexdigest() == r["sha1"], "flags.csv changed since the pull"
    df = pd.read_csv(CSV, dtype={k: str for k in ("objID", "flags", "flags_g", "flags_r", "flags_i", "parentID")}).set_index("objID")
    const = json.loads(CONST.read_text())
    F = {n: np.array([bool(int(x) & b) for x in df["flags"]]) for n, b in const.items()}
    return df, pd.DataFrame(F, index=df.index)



# --- fixed before any clustering (24 Nov) ------------------------------------------------------------------------------
# Single flags: those set on 5–95% of the sample. Groups: the pattern of SATURATED × EDGE × BLENDED (patterns under 10 galaxies
# merged into "other"). Nothing here reads an embedding. BRIGHT and DEBLENDED_AS_MOVING are never set; DEBLENDED_AT_EDGE ≡ CHILD.
KEY = ("SATURATED", "EDGE", "BLENDED")
KS = range(2, 9)
N_SUB, SUB_FRAC, N_PERM = 100, 0.8, 1000
# "clearly better": A1's ARI (pattern groups, silhouette k) beats every untrained seed, the 80%-subsample paired difference against
# the BEST seed has a 2.5th percentile > 0, and A1's ARI exceeds the 97.5th percentile of label permutations.


def groups(df, F):
    pat = np.array(["+".join(k for k in KEY if F.loc[i, k]) or "none" for i in df.index])
    u, n = np.unique(pat, return_counts=True)
    rare = set(u[n < 10])
    return np.array(["other" if p in rare else p for p in pat])


def footing(ids):
    """Raw and cleaned embeddings of the sample for A1 and the three untrained seeds, standardised on the union's train rows."""
    import almagest_axes as A
    D = A.Data()
    z = np.load(OUT / "untrained.npz")
    assert np.array_equal(z["ids"], ids)
    bank = np.load(BANK)
    pos = {int(o): k for k, o in enumerate(bank["ids"])}
    X = {"A1": bank["x"][[pos[int(o)] for o in ids]].astype(np.float64)}
    for s in range(3):
        X[f"untrained-s{s}"] = z[f"s{s}"].astype(np.float64)
    inu = np.array([int(o) in D.row for o in ids])
    chk = {"A1 embed vs bank": float(np.abs(z["A1"] - X["A1"]).max())}
    for n in X:
        chk[f"{n} vs union bank ({int(inu.sum())})"] = float(np.abs(X[n][inu] - D.mats[n][[D.row[int(o)] for o in ids[inu]]]).max())
    N = np.column_stack([D.lab.nuisance_value(n, ids) for n in A.S.NUIS])
    ok = np.isfinite(N).all(1)
    tr = np.isin(D.union, np.asarray(D.setup.train_ids)) & D.ok
    out = {}
    for n, x in X.items():
        mu, sd = D.mats[n][tr].mean(0), D.mats[n][tr].std(0)
        out["raw", n] = (x - mu) / sd
        c = D.cleaner[n](x[ok], N[ok])
        mu, sd = D.clean[n][tr].mean(0), D.clean[n][tr].std(0)
        full = np.full_like(x, np.nan)
        full[ok] = (c - mu) / sd
        out["cleaned", n] = full
    return out, ok, chk, N, list(A.S.NUIS)


def step_embed():
    import torch
    import almagest_axes as A
    from galaxy_jepa.probing.controls import untrained_encoder_matrix
    from galaxy_jepa.probing.extract import extract_matrix
    ids, _, _ = sample()
    st = A.R.prepare("runs/m_v2/encoder.pt", A.R.MAX_TRAIN, label="C", sources=1)
    fz = A.load_frozen_encoder(st.ckpt)
    ds = type(st.ds)(st.ds.cache, st.rows, [int(i) for i in ids]) if hasattr(st.ds, "cache") else None
    if ds is None:
        import j4_spread_controls as J
        ds = type(st.ds)(J.check(verbose=False)[1], st.rows, [int(i) for i in ids])
    out = {"ids": ids}
    m = extract_matrix(fz.encoder if hasattr(fz, "encoder") else fz, ds, device=st.device, batch_size=64)
    assert np.array_equal(np.asarray(m.object_ids), ids)
    out["A1"] = m.x.astype(np.float32)
    for s in range(3):
        m = untrained_encoder_matrix(fz.config, ds, device=st.device, seed=s, batch_size=64)
        assert np.array_equal(np.asarray(m.object_ids), ids)
        out[f"s{s}"] = m.x.astype(np.float32)
    np.savez(OUT / "untrained.npz", **out)
    print("embedded", {k: v.shape for k, v in out.items()})


def km(Z, k, seed=SEED, n_init=20):
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=n_init, random_state=seed).fit(Z)


def pick_k(Z):
    from sklearn.metrics import silhouette_score
    fits = {k: km(Z, k).labels_ for k in KS}
    sil = {k: float(silhouette_score(Z, l)) for k, l in fits.items()}
    return max(sil, key=sil.get), sil, fits


def step_run():
    from sklearn.metrics import adjusted_rand_score as ari
    from threadpoolctl import threadpool_limits
    ids, ctrl, srec = sample()
    df, F = table()
    sid, cid = [str(i) for i in ids], [str(i) for i in ctrl]
    d, fl = df.loc[sid], F.loc[sid]
    g = groups(d, fl)
    rate = fl.mean()
    use = [n for n in rate.index if 0.05 <= rate[n] <= 0.95]
    parts = {"pattern": g, **{n: fl[n].to_numpy().astype(int) for n in use}}
    emb, ok, chk, N, nuis = footing(ids)
    rng = np.random.default_rng(SEED)
    res = {"sample": srec, "checks": chk, "nuisances_ok": int(ok.sum()), "groups": {k: int((g == k).sum()) for k in np.unique(g)},
           "flag_rates": {n: {"star": float(rate[n]), "control": float(F.loc[cid][n].mean())} for n in F if max(rate[n], F.loc[cid][n].mean()) >= 0.01},
           "types": {"star": d.type.value_counts().to_dict(), "control": df.loc[cid].type.value_counts().to_dict()},
           "type_r": d.type_r.value_counts().to_dict(), "spec_class": d.spec_class.fillna("none").value_counts().to_dict(),
           "flags_used_single": use, "embeddings": {}}
    lab = {}
    with threadpool_limits(3):
        for space in ("raw", "cleaned"):
            o = ok if space == "cleaned" else np.ones(len(ids), bool)
            block = {}
            for n in ("A1", "untrained-s0", "untrained-s1", "untrained-s2"):
                Z = emb[space, n][o]
                k, sil, fits = pick_k(Z)
                L = fits[k]
                lab[space, n] = (o, L)
                block[n] = {"k": k, "silhouette": sil, "sizes": np.bincount(L).tolist(),
                            "ari": {p: ari(v[o], L) for p, v in parts.items()},
                            "ari_pattern_by_k": {kk: ari(g[o], l) for kk, l in fits.items()}}
            L = lab[space, "A1"][1]
            perm = [ari(g[o], rng.permutation(L)) for _ in range(N_PERM)]
            subs = {n: [] for n in block}
            for b in range(N_SUB):  # re-cluster on 80% subsamples at each encoder's own k (chosen on the full sample)
                s = rng.choice(o.sum(), int(SUB_FRAC * o.sum()), replace=False)
                for n in block:
                    subs[n].append(ari(g[o][s], km(emb[space, n][o][s], block[n]["k"], seed=b, n_init=5).labels_))
            best = max((n for n in block if n != "A1"), key=lambda n: block[n]["ari"]["pattern"])
            diff = np.array(subs["A1"]) - np.array(subs[best])
            a1, un = block["A1"]["ari"]["pattern"], [block[n]["ari"]["pattern"] for n in block if n != "A1"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            block["comparison"] = {"a1_ari": a1, "untrained_ari": un, "best_seed": best, "perm_97.5": float(np.percentile(perm, 97.5)),
                                   "sub_diff_mean": float(diff.mean()), "sub_diff_ci": [float(lo), float(hi)],
                                   "sub_ari_median": {n: float(np.median(v)) for n, v in subs.items()},
                                   "clearly_better": bool(a1 > max(un) and lo > 0 and a1 > np.percentile(perm, 97.5))}
            cl = []
            for c in range(block["A1"]["k"]):
                m = np.flatnonzero(o)[L == c]
                cl.append({"n": len(m), "flag_rates": {x: float(fl[x].iloc[m].mean()) for x in use}, "patterns": {p: int((g[m] == p).sum()) for p in np.unique(g)},
                           "spec_class": d.spec_class.iloc[m].fillna("none").value_counts().to_dict(), "type": d.type.iloc[m].value_counts().to_dict(),
                           "median_modelMag_r": float(d.modelMag_r.iloc[m].median()), "median_petroR50_r": float(d.petroR50_r.iloc[m].median()),
                           "median_psf_minus_model": float((d.psfMag_r - d.modelMag_r).iloc[m].median()),
                           "nuisance_medians": dict(zip(nuis, np.nanmedian(N[m], 0).round(3).tolist()))})
            block["A1_clusters"] = cl
            res["embeddings"][space] = block
            print(space, json.dumps(block["comparison"], indent=1), {n: (block[n]["k"], round(block[n]["silhouette"][block[n]["k"]], 3)) for n in subs})
    np.savez(OUT / "clusters.npz", ids=ids, groups=g, **{f"{sp}_{n}_{w}": v for (sp, n), t in lab.items() for w, v in zip(("rows", "labels"), t)})
    (DOC / "c_star.json").write_text(json.dumps(res, indent=1, default=float))


def step_sheet():
    from PIL import Image, ImageDraw, ImageFont
    import j4_spread_controls as J
    _, cache = J.check(verbose=False)
    z = np.load(OUT / "clusters.npz", allow_pickle=True)
    res = json.loads((DOC / "c_star.json").read_text())
    f = lambda n: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", n)  # noqa: E731
    T, HEAD, G = 96, 40, 14
    for space in ("raw", "cleaned"):
        rows, L, cl = z[f"{space}_A1_rows"], z[f"{space}_A1_labels"], res["embeddings"][space]["A1_clusters"]
        idx = np.flatnonzero(rows)
        rng = np.random.default_rng(SEED)
        H = len(cl) * (2 * T + HEAD + G) + G + 22
        im = Image.new("RGB", (8 * T + 2 * G, H), (250, 249, 246))
        dr = ImageDraw.Draw(im)
        dr.text((G, 8), f"star/artifact fraction >= {res['sample']['threshold']}: real galaxies, A1 {space}, 16 drawn at random per cluster", fill=(29, 27, 25), font=f(12))
        for c, r in enumerate(cl):
            y0 = 28 + c * (2 * T + HEAD + G)
            fr = r["flag_rates"]
            dr.text((G, y0), f"cluster {c + 1}: n = {r['n']}", fill=(29, 27, 25), font=f(13))
            dr.text((G, y0 + 18), "   ".join(f"{k.lower()} {fr[k]:.0%}" for k in ("SATURATED", "EDGE", "BLENDED", "SUBTRACTED", "NOTCHECKED") if k in fr) + f"   mag r {r['median_modelMag_r']:.1f}",
                    fill=(90, 85, 80), font=f(11))
            m = idx[L == c]
            for j, i in enumerate(rng.choice(m, min(16, len(m)), replace=False)):
                x = np.asarray(cache.get(int(z["ids"][i])), dtype=np.float32)
                rgb = np.stack([x[2], x[1], x[0]], -1)
                lo, hi = np.percentile(rgb, [1, 99.7])
                a = (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
                im.paste(Image.fromarray(a).resize((T, T), Image.LANCZOS), (G + (j % 8) * T, y0 + HEAD - 4 + (j // 8) * T))
        im.save(DOC / f"c_sheet_{space}.png", optimize=True)


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
