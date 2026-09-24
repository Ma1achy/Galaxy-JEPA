"""Brief Y (revised) — machine-measured pitch angle from public catalogues.

  --y1        acquire-and-verify manifest (MD5, rows, columns, join counts) → out/y1_manifest.json
  --planted2  D28 checks through Y2's comparison code → out/y2_planted.json
  --y2        provenance of the Hayes table → out/y2_provenance.json
  --bank      embed the replication galaxies outside P2's union (M + 3 untrained)
  --planted3  D28 checks through Y3's code → out/y3_planted.json
  --y3        the science → out/y3_pitch.json

Sources (all in out/ext/, gitignored):
- Hayes SpArcFiRe table `hayes_SF5-CS.5+axisRatio.5.tsv` (ics.uci.edu/~wayne/research/students/);
  `name` is the DR8+ objID, so it joins on our object_id (the brief said dr7objid; checked: 0).
- Galaxy PAnDa v1.0.1 (Zenodo 10.5281/zenodo.19704211, CC BY 4.0): Hart et al. 2017 machine
  (SpArcFiRe, r, SDSS) and Yu & Ho 2020 (2DFFT, R, SDSS) rows, matched on position within 3″.
- Shamir's SpArcFiRe chirality (McAdam & Shamir 2023), non-mirrored and mirrored runs, on dr7objid.
- GZ1 table 2 (X1b), on dr7objid.
Every ID is read as a string: int(float(id)) rounds an 18-digit objID (Brief X).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402
import v2_independent as V2  # noqa: E402
import w2_name_pcs as W  # noqa: E402
import x1_handedness as X  # noqa: E402

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402

EXT = R.OUT / "ext"
HAYES = EXT / "hayes_SF5-CS.5+axisRatio.5.tsv"
PANDA = EXT / "panda_galaxies_all_v1.0.1.csv"
SHAMIR = {"non_mirror": EXT / "shamir_small_non_mirror.csv",
          "mirror": EXT / "shamir_small_mirror.csv"}
PRIMARY = "pa_alenWtd_avg__abs"  # pre-registered; the rest are sensitivity only
RECORDED = PRIMARY  # the column Y2/Y3 were run on; `--dco` switches PRIMARY to "dco_abs" (Brief BB)
DCO = "pa_alenWtd_avg_domChiralityOnly"
AGREE_ONLY = False  # `--agree`: Hayes pitch kept only where top2_chirality_agreement is 'agree'


def switches(argv: list[str], path: Path) -> Path:
    """Brief BB estimator / reliability switches; returns the output path suffixed to match."""
    global PRIMARY, AGREE_ONLY
    tag = ""
    if "--dco" in argv:
        PRIMARY, tag = "dco_abs", tag + "_dco"
    if "--agree" in argv:
        AGREE_ONLY, tag = True, tag + "_agree"
    return path.with_name(path.stem + tag + path.suffix) if tag else path
SENSITIVITY = ("pa_avg__abs", "pa_alenWtd_median", "pa_longest", "pa_alenWtd_avg_domChiralityOnly")
VIS_COLS = ("modelMag_r", "snr_r", "petroRad_r", "specz")  # V1's composite; magnitude first
MIN_N = 100
N_CHI = 3000
SEED = 0
REFS = {"hart": "Hart et al 2017", "yuho": "Si-Yue Yu and Luis C. Ho 2020"}
# Y1 joined `name` (DR8+) to object_id: 37,381. Z2: `OBJID` (DR7) to dr7objid recovers 9,501 more,
# the same galaxies detected in an overlapping run (46,882). Set to "dr7objid" to use it.
HAYES_KEY = "object_id"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ the joined table


def partitions(setup) -> dict[int, str]:
    """P2's three-way split over the whole probe corpus, and which rows P2's union embedded."""
    cache = setup.ds.cache
    probe_ids = cache.present(sorted(int(o) for o in setup.rows))
    split = assign_three_way(probe_ids, seed=setup.cfg.seed, ratios=setup.cfg.ratios)
    part = {int(o): "train" for o in split.train}
    part.update({int(o): "val" for o in split.val})
    part.update({int(o): "test" for o in split.test})
    for o in setup.train_ids:
        part[int(o)] = "A"
    for o in setup.test_ids:
        part[int(o)] = "B"
    return part


def panda(which: str) -> pd.DataFrame:
    p = pd.read_csv(PANDA, low_memory=False)
    s = p[p.Reference.str.startswith(REFS[which], na=False)]
    return s[~s.Galaxy_name.duplicated(keep=False)]  # Yu & Ho: two galaxies listed twice, drop both


def table(setup) -> pd.DataFrame:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    t = pd.read_csv(Path(setup.cfg.paths.probe_dir) / "metadata.csv", dtype={"object_id": str,
                                                                               "dr7objid": str},
                    low_memory=False)
    part = partitions(setup)
    t["part"] = [part.get(int(o), "absent") for o in t.object_id]
    h = pd.read_csv(HAYES, sep="\t", dtype={"name": str, "OBJID": str}, low_memory=False)
    if HAYES_KEY == "dr7objid":
        h = h.drop(columns="name").rename(columns={"OBJID": "name"})
    keep = ["name", "P_CS", "diskAxisRatio", RECORDED, *SENSITIVITY, "chirality_alenWtd",
            "chirality_maj", "top2_chirality_agreement", "totalNumArcs", "alenAt50pct",
            *[c for c in h.columns if c.startswith("numDcoArcsGE")]]
    h = h[keep].rename(columns={c: f"H_{c}" for c in keep if c != "name"})
    t = t.merge(h, left_on=HAYES_KEY, right_on="name", how="left").drop(columns="name")
    # |DCO|: every retained arc shares the dominant sign, so the magnitude is the DCO pitch
    t["H_dco_abs"] = pd.to_numeric(t[f"H_{DCO}"], errors="coerce").abs()
    if AGREE_ONLY:
        bad = t.H_top2_chirality_agreement.fillna("").str.strip("'") != "agree"
        for c in ("H_dco_abs", f"H_{RECORDED}", *[f"H_{e}" for e in SENSITIVITY]):
            t[c] = pd.to_numeric(t[c], errors="coerce").where(~bad)
    cat = SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg)
    for which in REFS:
        s = panda(which)
        c = SkyCoord(s.RA_deg.values * u.deg, s.Dec_deg.values * u.deg)
        i, d2, _ = c.match_to_catalog_sky(cat)
        ok = d2.arcsec < 3
        col = np.full(len(t), np.nan)
        col[i[ok]] = s.PA_degrees.values[ok]
        t[f"{which}_pa"] = col
    for run, path in SHAMIR.items():
        s = pd.read_csv(path, dtype=str, skipinitialspace=True)
        s.columns = [c.strip() for c in s.columns]
        lab = dict(zip(s["name"].str.strip(), s["chirality_alenWtd"].str.strip(), strict=True))
        t[f"shamir_{run}"] = [lab.get(o) for o in t.dr7objid.fillna("")]
    g: dict[str, tuple[float, float, float]] = {}
    with gzip.open(X.GZ1, "rt") as fh:
        for r in csv.DictReader(fh):
            g[r["OBJID"]] = (float(r["P_CW"]), float(r["P_ACW"]), float(r["NVOTE"]))
    gz = [g.get(o, (np.nan,) * 3) for o in t.dr7objid.fillna("")]
    t["gz1_h"] = [a - b for a, b, _ in gz]
    t["gz1_n"] = [n for *_, n in gz]
    return t


def sz(v) -> float:
    """+1 S-wise, −1 Z-wise, NaN otherwise (EQ, missing)."""
    if not isinstance(v, str):
        return np.nan
    v = v.replace("-", "").lower()
    return 1.0 if v == "swise" else -1.0 if v == "zwise" else np.nan


def visibility(t: pd.DataFrame) -> np.ndarray:
    """V1's composite on the given rows: PC1 of the rank-z nuisances, + = fainter."""
    import v1_loose_ends as V1

    vis = t[list(VIS_COLS)].to_numpy(float)
    return V1.composite(vis)[0]


# ------------------------------------------------------------------ Y2


def pitch_compare(a: np.ndarray, b: np.ndarray, vis: np.ndarray, seed: int) -> dict:
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(vis)
    a, b, vis = a[ok], b[ok], vis[ok]
    out = {"n": int(ok.sum())}
    if ok.sum() < MIN_N:
        return out
    out["rho"], out["p"] = W.spearman_perm(a, b, seed=seed)
    out["partial"], out["p_partial"] = U2.partial_perm(a, b, vis[:, None], seed=seed + 1)
    out["ci"] = V2.boot_ci(a, b, seed)
    out["median_abs_diff_deg"] = float(np.median(np.abs(a - b)))
    out["median_offset_deg"] = float(np.median(a - b))
    return out


def pitch_state(c: dict, sig: bool) -> str:
    """First match applies (D27)."""
    if c["n"] < MIN_N:
        return "INSUFFICIENT"
    if not sig or c["rho"] <= 0:
        return "BROKEN"
    if c["rho"] >= 0.5 and c["partial"] >= 0.5:
        return "CONSISTENT"
    return "DIVERGENT"


def label_compare(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Two ±1 handedness labels: agreement fraction, and its permutation p against 0.5."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": int(ok.sum())}
    if ok.sum() < MIN_N:
        return out
    agree = float(np.mean(a == b))
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(W.N_PERM):
        hits += abs(np.mean(a == rng.permutation(b)) - 0.5) >= abs(agree - 0.5) - 1e-12
    out.update(agree=agree, p=(1 + hits) / (1 + W.N_PERM))
    return out


def chi_compare(label: np.ndarray, chi: np.ndarray, seed: int) -> dict:
    """χ against a ±1 label: AUC of χ for S-wise (+1) vs Z-wise (−1), permutation p."""
    ok = np.isfinite(label) & np.isfinite(chi)
    y, s = label[ok] > 0, chi[ok]
    out = {"n": int(ok.sum())}
    if ok.sum() < MIN_N or y.all() or (~y).all():
        return out
    out["auc"] = X._auc(y, s)
    rho, p = W.spearman_perm(s, y.astype(float), seed=seed)
    out["p"] = p
    return out


def hand_state(c: dict, sig: bool, kind: str) -> str:
    """Up to a parity flip: the sign of the agreement is reported, not graded (D27)."""
    if c["n"] < MIN_N:
        return "INSUFFICIENT"
    d = abs((c["agree"] if kind == "label" else c["auc"]) - 0.5)
    if not sig or d < 0.1:
        return "BROKEN"
    return "CONSISTENT" if d >= 0.3 else "DIVERGENT"


def chi_values(setup, ids: list[int]) -> np.ndarray:
    ds = StampDataset(setup.ds.cache, setup.rows, ids)
    idx = np.arange(len(ids))
    return extract_matrix(X.pixel_encoder(), X.Transformed(ds, idx, "orig"), device="cpu").x[:, 0]


def y2(setup, t: pd.DataFrame) -> dict:
    rec: dict = {}
    P = t[f"H_{PRIMARY}"].to_numpy(float)
    vis_all = np.full(len(t), np.nan)
    comps, sig_p = {}, {}
    for key, ref in (("hayes_vs_hart", "hart_pa"), ("hayes_vs_yuho", "yuho_pa")):
        rows = np.flatnonzero(np.isfinite(P) & np.isfinite(t[ref].to_numpy(float)))
        sub = t.iloc[rows]
        v = visibility(sub)
        vis_all[rows] = v
        comps[key] = pitch_compare(P[rows], sub[ref].to_numpy(float), v, seed=len(key))
        comps[key]["sensitivity"] = {
            e: V2.spearman(np.abs(sub[f"H_{e}"].to_numpy(float)), sub[ref].to_numpy(float))
            for e in SENSITIVITY}
        if "p" in comps[key]:
            sig_p[key] = comps[key]["p"]
    rows = np.flatnonzero(np.isfinite(t.hart_pa) & np.isfinite(t.yuho_pa))
    sub = t.iloc[rows]
    rec["hart_vs_yuho_reference"] = {"n": int(rows.size),
                                     "rho": V2.spearman(sub.hart_pa, sub.yuho_pa)
                                     if rows.size >= 10 else None}

    # handedness: Hayes, Shamir (both runs), χ on our stamps, GZ1
    hs = np.array([sz(v) for v in t.H_chirality_alenWtd])
    sn = np.array([sz(v) for v in t.shamir_non_mirror])
    sm = np.array([sz(v) for v in t.shamir_mirror])
    lab = {"hayes_vs_shamir_non_mirror": label_compare(hs, sn, 11),
           "hayes_vs_shamir_mirror": label_compare(hs, sm, 12),
           "shamir_non_mirror_vs_mirror": label_compare(sn, sm, 13)}
    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(np.isfinite(hs) & (t.part != "absent").to_numpy())
    pick = np.sort(rng.choice(pool, min(N_CHI, pool.size), replace=False))
    chi = np.full(len(t), np.nan)
    chi[pick] = chi_values(setup, [int(o) for o in t.object_id.values[pick]])
    chis = {"chi_vs_hayes": chi_compare(hs, chi, 21), "chi_vs_shamir_non_mirror":
            chi_compare(sn, chi, 22), "chi_vs_shamir_mirror": chi_compare(sm, chi, 23)}
    gz = t.gz1_h.to_numpy(float)
    gz_ok = np.isfinite(gz) & (t.gz1_n.to_numpy(float) >= 10) & (np.abs(gz) > 0)
    gzs = np.where(gz_ok, -np.sign(gz), np.nan)  # CW (h>0) ↔ clockwise-outward in our arrays (X1)
    lab["hayes_vs_gz1"] = label_compare(hs, gzs, 14)
    for k, c in {**lab, **chis}.items():
        if "p" in c:
            sig_p[k] = c["p"]
    fam = len(sig_p)
    nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=fam)
    sig = nulls_mod.family_significant(sig_p, alpha=W.ALPHA, method="benjamini_yekutieli",
                                       n_tests=fam)
    for k, c in comps.items():
        c["state"] = pitch_state(c, sig.get(k, False))
    for k, c in lab.items():
        c["state"] = hand_state(c, sig.get(k, False), "label")
    for k, c in chis.items():
        c["state"] = hand_state(c, sig.get(k, False), "chi")
    rec.update(pitch=comps, labels=lab, chi=chis, family=fam)
    rec["parity"] = parity(lab, chis)
    rec["localise"] = localise(t, P)
    rec["selection"] = selection(t)
    return rec


def parity(lab: dict, chis: dict) -> dict:
    """The Hayes run's image parity, read from which Shamir run it agrees with.

    Shamir's non-mirrored run used the SDSS JPEG cutouts (JPEG parity); our arrays mirror those
    (X1). Hayes agreeing with the non-mirrored run → JPEG parity; with the mirrored run → our
    array parity. The triangle closes if χ's sign against Hayes equals χ's sign against the run
    Hayes agrees with.
    """
    a, b = lab["hayes_vs_shamir_non_mirror"], lab["hayes_vs_shamir_mirror"]
    if "agree" not in a or "agree" not in b:
        return {"verdict": "UNDETERMINED"}
    ok = a["state"] == "CONSISTENT" or b["state"] == "CONSISTENT"
    if not ok:
        verdict = "UNDETERMINED"
    elif a["agree"] > b["agree"]:
        verdict = "JPEG PARITY"
    else:
        verdict = "ARRAY PARITY (mirrors the JPEGs)"
    ref = "chi_vs_shamir_non_mirror" if a["agree"] > b["agree"] else "chi_vs_shamir_mirror"
    closes = None
    if "auc" in chis["chi_vs_hayes"] and "auc" in chis[ref]:
        closes = bool(np.sign(chis["chi_vs_hayes"]["auc"] - 0.5)
                      == np.sign(chis[ref]["auc"] - 0.5))
    return {"verdict": verdict, "triangle_closes": closes}


def localise(t: pd.DataFrame, P: np.ndarray) -> dict:
    """Where Hayes and Hart disagree (descriptive): |ΔP| against the table's own arc stats."""
    ok = np.isfinite(P) & np.isfinite(t.hart_pa.to_numpy(float))
    d = np.abs(P[ok] - t.hart_pa.to_numpy(float)[ok])
    sub = t[ok]
    out = {k: V2.spearman(d, sub[f"H_{k}"].to_numpy(float)) for k in ("totalNumArcs",
                                                                      "alenAt50pct")}
    agree = sub.H_top2_chirality_agreement.str.strip("'").to_numpy()
    out["median_abs_diff_by_top2"] = {k: float(np.median(d[agree == k])) for k in np.unique(agree)
                                      if (agree == k).sum() >= 30}
    return out


def selection(t: pd.DataFrame) -> dict:
    """Matched Hayes sample against all our spirals: visibility, redshift, size (descriptive)."""
    f = lambda c: pd.to_numeric(t[c], errors="coerce").to_numpy(float)  # noqa: E731
    spir = ((f("t01_smooth_or_features_a02_features_or_disk_fraction") >= 0.5)
            & (f("t02_edgeon_a05_no_fraction") >= 0.5) & (f("t04_spiral_a08_spiral_fraction") >= 0.5))
    hay = np.isfinite(f(f"H_{PRIMARY}"))
    rows = spir | hay
    vis = np.full(len(t), np.nan)
    vis[rows] = visibility(t[rows])
    out = {"n_spirals": int(spir.sum()), "n_hayes": int(hay.sum()),
           "n_hayes_in_spirals": int((hay & spir).sum())}
    for name, v in (("visibility", vis), ("specz", f("specz")), ("petroRad_r", f("petroRad_r")),
                    ("expAB_r", f("expAB_r")), ("t04_spiral", f("t04_spiral_a08_spiral_fraction"))):
        a, b = v[hay & np.isfinite(v)], v[spir & np.isfinite(v)]
        out[name] = {"median_hayes": float(np.median(a)), "median_spirals": float(np.median(b)),
                     "cliffs_delta": cliffs(a, b)}
    return out


def cliffs(a: np.ndarray, b: np.ndarray) -> float:
    r = rankdata(np.r_[a, b])
    u = r[: a.size].sum() - a.size * (a.size + 1) / 2
    return float(2 * u / (a.size * b.size) - 1)


# ------------------------------------------------------------------ Y2 planted (D28)


def planted2(t: pd.DataFrame) -> dict:
    """Each state reached through the identical compare/state code and BY at Y2's family (9)."""
    rng = np.random.default_rng(5)
    rows = np.flatnonzero(np.isfinite(t.hart_pa.to_numpy(float)))
    sub = t.iloc[rows]
    hart = sub.hart_pa.to_numpy(float)
    v = visibility(sub)
    out, p = {}, {}
    for name, noise in (("pitch_strong", 0.3), ("pitch_moderate", 2.5), ("pitch_null", None)):
        a = rng.permutation(hart) if noise is None else hart + noise * hart.std() * rng.normal(
            size=hart.size)
        out[name] = pitch_compare(a, hart, v, seed=31)
    s = np.where(rng.random(4000) < 0.5, 1.0, -1.0)
    for name, flip in (("labels_10pct", 0.1), ("labels_30pct", 0.3), ("labels_random", 0.5)):
        out[name] = label_compare(s, np.where(rng.random(s.size) < flip, -s, s), 33)
    for name, gap in (("chi_strong", 3.0), ("chi_weak", 0.5), ("chi_null", 0.0)):
        out[name] = chi_compare(s, gap * s + rng.normal(size=s.size), 35)
    p = {k: c["p"] for k, c in out.items()}
    sig = nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli", n_tests=9)
    for k, c in out.items():
        c["state"] = (pitch_state(c, sig[k]) if k.startswith("pitch") else
                      hand_state(c, sig[k], "label" if k.startswith("labels") else "chi"))
    return out


# ------------------------------------------------------------------ modes


def y1(setup, t: pd.DataFrame) -> dict:
    files = {"hayes": HAYES, "panda": PANDA, **{f"shamir_{k}": v for k, v in SHAMIR.items()},
             "gz1_table2": X.GZ1}
    rec: dict = {"files": {}}
    for k, p in files.items():
        if p.suffix == ".gz":
            with gzip.open(p, "rt") as fh:
                head = fh.readline().strip().split(",")
                n = sum(1 for _ in fh)
        else:
            sep = "\t" if p.suffix == ".tsv" else ","
            with open(p) as fh:
                head = [c.strip() for c in fh.readline().strip().split(sep)]
                n = sum(1 for _ in fh)
        rec["files"][k] = {"path": str(p.relative_to(R.OUT)), "md5": md5(p), "rows": n,
                           "n_columns": len([c for c in head if c]), "columns": head[:40]}
    embedded = t.part.isin(["A", "B"])
    for k, col in (("hayes", f"H_{PRIMARY}"), ("hart", "hart_pa"), ("yuho", "yuho_pa")):
        has = t[col].notna()
        rec[k] = {"in_230k": int(has.sum()), "in_union": int((has & embedded).sum()),
                  "A": int((has & (t.part == "A")).sum()), "B": int((has & (t.part == "B")).sum()),
                  "train_not_union": int((has & (t.part == "train")).sum()),
                  "val": int((has & (t.part == "val")).sum())}
    rec["shamir_matched"] = int(t.shamir_non_mirror.notna().sum())
    rec["gz1_matched"] = int(t.gz1_h.notna().sum())
    rec["panda_subsets"] = {k: {"rows_after_dedup": int(len(panda(k))),
                                "method": sorted(panda(k).Method.unique().tolist())}
                            for k in REFS}
    return rec


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--y1"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="Y", sources=1)
    if "--z2" in sys.argv:
        global HAYES_KEY
        HAYES_KEY = "dr7objid"
    if mode == "--y2" and any(a in sys.argv for a in ("--dco", "--agree", "--z2")):
        path = switches(sys.argv, R.OUT / ("y2_provenance_z2.json" if "--z2" in sys.argv
                                           else "y2_provenance.json"))  # before the table: it masks
        t = table(setup)
        out = y2(setup, t)
        out["switches"] = {"primary": PRIMARY, "agree_only": AGREE_ONLY, "join": HAYES_KEY}
        path.write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out["pitch"], indent=1, default=float)[:3000], file=sys.stderr)
        return
    t = table(setup)
    if mode == "--y1":
        out, path = y1(setup, t), R.OUT / "y1_manifest.json"
    elif mode == "--planted2":
        out, path = planted2(t), R.OUT / "y2_planted.json"
    elif mode == "--y2":
        out, path = y2(setup, t), R.OUT / "y2_provenance.json"
    else:
        raise SystemExit(f"Y: unknown mode {mode}")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000], file=sys.stderr)


if __name__ == "__main__":
    main()
