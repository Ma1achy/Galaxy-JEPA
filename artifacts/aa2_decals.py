"""Brief AA2 — Galaxy Zoo DECaLS volunteer coverage of our 230k (coverage and mapping only).

Volunteer votes only (Walmsley et al. 2022, Zenodo 4573248: GZD-1/2, GZD-5; Walmsley et al.
2023, Zenodo 8360385: GZD-8 core). Never the Zoobot/DESI predictions — they are trained on votes
and would make the referee circular. Definitions and states: `aa_findings.md` §AA2.

  --planted   D28 — shifted-position chance-match rate; simulated power of the MDD formulae
  --aa2       the counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import y_pitch as Y  # noqa: E402

EXT = R.OUT / "ext"
CAMPAIGNS = {"gzd12": EXT / "gz_decals_volunteers_1_and_2.parquet",
             "gzd5": EXT / "gz_decals_volunteers_5.parquet",
             "gzd8": EXT / "gz_desi_gzd8_volunteer_core_catalog.parquet"}
RADIUS = 3.0          # arcsec, as the PAnDa match (Y1)
SDSS_REACH = 21       # U3.FLOOR
DECALS_REACH = 10     # a plurality of 0.8 means ≥ 8 agreeing volunteers
UNCERTAIN = 0.6       # plurality below this: no clear majority
CONFIDENT = 0.8
OUT = R.OUT / "aa2_decals.json"
PLANTED = R.OUT / "aa2_planted.json"

# Each question is a list of comparable categories; each category is a list of source answers.
# GZ2 columns are t.._a.._count; DECaLS columns are {question}_{answer} counts.
G = "t{}_count"
MAP = {
    "smooth": {"gz2": [["01_smooth_or_features_a01_smooth"], ["01_smooth_or_features_a02_features_or_disk"],
                       ["01_smooth_or_features_a03_star_or_artifact"]],
               "decals": {"*": [["smooth-or-featured_smooth"], ["smooth-or-featured_featured-or-disk"],
                                ["smooth-or-featured_artifact"]]},
               "flag": "GZ2 'star or artifact' vs DECaLS 'artifact' (stars are pre-excluded in DECaLS)"},
    "edge_on": {"gz2": [["02_edgeon_a04_yes"], ["02_edgeon_a05_no"]],
                "decals": {"*": [["disk-edge-on_yes"], ["disk-edge-on_no"]]}, "flag": ""},
    "bar": {"gz2": [["03_bar_a06_bar"], ["03_bar_a07_no_bar"]],
            "decals": {"gzd12": [["bar_yes"], ["bar_no"]],
                       "*": [["bar_strong", "bar_weak"], ["bar_no"]]},
            "flag": "GZD-5/8 split strong/weak (weak added to catch bars GZ2 volunteers call 'no'); "
                    "mapped bar = strong + weak"},
    "spiral": {"gz2": [["04_spiral_a08_spiral"], ["04_spiral_a09_no_spiral"]],
               "decals": {"*": [["has-spiral-arms_yes"], ["has-spiral-arms_no"]]}, "flag": ""},
    "bulge": {"gz2": [["05_bulge_prominence_a10_no_bulge", "05_bulge_prominence_a11_just_noticeable"],
                      ["05_bulge_prominence_a12_obvious"], ["05_bulge_prominence_a13_dominant"]],
              "decals": {"gzd12": [["bulge-size_none"], ["bulge-size_obvious"], ["bulge-size_dominant"]],
                         "*": [["bulge-size_none", "bulge-size_small"],
                               ["bulge-size_moderate", "bulge-size_large"], ["bulge-size_dominant"]]},
              "flag": "NOT EQUIVALENT: GZ2 4 options, GZD-1/2 3, GZD-5/8 5; collapsed to "
                      "none-or-small / obvious / dominant — GZ2 'just noticeable' → none-or-small"},
    "winding": {"gz2": [["10_arms_winding_a28_tight"], ["10_arms_winding_a29_medium"],
                        ["10_arms_winding_a30_loose"]],
                "decals": {"*": [["spiral-winding_tight"], ["spiral-winding_medium"],
                                 ["spiral-winding_loose"]]}, "flag": ""},
    "arm_count": {"gz2": [["11_arms_number_a31_1"], ["11_arms_number_a32_2"], ["11_arms_number_a33_3"],
                          ["11_arms_number_a34_4"], ["11_arms_number_a36_more_than_4"]],
                  "decals": {"*": [["spiral-arm-count_1"], ["spiral-arm-count_2"], ["spiral-arm-count_3"],
                                   ["spiral-arm-count_4"], ["spiral-arm-count_more-than-4"]]},
                  "flag": "GZD-1/2 has no 'can't tell'; compared on the five counts with 'can't tell' "
                          "dropped from GZ2, GZD-5 and GZD-8 alike"},
}


def ours(setup) -> pd.DataFrame:
    cols = ["object_id", "ra", "dec", *{G.format(a) for q in MAP.values() for c in q["gz2"] for a in c},
            *Y.VIS_COLS]
    t = pd.read_csv(Path(setup.cfg.paths.probe_dir) / "metadata.csv", usecols=cols,
                    dtype={"object_id": str}, low_memory=False)
    part = Y.partitions(setup)
    t["part"] = [part.get(int(o), "absent") for o in t.object_id]
    return t


def crossmatch(t: pd.DataFrame, d: pd.DataFrame, shift_arcsec: float = 0.0) -> pd.DataFrame:
    """Nearest of ours for each DECaLS row within RADIUS; one DECaLS row per galaxy of ours."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(d.ra.values * u.deg, (d.dec.values + shift_arcsec / 3600) * u.deg)
    i, d2, _ = c.match_to_catalog_sky(SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg))
    m = pd.DataFrame({"row": i, "sep": d2.arcsec, "src": np.arange(len(d))})
    m = m[m.sep < RADIUS].sort_values("sep").drop_duplicates("row")
    return m


def fractions(counts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(category fractions, total, plurality fraction) over the mapped categories."""
    c = np.column_stack(counts).astype(float)
    tot = c.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = c / tot[:, None]
    return f, tot, np.nanmax(np.where(tot[:, None] > 0, f, np.nan), 1)


def side(df: pd.DataFrame, cats: list[list[str]], fmt=str) -> tuple:
    return fractions([sum(pd.to_numeric(df[fmt(a)], errors="coerce").fillna(0).to_numpy()
                          for a in cat) for cat in cats])


def auc_se(auc: float, n1: int, n0: int) -> float:
    """Hanley & McNeil (1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2))
                         / (n1 * n0)))


def mdd_auc(n1: int, n0: int, auc: float = 0.75, r: float = 0.5) -> float:
    """Smallest paired AUC difference detectable at 80% power, two-sided α 0.05: 2.8·SE_diff,
    SE_diff = SE·√(2(1−r)) for two scores whose AUC errors correlate at r."""
    if min(n1, n0) < 10:
        return float("inf")
    return 2.8 * auc_se(auc, n1, n0) * np.sqrt(2 * (1 - r))


def mde_rho(n: int) -> float:
    return float("inf") if n < 30 else 2.8 / np.sqrt(n - 3)


def power_state(mdd: float, powered: float, possible: float) -> str:
    if mdd <= powered:
        return "POWERED"
    if mdd <= possible:
        return "UNDERPOWERED"
    return "NOT POSSIBLE"


def per_question(t, dec: dict, matches: dict) -> dict:
    """For each galaxy of ours and each question, the campaign with most votes on that question
    is its DECaLS referee (declared)."""
    out = {}
    n = len(t)
    for q, spec in MAP.items():
        fg, tg, pg = side(t, spec["gz2"], G.format)
        best_tot = np.zeros(n)
        best_p = np.full(n, np.nan)
        best_top = np.full(n, -1)
        best_camp = np.full(n, "", dtype=object)
        top_g = np.where(tg > 0, np.nanargmax(np.nan_to_num(fg, nan=-1), 1), -1)
        per_camp = {}
        for camp, d in dec.items():
            m = matches[camp]
            cats = spec["decals"].get(camp, spec["decals"]["*"])
            if not all(a in d.columns for cat in cats for a in cat):
                per_camp[camp] = "question absent"
                continue
            fd, td, pd_ = side(d.iloc[m.src.to_numpy()], cats)
            rows = m.row.to_numpy()
            per_camp[camp] = {"answered": int((td > 0).sum()), "reach_ge_10": int((td >= DECALS_REACH).sum()),
                              "median_votes": float(np.median(td[td > 0])) if (td > 0).any() else 0}
            better = td > best_tot[rows]
            r = rows[better]
            best_tot[r], best_p[r] = td[better], pd_[better]
            best_top[r] = np.nanargmax(np.nan_to_num(fd[better], nan=-1), 1)
            best_camp[r] = camp
        sd_ok = tg >= SDSS_REACH
        de_ok = best_tot >= DECALS_REACH
        unc = sd_ok & (pg < UNCERTAIN)
        conf_d = de_ok & (best_p >= CONFIDENT)
        sharp = unc & conf_d
        both = sd_ok & de_ok
        part = t.part.to_numpy()
        rec = {"flag": spec["flag"], "campaigns": per_camp,
               "overlap_answered": int((best_tot > 0).sum()),
               "both_reached": int(both.sum()),
               "sdss_uncertain": int(unc.sum()),
               "decals_confident": int(conf_d.sum()),
               "sharp": int(sharp.sum()),
               "sharp_by_part": {p: int((sharp & (part == p)).sum()) for p in ("A", "B", "train", "val", "test", "absent")},
               "sharp_by_campaign": {c: int((sharp & (best_camp == c)).sum()) for c in dec},
               "plurality_agree_when_both_confident": float(np.mean(top_g[both & (pg >= CONFIDENT) & conf_d]
                                                                   == best_top[both & (pg >= CONFIDENT) & conf_d]))
               if (both & (pg >= CONFIDENT) & conf_d).any() else None}
        # (a) probe vs SDSS voters on the sharp set, scored on B (embedded test) and on all test
        for key, sel in (("B", sharp & (part == "B")), ("B_or_test", sharp & np.isin(part, ["B", "test"]))):
            k = best_top[sel]
            if k.size:
                top = np.bincount(k[k >= 0]).argmax()
                n1, n0 = int((k == top).sum()), int((k != top).sum())
            else:
                n1 = n0 = 0
            mdd = mdd_auc(n1, n0)
            rec[f"a_{key}"] = {"n": int(sel.sum()), "n_pos": n1, "n_neg": n0, "mdd_auc": mdd,
                               "state": power_state(mdd, 0.05, 0.15)}
        rec["_arrays"] = (both, part, best_camp)
        out[q] = rec
    return out


def load_decals() -> dict[str, pd.DataFrame]:
    return {c: pd.read_parquet(p) for c, p in CAMPAIGNS.items()}


def aa2(setup) -> dict:
    t = ours(setup)
    dec = load_decals()
    matches = {c: crossmatch(t, d) for c, d in dec.items()}
    rec: dict = {"n_ours": len(t), "radius_arcsec": RADIUS, "campaigns": {}}
    any_match = np.zeros(len(t), bool)
    for c, d in dec.items():
        m = matches[c]
        any_match[m.row.to_numpy()] = True
        rec["campaigns"][c] = {"rows": len(d), "matched": len(m),
                               "sep_arcsec_q50_q95": np.percentile(m.sep, [50, 95]).tolist(),
                               "within_1": int((m.sep < 1).sum()), "within_2": int((m.sep < 2).sum())}
    rec["overlap_any_campaign"] = int(any_match.sum())
    rec["overlap_by_part"] = {p: int((any_match & (t.part == p)).sum())
                              for p in ("A", "B", "train", "val", "test", "absent")}
    sets = {c: set(matches[c].row) for c in dec}
    rec["campaign_overlaps"] = {"gzd12&gzd5": len(sets["gzd12"] & sets["gzd5"]),
                                "gzd12&gzd8": len(sets["gzd12"] & sets["gzd8"]),
                                "gzd5&gzd8": len(sets["gzd5"] & sets["gzd8"]),
                                "all_three": len(sets["gzd12"] & sets["gzd5"] & sets["gzd8"])}
    q = per_question(t, dec, matches)
    # Match sanity: smooth fraction, SDSS against DECaLS, where both reached
    fg, tg, _ = side(t, MAP["smooth"]["gz2"], G.format)
    from scipy.stats import spearmanr

    m = matches["gzd12"]
    fd, td, _ = side(dec["gzd12"].iloc[m.src.to_numpy()], MAP["smooth"]["decals"]["*"])
    ok = (tg[m.row.to_numpy()] >= SDSS_REACH) & (td >= DECALS_REACH)
    rec["match_sanity_smooth_rho"] = float(spearmanr(fg[m.row.to_numpy()][ok, 0], fd[ok, 0])[0])
    rec["match_sanity_n"] = int(ok.sum())

    # (b) winding and (c) imaging depth — rough power from the n that each test would use
    both_w, part, _ = q["winding"].pop("_arrays")
    wB = int((both_w & (part == "B")).sum())
    rec["b_winding"] = {"n_both_reached": int(both_w.sum()), "n_B": wB, "mde_rho_B": mde_rho(wB),
                        "state": power_state(mde_rho(wB), 0.10, 0.25),
                        "sharp": q["winding"]["sharp"], "sharp_B": q["winding"]["sharp_by_part"]["B"]}
    depth = {}
    for name in ("smooth", "bar", "spiral", "bulge", "arm_count", "edge_on"):
        both, part, _ = q[name].pop("_arrays")
        n = int(both.sum())
        depth[name] = {"n_both_reached": n, "mde_rho": mde_rho(n),
                       "state": power_state(mde_rho(n), 0.10, 0.25)}
    rec["c_depth"] = depth
    rec["questions"] = q
    return rec


def planted(setup) -> dict:
    """(1) Chance-match rate: shift DECaLS positions 60″ north and rematch. (2) The MDD formula's
    power: simulate paired scores with a true ΔAUC equal to the formula's MDD, test by DeLong."""
    t = ours(setup)
    dec = load_decals()
    out: dict = {"chance": {}}
    for c, d in dec.items():
        out["chance"][c] = {"shifted_matches": len(crossmatch(t, d, shift_arcsec=60.0)),
                            "rows": len(d)}
    rng = np.random.default_rng(3)
    sims = {}
    for n1, n0 in ((100, 100), (300, 700), (1000, 3000)):
        mdd = mdd_auc(n1, n0)
        rej = []
        for _ in range(300):
            y = np.r_[np.ones(n1), np.zeros(n0)]
            # two correlated scores; score b's AUC exceeds a's by ≈ mdd
            z = rng.normal(size=y.size)
            ea, eb = rng.normal(size=y.size), rng.normal(size=y.size)
            from scipy.stats import norm

            d_a = np.sqrt(2) * norm.ppf(0.75)
            d_b = np.sqrt(2) * norm.ppf(min(0.75 + mdd, 0.999))
            sa = d_a * y + (z + ea) / np.sqrt(2)
            sb = d_b * y + (z + eb) / np.sqrt(2)
            rej.append(delong_p(y, sa, sb) < 0.05)
        sims[f"{n1}/{n0}"] = {"mdd": mdd, "power": float(np.mean(rej))}
    out["mdd_power"] = sims
    rng2 = np.random.default_rng(4)
    rho_sims = {}
    for n in (100, 400, 1600):
        mde = mde_rho(n)
        from scipy.stats import spearmanr

        hits = 0
        for _ in range(300):
            x = rng2.normal(size=n)
            r = 2 * np.sin(np.pi * mde / 6)
            y = r * x + np.sqrt(1 - r**2) * rng2.normal(size=n)
            hits += spearmanr(x, y)[1] < 0.05
        rho_sims[str(n)] = {"mde": mde, "power": hits / 300}
    out["mde_power"] = rho_sims
    return out


def delong_p(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided DeLong test for two correlated AUCs."""
    from scipy.stats import norm, rankdata

    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(s):
        r_all = rankdata(s)
        r_pos, r_neg = rankdata(s[pos]), rankdata(s[neg])
        v10 = (r_all[pos] - r_pos) / n
        v01 = 1 - (r_all[neg] - r_neg) / m
        return v10.mean(), v10, v01

    ta, va10, va01 = comps(a)
    tb, vb10, vb01 = comps(b)
    s10 = np.cov(np.vstack([va10, vb10]))
    s01 = np.cov(np.vstack([va01, vb01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    return float(2 * norm.sf(abs(ta - tb) / np.sqrt(var)))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--aa2"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="AA2", sources=1)
    out, path = (planted(setup), PLANTED) if mode == "--planted" else (aa2(setup), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
