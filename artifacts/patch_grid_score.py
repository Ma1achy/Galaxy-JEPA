"""Patch-size grid 16 → 8 → 4: the scoring code for docs/patch_grid/prereg.md (DRAFT, not hashed) and its D28 plants.

    .venv/bin/python artifacts/patch_grid_score.py plants   # synthetic per-decile AUCs through the identical path; every state reached
    .venv/bin/python artifacts/patch_grid_score.py power    # power at the observed patch-16 seed spread (patch_grid_spread.py's record)

Pure functions of per-decile AUC tables; this module reads no image, embedding or checkpoint. Where the draft is ambiguous the
choice made here is a named constant/comment marked [AMBIGUOUS n], listed in docs/patch_grid/flags.md for the user.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DOC, OUT = REPO / "docs/patch_grid", REPO / "runs/almagest/patch_grid"
OBJ, PATCH = ("jepa", "moco", "mae"), (16, 8, 4)
STEPS = ((16, 8), (8, 4))
SESOI, NEGLIGIBLE, MIN_ANSWERS, LOWDEC = 0.02, 0.005, 5, 5
GROUPS = {
    "G1": ["smooth", "features", "edge-on yes", "edge-on no", "completely round", "cigar", "odd yes", "odd no", "no bulge"],
    "G2": ["bar", "no bar", "spiral", "no spiral", "ring", "lens/arc", "obvious bulge", "in between", "loose winding", "2 arms", "bulge-shape no bulge"],
    "G3": ["dominant bulge", "dust lane", "disturbed", "irregular", "other", "merger", "bulge rounded", "tight winding", "just noticeable", "can't tell (arms)"],
}
KEY = {  # group name → the repo's answer id
    "smooth": "t01_smooth_or_features_a01_smooth", "features": "t01_smooth_or_features_a02_features_or_disk", "edge-on yes": "t02_edgeon_a04_yes",
    "edge-on no": "t02_edgeon_a05_no", "completely round": "t07_rounded_a16_completely_round", "cigar": "t07_rounded_a18_cigar_shaped",
    "odd yes": "t06_odd_a14_yes", "odd no": "t06_odd_a15_no", "no bulge": "t05_bulge_prominence_a10_no_bulge", "bar": "t03_bar_a06_bar",
    "no bar": "t03_bar_a07_no_bar", "spiral": "t04_spiral_a08_spiral", "no spiral": "t04_spiral_a09_no_spiral", "ring": "t08_odd_feature_a19_ring",
    "lens/arc": "t08_odd_feature_a20_lens_or_arc", "obvious bulge": "t05_bulge_prominence_a12_obvious", "in between": "t07_rounded_a17_in_between",
    "loose winding": "t10_arms_winding_a30_loose", "2 arms": "t11_arms_number_a32_2", "bulge-shape no bulge": "t09_bulge_shape_a27_no_bulge",
    "dominant bulge": "t05_bulge_prominence_a13_dominant", "dust lane": "t08_odd_feature_a38_dust_lane", "disturbed": "t08_odd_feature_a21_disturbed",
    "irregular": "t08_odd_feature_a22_irregular", "other": "t08_odd_feature_a23_other", "merger": "t08_odd_feature_a24_merger",
    "bulge rounded": "t09_bulge_shape_a25_rounded", "tight winding": "t10_arms_winding_a28_tight", "just noticeable": "t05_bulge_prominence_a11_just_noticeable",
    "can't tell (arms)": "t11_arms_number_a37_cant_tell"}
MIN_CLASS = 10  # [AMBIGUOUS 1] a decile's AUC is defined if both classes have ≥ 10 test galaxies; L is defined iff deciles 1–5 all are


# --- L and Δ -----------------------------------------------------------------------------------------------------------
def low_margin(auc, un, counts):
    """auc (10,), un (K, 10), counts (10, 2) = (n_pos, n_neg) → L, or NaN when any of deciles 1–5 lacks MIN_CLASS of a class."""
    k = slice(0, LOWDEC)
    if (np.asarray(counts)[k] < MIN_CLASS).any() or not np.isfinite(np.asarray(auc)[k]).all() or not np.isfinite(np.asarray(un)[:, k]).all():
        return float("nan")
    return float(np.mean(np.asarray(auc)[k] - np.asarray(un)[:, k].max(0)))


def all_L(auc, un, counts):
    """auc[(o, p, s)][answer] (10,); un[(o, p)][answer] (K, 10); counts[answer] (10, 2) → {(o, p, s): {answer: L}} (missing arm = void)."""
    return {e: {a: low_margin(v, un[e[0], e[1]][a], counts[a]) for a, v in r.items()} for e, r in auc.items()}


def seeds_of(L, o, p):
    return sorted(s for (oo, pp, s) in L if (oo, pp) == (o, p))


def seed_mean(L, o, p, a):
    v = [L[o, p, s][a] for s in seeds_of(L, o, p)]
    return float(np.mean(v)) if v and np.isfinite(v).all() else float("nan")


def spread(L, o, p, answers):
    """Per-answer |L(s0) − L(s1)| of the pair; [AMBIGUOUS 2] with 3 seeds the per-answer value is the mean over the pairs."""
    ss = seeds_of(L, o, p)
    if len(ss) < 2:
        return []
    out = []
    for a in answers:
        v = [L[o, p, s][a] for s in ss]
        if np.isfinite(v).all():
            out.append(float(np.mean([abs(v[i] - v[j]) for i in range(len(v)) for j in range(i)])))
    return out


def step(L, o, p, q, answers):
    """One group, objective, step: medians, bar and state (first match)."""
    d = {a: seed_mean(L, o, q, a) - seed_mean(L, o, p, a) for a in answers}
    ok = [a for a in answers if np.isfinite(d[a])]
    # s_g: the median over the group and both patch sizes of the per-answer spreads ([AMBIGUOUS 3] pooled = one median over both lists)
    sp = spread(L, o, p, ok) + spread(L, o, q, ok)
    n = [len(seeds_of(L, o, x)) for x in (p, q)]
    rec = {"n_answers": len(ok), "n_seeds": n, "delta": {a: d[a] for a in ok}}
    if len(ok) < MIN_ANSWERS or min(n) == 0 or not sp:  # [AMBIGUOUS 4] no spread at all (a size with one seed on both sides) is INSUFFICIENT
        return {**rec, "state": "INSUFFICIENT"}
    med, s_g = float(np.median([d[a] for a in ok])), float(np.median(sp))
    bar = max(SESOI / 2, 2 * s_g)
    pos, neg = np.mean([d[a] > 0 for a in ok]), np.mean([d[a] < 0 for a in ok])
    st = ("GAIN" if med > bar and pos >= 2 / 3 else "LOSS" if med < -bar and neg >= 2 / 3 else
          "NO CHANGE" if abs(med) <= NEGLIGIBLE and s_g <= SESOI / 2 else "UNRESOLVED")
    return {**rec, "median": med, "s_g": s_g, "bar": bar, "frac_pos": float(pos), "frac_neg": float(neg), "state": st,
            "marked_single_seed": bool(min(n) < 2)}


def group_reading(a, b):
    """Combine the states of 16 → 8 (a) and 8 → 4 (b), first match of §6's table."""
    if "INSUFFICIENT" in (a, b):
        return "INSUFFICIENT"
    if "LOSS" in (a, b):
        return "WORSE WITH SMALLER PATCHES (" + "; ".join(n for n, s in (("16->8", a), ("8->4", b)) if s == "LOSS") + ")"
    return {("GAIN", "NO CHANGE"): "PATCH-LIMITED TO A PSF FLOOR", ("GAIN", "GAIN"): "PATCH-LIMITED BELOW 8 px",
            ("GAIN", "UNRESOLVED"): "PATCH-LIMITED; FLOOR UNRESOLVED", ("NO CHANGE", "NO CHANGE"): "DATA-LIMITED AT 16 px",
            ("NO CHANGE", "GAIN"): "LIMITED BELOW 8 px ONLY (unexpected)"}.get((a, b), "UNRESOLVED")


def objective_pair(s1, s2, answers):
    """D(o1, o2) for one group and step; [AMBIGUOUS 5] D = median over answers of the per-answer Δ difference, not the difference of medians."""
    if "INSUFFICIENT" in (s1["state"], s2["state"]):
        return {"state": "INSUFFICIENT"}
    same = [a for a in answers if a in s1["delta"] and a in s2["delta"]]
    if len(same) < MIN_ANSWERS:
        return {"state": "INSUFFICIENT"}
    D = float(np.median([s1["delta"][a] - s2["delta"][a] for a in same]))
    bar = max(SESOI / 2, 2 * np.sqrt(s1["s_g"] ** 2 + s2["s_g"] ** 2))
    st = ("DIFFERS" if abs(D) > bar else "SAME" if abs(D) <= NEGLIGIBLE and max(s1["s_g"], s2["s_g"]) <= SESOI / 2 else "UNRESOLVED")
    return {"D": D, "bar": float(bar), "state": st, "sign": "+" if D > 0 else "-"}


def device(gpu, mac, mac_seed_auc):
    """gpu/mac: {(o, s): {answer: full-sample AUC}} paired by seed; mac_seed_auc: {seed: {answer: AUC}} of the Mac A1–A3.
    Median |paired difference| over answers and seeds vs the Mac seed spread ([AMBIGUOUS 6] spread = median over answers of max − min over the three seeds)."""
    out = {}
    spreads = np.median([max(mac_seed_auc[s][a] for s in mac_seed_auc) - min(mac_seed_auc[s][a] for s in mac_seed_auc) for a in mac_seed_auc[0]])
    for o in sorted({k[0] for k in gpu} & {k[0] for k in mac}):
        d = [abs(gpu[o, s][a] - mac[o, s][a]) for (oo, s) in gpu if oo == o and (o, s) in mac for a in gpu[o, s]]
        out[o] = {"median_abs_diff": float(np.median(d)), "mac_spread": float(spreads), "state": "DEVICE EFFECT" if np.median(d) > spreads else "NO DEVICE EFFECT"}
    return out


def score(auc, un, counts, groups=GROUPS):
    L = all_L(auc, un, counts)
    out = {"steps": {}, "readings": {}, "objective_pairs": {}}
    for o in OBJ:
        for g, ans in groups.items():
            st = {f"{p}->{q}": step(L, o, p, q, ans) for p, q in STEPS}
            out["steps"][o, g] = st
            out["readings"][o, g] = group_reading(st["16->8"]["state"], st["8->4"]["state"])
    for g, ans in groups.items():
        for (o1, o2) in (("jepa", "mae"), ("jepa", "moco"), ("mae", "moco")):
            for st in ("16->8", "8->4"):
                out["objective_pairs"][o1, o2, g, st] = objective_pair(out["steps"][o1, g][st], out["steps"][o2, g][st], ans)
    return out


# --- synthetic inputs for the plants: per-decile AUC = untrained max + margin, the margin set so that L is what the plant asks for ------
def synth(target, rng, seeds=2, sd=0.0, shared=0.0, drop=(), n_pos=400, few=()):
    """target[(o, p)][group] → the group's L at that patch size (answers get group L ± a small offset); per-seed answer noise sd, shared
    fraction `shared`; drop = arms voided; few = answers with too few positives."""
    auc, un, counts = {}, {}, {}
    ans = [a for g in GROUPS.values() for a in g]
    gof = {a: g for g, v in GROUPS.items() for a in v}
    base = {a: 0.62 + 0.004 * k for k, a in enumerate(ans)}
    for a in ans:
        counts[a] = np.full((10, 2), 8 if a in few else n_pos)
    for o in OBJ:
        for p in PATCH:
            u = {a: np.tile(base[a], (3, 10)) - np.array([0.0, 0.01, 0.02])[:, None] for a in ans}  # the max is seed 0's
            un[o, p] = u
            for s in range(seeds):
                if (o, p, s) in drop:
                    continue
                # sd: one number, or {objective: {answer: sd}} with shared {objective: fraction} (the observed patch-16 spread)
                sda = {a: (sd[o][a] if isinstance(sd, dict) else sd) for a in ans}
                r = shared[o] if isinstance(shared, dict) else shared
                z = rng.normal() * float(np.sqrt(np.mean(np.square(list(sda.values()))))) * np.sqrt(r)
                auc[o, p, s] = {a: base[a] + target[o, p][gof[a]] + z + rng.normal() * sda[a] * np.sqrt(1 - r) for a in ans}
                auc[o, p, s] = {a: np.full(10, v) for a, v in auc[o, p, s].items()}
    return auc, un, counts


def step_plants():
    """Every step state, group reading, objective state and device state, by construction; sd = the observed spread's scale is tested in `power`."""
    rng = np.random.default_rng(1)
    cells = {}
    base = {"G1": 0.10, "G2": 0.06, "G3": 0.03}
    f = lambda L16, L8, L4: {"16": L16, "8": L8, "4": L4}  # noqa: E731

    def plant(name, per_obj, **kw):  # per_obj: {obj: {group: (L16, L8, L4)}} or a single dict for all objectives
        t = {(o, p): {} for o in OBJ for p in PATCH}
        for o in OBJ:
            spec = per_obj.get(o, per_obj["*"])
            for g in GROUPS:
                for p, v in zip(PATCH, spec[g]):
                    t[o, p][g] = v
        cells[name] = score(*synth(t, rng, **kw))
        return cells[name]
    stay = lambda g: (base[g],) * 3  # noqa: E731
    up = lambda a, b=None: {g: (base[g], base[g] + a, base[g] + (b if b is not None else a)) for g in GROUPS}  # noqa: E731
    noise = {"sd": 0.001}
    plant("no change", {"*": {g: stay(g) for g in GROUPS}}, **noise)
    plant("gain then floor", {"*": up(0.05)}, **noise)
    plant("gain then gain", {"*": up(0.05, 0.10)}, **noise)
    plant("no change then gain", {"*": {g: (base[g], base[g], base[g] + 0.05) for g in GROUPS}}, **noise)
    plant("loss at 8->4", {"*": {g: (base[g], base[g], base[g] - 0.05) for g in GROUPS}}, **noise)
    plant("loss at 16->8", {"*": {g: (base[g], base[g] - 0.05, base[g] - 0.05) for g in GROUPS}}, **noise)
    plant("gain then unresolved", {"*": {g: (base[g], base[g] + 0.06, base[g] + 0.067) for g in GROUPS}}, sd=0.001)  # 8→4 Δ ≈ 0.007: between NEGLIGIBLE and B_g
    plant("unresolved", {"*": {g: (base[g], base[g] + 0.008, base[g] + 0.009) for g in GROUPS}}, **noise)
    plant("insufficient (rare answers)", {"*": up(0.05)}, few=GROUPS["G1"][:5] + GROUPS["G2"][:7] + GROUPS["G3"][:6], **noise)
    plant("insufficient (both seeds voided at 8)", {"*": up(0.05)}, drop=tuple((o, 8, s) for o in OBJ for s in (0, 1)), **noise)
    plant("one seed voided", {"*": up(0.05)}, drop=(("jepa", 4, 1),), **noise)
    plant("MAE > JEPA > MoCo", {"mae": up(0.12), "jepa": up(0.05), "moco": up(0.0), "*": up(0.0)}, **noise)
    plant("objectives differ by SESOI-ish (unresolved)", {"mae": up(0.058), "jepa": up(0.05), "moco": up(0.05), "*": up(0.05)}, **noise)
    readings = {n: {f"{o}/{g}": c["readings"][o, g] for o in OBJ for g in GROUPS} for n, c in cells.items()}
    steps = {n: sorted({v["state"] for st in c["steps"].values() for v in st.values()}) for n, c in cells.items()}
    obj = {n: sorted({v["state"] for v in c["objective_pairs"].values()}) for n, c in cells.items()}
    a = list(KEY)
    mac = {("jepa", s): {x: 0.80 + 0.001 * s for x in a} for s in (0, 1)}
    dev = {"no effect": device({k: {x: v + 0.0002 for x, v in d.items()} for k, d in mac.items()}, mac, {s: {x: 0.80 + 0.001 * s for x in a} for s in (0, 1, 2)}),
           "effect": device({k: {x: v + 0.05 for x, v in d.items()} for k, d in mac.items()}, mac, {s: {x: 0.80 + 0.001 * s for x in a} for s in (0, 1, 2)})}
    reached = {"step states": sorted({s for v in steps.values() for s in v}), "group readings": sorted({r.split(" (")[0] if r.startswith("WORSE") else r for v in readings.values() for r in v.values()}),
               "objective states": sorted({s for v in obj.values() for s in v}), "device states": sorted({v["jepa"]["state"] for v in dev.values()})}
    need = {"step states": {"INSUFFICIENT", "GAIN", "LOSS", "NO CHANGE", "UNRESOLVED"},
            "group readings": {"INSUFFICIENT", "WORSE WITH SMALLER PATCHES", "PATCH-LIMITED TO A PSF FLOOR", "PATCH-LIMITED BELOW 8 px", "PATCH-LIMITED; FLOOR UNRESOLVED",
                               "DATA-LIMITED AT 16 px", "LIMITED BELOW 8 px ONLY (unexpected)", "UNRESOLVED"},
            "objective states": {"INSUFFICIENT", "DIFFERS", "SAME", "UNRESOLVED"}, "device states": {"DEVICE EFFECT", "NO DEVICE EFFECT"}}
    missing = {k: sorted(need[k] - set(reached[k])) for k in need}
    out = {"reached": reached, "missing": missing, "readings": readings, "step_states": steps, "objective_states": obj, "device": dev}
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "plants.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"reached": reached, "missing": missing, "one_seed_voided": readings["one seed voided"]["jepa/G1"]}, indent=1))
    assert not any(missing.values()), missing


ENCS = {"jepa": ("A1", "A2", "A3"), "moco": ("moco0", "moco1", "moco2"), "mae": ("mae0", "mae1", "mae2")}


def seed_stats(sp, kind):
    """{objective: (per-answer seed SD {answer: sd}, shared fraction, pooled SD)} from the three patch-16 Mac seeds. Shared fraction r = the part
    of the variance common to every answer of an encoder (a seed's whole-embedding quality), estimated over all answers of that objective.
    Answers whose L is undefined in any seed are dropped and named."""
    out, dropped = {}, []
    for o, encs in ENCS.items():
        v = {a: np.array([sp["L"][kind][e][KEY[a]] for e in encs]) for a in KEY}
        bad = [a for a in KEY if not np.isfinite(v[a]).all()]
        dropped += [(o, a) for a in bad]
        keep = [a for a in KEY if a not in bad]
        m = np.array([v[a] for a in keep])  # (answers, 3)
        var = m.var(1, ddof=1)
        d = np.array([m[:, i] - m[:, j] for i in range(3) for j in range(i)])  # (pairs, answers)
        r = max(0.0, (np.mean(d.mean(1) ** 2) / 2 - var.mean() / len(keep)) / var.mean())
        sds = dict(zip(keep, np.sqrt(var)))
        med = float(np.median(list(sds.values())))
        sds.update({a: med for a in bad})  # an undefined answer needs an SD to be drawn; it is INSUFFICIENT in any real run
        out[o] = {"sd": sds, "shared": float(np.clip(r, 0, 0.99)), "pooled": float(np.sqrt(var.mean())), "median_sd": med,
                  "observed_s_g": {g: float(np.median([np.mean(np.abs(d[:, keep.index(a)])) for a in ans if a in keep])) for g, ans in GROUPS.items()}}
    return out, dropped


def step_power():
    """Monte Carlo at the observed patch-16 seed spread, per objective and kind: 2 and 3 seeds; the same seed SD is assumed at 8 and 4 (unknown)."""
    sp = json.loads((OUT / "spread.json").read_text())
    out = {"source": sp["note"], "assumption": "seed SD and shared fraction at patch 16 are applied at 8 and 4; neither is known", "kinds": {}}
    grid = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08]
    for kind in ("raw", "cleaned"):
        st, dropped = seed_stats(sp, kind)
        out["kinds"][kind] = {"dropped_answers": dropped, "observed": {o: {k: v for k, v in x.items() if k != "sd"} for o, x in st.items()}, "power": {}}
        for o in OBJ:
            for n_seed in (2, 3):
                for g, ans in GROUPS.items():
                    rng, N, tab = np.random.default_rng(7), 400, {}
                    sig = np.array([st[o]["sd"][a] for a in ans])
                    r, pooled = st[o]["shared"], st[o]["pooled"]
                    for dlt in grid:
                        cnt = {k: 0 for k in ("GAIN", "NO CHANGE", "UNRESOLVED", "LOSS", "INSUFFICIENT")}
                        for _ in range(N):
                            L = {}
                            for p, off in ((16, 0.0), (8, dlt)):
                                for s in range(n_seed):
                                    z = rng.normal() * pooled * np.sqrt(r) + rng.normal(size=len(ans)) * np.sqrt(1 - r) * sig
                                    L[o, p, s] = dict(zip(ans, off + z))
                            cnt[step(L, o, 16, 8, ans)["state"]] += 1
                        tab[dlt] = {k: c / N for k, c in cnt.items()}
                    pg = [tab[x]["GAIN"] for x in grid]
                    d80 = next((float(np.interp(0.8, [pg[k - 1], pg[k]], [grid[k - 1], grid[k]])) for k in range(1, len(pg)) if pg[k] >= 0.8 > pg[k - 1]), None)
                    out["kinds"][kind]["power"][f"{o}/{n_seed}/{g}"] = {"s_g_observed_pairs": st[o]["observed_s_g"][g], "delta_80pct_GAIN": d80, "P_at_0": tab[0.0],
                                                                         "P_GAIN": dict(zip(map(str, grid), pg)), "P_NO_CHANGE_at_0": tab[0.0]["NO CHANGE"]}
    (DOC / "power.json").write_text(json.dumps(out, indent=1, default=float))
    for kind, b in out["kinds"].items():
        for k, v in b["power"].items():
            print(kind, k, f"s_g {v['s_g_observed_pairs']:.4f} Δ80 {v['delta_80pct_GAIN']} P(NO CHANGE|0) {v['P_NO_CHANGE_at_0']:.2f} P(UNRES|0) {v['P_at_0']['UNRESOLVED']:.2f}")


def step_observed():
    """The step plants again at the observed seed spread (§9): what share of draws give the intended state, and which states are reached at all."""
    sp = json.loads((OUT / "spread.json").read_text())
    out = {"kinds": {}}
    b = {"G1": 0.10, "G2": 0.06, "G3": 0.03}
    cases = {"no change": (0, 0), "gain then floor": (.05, .05), "gain then gain": (.05, .10), "no change then gain": (0, .05), "loss then loss-recover": (-.05, -.05),
             "gain then small": (.06, .067), "sub-bar": (.008, .009), "large gain then floor": (.08, .08)}
    for kind in ("raw", "cleaned"):
        st, _ = seed_stats(sp, kind)
        sd, sh = {o: st[o]["sd"] for o in OBJ}, {o: st[o]["shared"] for o in OBJ}
        res = {}
        for name, (a8, a4) in cases.items():
            rng, N = np.random.default_rng(11), 200
            t = {(o, p): {g: b[g] + {16: 0, 8: a8, 4: a4}[p] for g in GROUPS} for o in OBJ for p in PATCH}
            cnt, pairs, stp = {}, {}, {}
            for _ in range(N):
                r = score(*synth(t, rng, sd=sd, shared=sh, seeds=2))
                for o in OBJ:
                    for g in GROUPS:
                        key = r["readings"][o, g].split(" (")[0]
                        cnt[f"{o}/{g}"] = {**cnt.get(f"{o}/{g}", {}), key: cnt.get(f"{o}/{g}", {}).get(key, 0) + 1}
                        for s in ("16->8", "8->4"):
                            k = r["steps"][o, g][s]["state"]
                            stp[k] = stp.get(k, 0) + 1
                for v in r["objective_pairs"].values():
                    pairs[v["state"]] = pairs.get(v["state"], 0) + 1
            res[name] = {"reading_share": {k: {kk: vv / N for kk, vv in v.items()} for k, v in cnt.items()}, "step_states_seen": stp, "objective_states_seen": pairs}
        out["kinds"][kind] = res
        print(kind, {n: {k: max(v, key=v.get) for k, v in r["reading_share"].items() if k.startswith("jepa")} for n, r in res.items()})
    (DOC / "plants_observed.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
