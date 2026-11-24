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
EQUIV = SESOI  # NO CHANGE iff |G| + b <= EQUIV (user, 24 Nov)
RHO_GATE = 0.5
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


# Paired rule (user, 24 Nov): Δ_s(a) = L(seed s at q) − L(seed s at p); G_s = median over the group's answers of Δ_s(a); G = mean over seeds of G_s.
# Bar b = max(SESOI/2, K_n · S / √n), S = the larger of the two steps' SDs of G_s over seeds (the larger noise sets the bar, as max(S_A, S_X) does in
# baselines_comparison.md), n = seeds valid at both sizes. K_n is calibrated by Monte Carlo at the observed noise (`calibrate`): the smallest K on a
# 0.05 grid whose false GAIN-or-LOSS rate and false DIFFERS rate are ≤ 2.5% in every regime (docs/patch_grid/calibration.json).
K_N = {3: None, 2: None, "3s": None, "2s": None}  # "3s"/"2s": one step only (8 → 4 not run), S from that step alone; filled from calibration.json by load_K(); None means not calibrated and the code refuses to score
CAL = DOC / "calibration.json"


def load_K():
    c = json.loads(CAL.read_text())
    K_N.update({(int(k) if k.isdigit() else k): v for k, v in c["K"].items()})
    return K_N


def paired(L, o, p, q, answers):
    """(seeds used, answers used, Δ (n, A)): seeds valid at both sizes; answers defined (finite L) in every one of them."""
    ss = [s for s in seeds_of(L, o, p) if s in seeds_of(L, o, q)]
    ok = [a for a in answers if ss and all(np.isfinite(L[o, p, s][a]) and np.isfinite(L[o, q, s][a]) for s in ss)]
    D = np.array([[L[o, q, s][a] - L[o, p, s][a] for a in ok] for s in ss]) if ss and ok else np.zeros((0, 0))
    return ss, ok, D


def step_raw(L, o, p, q, answers):
    ss, ok, D = paired(L, o, p, q, answers)
    rec = {"n_answers": len(ok), "seeds": ss, "answers": ok, "n_pairs": len(ss)}
    if len(ok) < MIN_ANSWERS or len(ss) < 2:  # [AMBIGUOUS 4'] fewer than 2 paired seeds: no noise estimate, so INSUFFICIENT
        return {**rec, "state": "INSUFFICIENT"}
    Gs = np.median(D, 1)
    return {**rec, "G_s": Gs.tolist(), "G": float(Gs.mean()), "S": float(Gs.std(ddof=1)), "delta_answers": dict(zip(ok, D.mean(0))), "state": None}


def step_state(rec, S, single=False):
    """First match, with the group's shared noise S (the larger of the two steps' SDs) and K for this many pairs."""
    if rec["state"] == "INSUFFICIENT":
        return rec
    k = K_N[f"{rec['n_pairs']}s" if single else rec["n_pairs"]]
    if k is None:
        raise RuntimeError("K is not calibrated: run `calibrate` first")
    G, n = rec["G"], rec["n_pairs"]
    b = max(SESOI / 2, k * S / np.sqrt(n))
    d = np.array(list(rec["delta_answers"].values()))
    pos, neg = float((d > 0).mean()), float((d < 0).mean())
    st = ("GAIN" if G > b and pos >= 2 / 3 else "LOSS" if G < -b and neg >= 2 / 3 else "NO CHANGE" if abs(G) + b <= EQUIV else "UNRESOLVED")
    if st == "UNRESOLVED" and (min(rec["G_s"]) <= -SESOI / 2 and max(rec["G_s"]) >= SESOI / 2):
        rec = {**rec, "label": "the seeds disagree in sign"}
    return {**rec, "S_used": float(S), "bar": float(b), "frac_pos": pos, "frac_neg": neg, "state": st, "marked": n < 3}


def group_reading(a, b):
    """Combine the states of 16 → 8 (a) and 8 → 4 (b), first match of §6's table."""
    if b == "NOT RUN":  # staged design: the 8 → 4 arms were not run for this objective (gate failed)
        return {"INSUFFICIENT": "INSUFFICIENT", "LOSS": "WORSE WITH SMALLER PATCHES (16->8; 8->4 not run)", "GAIN": "PATCH-LIMITED AT 16 -> 8; 8->4 NOT RUN",
                "NO CHANGE": "NO CHANGE 16->8; 8->4 NOT RUN"}.get(a, "UNRESOLVED (8->4 not run)")
    if "INSUFFICIENT" in (a, b):
        return "INSUFFICIENT"
    if "LOSS" in (a, b):
        return "WORSE WITH SMALLER PATCHES (" + "; ".join(n for n, s in (("16->8", a), ("8->4", b)) if s == "LOSS") + ")"
    return {("GAIN", "NO CHANGE"): "PATCH-LIMITED TO A PSF FLOOR", ("GAIN", "GAIN"): "PATCH-LIMITED BELOW 8 px",
            ("GAIN", "UNRESOLVED"): "PATCH-LIMITED; FLOOR UNRESOLVED", ("NO CHANGE", "NO CHANGE"): "DATA-LIMITED AT 16 px",
            ("NO CHANGE", "GAIN"): "LIMITED BELOW 8 px ONLY (unexpected)"}.get((a, b), "UNRESOLVED")


def objective_pair(r1, r2, S1, S2):
    """D(o1, o2) for one group and step, paired by seed: D_s = G_s(o1) − G_s(o2) over seeds valid in both."""
    if "NOT RUN" in (r1["state"], r2["state"]):
        return {"state": "NOT RUN"}
    if "INSUFFICIENT" in (r1["state"], r2["state"]):
        return {"state": "INSUFFICIENT"}
    ss = [s for s in r1["seeds"] if s in r2["seeds"]]
    if len(ss) < 2:
        return {"state": "INSUFFICIENT"}
    a = np.array([r1["G_s"][r1["seeds"].index(s)] - r2["G_s"][r2["seeds"].index(s)] for s in ss])
    n = len(ss)
    k = max(K_N[n], K_N[f"{n}s"] if r1.get("single") or r2.get("single") else 0)
    D, b = float(a.mean()), max(SESOI / 2, k * max(S1, S2) * np.sqrt(2 / n))
    st = "DIFFERS" if abs(D) > b else "SAME" if abs(D) <= NEGLIGIBLE and r1["bar"] <= SESOI and r2["bar"] <= SESOI else "UNRESOLVED"
    return {"D": D, "bar": float(b), "state": st, "sign": "+" if D > 0 else "-"}


def device(gpu, mac, mac_seed_auc):
    """gpu/mac: {(o, s): {answer: full-sample AUC}} paired by seed; mac_seed_auc: {seed: {answer: AUC}} of the Mac A1–A3.
    Median |paired difference| over answers and seeds vs the Mac seed spread ([AMBIGUOUS 6] spread = median over answers of max − min over the three seeds)."""
    out = {}
    spreads = np.median([max(mac_seed_auc[s][a] for s in mac_seed_auc) - min(mac_seed_auc[s][a] for s in mac_seed_auc) for a in mac_seed_auc[0]])
    for o in sorted({k[0] for k in gpu} & {k[0] for k in mac}):
        d = [abs(gpu[o, s][a] - mac[o, s][a]) for (oo, s) in gpu if oo == o and (o, s) in mac for a in gpu[o, s]]
        out[o] = {"median_abs_diff": float(np.median(d)), "mac_spread": float(spreads), "state": "DEVICE EFFECT" if np.median(d) > spreads else "NO DEVICE EFFECT"}
    return out


def score(auc, un, counts, groups=GROUPS, not_run=()):
    return score_L(all_L(auc, un, counts), groups, not_run)


# --- the blind ρ gate (staged design, user 24 Nov) ---------------------------------------------------------------------
def rho_hat(M16, M8):
    """M: (n seeds, A answers) of L (or of the deciles-1–5 mean AUC: the untrained maximum is the same for every seed and centres away) at each
    patch size. Seed effects = M minus each answer's mean over the seeds, at each size; ρ̂ = the Pearson correlation over all (answer, seed) pairs.
    Answers undefined in any seed at either size are dropped. NaN when nothing varies or < 2 seeds. Returns ρ̂ alone: nothing else is computed."""
    M16, M8 = np.asarray(M16, float), np.asarray(M8, float)
    ok = np.isfinite(M16).all(0) & np.isfinite(M8).all(0)
    if M16.shape[0] < 2 or ok.sum() < MIN_ANSWERS:
        return float("nan")
    a = (M16[:, ok] - M16[:, ok].mean(0)).ravel()
    b = (M8[:, ok] - M8[:, ok].mean(0)).ravel()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def gate_decision(o, r):
    """The staged rule on ρ̂ alone; MoCo's 4×4 runs whatever ρ̂ is."""
    if o == "moco":
        return "RUN 4x4 (always)"
    return "RUN 4x4" if r >= RHO_GATE else "NOT RUN: planted power at rho-hat cannot resolve" if np.isfinite(r) else "NOT RUN: rho-hat not estimable"


def gate(L, o):
    """(ρ̂, decision) for one objective from {(o, p, s): {answer: L}}: only seeds present at both sizes are used."""
    ss = [s for s in seeds_of(L, o, 16) if s in seeds_of(L, o, 8)]
    ans = list(KEY)
    r = rho_hat([[L[o, 16, s][a] for a in ans] for s in ss], [[L[o, 8, s][a] for a in ans] for s in ss]) if ss else float("nan")
    return r, gate_decision(o, r)


def score_L(L, groups=GROUPS, not_run=()):
    """not_run: objectives whose 8 → 4 arms the gate stopped."""
    out = {"steps": {}, "readings": {}, "objective_pairs": {}}
    Sg = {}
    for o in OBJ:
        for g, ans in groups.items():
            raw = {f"{p}->{q}": step_raw(L, o, p, q, ans) for p, q in STEPS if not (o in not_run and q == 4)}
            single = o in not_run
            Sg[o, g] = max((r["S"] for r in raw.values() if r["state"] is None), default=0.0)
            st = {k: {**step_state(r, Sg[o, g], single), "single": single} for k, r in raw.items()}
            if single:
                st["8->4"] = {"state": "NOT RUN"}
            out["steps"][o, g] = st
            out["readings"][o, g] = group_reading(st["16->8"]["state"], st["8->4"]["state"])
    for g in groups:
        for (o1, o2) in (("jepa", "mae"), ("jepa", "moco"), ("mae", "moco")):
            for st in ("16->8", "8->4"):
                out["objective_pairs"][o1, o2, g, st] = objective_pair(out["steps"][o1, g][st], out["steps"][o2, g][st], Sg[o1, g], Sg[o2, g])
    return out


# --- synthetic inputs for the plants: per-decile AUC = untrained max + margin, the margin set so that L is what the plant asks for ------
def synth(target, rng, seeds=3, sd=0.0, shared=0.0, drop=(), n_pos=400, few=()):
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
    load_K()
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
        not_run = kw.pop("not_run", ())
        cells[name] = score(*synth(t, rng, **kw), not_run=not_run)
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
    # With |G| + b <= 0.02 and almost no noise the bar is 0.01, so NO CHANGE and GAIN meet at 0.01 and leave no gap: UNRESOLVED needs real seed noise.
    for k in range(6):
        plant(f"gain then unresolved {k}", {"*": {g: (base[g], base[g] + 0.30, base[g] + 0.312) for g in GROUPS}}, sd=0.02, shared=0.8)
        plant(f"unresolved {k}", {"*": {g: (base[g], base[g] + 0.012, base[g] + 0.02) for g in GROUPS}}, sd=0.02, shared=0.8)
    plant("insufficient (rare answers)", {"*": up(0.05)}, few=GROUPS["G1"][:5] + GROUPS["G2"][:7] + GROUPS["G3"][:6], **noise)
    plant("insufficient (two of three seeds voided at 8)", {"*": up(0.05)}, drop=tuple((o, 8, s) for o in OBJ for s in (0, 1)), **noise)
    plant("one seed voided", {"*": up(0.05)}, drop=(("jepa", 4, 1),), **noise)
    plant("MAE > JEPA > MoCo", {"mae": up(0.12), "jepa": up(0.05), "moco": up(0.0), "*": up(0.0)}, **noise)
    plant("objectives differ by SESOI-ish (unresolved)", {"mae": up(0.058), "jepa": up(0.05), "moco": up(0.05), "*": up(0.05)}, **noise)
    # the gate's NOT RUN readings: JEPA and MAE stopped at 8, one planted case for each 16 → 8 state
    for nm_, spec in (("gain", up(0.05)), ("no change", {g: stay(g) for g in GROUPS}), ("loss", {g: (base[g], base[g] - 0.05, base[g]) for g in GROUPS}),
                      ("noisy", {g: (base[g], base[g] + 0.02, base[g]) for g in GROUPS})):
        plant(f"8->4 not run: {nm_}", {"*": spec}, not_run=("jepa", "mae"), **({"sd": 0.02, "shared": 0.8} if nm_ == "noisy" else noise))
    plant("8->4 not run: insufficient", {"*": up(0.05)}, not_run=("jepa", "mae"), few=GROUPS["G1"][:5] + GROUPS["G2"][:7] + GROUPS["G3"][:6], **noise)
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
                               "DATA-LIMITED AT 16 px", "LIMITED BELOW 8 px ONLY (unexpected)", "UNRESOLVED", "PATCH-LIMITED AT 16 -> 8; 8->4 NOT RUN",
                               "NO CHANGE 16->8; 8->4 NOT RUN", "UNRESOLVED (8->4 not run)"},
            "objective states": {"INSUFFICIENT", "DIFFERS", "SAME", "UNRESOLVED", "NOT RUN"}, "device states": {"DEVICE EFFECT", "NO DEVICE EFFECT"}}
    missing = {k: sorted(need[k] - set(reached[k])) for k in need}
    out = {"reached": reached, "missing": missing, "readings": readings, "step_states": steps, "objective_states": obj, "device": dev}
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "plants.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"reached": reached, "missing": missing, "one_seed_voided": readings["one seed voided"]["jepa/G1"]}, indent=1))
    assert not any(missing.values()), missing


ENCS = {"jepa": ("A1", "A2", "A3"), "moco": ("moco0", "moco1", "moco2"), "mae": ("mae0", "mae1", "mae2")}
RHOS, KINDS, SCALES, FALSE = (0.0, 0.5, 1.0), ("gauss", "outlier"), (0.5, 1.0, 2.0), 0.025


def noise_model(sp, kind):
    """{(objective, group): (tau, sigma)} from the three patch-16 Mac seeds: tau = run-level SD of the group mean of L over seeds (the encoder-wide
    shift, answer noise removed); sigma = answer-level residual SD (pooled over answers, 2 df each). Also the raw per-seed group means, for the record."""
    out = {}
    for o, encs in ENCS.items():
        for g, ans in GROUPS.items():
            v = np.array([[sp["L"][kind][e][KEY[a]] for e in encs] for a in ans])  # (A, 3)
            assert np.isfinite(v).all(), (o, g)
            A = v.shape[0]
            m = v.mean(0)  # group mean per seed
            res = v - v.mean(1, keepdims=True) - m[None] + m.mean()
            sig = float(np.sqrt((res ** 2).sum() / ((A - 1) * 2)))
            tau = float(np.sqrt(max(0.0, m.var(ddof=1) - sig ** 2 / A)))
            out[o, g] = (tau, sig)
    return out


def draw(rng, R, n, A, tau, sig, rho, kind, sizes=3):
    """Noise of L: (R, n seeds, sizes, A answers) = run-level u (correlation rho across patch sizes within a seed) + answer-level jitter (independent)."""
    def z(shape):
        if kind == "gauss":
            return rng.normal(size=shape)
        return (rng.random(shape) < 1 / 3) * np.sqrt(3) * rng.normal(size=shape)  # one seed in three runs far off; unit variance
    common = z((R, n, 1))
    u = np.sqrt(rho) * common + np.sqrt(1 - rho) * z((R, n, sizes))
    return tau * u[..., None] + sig * rng.normal(size=(R, n, sizes, A))


def steps_vec(N, e8=0.0, e4=0.0):
    """Both steps of one objective and group from noise N: per-seed medians, G, S (shared), and the sign fractions. Vector form of step_raw/step_state."""
    D1 = N[:, :, 1] - N[:, :, 0] + e8
    D2 = N[:, :, 2] - N[:, :, 1] + (e4 - e8)
    out = []
    for D in (D1, D2):
        Gs = np.median(D, 2)
        d = D.mean(1)
        out.append((Gs.mean(1), Gs.std(1, ddof=1), (d > 0).mean(1), (d < 0).mean(1), Gs))
    S = np.maximum(out[0][1], out[1][1])
    return out, S


def t_stats(N, n, single=False):
    """Per draw: the K above which GAIN or LOSS stops being called at each step (0 when the sign or size floor already blocks it)."""
    steps, S = steps_vec(N)
    if single:
        steps, S = steps[:1], steps[0][1]
    t = []
    for G, _, pos, neg, _ in steps:
        fl = (np.abs(G) > SESOI / 2) & np.where(G > 0, pos >= 2 / 3, neg >= 2 / 3)
        t.append(np.where(fl, np.abs(G) * np.sqrt(n) / np.maximum(S, 1e-12), 0.0))
    return np.array(t)


def step_calibrate():
    sp = json.loads((OUT / "spread.json").read_text())
    rng = np.random.default_rng(20261124)
    R, rows = 20000, []
    worst = {2: [], 3: [], "2s": [], "3s": []}
    for n in (2, 3):
        for kind in ("raw", "cleaned"):
            nm = noise_model(sp, kind)
            for (o, g), (tau, sig) in nm.items():
                A = len(GROUPS[g])
                for nz in KINDS:
                    for rho in RHOS:
                        for sc in SCALES:
                            t = t_stats(draw(rng, R, n, A, tau * sc, sig * sc, rho, nz), n)
                            k = float(max(np.quantile(x, 1 - FALSE) for x in t))
                            rows.append({"n": n, "kind": kind, "cell": f"{o}/{g}", "noise": nz, "rho": rho, "scale": sc, "K_needed": k})
                            worst[n].append(k)
                            ts = t_stats(draw(rng, R, n, A, tau * sc, sig * sc, rho, nz), n, single=True)
                            ks = float(np.quantile(ts[0], 1 - FALSE))
                            rows.append({"n": f"{n}s", "kind": kind, "cell": f"{o}/{g}", "noise": nz, "rho": rho, "scale": sc, "K_needed": ks})
                            worst[f"{n}s"].append(ks)
            for (o1, o2) in (("jepa", "mae"), ("jepa", "moco"), ("mae", "moco")):
                for g in GROUPS:
                    A = len(GROUPS[g])
                    for nz in KINDS:
                        for rho in RHOS:
                            s1, S1 = steps_vec(draw(rng, R, n, A, *nm[o1, g], rho, nz))
                            s2, S2 = steps_vec(draw(rng, R, n, A, *nm[o2, g], rho, nz))
                            for a, b in zip(s1, s2):
                                D = a[0] - b[0]
                                t = np.where(np.abs(D) > SESOI / 2, np.abs(D) / np.maximum(np.maximum(S1, S2) * np.sqrt(2 / n), 1e-12), 0.0)
                                k = float(np.quantile(t, 1 - FALSE))
                                rows.append({"n": n, "kind": kind, "cell": f"{o1}-{o2}/{g}", "noise": nz, "rho": rho, "scale": 1.0, "K_needed": k})
                                worst[n].append(k)
    # 3 decimals of headroom are meaningless at R = 20,000: round up to the 0.05 grid
    K = {str(n): float(np.ceil(max(v) / 0.05) * 0.05) for n, v in worst.items()}
    top = {str(n): sorted((r for r in rows if r["n"] == n), key=lambda r: -r["K_needed"])[:5] for n in (2, 3, "2s", "3s")}
    out = {"rule": "smallest K on a 0.05 grid with false GAIN-or-LOSS and false DIFFERS rates <= 2.5% in every regime", "K": K, "draws_per_regime": R, "n_regimes": len(rows), "worst_regimes": top,
           "quantiles_of_K_needed_n3": np.percentile([r["K_needed"] for r in rows if r["n"] == 3], [0, 25, 50, 75, 90, 99, 100]).tolist(),
           "noise_model_raw": {f"{o}/{g}": v for (o, g), v in noise_model(sp, "raw").items()}, "noise_model_cleaned": {f"{o}/{g}": v for (o, g), v in noise_model(sp, "cleaned").items()}}
    CAL.write_text(json.dumps(out, indent=1, default=float))
    print(K, out["quantiles_of_K_needed_n3"]); print(top["3"][:2], top["3s"][:2])


def decide(N, n, e8=0.0, e4=0.0):
    """Vector states for both steps: integer codes 0 GAIN, 1 LOSS, 2 NO CHANGE, 3 UNRESOLVED; plus |G| and the bar of step 1."""
    steps, S = steps_vec(N, e8, e4)
    b = np.maximum(SESOI / 2, K_N[n] * S / np.sqrt(n))
    codes = []
    for G, _, pos, neg, _ in steps:
        codes.append(np.select([(G > b) & (pos >= 2 / 3), (G < -b) & (neg >= 2 / 3), np.abs(G) + b <= EQUIV], [0, 1, 2], 3))
    return codes, steps, b


def step_check():
    """The scalar path (score_L) and the vector path must agree draw by draw."""
    load_K()
    rng = np.random.default_rng(5)
    bad = 0
    for trial in range(300):
        n = int(rng.integers(2, 4))
        L = {}
        Ns = {}
        for o in OBJ:
            A = [a for v in GROUPS.values() for a in v]
            N = draw(rng, 1, n, len(A), 0.02, 0.01, 0.5, "gauss")[0]
            off = rng.normal(size=3) * 0.03
            for pi, p in enumerate(PATCH):
                for s in range(n):
                    L[o, p, s] = dict(zip(A, 0.1 + off[pi] + N[s, pi]))
            Ns[o] = (N, off, A)
        r = score_L(L)
        for o in OBJ:
            N, off, A = Ns[o]
            for g, ans in GROUPS.items():
                idx = [A.index(a) for a in ans]
                codes, st, b = decide(N[None][:, :, :, idx], n, e8=off[1] - off[0], e4=off[2] - off[0])
                for k, key in enumerate(("16->8", "8->4")):
                    name = ("GAIN", "LOSS", "NO CHANGE", "UNRESOLVED")[int(codes[k][0])]
                    bad += name != r["steps"][o, g][key]["state"]
    print("scalar and vector paths disagree on", bad, "of", 300 * 18, "steps")
    assert bad == 0


def step_power():
    load_K()
    sp = json.loads((OUT / "spread.json").read_text())
    grid = [0, .005, .01, .015, .02, .025, .03, .035, .04, .05, .06, .07, .08, .1, .12, .15, .2, .3]
    out = {"K": K_N, "note": "n = 3 paired seeds unless stated; noise model from the three Mac patch-16 seeds per objective (tau run-level, sigma answer-level); rho = how much of a "
           "seed's run-level effect is shared between patch sizes (unknown; 0 none, 1 all)", "cells": {}}
    R = 20000
    for kind in ("raw", "cleaned"):
        nm = noise_model(sp, kind)
        for (o, g), (tau, sig) in nm.items():
            A = len(GROUPS[g])
            for nz in KINDS:
                for rho in RHOS:
                    for n in (3, 2):
                        rng = np.random.default_rng(99)
                        N = draw(rng, R, n, A, tau, sig, rho, nz)
                        codes, _, _ = decide(N, n)
                        pn = {0: 0, 1: 0, 2: 0, 3: 0}
                        for c, v in zip(*np.unique(codes[0], return_counts=True)):
                            pn[int(c)] = v / R
                        rec = {"tau": tau, "sigma": sig, "P_at_0": {"GAIN": pn[0], "LOSS": pn[1], "NO CHANGE": pn[2], "UNRESOLVED": pn[3]}}
                        pg = []
                        for dlt in grid:
                            c, _, _ = decide(N, n, e8=dlt, e4=dlt)
                            pg.append(float((c[0] == 0).mean()))
                        rec["P_GAIN"] = dict(zip(map(str, grid), pg))
                        rec["delta80"] = next((float(np.interp(0.8, [pg[k - 1], pg[k]], [grid[k - 1], grid[k]])) for k in range(1, len(pg)) if pg[k] >= 0.8 > pg[k - 1]), None)
                        # equivalence bound: the smallest E at which NO CHANGE would be reached, under the equivalence form |G| + b <= E (the baselines' SAME)
                        st, S = steps_vec(N)
                        G = np.abs(st[0][0]); b = np.maximum(SESOI / 2, K_N[n] * S / np.sqrt(n))
                        rec["E_equiv_for_P80"] = float(np.quantile(G + b, 0.8))
                        rec["E_equiv_for_P50"] = float(np.quantile(G + b, 0.5))
                        rec["E_bar_for_P50_given_small_G"] = float(np.quantile(b[G <= NEGLIGIBLE], 0.5)) if (G <= NEGLIGIBLE).any() else None
                        rec["P_absG_le_NEGLIGIBLE"] = float((G <= NEGLIGIBLE).mean())
                        G1, S1 = st[0][0], st[0][1]  # one step only (the 8 → 4 arms not run): S from 16 → 8 alone, K single
                        bs = np.maximum(SESOI / 2, K_N[f"{n}s"] * S1 / np.sqrt(n))
                        rec["single_step"] = {"P_NO_CHANGE_at_0": float((np.abs(G1) + bs <= EQUIV).mean()), "P_GAIN_at": {}}
                        for dlt in grid:
                            s2, S2 = steps_vec(N, e8=dlt, e4=dlt)
                            g2, p2 = s2[0][0], s2[0][2]
                            bb = np.maximum(SESOI / 2, K_N[f"{n}s"] * s2[0][1] / np.sqrt(n))
                            rec["single_step"]["P_GAIN_at"][str(dlt)] = float(((g2 > bb) & (p2 >= 2 / 3)).mean())
                        pg = list(rec["single_step"]["P_GAIN_at"].values())
                        rec["single_step"]["delta80"] = next((float(np.interp(0.8, [pg[k - 1], pg[k]], [grid[k - 1], grid[k]])) for k in range(1, len(pg)) if pg[k] >= 0.8 > pg[k - 1]), None)
                        out["cells"][f"{kind}/{o}/{g}/{nz}/rho{rho}/n{n}"] = rec
    (DOC / "power.json").write_text(json.dumps(out, indent=1, default=float))
    for kind in ("raw",):
        for (o, g) in [(o, g) for o in OBJ for g in GROUPS]:
            print(o, g, *(f"| {nz} rho {r}: Δ80 {out['cells'][f'{kind}/{o}/{g}/{nz}/rho{r}/n3']['delta80']} P(NC|0) {out['cells'][f'{kind}/{o}/{g}/{nz}/rho{r}/n3']['P_at_0']['NO CHANGE']:.2f} E80 {out['cells'][f'{kind}/{o}/{g}/{nz}/rho{r}/n3']['E_equiv_for_P80']:.3f}" for nz in KINDS for r in (0.0, 1.0)))


def rho_vec(L16, L8):
    """Vector form of rho_hat on (R, n, A) noise."""
    a = (L16 - L16.mean(1, keepdims=True)).reshape(len(L16), -1)
    b = (L8 - L8.mean(1, keepdims=True)).reshape(len(L8), -1)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def step_gate():
    """Plants for the ρ gate: (1) planted Δ never moves ρ̂; (2) bias and spread of ρ̂ against the true correlation; (3) ρ̂ against the mean Δ;
    (4) every gate decision reached; (5) the gate's pass rate by true ρ."""
    load_K()
    sp = json.loads((OUT / "spread.json").read_text())
    rng = np.random.default_rng(13)
    nm = noise_model(sp, "raw")
    allA = [a for g in GROUPS.values() for a in g]
    out = {"1_invariance": {}, "2_bias": {}, "3_independence": {}, "4_decisions": {}, "5_pass_rate": {}}
    # 1: exact invariance to any planted shift, uniform, per-answer, or per-objective
    worst = 0.0
    for _ in range(2000):
        n = 3
        a16, a8 = rng.normal(size=(n, 30)) * 0.05, rng.normal(size=(n, 30)) * 0.05
        base = rho_hat(a16, a8)
        for shift in (0.05, -0.2, rng.normal(size=30) * 0.1):
            worst = max(worst, abs(rho_hat(a16, a8 + shift) - base), abs(rho_hat(a16 + shift, a8) - base))
    out["1_invariance"] = {"largest_change_over_6000_planted_shifts": worst, "draws": 2000}
    # 2–3, 5 at the observed noise, per objective
    for o in OBJ:
        tg = np.concatenate([np.full(len(GROUPS[g]), nm[o, g][0]) for g in GROUPS])
        sg = np.concatenate([np.full(len(GROUPS[g]), nm[o, g][1]) for g in GROUPS])
        R = 20000
        for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
            c = rng.normal(size=(R, 3, 1))
            u = [np.sqrt(rho) * c + np.sqrt(1 - rho) * rng.normal(size=(R, 3, 1)) for _ in range(2)]
            L16 = tg * u[0] + sg * rng.normal(size=(R, 3, 30))
            L8 = tg * u[1] + sg * rng.normal(size=(R, 3, 30))
            r = rho_vec(L16, L8)
            meanD = np.median(L8 - L16, axis=2).mean(1)  # the group-style mean Δ of a null, per draw
            truth = float(rho * (tg ** 2).sum() / ((tg ** 2) + (sg ** 2)).sum())
            out["2_bias"][f"{o}/rho{rho}"] = {"true_run_level_rho": rho, "true_pooled_correlation": truth, "mean_rhohat": float(r.mean()), "sd_rhohat": float(r.std()),
                                              "bias_vs_pooled": float(r.mean() - truth), "pass_rate_ge_0.5": float((r >= RHO_GATE).mean())}
            out["3_independence"][f"{o}/rho{rho}"] = {"pearson_rhohat_vs_mean_delta": float(np.corrcoef(r, meanD)[0, 1]),
                                                      "mean_rhohat_when_mean_delta_planted_0.05": float(rho_vec(L16, L8 + 0.05).mean() - r.mean())}
    # 4: decisions through the scalar `gate`, with three, two, one seeds and undefined answers
    def mk(n16, n8, drop=0, noise=0.0, share=0.0):
        L, c = {}, rng.normal(size=(3, 30)) * 0.05
        for o in OBJ:
            for s in range(3):
                if s < n16:
                    L[o, 16, s] = dict(zip(allA, 0.1 + c[s]))
                if s < n8:
                    L[o, 8, s] = dict(zip(allA, 0.1 + (c[s] if share else rng.normal(size=30) * 0.05)))
                    if drop and s == 0:
                        for a in allA[:26]:
                            L[o, 8, s][a] = float("nan")
        return L
    for name, L in (("shared seeds (rho 1)", mk(3, 3, share=1)), ("no sharing, 2 seeds at 8", mk(3, 2)), ("one seed at 8", mk(3, 1)), ("26 answers undefined", mk(3, 3, drop=1))):
        out["4_decisions"][name] = {o: list(gate(L, o)) for o in OBJ}
    seen = {d for v in out["4_decisions"].values() for (_, d) in v.values()}
    # a draw whose ρ̂ is finite and below 0.5, for JEPA
    for _ in range(200):
        L = mk(3, 3)
        r, d = gate(L, "jepa")
        if np.isfinite(r) and r < RHO_GATE:
            out["4_decisions"]["independent seeds, low rho-hat"] = {"jepa": [r, d]}
            seen.add(d)
            break
    need = {"RUN 4x4 (always)", "RUN 4x4", "NOT RUN: planted power at rho-hat cannot resolve", "NOT RUN: rho-hat not estimable"}
    out["missing_decisions"] = sorted(need - seen)
    # 5: conditional false-call rate at 8 → 4 given the gate passes (Δ = 0, every group; JEPA, rho 0 / 0.5 / 1)
    cond = {}
    for rho in (0.0, 0.5, 1.0):
        A = len(allA)
        N = draw(rng, 20000, 3, A, 1.0, 0.0, rho, "gauss")  # unit run-level effect; scaled per answer below
        tg = np.concatenate([np.full(len(GROUPS[g]), nm["jepa", g][0]) for g in GROUPS])
        sg = np.concatenate([np.full(len(GROUPS[g]), nm["jepa", g][1]) for g in GROUPS])
        Z = N * tg + sg * rng.normal(size=N.shape)
        r = rho_vec(Z[:, :, 0], Z[:, :, 1])
        for g, ans in GROUPS.items():
            idx = [allA.index(a) for a in ans]
            codes, _, _ = decide(Z[:, :, :, idx], 3)
            p = r >= RHO_GATE
            cond[f"rho{rho}/{g}"] = {"pass_rate": float(p.mean()), "false_GAIN_or_LOSS_8to4_given_pass": float(np.isin(codes[1][p], [0, 1]).mean()) if p.any() else None,
                                     "false_unconditional": float(np.isin(codes[1], [0, 1]).mean())}
    out["6_conditional_false_call_after_the_gate"] = cond
    (DOC / "plants_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print("invariance: largest change", worst)
    print("missing decisions:", out["missing_decisions"])
    for k, v in out["2_bias"].items():
        print(k, "true pooled %.2f mean rhohat %.2f sd %.2f pass %.2f | corr with mean Δ %.3f" % (v["true_pooled_correlation"], v["mean_rhohat"], v["sd_rhohat"], v["pass_rate_ge_0.5"], out["3_independence"][k]["pearson_rhohat_vs_mean_delta"]))
    print(cond)


def step_observed():
    """The planted readings at the observed noise: for each case, the share of draws in each group reading (vector states composed by group_reading)."""
    load_K()
    sp = json.loads((OUT / "spread.json").read_text())
    cases = {"no change": (0, 0), "gain then floor": (.06, .06), "gain then gain": (.06, .12), "no change then gain": (0, .06), "loss then stay": (-.06, -.06), "stay then loss": (0, -.06),
             "large gain then floor": (.15, .15), "huge gain then gain": (.25, .5), "gain then small": (.06, .067)}
    name = ("GAIN", "LOSS", "NO CHANGE", "UNRESOLVED")
    out = {"kinds": {}}
    for kind in ("raw", "cleaned"):
        nm = noise_model(sp, kind)
        res = {}
        for cn, (a8, a4) in cases.items():
            for (o, g), (tau, sig) in nm.items():
                for rho in (0.0, 1.0):
                    rng = np.random.default_rng(3)
                    R = 4000
                    N = draw(rng, R, 3, len(GROUPS[g]), tau, sig, rho, "gauss")
                    codes, _, _ = decide(N, 3, a8, a4)
                    cnt = {}
                    for i, j in zip(*codes):
                        r = group_reading(name[i], name[j]).split(" (")[0]
                        cnt[r] = cnt.get(r, 0) + 1
                    res[f"{cn}/{o}/{g}/rho{rho}"] = {k: v / R for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
        out["kinds"][kind] = res
    reach = {}
    for kind, res in out["kinds"].items():
        for o in OBJ:
            for rho in (0.0, 1.0):
                for k, v in res.items():
                    if f"/{o}/" in k and k.endswith(f"rho{rho}"):
                        for rd, p in v.items():
                            reach.setdefault(f"{kind}/{o}/rho{rho}", {})[rd] = max(reach.get(f"{kind}/{o}/rho{rho}", {}).get(rd, 0), p)
    out["max_share_of_each_reading_over_cases"] = reach
    (DOC / "plants_observed.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in reach.items():
        print(k, {a: round(b, 2) for a, b in sorted(v.items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
