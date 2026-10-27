"""D12's baselines against their paired JEPA seeds — the states, the plants, and (after the hash) the
scoring (`artifacts/baselines_comparison.md`).

One decision path, `decide()`, used identically by the plants and the real scoring. Per objective
X ∈ {MoCo, MAE}, per epoch point, three seed pairs (user, 2026-10-23: a third seed): A1/X0, A2/X1,
A3/X2 — JEPA and baseline with the same init stream and data order. Inputs per encoder and answer:
the held-out AUC and its 2,000 Poisson galaxy-bootstrap AUCs under weights shared across encoders and
answers (`aligned_c2`: `auc`, `boot_aucs`, `weights`) — the E2 protocol the trajectory probes ran.

    calibrate   decision-layer Monte Carlo: K and Z by a fixed rule, power, SPLIT's null rate (no scores)
    plants      synthetic scores on the real labels through auc / boot_aucs / weights, then decide()
    score       (after the hash only) refit from the banks and decide; refuses without the hash

Plants use **no real result**: base AUCs, seed noise and effects are synthetic draws; only the
probe corpus's labels and the identical split's test ids are real (counts, not representations).
The two-seed version of this file is e545a0c.

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --- the pre-registered constants (baselines_comparison.md §C) --------------------------------
PAIRS = (("A1", "X0"), ("A2", "X1"), ("A3", "X2"))  # (JEPA, baseline) per training seed 0, 1, 2
K_FAM = 3.0          # family bar = K × SE of D̄ from the run-level spread (calibrated for 3 pairs, §F)
Z_ANS = 2.1          # per-answer idiosyncratic multiplier (calibrated for 3 pairs, §F)
CI_FAM = 97.5        # family CI level: 95% Bonferroni-split over the two objectives
CI_ANS = 95.0
NEGLIGIBLE = 0.005   # |effect| below this is negligible however significant
SESOI = 0.02         # smallest effect of interest: SAME needs a bar that could have seen it
MIN_ANSWERS = 30     # of the 35: fewer scoreable for any encoder -> VOID
SPREAD_FLAG = 2.0    # one arm's spread > 2× the other's -> flagged, reported (the bar already takes the max)
EPOCHS = (0.5, 1.0, 2.0, 4.0)
PRIMARY = 4.0
# the calibration rule (fixed before the sweep): the smallest K, then Z, on a 0.05 grid, keeping the
# false-call rate ≤ 2.5% in every equal-spread regime
FALSE_BOUND = 0.025

ANSWER_SET = HERE / "out" / "baselines_answer_set.json"


def _w() -> np.ndarray:
    a = json.loads(ANSWER_SET.read_text())
    se = np.array([a["answers"][f]["se_at_ref"] for f in a["answer_set"]])
    return se ** 2 / np.mean(se ** 2)


def spreads(A: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple:
    """A, X (..., n, J) point AUCs per seed and answer -> (S_A, S_X, σ̂²_A, σ̂²_X).

    S: the sample SD over seeds of each arm's family mean (run-level spread; answer noise averages
    out over J). σ̂²: the two-way (seed × answer) residual variance, each answer's term scaled by its
    count weight w_j — the answer-level spread at mean weight."""
    n, J = A.shape[-2], A.shape[-1]
    out = [M.mean(-1).std(-1, ddof=1) for M in (A, X)]
    for M in (A, X):
        E = M - M.mean(-1, keepdims=True) - M.mean(-2, keepdims=True) + M.mean((-2, -1), keepdims=True)
        out.append((E ** 2 / w).sum((-2, -1)) / ((n - 1) * (J - 1)))
    return tuple(out)


def split_of(pairs) -> bool:
    """The seed pairs disagree: some pair ≥ +SESOI/2 and some pair ≤ −SESOI/2 (§C)."""
    return max(pairs) >= SESOI / 2 and min(pairs) <= -SESOI / 2


def state(d: float, lo: float, hi: float, b: float, pr: list[float]) -> tuple[str, str]:
    """The §C state table, first match wins — family, per answer and §N alike."""
    if split_of(pr):
        return "SPLIT", "the seed pairs disagree in sign, by ≥ SESOI/2 each way; no direction claimed"
    if lo > b and d >= NEGLIGIBLE:
        return "BETTER", ""
    if hi < -b and d <= -NEGLIGIBLE:
        return "WORSE", ""
    if lo > b or hi < -b:
        return "NEGLIGIBLE", "beyond the seed bar but |D| < NEGLIGIBLE"
    # SAME in the equivalence form (user, 25 Oct): the whole seed-noise interval |D| + b inside ±SESOI,
    # so a real SESOI effect is not called SAME (false SAME ≤ 1.8% in every regime, §C)
    inside = -b <= lo and hi <= b
    if inside and abs(d) + b <= SESOI:
        return "SAME", "CI inside ±bar; detectable-but-tiny" if (lo > 0 or hi < 0) else "SAME"
    why = "underpowered: bar > SESOI" if b > SESOI else (
        "not equivalent: |D| + bar > SESOI" if inside else (
            "large but not resolved" if abs(d) >= SESOI else "CI straddles the bar"))
    return "UNRESOLVED", why


def decide(auc: dict, boot: dict, answers: list[str], w: np.ndarray, pairs=PAIRS) -> dict:
    """auc[e][j] float, boot[e][j] (B,) for every encoder in `pairs`. First-match state table (§C)."""
    if K_FAM is None or Z_ANS is None:
        raise SystemExit("decide: K_FAM / Z_ANS unset — run `calibrate` and fix them first")
    encs = [e for p in pairs for e in p]
    n = len(pairs)
    for e in encs:
        if sum(j in auc[e] for j in answers) < MIN_ANSWERS:
            return {"state": "VOID", "why": f"{e}: fewer than {MIN_ANSWERS} answers scoreable"}
    have = [j for j in answers if all(j in auc[e] and j in boot[e] for e in encs)]
    ix = [answers.index(j) for j in have]
    wj, J = w[ix], len(have)
    P = {e: np.array([auc[e][j] for j in have]) for e in encs}
    Bt = {e: np.stack([boot[e][j] for j in have], 1) for e in encs}  # (B, J)
    d_s = [P[x] - P[a] for a, x in pairs]                              # per seed pair, (J,)
    Dj = np.mean(d_s, 0)
    D = float(Dj.mean())
    Ds = [float(d.mean()) for d in d_s]
    bDj = np.mean([Bt[x] - Bt[a] for a, x in pairs], 0)
    bD = bDj.mean(1)
    a = (100 - CI_FAM) / 2
    lo, hi = np.percentile(bD, [a, 100 - a])
    Am = np.stack([P[a] for a, _ in pairs])
    Xm = np.stack([P[x] for _, x in pairs])
    S_A, S_X, vA, vX = (float(v) for v in spreads(Am, Xm, wj))
    bar = K_FAM * max(S_A, S_X) * math.sqrt(2 / n)
    sj = np.sqrt(wj * (vA + vX) / n)
    barj = np.sqrt(bar ** 2 + (Z_ANS * sj) ** 2)
    flags = []
    run_ratio = max(S_A, S_X) / max(min(S_A, S_X), 1e-12)
    idio_ratio = math.sqrt(max(vA, vX) / max(min(vA, vX), 1e-24))
    if run_ratio > SPREAD_FLAG or idio_ratio > SPREAD_FLAG:
        flags.append(f"SPREADS DIFFER (run {run_ratio:.2f}×, answer-level {idio_ratio:.2f}×)")

    st, why = state(D, lo, hi, bar, Ds)
    a2 = (100 - CI_ANS) / 2
    per = {}
    for i, j in enumerate(have):
        l, h = np.percentile(bDj[:, i], [a2, 100 - a2])
        pr = [float(d[i]) for d in d_s]
        s, y = state(float(Dj[i]), float(l), float(h), float(barj[i]), pr)
        per[j] = {"D": float(Dj[i]), "ci": [float(l), float(h)], "bar": float(barj[i]), "state": s,
                  "why": y, "pairs": pr}
    return {"state": st, "why": why, "D": D, "ci": [float(lo), float(hi)], "bar": bar,
            "S_run": [S_A, S_X], "pairs": Ds, "idio_sd": [math.sqrt(vA), math.sqrt(vX)],
            "n_pairs": n, "n_answers": J, "flags": flags, "answers": per}


TRAJ_ORDER = ("VOID", "CROSSES", "CONSISTENT", "DIVERGES", "CONVERGES", "MIXED")


def trajectory(states: list[str]) -> str:
    """Secondary: the family states at 0.5, 1, 2, 4 epochs -> one trajectory state (first match)."""
    if "VOID" in states:
        return "VOID"
    if "BETTER" in states and "WORSE" in states:
        return "CROSSES"
    if len(set(states)) == 1:
        return "CONSISTENT"
    last, early = states[-1], states[:-1]
    if last in ("BETTER", "WORSE") and all(s not in ("BETTER", "WORSE") for s in early[:2]):
        return "DIVERGES"
    if last not in ("BETTER", "WORSE") and any(s in ("BETTER", "WORSE") for s in early[:2]):
        return "CONVERGES"
    return "MIXED"


# --- calibrate: decision-layer Monte Carlo (no scores, no bootstrap) ---------------------------
REGIMES = [  # τ (run), σ (answer), X/A spread ratio, answer-noise shape
    (0.006, 0.006, 1, "se"), (0.006, 0.0, 1, "se"), (0.0, 0.006, 1, "se"), (0.002, 0.008, 1, "se"),
    (0.004, 0.006, 1, "lognormal"), (0.002, 0.004, 1, "se"), (0.001, 0.003, 1, "se"),
    (0.004, 0.004, 3, "se"),
]
GRID = np.round(np.arange(1.0, 12.0001, 0.05), 2)


def _draw(rng, R, n, w, tau, sig, ratio, het, delta):
    J = len(w)
    sj = sig * np.sqrt(w) if het == "se" else sig * np.exp(rng.normal(0, 0.7, (R, 1, J)) - 0.245)
    A = rng.normal(0, tau, (R, n, 1)) + rng.normal(0, 1, (R, n, J)) * sj
    X = rng.normal(0, tau * ratio, (R, n, 1)) + rng.normal(0, 1, (R, n, J)) * sj * ratio + delta
    return A, X


def _stats(A, X, w):
    n = A.shape[1]
    S_A, S_X, vA, vX = spreads(A, X, w)
    se_fam = np.maximum(S_A, S_X) * math.sqrt(2 / n)                 # (R,)
    sj = np.sqrt(np.multiply.outer((vA + vX) / n, w))                # (R, J)
    dij = X - A                                                      # (R, n, J)
    Dj, Ds = dij.mean(1), dij.mean(2)                                # (R, J), (R, n)
    D = Dj.mean(1)
    split_f = (Ds.max(1) >= SESOI / 2) & (Ds.min(1) <= -SESOI / 2)
    split_j = (dij.max(1) >= SESOI / 2) & (dij.min(1) <= -SESOI / 2)
    return se_fam, sj, D, Dj, split_f, split_j


def _fam_rate(st, k, sign=0):
    se_fam, _, D, _, split_f, _ = st
    b = k * se_fam
    hit = (np.abs(D) > b) if sign == 0 else (sign * D > b) & (sign * D >= NEGLIGIBLE)
    return float((hit & ~split_f).mean())


def _ans_rate(st, k, z, sign=0):
    se_fam, sj, _, Dj, _, split_j = st
    bj = np.sqrt((k * se_fam)[:, None] ** 2 + (z * sj) ** 2)
    hit = (np.abs(Dj) > bj) if sign == 0 else (sign * Dj > bj) & (sign * Dj >= NEGLIGIBLE)
    return float((hit & ~split_j).mean())


def calibrate() -> dict:
    w = _w()
    rng = np.random.default_rng(20261024)
    R = 60_000
    out = {"R": R, "J": len(w), "rule": f"smallest K then Z on a 0.05 grid with false calls ≤ {FALSE_BOUND} "
           "in every equal-spread regime (ratio 1)", "designs": {}}
    for n in (2, 3):
        nulls = {r: _stats(*_draw(rng, R, n, w, *r, 0.0), w) for r in REGIMES}
        eq = [r for r in REGIMES if r[2] == 1]
        k = next(float(g) for g in GRID if all(_fam_rate(nulls[r], g) <= FALSE_BOUND for r in eq))
        z = next(float(g) for g in GRID if all(_ans_rate(nulls[r], k, g) <= FALSE_BOUND for r in eq))
        rows = []
        for r in REGIMES:
            st = nulls[r]
            row = {"tau": r[0], "sigma": r[1], "ratio": r[2], "het": r[3],
                   "false_fam": _fam_rate(st, k), "false_ans": _ans_rate(st, k, z),
                   "split_fam_null": float(st[4].mean()), "split_ans_null": float(st[5].mean()),
                   "median_bar": float(np.median(k * st[0]))}
            for d in (0.01, 0.02, 0.03):
                sd = _stats(*_draw(rng, R, n, w, *r, d), w)
                row[f"power_fam_{d}"] = _fam_rate(sd, k, +1)
                row[f"power_ans_{d}"] = _ans_rate(sd, k, z, +1)
            rows.append(row)
            print(json.dumps({"n": n, "K": k, "Z": z, **row}), flush=True)
        out["designs"][str(n)] = {"K": k, "Z": z, "rows": rows}
    return out


# --- plants: synthetic scores through the identical scoring path -------------------------------
def _labels():
    """The identical split's test labels. Called before anything imports aligned_c2: that import
    reads F0's cache at import time, and before baseline_run has set the v2 environment it reads v1's
    (the two-seed plants did, 23 Oct). The split's SHA-1 must equal the answer set's."""
    if "aligned_c2" in sys.modules or "j4_spread_controls" in sys.modules:
        raise SystemExit("plants: aligned_c2 / j4 imported before the v2 environment was set")
    import baseline_run  # noqa: F401  — A1's v2 environment, before F0 reads it
    os.environ.setdefault("LADDER_SPLIT", "identical")
    from j4_spread_controls import prepare
    s = prepare(None, 0, label="baselines-plants", sources=1)
    a = json.loads(ANSWER_SET.read_text())
    sha = hashlib.sha1(json.dumps([sorted(map(int, s.train_ids)), sorted(map(int, s.test_ids))])
                       .encode()).hexdigest()
    if sha != a["split_ids_sha1"]:
        raise SystemExit(f"plants: split SHA-1 {sha[:12]} is not the answer set's {a['split_ids_sha1'][:12]}")
    test = [int(o) for o in s.test_ids]
    ys, rows = {}, {}
    for f in a["answer_set"]:
        el = s.labels.eligible(f, test)
        ys[f], rows[f] = np.asarray(s.labels.binary_label(f, el), dtype=int), el
    return a["answer_set"], ys, rows, test, sha


def _realise(rng, answers, ys, cols, wt, base, tau, sig, delta, ratio=1.0, drop=None, rho=0.5):
    """One planted world: 6 encoders' scores per answer -> auc + boot under the shared weights."""
    from aligned_c2 import auc, boot_aucs
    from scipy.stats import norm
    wj = _w()
    encs = [e for p in PAIRS for e in p]
    c = {e: rng.normal(0, tau * (ratio if e[0] == "X" else 1.0)) for e in encs}
    A, Bt = {e: {} for e in encs}, {e: {} for e in encs}
    for i, f in enumerate(answers):
        y = ys[f]
        shared = rng.normal(size=len(y))  # galaxy noise common to all encoders (pairing in the bootstrap)
        for e in encs:
            if drop and e in drop and f in drop[e]:
                continue
            s_i = sig * math.sqrt(wj[i]) * (ratio if e[0] == "X" else 1.0)
            d = delta[e][i] if isinstance(delta[e], np.ndarray) else delta[e]
            t = float(np.clip(base[i] + c[e] + rng.normal(0, s_i) + d, 0.505, 0.995))
            mu = math.sqrt(2) * norm.ppf(t)
            sc = mu * y + math.sqrt(rho) * shared + math.sqrt(1 - rho) * rng.normal(size=len(y))
            A[e][f] = auc(sc, y)
            Bt[e][f] = boot_aucs(sc, y, wt[:, cols[f]])
    return A, Bt


def plants() -> dict:
    answers, ys, rows, test, sha = _labels()
    from aligned_c2 import weights
    wt, pos = weights(test)
    cols = {f: [pos[int(o)] for o in rows[f]] for f in answers}
    w = _w()
    J = len(answers)
    rng = np.random.default_rng(20261026)
    base = rng.uniform(0.70, 0.95, J)  # synthetic, fixed: no encoder's AUCs
    flat = lambda v: {e: v for p in PAIRS for e in p}  # noqa: E731
    on_x = lambda v: {e: (v if e[0] == "X" else 0.0) for p in PAIRS for e in p}  # noqa: E731
    some = np.zeros(J)
    some[:5] = 0.03
    N = dict(tau=0.002, sig=0.006)
    spec = [  # name, expect, realisations, kwargs — the expectations are fixed before the run
        ("null", "SAME", 10, dict(**N, delta=flat(0.0))),
        ("better_0.03", "BETTER", 10, dict(**N, delta=on_x(0.03))),
        ("worse_0.03", "WORSE", 10, dict(**N, delta=on_x(-0.03))),
        ("split", "SPLIT", 10, dict(**N, delta={**on_x(0.0), "X0": 0.03, "X1": -0.03})),
        ("negligible", "NEGLIGIBLE", 10, dict(tau=0.0003, sig=0.0005, delta=on_x(0.004), rho=0.9)),
        ("underpowered", "UNRESOLVED or SPLIT", 10, dict(tau=0.015, sig=0.006, delta=flat(0.0))),
        ("void", "VOID", 3, dict(**N, delta=flat(0.0), drop={"X2": set(answers[:6])})),
        ("extreme", "BETTER", 5, dict(**N, delta=on_x(0.15))),
        ("spreads_differ_null", "not BETTER/WORSE", 10, dict(tau=0.002, sig=0.006, ratio=3.0, delta=flat(0.0))),
        ("five_answers_0.03", "per-answer: 5 BETTER", 10, dict(**N, delta={**on_x(0.0), "X0": some, "X1": some, "X2": some})),
        # 25 Oct, the equivalence-form SAME: how often SAME is wrongly called on a real SESOI effect, and
        # how often it is reached at a true null, at the seed noise where the old SAME failed (τ 0.006)
        ("real_0.02", "not SAME", 20, dict(tau=0.006, sig=0.006, delta=on_x(0.02))),
        ("null_tau0.006", "SAME reachable", 20, dict(tau=0.006, sig=0.006, delta=flat(0.0))),
    ]
    out = {"pairs": PAIRS, "K_FAM": K_FAM, "Z_ANS": Z_ANS, "split_ids_sha1": sha,
           "base_auc_draw": "U(0.70, 0.95), seed 20261026", "same_rule": "equivalence: CI inside ±b and |D| + b <= SESOI", "noise": N, "rho": 0.5, "plants": {}}
    for name, expect, R, kw in spec:
        t0 = time.time()
        res = []
        for _ in range(R):
            A, Bt = _realise(rng, answers, ys, cols, wt, base, **kw)
            r = decide(A, Bt, answers, w)
            res.append({"state": r["state"], "D": r.get("D"), "bar": r.get("bar"), "flags": r.get("flags"),
                        "answer_states": {j: v["state"] for j, v in r.get("answers", {}).items()}})
        fam = {s: sum(x["state"] == s for x in res) for s in sorted({x["state"] for x in res})}
        ans = {}
        for x in res:
            for j, s in x["answer_states"].items():
                ans.setdefault("planted" if (name == "five_answers_0.03" and j in answers[:5]) else "rest", []).append(s)
        ans_rates = {g: {s: round(v.count(s) / len(v), 3) for s in sorted(set(v))} for g, v in ans.items()}
        bars_ = [x["bar"] for x in res if x["bar"] is not None]
        out["plants"][name] = {"expect": expect, "R": R, "family": fam, "answers": ans_rates,
                               "median_bar": float(np.median(bars_)) if bars_ else None,
                               "seconds": round(time.time() - t0), "runs": res}
        print(f"{name:22s} expect {expect:24s} family {fam}  answers {ans_rates}  "
              f"median bar {out['plants'][name]['median_bar']}", flush=True)
    seqs = {"CONSISTENT": (0.03, 0.03, 0.03, 0.03), "CONVERGES": (0.03, 0.03, 0.0, 0.0),
            "DIVERGES": (0.0, 0.0, 0.03, 0.03), "CROSSES": (0.03, 0.03, -0.03, -0.03)}
    out["trajectory"] = {}
    for name, ds in seqs.items():
        got = []
        for _ in range(3):
            st = []
            for d in ds:
                A, Bt = _realise(rng, answers, ys, cols, wt, base, tau=0.0003, sig=0.002, delta=on_x(d))
                st.append(decide(A, Bt, answers, w)["state"])
            got.append({"states": st, "trajectory": trajectory(st)})
        out["trajectory"][name] = got
        print(f"trajectory {name:11s} -> {[g['trajectory'] for g in got]}  {[g['states'] for g in got]}", flush=True)
    return out


# --- §N: the nuisance readout -------------------------------------------------------------------
NUISANCE = {"psf": "psfWidth_r", "brightness": "modelMag_r", "sky": "sky_r"}
N_NAMES = {"BETTER": "MORE", "WORSE": "LESS"}  # X carries more / less of the nuisance than the JEPA


def decide_nuisance(auc: dict, boot: dict, variables=tuple(NUISANCE), pairs=PAIRS) -> dict:
    """auc[e][v], boot[e][v] for every encoder -> one §N state per variable (the §C table, K, 97.5%)."""
    n = len(pairs)
    out = {}
    for v in variables:
        if any(v not in auc[e] or v not in boot[e] for p in pairs for e in p):
            out[v] = {"state": "VOID", "why": "an encoder lacks this variable"}
            continue
        pr = [float(auc[x][v] - auc[a][v]) for a, x in pairs]
        d = float(np.mean(pr))
        bd = np.mean([boot[x][v] - boot[a][v] for a, x in pairs], 0)
        a_ = (100 - CI_FAM) / 2
        lo, hi = (float(q) for q in np.percentile(bd, [a_, 100 - a_]))
        S_A = float(np.std([auc[a][v] for a, _ in pairs], ddof=1))
        S_X = float(np.std([auc[x][v] for _, x in pairs], ddof=1))
        b = K_FAM * max(S_A, S_X) * math.sqrt(2 / n)
        st, why = state(d, lo, hi, b, pr)
        out[v] = {"state": N_NAMES.get(st, st), "why": why, "D": d, "ci": [lo, hi], "bar": b,
                  "S_run": [S_A, S_X], "pairs": pr}
    return out


def nplants() -> dict:
    """§N's plants: synthetic scores on the median-split labels of the pulled nuisance values for the
    identical split's test galaxies, through auc / boot_aucs / weights, then decide_nuisance()."""
    import pandas as pd
    answers, ys, rows, test, sha = _labels()
    from aligned_c2 import auc, boot_aucs, weights
    from scipy.stats import norm
    wt, pos = weights(test)
    rec = json.loads((HERE / "out" / "baselines_nuisance.json").read_text())
    csv = HERE / "out" / "baselines_nuisance.csv"
    if hashlib.sha1(csv.read_bytes()).hexdigest() != rec["output_sha1"]:
        raise SystemExit("nplants: baselines_nuisance.csv does not match its record")
    df = pd.read_csv(csv, dtype={"object_id": str}).set_index("object_id")
    labels = {}
    for v, col in NUISANCE.items():
        x = pd.to_numeric(df[col].reindex([str(o) for o in test]), errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & (x > -9000) & ((x > 0) if v == "sky" else True)
        idx = np.flatnonzero(ok)
        xv = x[idx]
        labels[v] = ((xv >= np.nanmedian(xv)).astype(int), [pos[int(test[i])] for i in idx])
    rng = np.random.default_rng(20261027)
    base = dict(zip(NUISANCE, rng.uniform(0.60, 0.90, len(NUISANCE)), strict=True))
    encs = [e for p in PAIRS for e in p]

    def world(tau, delta, rho=0.5):
        A, Bt = {e: {} for e in encs}, {e: {} for e in encs}
        for v, (y, cols) in labels.items():
            shared = rng.normal(size=len(y))
            for e in encs:
                t = float(np.clip(base[v] + rng.normal(0, tau) + delta.get(e, 0.0), 0.505, 0.995))
                sc = math.sqrt(2) * norm.ppf(t) * y + math.sqrt(rho) * shared + math.sqrt(1 - rho) * rng.normal(size=len(y))
                A[e][v], Bt[e][v] = auc(sc, y), boot_aucs(sc, y, wt[:, cols])
        return A, Bt

    X = lambda d: {e: d for e in encs if e[0] == "X"}  # noqa: E731
    spec = [("null", "not MORE/LESS", 0.004, {}), ("more_0.03", "MORE", 0.004, X(0.03)),
            ("less_0.03", "LESS", 0.004, X(-0.03)), ("split", "SPLIT", 0.004, {"X0": 0.03, "X1": -0.03})]
    out = {"split_ids_sha1": sha, "nuisance_output_sha1": rec["output_sha1"], "K_FAM": K_FAM,
           "n_test_per_variable": {v: len(y) for v, (y, _) in labels.items()},
           "base_auc_draw": "U(0.60, 0.90), seed 20261027", "plants": {}}
    for name, expect, tau, delta in spec:
        got = [decide_nuisance(*world(tau, delta)) for _ in range(10)]
        cnt = {}
        for g in got:
            for v, r in g.items():
                cnt[r["state"]] = cnt.get(r["state"], 0) + 1
        out["plants"][name] = {"expect": expect, "tau": tau, "R": 10, "readings": cnt, "runs": got}
        print(f"N {name:10s} expect {expect:14s} readings (10 × 3 variables) {cnt}", flush=True)
    return out


def score() -> None:
    raise SystemExit("score: refused — baselines_comparison.md is not hashed (D27). Nothing is read.")


if __name__ == "__main__":
    mode = sys.argv[1]
    OUTP = HERE / "out"
    if mode == "calibrate":
        (OUTP / "baselines_calibration_3seed.json").write_text(json.dumps(calibrate(), indent=1))
    elif mode == "plants":
        (OUTP / "baselines_plants_3seed_equiv.json").write_text(json.dumps(plants(), indent=1, default=list))
    elif mode == "nplants":
        (OUTP / "baselines_nplants_3seed_equiv.json").write_text(json.dumps(nplants(), indent=1, default=list))
    elif mode == "score":
        score()
