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
PREREG = HERE / "baselines_comparison.md"
PREREG_SHA1: str | None = "b021e8688970f684ad429babfa44a2fcebe88a9c"  # hashed 2026-10-27; `score` refuses unless it matches


def prereg_section() -> str:
    """From '## Pre-registration' to the next level-2 heading, trailing newline (as aligned_comparison)."""
    lines = PREREG.read_text().splitlines()
    a = lines.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip() + "\n"


def assert_hashed() -> None:
    if PREREG_SHA1 is None:
        raise SystemExit("baselines_comparison: the pre-registration is not hashed; nothing is scored")
    got = hashlib.sha1(prereg_section().encode()).hexdigest()
    if got != PREREG_SHA1:
        raise SystemExit(f"baselines_comparison: pre-registration changed since the hash ({got} != {PREREG_SHA1})")


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


# --- score: the hashed rules on the real inputs (after the hash, and only on the user's go) ----------
# The helpers take their paths as parameters, so tests drive them on synthetic files and read no real one;
# score() itself wires the real paths and is exercised only through its refusals.
STEPS = (12_664, 25_327, 50_654, 101_308)  # 0.5, 1, 2, 4 epochs (Common ground)
KINDS = ("moco", "mae")
P3_MANIFESTS = ("baselines_p3.sha1", "baselines_p3_s2.sha1")  # hashed unread (27 Oct; 4 Nov, third pair)
P1_ROOT = HERE.parent / "runs" / "_p1"
REPORTED_NUISANCE = {"snr": "snr_r", "redshift": "specz", "size": "petroRad_r"}  # §N: reported, no state


class P3Mismatch(SystemExit):
    """P3: a file or a refitted AUC differs from its record. Scoring stops; that epoch is VOID."""


def traj_tag(role: str, kind: str, seed: int, k: int) -> str:
    """The trajectory probe's tag: A side a{s+1}_traj{k} (P2), baselines {kind}_s{s}{k} (baseline_run)."""
    return f"a{seed + 1}_traj{k}" if role == "A" else f"{kind}_s{seed}{k}"


def encoders_for(kind: str) -> dict[str, tuple[str, int]]:
    """PAIRS' names -> (role, seed): A1..A3 are the JEPA seeds, X0..X2 this objective's."""
    return {**{a: ("A", s) for s, (a, _) in enumerate(PAIRS)}, **{x: ("X", s) for s, (_, x) in enumerate(PAIRS)}}


def needed_files(out_dir: Path, p1_fail: tuple = ()) -> list[Path]:
    """Every file scoring opens. A run whose P1 fails has no valid 0.5–2 epoch points: not required."""
    fs = []
    for k, step in enumerate(STEPS, 1):
        for s in range(len(PAIRS)):
            a_side = [] if (PAIRS[s][0] in p1_fail and step != STEPS[-1]) else [traj_tag("A", "", s, k)]
            for t in {*a_side, *(traj_tag("X", kd, s, k) for kd in KINDS)}:
                fs += [out_dir / f"{t}_trajectory.json", out_dir / f"c2_{t}_{step:09d}_embeddings.npz"]
    return sorted(set(fs))


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


N_PREFIXES = 16  # baselines.md §6.1: four runs × four epochs


def p3_hashes(out_dir: Path, manifests: list[Path], table_md: Path | None = None, p1_fail: tuple = ()) -> dict:
    """P3, files: every trajectory file and bank scoring will open matches its recorded SHA-1 (the
    manifests hashed unread; for the first two baseline pairs also `baselines.md` §6.1's prefixes)."""
    rec = {}
    for m in manifests:
        for ln in m.read_text().splitlines():
            if ln and not ln.startswith("#"):
                sha, path = ln.split(None, 1)
                if Path(path).name in rec:
                    raise P3Mismatch(f"P3: {Path(path).name} is listed twice across the manifests")
                rec[Path(path).name] = sha
    prefix = {}
    if table_md is not None:  # | `moco_s0` | `f704…` | … |, i = 1..4
        import re
        for m_ in re.finditer(r"^\| `((?:moco|mae)_s\d)` \|(.*)\|$", table_md.read_text(), re.M):
            for k, cell in enumerate(re.findall(r"`([0-9a-f]{12})`", m_.group(2)), 1):
                prefix[f"{m_.group(1)}{k}_trajectory.json"] = cell
        if len(prefix) != N_PREFIXES:
            raise P3Mismatch(f"P3: parsed {len(prefix)} §6.1 prefixes from {table_md.name}, not {N_PREFIXES}")
    bad = []
    files = needed_files(out_dir, p1_fail)
    for f in files:
        want = rec.get(f.name)
        if want is None:
            bad.append(f"{f.name}: no recorded SHA-1")
            continue
        got = _sha1(f) if f.exists() else None
        if got != want:
            bad.append(f"{f.name}: {'missing' if got is None else got[:12]} != recorded {want[:12]}")
        if f.name in prefix and not want.startswith(prefix[f.name]):
            bad.append(f"{f.name}: manifest {want[:12]} != baselines.md §6.1 {prefix[f.name]}")
    return {"checked": len(files), "mismatches": bad}


def p1_status(root: Path = P1_ROOT) -> dict[str, dict]:
    """P1 per A run: the identity resume and the end-to-end resume both pass the amended rule."""
    out = {}
    for s in range(len(PAIRS)):
        a = PAIRS[s][0]
        got = {}
        for part in ("identity", "e2e"):
            f = root / f"{a}_{part}" / "p1_record.json"
            if not f.exists():
                raise SystemExit(f"score: P1 record {f} is missing (wrong root or disk unmounted?)")
            got[part] = json.loads(f.read_text())["amended"]["passes"] is True
        out[a] = {**got, "passes": all(got.values())}
    return out


def refit(tag: str, step: int, out_dir: Path, answers: list[str], fit, wt, pos) -> tuple[dict, dict]:
    """P3, values: refit from the bank (`fit` = aligned_c2.fit_scores, the probe's own call); every
    refitted held-out AUC must equal the trajectory file's exactly. Returns auc[j], boot[j]."""
    from aligned_c2 import auc as _auc
    from aligned_c2 import boot_aucs
    traj = json.loads((out_dir / f"{tag}_trajectory.json").read_text())
    rec = next((c for c in traj["checkpoints"] if c["step"] == step), None)
    if rec is None:
        raise P3Mismatch(f"P3: {tag}_trajectory.json has no step {step}")
    bank = f"{tag}_{step:09d}"
    sc, meta, test_ids = fit([bank])
    if [int(o) for o in test_ids] != list(pos):
        raise P3Mismatch(f"P3: {bank}'s test split is not the shared weights' split")
    filed = {j for j in answers if j in rec["answers"]}
    if filed != {j for j in answers if j in sc[bank]}:
        raise P3Mismatch(f"P3: {bank}: the refit scores a different answer set from the trajectory file")
    A, B = {}, {}
    for j in answers:
        if j not in sc[bank]:
            continue  # not scoreable for this encoder: decide() counts it against MIN_ANSWERS
        y = meta["y"][j]
        a = _auc(sc[bank][j], y)
        want = rec["answers"].get(j, {}).get("auc")
        if want is None or a != want:
            raise P3Mismatch(f"P3: {bank} {j}: refitted AUC differs from the trajectory file")
        A[j] = a
        B[j] = boot_aucs(sc[bank][j], y, wt[:, [pos[int(o)] for o in meta["rows"][j]]])
    return A, B


def score_core(tables: dict, answers: list[str], w: np.ndarray, p1: dict) -> dict:
    """tables[kind][epoch] = (auc, boot) keyed by A1..A3, X0..X2, as decide() takes them; plus
    tables["moco_vs_mae"][4.0] with the JEPA slot taken by MoCo. P1: a failed A run voids 0.5–2 ep."""
    p1_fail = [a for a, r in p1.items() if not r["passes"]]
    out = {}
    for kind in KINDS:
        per = {}
        for ep in EPOCHS:
            if ep != PRIMARY and p1_fail:
                per[str(ep)] = {"state": "VOID", "why": f"P1 failed for {', '.join(p1_fail)}"}
                continue
            if ep not in tables[kind]:
                per[str(ep)] = {"state": "VOID", "why": "P3: scoring stopped before this epoch"}
                continue
            per[str(ep)] = decide(*tables[kind][ep], answers, w)
            if ep != PRIMARY:  # §B: per-answer states at 4 epochs only; earlier, D_j and its CI reported
                for v in per[str(ep)].get("answers", {}).values():
                    v.pop("state", None)
                    v.pop("why", None)
        out[kind] = {"primary": per[str(PRIMARY)], "epochs": per,
                     "trajectory": trajectory([per[str(e)]["state"] for e in EPOCHS])}
    if "moco_vs_mae" in tables:
        out["moco_vs_mae"] = {}
        for ep, tb in tables["moco_vs_mae"].items():
            r = decide(*tb, answers, w)
            for v in [r, *r.get("answers", {}).values()]:  # reported, no state (§B)
                v.pop("state", None)
                v.pop("why", None)
            out["moco_vs_mae"][str(ep)] = r
    return out


def all_void(why: str) -> dict:
    """P3: "a mismatch stops scoring" — read literally: nothing is decided, every state is VOID, named."""
    v = {"state": "VOID", "why": why}
    return {"comparison": {k: {"primary": v, "epochs": {str(e): v for e in EPOCHS}, "trajectory": "VOID"}
                           for k in KINDS},
            "nuisance": {k: {"states": {n: v for n in NUISANCE}} for k in KINDS}}


def nuisance_labels(values: np.ndarray, var: str) -> tuple[np.ndarray, np.ndarray]:
    """§N: (mask of usable galaxies, 0/1 label at this split's own median, ≥ is 1). Sentinels
    (≤ −9000), non-positive sky and missing values are excluded, per variable."""
    v = np.asarray(values, float)
    ok = np.isfinite(v) & (v > -9000) & ((v > 0) if var == "sky" else True)
    y = (v[ok] >= np.nanmedian(v[ok])).astype(int)
    return ok, y


def nuisance_tables(banks: dict, train_ids, test_ids, values: dict, wt, pos, c: float) -> tuple[dict, dict]:
    """§N's statistic per encoder and variable: `_fit` on probe-train, AUC + shared bootstrap on probe-test.
    banks[e] = (ids, x) at 4 epochs; values[var] = (train values, test values) aligned to the ids."""
    from aligned_c2 import auc as _auc
    from aligned_c2 import boot_aucs
    from galaxy_jepa.probing.logistic import _fit
    from types import SimpleNamespace
    A, B = {e: {} for e in banks}, {e: {} for e in banks}
    for e, (ids, x) in banks.items():
        row = {int(o): k for k, o in enumerate(ids)}
        for var, (vtr, vte) in values.items():
            ktr, ytr = nuisance_labels(vtr, var)
            kte, yte = nuisance_labels(vte, var)
            tr_ids = np.asarray(train_ids)[ktr]
            te_ids = np.asarray(test_ids)[kte]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            tr = SimpleNamespace(x=x[[row[int(o)] for o in tr_ids]].astype(np.float64), y=ytr)
            sc, clf = _fit(tr, c=c)
            s = clf.decision_function(sc.transform(x[[row[int(o)] for o in te_ids]].astype(np.float64)))
            A[e][var] = _auc(s, yte)
            B[e][var] = boot_aucs(s, yte, wt[:, [pos[int(o)] for o in te_ids]])
    return A, B


def representation_shares(x_test: np.ndarray, fam: dict) -> dict:
    """§N, draft v1's criterion 2, reported with no state: the top-10 PC variance share tracking a
    nuisance family (brightness, observing) and the share tracking morphology, plus PR
    (aligned_c13.representation's machinery; offsets are gone on aligned stamps)."""
    from aligned_c13 import representation
    r = representation(x_test, fam)
    r["N"] = float(sum(p["share"] for p in r["pcs"] if p["tracks"] in ("brightness", "observing")))
    r.pop("offset_pc", None)
    return r


LADDER_ROOT = HERE.parent / "runs" / "ladder_v2"
LADDER_STATES = (HERE.parent / "docs/ladder_v2/state.json", HERE.parent / "docs/ladder_v2_baselines/state.json")
LADDER_FILES = ("p2_ladder.json", "s2_ladder.json", "t1_ladder.json")
LADDER_GROUPS = {"JEPA": ("a1", "a2", "a3"), "MoCo": ("moco_s0", "moco_s1", "moco_s2"),
                 "MAE": ("mae_s0", "mae_s1", "mae_s2")}
N_FEATURES = 37  # every Scheme-1 answer gets a verdict in both populations (run_ladder)


def ladder_table(root: Path = LADDER_ROOT, states: tuple = LADDER_STATES, groups: dict = LADDER_GROUPS) -> dict:
    """§D, "how it is read": each encoder's own ladder states per answer and population, beside the
    others', with an agreement count per objective (3/3, or 2/1 naming the seed that differs). **No
    comparative state** is formed across objectives. Refuses unless both ladders have finished with
    every entry done, and every file read matches the runner's recorded SHA-1."""
    recorded = {}
    for st in states:
        d = json.loads(Path(st).read_text())
        bad = {k: v["status"] for k, v in d["entries"].items() if v["status"] != "done"}
        if bad:
            raise SystemExit(f"ladder: {st} has entries not done: {sorted(bad.items())[:5]}")
        recorded.update({str(Path(p_).resolve()): v[0] for p_, v in d["files"].items()})
    encs = [e for g in groups.values() for e in g]
    rec = {}
    for e in encs:
        for f in LADDER_FILES:
            p_ = (Path(root) / e / f).resolve()
            if recorded.get(str(p_)) != _sha1(p_):
                raise SystemExit(f"ladder: {e}/{f} does not match the runner's recorded SHA-1")
        rec[e] = {f: json.loads((Path(root) / e / f).read_text()) for f in LADDER_FILES}
        want = f"{e}.pt" if e not in groups["JEPA"] else None
        if want and not str(rec[e]["t1_ladder.json"]["checkpoint"]).endswith(want):
            raise SystemExit(f"ladder: {e}'s t1 record names checkpoint {rec[e]['t1_ladder.json']['checkpoint']}")
    out: dict = {"encoders": encs, "groups": groups, "populations": {}}
    for pop in ("full", "conditional"):
        rows: dict = {}
        for e in encs:
            got = {f: {r["feature"]: r for r in rec[e][f][pop]} for f in LADDER_FILES}
            for f in LADDER_FILES:
                if len(got[f]) != N_FEATURES:
                    raise SystemExit(f"ladder: {e}/{f} {pop} has {len(got[f])} features, not {N_FEATURES}")
            for feat, t1 in got["t1_ladder.json"].items():
                rungs = {f: got[f][feat]["rung"] for f in LADDER_FILES}
                rows.setdefault(feat, {})[e] = {
                    "rung": t1["rung"], "mechanism": t1["mechanism"], "retention": t1.get("retention"),
                    "exceeds_null": t1.get("exceeds_null"), "clean": t1.get("clean"),
                    "underpowered": t1.get("underpowered"),
                    "files_agree": len(set(rungs.values())) == 1, "rungs_by_file": rungs}
        for feat, by in rows.items():
            agree = {}
            for g, members in groups.items():
                vals = [by[m]["rung"] for m in members]
                top = max(sorted(set(vals)), key=vals.count)
                if vals.count(top) == 1:
                    agree[g] = f"0/{len(members)} (all differ)"
                    continue
                odd = [m for m, v in zip(members, vals, strict=True) if v != top]
                agree[g] = f"{len(members) - len(odd)}/{len(odd)} ({', '.join(odd)} differs)" if odd else f"{len(members)}/{len(members)}"
            by["agreement"] = agree
        out["populations"][pop] = rows
    return out


NUISANCE_CSV_SHA1 = "6bea1443"  # the pull's output SHA-1 prefix as recorded beside the hash ("Nuisance pull")
SCORE_OUT = "baselines_score.json"


def score(go: bool = False, out_dir: Path = HERE / "out") -> dict:
    """The real scoring. Refuses without the hash, without `--go` (the user's word, after the ladder), with
    a ladder environment left set, or before both ladders have finished. Any P3 mismatch (a file or a
    refitted value) stops scoring: a record is written with every state VOID and the mismatch named."""
    assert_hashed()
    if not go:
        raise SystemExit("score: refuses without --go (user, 4 Nov: no real scoring until the ladder has "
                         "finished and the user says go). Nothing is read.")
    if "aligned_c2" in sys.modules or "j4_spread_controls" in sys.modules:
        raise SystemExit("score: aligned_c2 / j4 imported before the v2 environment was set (F0 would read v1)")
    import ladder_env
    if ladder_env.active():
        raise SystemExit("score: LADDER_ENCODER / LADDER_OUT are set; unset them (the banks live in artifacts/out)")
    import baseline_run  # noqa: F401  — A1's v2 environment, before F0 reads it (the import-order trap)
    os.environ.setdefault("LADDER_SPLIT", "identical")
    from j4_spread_controls import OUT
    if OUT.resolve() != Path(out_dir).resolve():
        raise SystemExit(f"score: the banks would be read from {OUT}, not the hashed {out_dir}")
    ladder = ladder_table()  # refuses unless both ladders finished and every file matches: before any refit
    p1 = p1_status()
    p1_fail = tuple(a for a, r in p1.items() if not r["passes"])
    res = {"prereg_sha1": PREREG_SHA1, "p1": p1, "reads": "trajectory files (smoke: trajectories only) and banks"}

    def stop(why: str) -> dict:
        res.update(all_void(why), p3_stop=why)
        (Path(out_dir) / SCORE_OUT).write_text(json.dumps(res, indent=1, default=float))
        raise P3Mismatch(f"score: P3 stop, every state VOID ({why}); record {SCORE_OUT}")

    try:
        h = p3_hashes(out_dir, [HERE / m for m in P3_MANIFESTS], HERE / "baselines.md", p1_fail)
    except P3Mismatch as e:
        stop(str(e))
    res["p3_files"] = h
    if h["mismatches"]:
        stop("P3 files: " + "; ".join(h["mismatches"]))
    from aligned_c2 import fit_scores, weights
    from j4_spread_controls import prepare
    sp = prepare(None, 0, label="baselines-score", sources=1)
    a = json.loads(ANSWER_SET.read_text())
    sha = hashlib.sha1(json.dumps([sorted(map(int, sp.train_ids)), sorted(map(int, sp.test_ids))]).encode()).hexdigest()
    if sha != a["split_ids_sha1"]:
        raise SystemExit(f"score: split SHA-1 {sha[:12]} is not the answer set's {a['split_ids_sha1'][:12]}")
    res["split_ids_sha1"] = sha
    answers, w = a["answer_set"], _w()
    wt, pos = weights(list(sp.test_ids))
    tables: dict = {k: {} for k in (*KINDS, "moco_vs_mae")}
    cache: dict = {}
    try:
        for k, (ep, step) in enumerate(zip(EPOCHS, STEPS, strict=True), 1):
            if ep != PRIMARY and p1_fail:
                continue  # VOID by P1 (score_core names it); those A files were not required
            for kind in KINDS:
                A, B = {}, {}
                for name, (role, seed) in encoders_for(kind).items():
                    t = traj_tag(role, kind, seed, k)
                    if t not in cache:
                        cache[t] = refit(t, step, out_dir, answers, fit_scores, wt, pos)
                    A[name], B[name] = cache[t]
                tables[kind][ep] = (A, B)
            A, B = {}, {}  # MoCo against MAE: X = MAE, the JEPA slot taken by MoCo (reported, no state)
            for sd, (a_, x_) in enumerate(PAIRS):
                A[a_], B[a_] = cache[traj_tag("X", "moco", sd, k)]
                A[x_], B[x_] = cache[traj_tag("X", "mae", sd, k)]
            tables["moco_vs_mae"][ep] = (A, B)
    except P3Mismatch as e:
        stop(str(e))
    del cache
    res["comparison"] = score_core(tables, answers, w, p1)
    res["nuisance"] = _score_nuisance(sp, out_dir, wt, pos)
    res["ladder"] = ladder
    (Path(out_dir) / SCORE_OUT).write_text(json.dumps(res, indent=1, default=float))
    return res


def _score_nuisance(s, out_dir: Path, wt, pos) -> dict:
    """§N at 4 epochs on the same banks, one encoder in memory at a time: three variables with states,
    three reported, and draft v1's criterion 2 reported."""
    import pandas as pd
    from aligned_c2 import _bank
    from aligned_c13 import _r_flux
    from probe_bank import load_bank
    rec = json.loads((Path(out_dir) / "baselines_nuisance.json").read_text())
    csv = Path(out_dir) / "baselines_nuisance.csv"
    got = _sha1(csv)
    if got != rec["output_sha1"] or not got.startswith(NUISANCE_CSV_SHA1):
        raise SystemExit("score: baselines_nuisance.csv does not match its record / the hashed prefix")
    sky = pd.read_csv(csv, dtype={"object_id": str}).set_index("object_id")["sky_r"]
    meta = pd.read_csv(HERE.parent / "data/probe_v2/metadata.csv", dtype={"object_id": str}).set_index("object_id")
    te = [int(o) for o in s.test_ids]
    if te != list(pos):
        raise SystemExit("score: the nuisance split is not the shared weights' split")
    tr = [int(o) for o in s.train_ids]

    def col(series, ids):
        return pd.to_numeric(series.reindex([str(o) for o in ids]), errors="coerce").to_numpy(float)
    cols = {**{v: (meta[c] if v != "sky" else sky) for v, c in NUISANCE.items()},
            **{v: meta[c] for v, c in REPORTED_NUISANCE.items()}}
    values = {v: (col(sr, tr), col(sr, te)) for v, sr in cols.items()}
    coverage = {}
    for v, (vtr, vte) in values.items():  # a failed join reads as all-NaN: refuse rather than VOID quietly
        ntr, nte = int(nuisance_labels(vtr, v)[0].sum()), int(nuisance_labels(vte, v)[0].sum())
        if ntr == 0 or nte == 0:
            raise SystemExit(f"score: §N variable {v} has no usable values (train {ntr}, test {nte}): a join failed")
        coverage[v] = {"train": ntr, "test": nte}
    fam = {"brightness": [values["brightness"][1], _r_flux({"test": te})],
           "observing": [values["psf"][1], values["sky"][1]], "morphology": []}
    for f in s.labels.features:
        el = set(s.labels.eligible(f, te))
        v = s.labels.vote_fraction(f, te).astype(float)
        v[[i for i, o in enumerate(te) if o not in el]] = np.nan
        fam["morphology"].append(v)
    k4 = len(STEPS)
    A, B, crit2 = {}, {}, {}
    tags = sorted({traj_tag(role, kind, seed, k4) for kind in KINDS for role, seed in encoders_for(kind).values()})
    for t in tags:
        b = load_bank(_bank(f"{t}_{STEPS[-1]:09d}"))
        ids, x = b["ids"].astype(np.int64), b["x"]
        a_, b_ = nuisance_tables({t: (ids, x)}, tr, te, values, wt, pos, s.pc.c)
        A[t], B[t] = a_[t], b_[t]
        row = {int(o): i for i, o in enumerate(ids)}
        crit2[t] = representation_shares(x[[row[o] for o in te]].astype(np.float64), fam)
        del b, x
    out: dict = {"coverage": coverage, "criterion2": crit2}
    for kind in KINDS:
        m = {name: traj_tag(role, kind, seed, k4) for name, (role, seed) in encoders_for(kind).items()}
        Ak, Bk = {n: A[t] for n, t in m.items()}, {n: B[t] for n, t in m.items()}
        rep = decide_nuisance(Ak, Bk, tuple(REPORTED_NUISANCE))
        for r in rep.values():
            r.pop("state", None)
            r.pop("why", None)
        out[kind] = {"states": decide_nuisance(Ak, Bk, tuple(NUISANCE)), "reported": rep}
    return out


def score_plants() -> dict:
    """The plants through score_core (the scoring's own path from tables to states): synthetic scores
    on the real labels, four epochs, both objectives; expectations fixed before the run."""
    answers, ys, rows, test, sha = _labels()
    from aligned_c2 import weights
    wt, pos = weights(test)
    cols = {f: [pos[int(o)] for o in rows[f]] for f in answers}
    w = _w()
    rng = np.random.default_rng(20261104)
    base = rng.uniform(0.70, 0.95, len(answers))
    on_x = lambda v: {e: (v if e[0] == "X" else 0.0) for p in PAIRS for e in p}  # noqa: E731
    ok = {a: {"identity": True, "e2e": True, "passes": True} for a, _ in PAIRS}
    spec = [  # name, moco Δ per epoch, mae Δ per epoch, P1, expectations (fixed here, before the run)
        ("better_vs_same", (0.03,) * 4, (0.0,) * 4, ok,
         {"moco": ("BETTER", "CONSISTENT"), "mae": ("SAME", "CONSISTENT")}),
        ("worse_crosses", (0.03, 0.03, -0.03, -0.03), (-0.03,) * 4, ok,
         {"moco": ("WORSE", "CROSSES"), "mae": ("WORSE", "CONSISTENT")}),
        ("p1_fails_A2", (0.03,) * 4, (0.0,) * 4, {**ok, "A2": {"identity": True, "e2e": False, "passes": False}},
         {"moco": ("BETTER", "VOID"), "mae": ("SAME", "VOID")}),
    ]
    out = {"split_ids_sha1": sha, "plants": {}}
    for name, dm, da, p1, expect in spec:
        runs = []
        for _ in range(3):
            tables = {k: {} for k in KINDS}
            for ep, a_, b_ in zip(EPOCHS, dm, da, strict=True):
                for kind, d in (("moco", a_), ("mae", b_)):
                    tables[kind][ep] = _realise(rng, answers, ys, cols, wt, base, tau=0.0003, sig=0.002, delta=on_x(d))
            r = score_core(tables, answers, w, p1)
            runs.append({k: (r[k]["primary"]["state"], r[k]["trajectory"]) for k in KINDS})
        hits = sum(all(tuple(x[k]) == expect[k] for k in KINDS) for x in runs)
        out["plants"][name] = {"expect": expect, "runs": runs, "fired": f"{hits}/3"}
        print(f"{name:16s} expect {expect}  fired {hits}/3  {runs}", flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    OUTP = HERE / "out"
    if mode == "calibrate":
        (OUTP / "baselines_calibration_3seed.json").write_text(json.dumps(calibrate(), indent=1))
    elif mode == "plants":
        (OUTP / "baselines_plants_3seed_equiv.json").write_text(json.dumps(plants(), indent=1, default=list))
    elif mode == "nplants":
        (OUTP / "baselines_nplants_3seed_equiv.json").write_text(json.dumps(nplants(), indent=1, default=list))
    elif mode == "scoreplants":
        (OUTP / "baselines_score_plants.json").write_text(json.dumps(score_plants(), indent=1, default=list))
    elif mode == "score":
        score(go="--go" in sys.argv[2:])  # writes artifacts/out/baselines_score.json
