"""D12's baselines against their paired JEPA seeds — the states, the plants, and (after the hash) the
scoring (`artifacts/baselines_comparison.md`).

One decision path, `decide()`, used identically by the plants and the real scoring. Per objective
X ∈ {MoCo, MAE}, per epoch point, four encoders: A1, A2 (JEPA, seeds 0 and 1) and X0, X1 (paired
with them: same init stream and data order). Inputs per encoder and answer: the held-out AUC and its
2,000 Poisson galaxy-bootstrap AUCs under weights shared across encoders and answers (`aligned_c2`:
`auc`, `boot_aucs`, `weights`) — the E2 protocol the trajectory probes ran.

    calibrate   decision-layer Monte Carlo: k and z against seed-noise nulls, power (no scores)
    plants      synthetic scores on the real labels through auc / boot_aucs / weights, then decide()
    score       (after the hash only) refit from the banks and decide; refuses without the hash

Plants use **no real result**: base AUCs, seed noise and effects are synthetic draws; only the
probe corpus's labels and the identical split's test ids are real (counts, not representations).

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

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
K_FAM = 3.5          # family bar = K_FAM × run-level spread (calibrated: ≤ 2.5% false calls per objective)
Z_ANS = 2.24         # per-answer idiosyncratic multiplier (≤ 2.5% false calls per answer)
CI_FAM = 97.5        # family CI level: 95% Bonferroni-split over the two objectives
CI_ANS = 95.0
NEGLIGIBLE = 0.005   # |effect| below this is negligible however significant
SESOI = 0.02         # smallest effect of interest: SAME needs a bar that could have seen it
MIN_ANSWERS = 30     # of the 35: fewer scoreable for any encoder -> VOID
SPREAD_FLAG = 2.0    # one arm's spread > 2× the other's -> flagged, reported (the bar already takes the max)
EPOCHS = (0.5, 1.0, 2.0, 4.0)
PRIMARY = 4.0

ANSWER_SET = HERE / "out" / "baselines_answer_set.json"


def _w() -> np.ndarray:
    a = json.loads(ANSWER_SET.read_text())
    se = np.array([a["answers"][f]["se_at_ref"] for f in a["answer_set"]])
    return se ** 2 / np.mean(se ** 2)


def decide(auc: dict, boot: dict, answers: list[str], w: np.ndarray) -> dict:
    """auc[e][j] float, boot[e][j] (B,) for e in A1 A2 X0 X1. First-match state table (§C)."""
    encs = ("A1", "A2", "X0", "X1")
    have = [j for j in answers if all(j in auc[e] and j in boot[e] for e in encs)]
    for e in encs:
        if sum(j in auc[e] for j in answers) < MIN_ANSWERS:
            return {"state": "VOID", "why": f"{e}: fewer than {MIN_ANSWERS} answers scoreable"}
    ix = [answers.index(j) for j in have]
    wj, J = w[ix], len(have)
    P = {e: np.array([auc[e][j] for j in have]) for e in encs}
    Bt = {e: np.stack([boot[e][j] for j in have], 1) for e in encs}  # (B, J)
    d_s = [P["X0"] - P["A1"], P["X1"] - P["A2"]]                     # paired by seed
    Dj = (d_s[0] + d_s[1]) / 2
    D = float(Dj.mean())
    Ds = [float(d.mean()) for d in d_s]
    bDj = ((Bt["X0"] - Bt["A1"]) + (Bt["X1"] - Bt["A2"])) / 2
    bD = bDj.mean(1)
    a = (100 - CI_FAM) / 2
    lo, hi = np.percentile(bD, [a, 100 - a])
    # run-level spread per arm: the seed difference of the family means (answer noise averages out)
    dA, dX = P["A1"] - P["A2"], P["X0"] - P["X1"]
    rA, rX = float(dA.mean()), float(dX.mean())
    s_run = max(abs(rA), abs(rX))
    bar = K_FAM * s_run
    # answer-level residual spread, pooled over answers, scaled by each answer's count-based SE
    sA2 = float(((dA - rA) ** 2 / wj).mean() / 2 * J / (J - 1))
    sX2 = float(((dX - rX) ** 2 / wj).mean() / 2 * J / (J - 1))
    sj = np.sqrt(wj * (sA2 + sX2) / 2)
    barj = np.sqrt(bar ** 2 + (Z_ANS * sj) ** 2)
    flags = []
    run_ratio = max(abs(rA), abs(rX)) / max(min(abs(rA), abs(rX)), 1e-12)
    idio_ratio = math.sqrt(max(sA2, sX2) / max(min(sA2, sX2), 1e-24))
    if run_ratio > SPREAD_FLAG or idio_ratio > SPREAD_FLAG:
        flags.append(f"SPREADS DIFFER (run {run_ratio:.2f}×, answer-level {idio_ratio:.2f}×)")

    def state(d: float, lo: float, hi: float, b: float, pair: list[float] | None) -> tuple[str, str]:
        if pair is not None and pair[0] * pair[1] < 0 and min(abs(pair[0]), abs(pair[1])) >= SESOI / 2:
            return "SPLIT", "the two seed pairs differ in sign, each by ≥ SESOI/2"
        if lo > b and d >= NEGLIGIBLE:
            return "BETTER", ""
        if hi < -b and d <= -NEGLIGIBLE:
            return "WORSE", ""
        if lo > b or hi < -b:
            return "NEGLIGIBLE", "beyond the seed bar but |D| < NEGLIGIBLE"
        if -b <= lo and hi <= b and b <= SESOI:
            return "SAME", "CI inside ±bar; detectable-but-tiny" if (lo > 0 or hi < 0) else "SAME"
        why = "underpowered: bar > SESOI" if b > SESOI else (
            "large but not resolved" if abs(d) >= SESOI else "CI straddles the bar")
        return "UNRESOLVED", why

    st, why = state(D, lo, hi, bar, Ds)
    a2 = (100 - CI_ANS) / 2
    per = {}
    for i, j in enumerate(have):
        l, h = np.percentile(bDj[:, i], [a2, 100 - a2])
        s, y = state(float(Dj[i]), float(l), float(h), float(barj[i]), [float(d_s[0][i]), float(d_s[1][i])])
        per[j] = {"D": float(Dj[i]), "ci": [float(l), float(h)], "bar": float(barj[i]), "state": s, "why": y,
                  "pairs": [float(d_s[0][i]), float(d_s[1][i])]}
    return {"state": st, "why": why, "D": D, "ci": [float(lo), float(hi)], "bar": bar, "s_run": s_run,
            "run_A": rA, "run_X": rX, "pairs": Ds, "idio_sd": [math.sqrt(sA2), math.sqrt(sX2)],
            "n_answers": J, "flags": flags, "answers": per}


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
def calibrate() -> dict:
    w = _w()
    J = len(w)
    rng = np.random.default_rng(20261023)
    R = 60_000
    out = {"R": R, "J": J, "K_FAM": K_FAM, "Z_ANS": Z_ANS, "rows": []}

    def draw(tau, sig, delta, ratio, het):
        sj = sig * np.sqrt(w) if het == "se" else sig * np.exp(rng.normal(0, 0.7, (R, 1, J)) - 0.245)
        cA, cX = rng.normal(0, tau, (R, 2, 1)), rng.normal(0, tau * ratio, (R, 2, 1))
        A = cA + rng.normal(0, 1, (R, 2, J)) * sj
        X = cX + rng.normal(0, 1, (R, 2, J)) * sj * ratio + delta
        return A, X

    for tau, sig, ratio, het in [(0.006, 0.006, 1, "se"), (0.006, 0, 1, "se"), (0, 0.006, 1, "se"),
                                 (0.002, 0.008, 1, "se"), (0.004, 0.006, 1, "lognormal"),
                                 (0.004, 0.004, 3, "se"), (0.002, 0.004, 1, "se"), (0.001, 0.003, 1, "se")]:
        row = {"tau": tau, "sigma": sig, "ratio": ratio, "het": het}
        for d in (0.0, 0.01, 0.02, 0.03):
            A, X = draw(tau, sig, d, ratio, het)
            dA, dX = A[:, 0] - A[:, 1], X[:, 0] - X[:, 1]
            rA, rX = dA.mean(1, keepdims=True), dX.mean(1, keepdims=True)
            bar = K_FAM * np.maximum(np.abs(rA), np.abs(rX))[:, 0]
            sA2 = ((dA - rA) ** 2 / w).mean(1) / 2 * J / (J - 1)
            sX2 = ((dX - rX) ** 2 / w).mean(1) / 2 * J / (J - 1)
            barj = np.sqrt(bar[:, None] ** 2 + (Z_ANS * np.sqrt(w[None] * (sA2 + sX2)[:, None] / 2)) ** 2)
            Dj = (X - A).mean(1)
            D = Dj.mean(1)
            if d == 0:
                row["false_fam"] = float((np.abs(D) > bar).mean())
                row["false_ans"] = float((np.abs(Dj) > barj).mean())
            else:
                row[f"power_fam_{d}"] = float((D > bar).mean())
                row[f"power_ans_{d}"] = float((Dj > barj).mean())
            row.setdefault("median_bar", float(np.median(bar)))
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    return out


# --- plants: synthetic scores through the identical scoring path -------------------------------
def _labels():
    import baseline_run  # noqa: F401  — A1's v2 environment, before F0 reads it
    os.environ.setdefault("LADDER_SPLIT", "identical")
    from j4_spread_controls import prepare
    s = prepare(None, 0, label="baselines-plants", sources=1)
    a = json.loads(ANSWER_SET.read_text())
    test = [int(o) for o in s.test_ids]
    ys, rows = {}, {}
    for f in a["answer_set"]:
        el = s.labels.eligible(f, test)
        ys[f], rows[f] = np.asarray(s.labels.binary_label(f, el), dtype=int), el
    return a["answer_set"], ys, rows, test


def _realise(rng, answers, ys, cols, wt, base, tau, sig, delta, ratio=1.0, drop=None, rho=0.5):
    """One planted world: 4 encoders' scores per answer -> auc + boot under the shared weights."""
    from aligned_c2 import auc, boot_aucs
    from scipy.stats import norm
    wj = _w()
    c = {"A1": rng.normal(0, tau), "A2": rng.normal(0, tau),
         "X0": rng.normal(0, tau * ratio), "X1": rng.normal(0, tau * ratio)}
    A, Bt = {e: {} for e in c}, {e: {} for e in c}
    for i, f in enumerate(answers):
        y = ys[f]
        shared = rng.normal(size=len(y))  # galaxy noise common to all encoders (pairing in the bootstrap)
        for e in c:
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
    from aligned_c2 import weights
    answers, ys, rows, test = _labels()
    wt, pos = weights(test)
    cols = {f: [pos[int(o)] for o in rows[f]] for f in answers}
    w = _w()
    J = len(answers)
    rng = np.random.default_rng(20261024)
    base = rng.uniform(0.70, 0.95, J)  # synthetic, fixed: no encoder's AUCs
    flat = lambda v: {e: v for e in ("A1", "A2", "X0", "X1")}  # noqa: E731
    on_x = lambda v: {"A1": 0.0, "A2": 0.0, "X0": v, "X1": v}  # noqa: E731
    some = np.zeros(J)
    some[:5] = 0.03
    N = dict(tau=0.002, sig=0.006)
    spec = [  # name, expect, realisations, kwargs
        ("null", "SAME", 10, dict(**N, delta=flat(0.0))),
        ("better_0.03", "BETTER", 10, dict(**N, delta=on_x(0.03))),
        ("worse_0.03", "WORSE", 10, dict(**N, delta=on_x(-0.03))),
        ("split", "SPLIT", 10, dict(**N, delta={"A1": 0.0, "A2": 0.0, "X0": 0.03, "X1": -0.03})),
        ("negligible", "NEGLIGIBLE", 10, dict(tau=0.0003, sig=0.0005, delta=on_x(0.003))),
        ("underpowered", "UNRESOLVED", 10, dict(tau=0.015, sig=0.006, delta=flat(0.0))),
        ("void", "VOID", 3, dict(**N, delta=flat(0.0), drop={"X1": set(answers[:6])})),
        ("extreme", "BETTER", 5, dict(**N, delta=on_x(0.15))),
        ("spreads_differ_null", "not BETTER/WORSE/SPLIT", 10, dict(tau=0.002, sig=0.006, ratio=3.0, delta=flat(0.0))),
        ("five_answers_0.03", "per-answer: 5 BETTER", 10, dict(**N, delta={"A1": 0.0, "A2": 0.0, "X0": some, "X1": some})),
    ]
    out = {"base_auc_draw": "U(0.70, 0.95), seed 20261024", "noise": N, "rho": 0.5, "plants": {}}
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
        out["plants"][name] = {"expect": expect, "R": R, "family": fam, "answers": ans_rates,
                               "median_bar": float(np.median([x["bar"] for x in res if x["bar"] is not None]))
                               if any(x["bar"] is not None for x in res) else None,
                               "seconds": round(time.time() - t0)}
        print(f"{name:22s} expect {expect:24s} family {fam}  answers {ans_rates}", flush=True)
    # trajectory states: the state sequence -> trajectory(), on planted family outcomes
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


def score() -> None:
    raise SystemExit("score: refused — baselines_comparison.md is not hashed (D27). Nothing is read.")


if __name__ == "__main__":
    mode = sys.argv[1]
    OUTP = HERE / "out"
    if mode == "calibrate":
        (OUTP / "baselines_calibration.json").write_text(json.dumps(calibrate(), indent=1))
    elif mode == "plants":
        (OUTP / "baselines_plants.json").write_text(json.dumps(plants(), indent=1))
    elif mode == "score":
        score()
