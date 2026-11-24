"""A-bis (user, 24 Nov 2026): arm count with a marginal + "can't tell" baseline, so the gap and share measure only what A1 adds.

Pre-registration: docs/axes/prereg_a_bis.md (hashed; `run` refuses on a mismatch). Fits are Part A's, unchanged (axes_a_ordinal imported).

    .venv/bin/python artifacts/axes_a_bis.py plants
    .venv/bin/python artifacts/axes_a_bis.py run
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

import axes_a_ordinal as A

PREREG = A.DOC / "prereg_a_bis.md"
PREREG_SHA1: str | None = "659d254042f63c359a8902fbf645ae82079fcd54"
SEED = A.SEED  # Part A's seed: the folds, and so the fits, must reproduce


def cov_only(kind, c, p, n):
    """The same covariate model with no embeddings: one zero column (its weight stays 0), one slope (ord) or K (free)."""
    M = A.Model(kind, np.zeros((len(c), 1)), c, p, n)
    return M.fit(1.0)


def reading2(lls, n, rng):
    """Part A's reading with the matched baselines: gap = free − freecov; share = (ord − ordcov) / (free − freecov). First match applies."""
    N = len(n)
    P = {k: float(n @ v / n.sum()) for k, v in lls.items()}
    B = {k: np.empty(A.N_BOOT) for k in lls}
    for b in range(A.N_BOOT):
        w = np.bincount(rng.integers(0, N, N), minlength=N) * n
        for k, v in lls.items():
            B[k][b] = w @ v / w.sum()

    def stat(nb, db):
        gap, gp = B["free"] - B[db], P["free"] - P[db]
        ok = gap > 0
        sh = (B["ord"][ok] - B[nb][ok]) / gap[ok]
        ci = np.percentile(sh, [2.5, 97.5]) if ok.sum() >= A.N_BOOT // 2 else (np.nan, np.nan)
        return gap, gp, (P["ord"] - P[nb]) / gp, ci, int((~ok).sum())
    gap, gp, share, sci, und = stat("ordcov", "freecov")
    lo, hi = np.percentile(gap, [2.5, 97.5])
    r = "NO SIGNAL" if lo <= 0 else "ONE AXIS" if sci[0] >= 0.9 else "MOSTLY ONE AXIS" if share >= 0.75 else "SEVERAL"
    out = {"LL": P, "gap_free": gp, "gap_free_ci95": [float(lo), float(hi)], "gap_ord": P["ord"] - P["ordcov"], "share": float(share),
           "share_ci95": [float(x) for x in sci], "share_undefined_resamples": und, "reading": r, "cov_gain_free": P["freecov"] - P["base"],
           "cov_gain_ord": P["ordcov"] - P["base"]}
    _, g2, s2, c2, u2 = stat("freecov", "freecov")
    out["sensitivity_common_free_cov_baseline (no reading)"] = {"share": float(s2), "share_ci95": [float(x) for x in c2], "undefined": u2}
    _, g3, s3, c3, _ = stat("base", "base")
    out["part_a_marginal_baseline (no reading)"] = {"share": float(s3), "share_ci95": [float(x) for x in c3], "gap": g3}
    return out


def analyse2(X, ct, p, n, tr, te, seed):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    c = (ct - ct[tr].mean()) / (ct[tr].std() or 1.0)
    rec, lls = {}, {}
    for kind in ("ord", "free"):
        m, rec[kind] = A.cv_fit(kind, Z[tr], c[tr], p[tr], n[tr], np.random.default_rng(seed))
        lls[kind] = (p[te] * A.logp(kind, m, Z[te], c[te])).sum(1)
        rec[kind]["cov_coef"] = np.atleast_1d(m["g"]).tolist()
        m0 = cov_only(kind, c[tr], p[tr], n[tr])
        assert np.abs(m0["w"] if kind == "ord" else m0["W"]).max() == 0, "the covariate-only fit moved an embedding weight"
        lls[kind + "cov"] = (p[te] * A.logp(kind, m0, np.zeros((te.sum(), 1)), c[te])).sum(1)
        rec[kind]["cov_only_coef"] = np.atleast_1d(m0["g"]).tolist()
    marg = n[tr] @ p[tr] / n[tr].sum()
    lls["base"] = p[te] @ np.log(np.clip(marg, 1e-12, None))
    rec.update(reading2(lls, n[te], np.random.default_rng(seed + 1)))
    return rec


# --- plants (D28): synthetic votes with a real covariate effect, real vote counts, A1's cleaned rows; no arm-count fraction enters
def step_plants():
    AXS, d = A.load()
    r = A.question_rows(AXS, d, "arm count")
    X, tr, te, n = d.clean["A1"][r["rows"]], r["tr"], r["te"], r["n"]
    ct = r["ct"]
    c = (ct - ct[tr].mean()) / ct[tr].std()
    Z = (X - X[tr].mean(0)) / X[tr].std(0)
    ev, V = np.linalg.eigh(Z[tr].T @ Z[tr] / tr.sum())
    U = Z @ V[:, ::-1][:, :40]
    U /= U[tr].std(0)
    rng = np.random.default_rng(SEED + 99)
    a = rng.normal(size=(40, 5)); dirs = U @ a; dirs /= dirs[tr].std(0)
    soft = lambda L: np.exp(L - np.logaddexp.reduce(L, axis=1, keepdims=True))  # noqa: E731
    G = 0.45 * np.array([1.0, -1.0, 0.6, -0.2, 0.9])  # a category-specific covariate slope, the shape the ordinal model cannot carry
    th = np.array([-2.0, -0.5, 0.7, 2.0])
    cov_lp = np.outer(c, G)
    out = {}

    def run(name, P):
        P = P / P.sum(1, keepdims=True)
        y = np.array([rng.multinomial(int(m), q / q.sum()) for m, q in zip(n, P)], float)
        t0 = time.perf_counter()
        rec = analyse2(X, ct, y / y.sum(1, keepdims=True), n, tr, te, SEED)
        out[name] = {k: rec[k] for k in ("LL", "gap_free", "gap_free_ci95", "share", "share_ci95", "reading", "cov_gain_free", "cov_gain_ord",
                                         "sensitivity_common_free_cov_baseline (no reading)", "part_a_marginal_baseline (no reading)")}
        s = out[name]
        A.log(f"plant {name}: {s['reading']} share {s['share']:.3f} {np.round(s['share_ci95'], 3)} gap {s['gap_free']:.4f} cov gain {s['cov_gain_free']:.4f}/{s['cov_gain_ord']:.4f} "
              f"part-A {s['part_a_marginal_baseline (no reading)']['share']:.3f} ({time.perf_counter() - t0:.0f}s)")
        (A.DOC / "a_bis_plants.json").write_text(json.dumps({"plants": out, "readings_reached": sorted({v["reading"] for v in out.values()})}, indent=1, default=float))

    ordp = lambda amp: A.ord_logp(amp * dirs[:, 0], th)  # noqa: E731
    run("one axis, no covariate effect", np.exp(ordp(2.0)))
    run("one axis + covariate, strong", np.exp(ordp(2.0) + cov_lp))
    run("one axis + covariate, weak (real-size gap)", np.exp(ordp(0.3) + cov_lp))
    run("independent directions + covariate (several)", np.exp(1.2 * dirs + cov_lp))
    for b in (0.3, 0.5, 0.7):
        run(f"one axis + independent b={b} + covariate", soft(np.outer(2.0 * dirs[:, 0], np.linspace(-1, 1, 5)) + b * dirs[:, 1:] @ np.eye(4, 5) + cov_lp))
    run("weak mixture + covariate (real-size gap)", soft(np.outer(0.3 * dirs[:, 0], np.linspace(-1, 1, 5)) + 0.12 * dirs[:, 1:] @ np.eye(4, 5) + cov_lp))
    run("null in X, covariate only", np.exp(np.log(np.exp(ordp(2.0)).mean(0))[None] + cov_lp))
    run("null in X, covariate only, A1-correlated covariate kept", np.exp(np.log(np.exp(ordp(2.0)).mean(0))[None] + 2 * cov_lp))


def section() -> str:
    L = PREREG.read_text().splitlines()
    a = L.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(L)) if L[k].startswith("## ")), len(L))
    return "\n".join(L[a:b]).rstrip() + "\n"


def step_hash():
    print(hashlib.sha1(section().encode()).hexdigest())


def step_run():
    if PREREG_SHA1 is None or hashlib.sha1(section().encode()).hexdigest() != PREREG_SHA1:
        raise SystemExit("a_bis: pre-registration not hashed, or changed since")
    AXS, d = A.load()
    old = json.loads((A.DOC / "a_ordinal.json").read_text())["questions"]["arm count"]["embeddings"]
    r = A.question_rows(AXS, d, "arm count")
    rows, p, n, ct, tr, te = (r[k] for k in ("rows", "p", "n", "ct", "tr", "te"))
    out = {"prereg_sha1": PREREG_SHA1, "n_train": int(tr.sum()), "n_test": int(te.sum()), "embeddings": {}}
    for sp in ("cleaned", "raw"):
        out["embeddings"][sp] = {}
        for enc in ["A1"] + [k for k in d.mats if k != "A1"]:
            X = (d.clean if sp == "cleaned" else d.mats)[enc][rows]
            a = analyse2(X, ct, p, n, tr, te, SEED)
            o = old[sp][enc]["LL"]
            a["reproduces_part_a"] = {k: a["LL"][k] - o[k] for k in ("base", "ord", "free")}
            out["embeddings"][sp][enc] = a
            A.log(f"{sp} {enc}: {a['reading']} share {a['share']:.3f} {np.round(a['share_ci95'], 3)} gap {a['gap_free']:.4f} Δ vs A {a['reproduces_part_a']}")
    (A.DOC / "a_bis.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
