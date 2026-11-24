"""Part A (brief, 24 Nov 2026): are ordered GZ2 questions one coordinate in A1, or several? Cumulative logit vs free multinomial.

Pre-registration: docs/axes/prereg_part_a.md (hashed; `run` refuses on a mismatch). Frozen A1 embeddings only, no training.

    .venv/bin/python artifacts/axes_a_ordinal.py test     # gradient + planted unit checks (no vote fractions read)
    .venv/bin/python artifacts/axes_a_ordinal.py plants   # D28: synthetic labels through the identical path, every reading reached
    .venv/bin/python artifacts/axes_a_ordinal.py run      # the real fits (A1 and three untrained, cleaned and raw)
    .venv/bin/python artifacts/axes_a_ordinal.py sheet    # strips

Investigation code: terse, excluded from lint/CI. Writes docs/axes/ and runs/almagest/axes_phys/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/axes"
OUT = REPO / "runs/almagest/axes_phys"
PREREG = DOC / "prereg_part_a.md"
PREREG_SHA1: str | None = "058e4dd16809c9db92a947e97e5f7b53afa8701a"  # hashed 24 Nov 2026, before any real fit
SEED = 20261124
N_BOOT = 1000
LAMS = 10.0 ** np.arange(1, -5.25, -0.5)  # 1e1 … 1e-5, per-vote-mean loss, standardised features
QUESTIONS = {  # ordered categories; `cov` = excluded answer whose fraction is a covariate
    "arm count": dict(q="t11_arms_number", cats=["a31_1", "a32_2", "a33_3", "a34_4", "a36_more_than_4"], cov="a37_cant_tell"),
    "winding": dict(q="t10_arms_winding", cats=["a28_tight", "a29_medium", "a30_loose"], cov=None),
    "bulge prominence": dict(q="t05_bulge_prominence", cats=["a10_no_bulge", "a11_just_noticeable", "a12_obvious", "a13_dominant"], cov=None),
    "roundness": dict(q="t07_rounded", cats=["a16_completely_round", "a17_in_between", "a18_cigar_shaped"], cov=None),
}


def log(*a):
    print(*a, f"[load {os.getloadavg()[0]:.1f}]", file=sys.stderr, flush=True)


def prereg_section() -> str:
    lines = PREREG.read_text().splitlines()
    a = lines.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip() + "\n"


def assert_hashed() -> None:
    if PREREG_SHA1 is None:
        raise SystemExit("axes_a: pre-registration not hashed yet")
    got = hashlib.sha1(prereg_section().encode()).hexdigest()
    if got != PREREG_SHA1:
        raise SystemExit(f"axes_a: pre-registration changed since the hash ({got} != {PREREG_SHA1})")


# --- losses (value, gradient); data terms are per-vote means, penalty λ/2·||W||² on standardised features -----------
def sig(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def logsig(x):
    return -np.logaddexp(0, -x)


def ord_logp(eta, th):
    """(N,), (K−1,) → (N, K) log P_k of the cumulative logit P(Y ≤ k) = σ(θ_k − η)."""
    a = th[None, :] - eta[:, None]
    N, m = a.shape
    out = np.empty((N, m + 1))
    out[:, 0] = logsig(a[:, 0])
    out[:, m] = logsig(-a[:, m - 1])
    for k in range(1, m):
        out[:, k] = logsig(a[:, k]) + logsig(-a[:, k - 1]) + np.log(-np.expm1(th[k - 1] - th[k]))
    return out


def ord_terms(eta, th, p, n):
    """Σ n_i Σ_k p_ik log P_ik and its derivatives in η and θ (n already normalised by its sum)."""
    lp = ord_logp(eta, th)
    ll = float(n @ (p * lp).sum(1))
    a = th[None, :] - eta[:, None]
    s = sig(a)
    g = p[:, :-1] * (1 - s) - p[:, 1:] * s  # ∂/∂θ_j from the sigmoid terms
    dth = n @ g
    deta = -n * g.sum(1)
    for j in range(1, len(th)):  # log(1 − e^{−δ_j}), δ_j = θ_j − θ_{j−1}, weighted by category j's mass
        d = th[j] - th[j - 1]
        h = float(n @ p[:, j]) / np.expm1(d)
        dth[j] += h
        dth[j - 1] -= h
    return ll, deta, dth


def to_th(t):  # θ1, log gaps → ordered thresholds
    return np.concatenate([[t[0]], t[0] + np.cumsum(np.exp(t[1:]))])


def d_th(t, dth):  # chain rule back to (θ1, log gaps)
    m = len(t)
    out = np.empty(m)
    out[0] = dth.sum()
    for j in range(1, m):
        out[j] = np.exp(t[j]) * dth[j:].sum()
    return out


class Model:
    """L2 fit in PCA coordinates with a per-λ diagonal preconditioner (w = v·D), warm-started down the λ path."""

    def __init__(self, kind, X, ct, p, n):
        self.kind, self.K = kind, p.shape[1]
        self.n = n / n.sum()
        self.p, self.ct = p, ct
        ev, V = np.linalg.eigh(X.T @ X / len(X))
        self.V, self.ev = V, np.maximum(ev, 0)
        self.Z = X @ V
        self.par = None

    def _init(self):
        q = np.clip(self.n @ self.p, 1e-9, None)
        if self.kind == "ord":
            c = np.cumsum(q)[:-1] / q.sum()
            th = np.log(c / (1 - c))
            t = np.concatenate([[th[0]], np.log(np.diff(th))]) if self.K > 2 else th
            return [np.zeros(self.Z.shape[1]), 0.0, t]
        return [np.zeros((self.Z.shape[1], self.K)), np.zeros(self.K), np.log(q / q.sum())]

    def fit(self, lam, maxiter=2000):
        from scipy.optimize import minimize
        if self.par is None:
            self.par = self._init()
        D = 1 / np.sqrt(self.ev * 0.2 + lam)
        W, g, t = self.par
        d, K = self.Z.shape[1], self.K
        if self.kind == "ord":
            x0 = np.concatenate([W / D, [g], t])

            def fun(x):
                w = x[:d] * D
                eta = self.Z @ w + x[d] * self.ct
                th = to_th(x[d + 1:])
                ll, deta, dth = ord_terms(eta, th, self.p, self.n)
                gw = -(deta @ self.Z) + lam * w
                return (-ll + lam / 2 * w @ w, np.concatenate([gw * D, [-(deta @ self.ct)], -d_th(x[d + 1:], dth)]))
        else:
            x0 = np.concatenate([(W / D[:, None]).ravel(), g, t])

            def fun(x):
                W_ = x[:d * K].reshape(d, K) * D[:, None]
                gc, b = x[d * K:d * K + K], x[d * K + K:]
                L = self.Z @ W_ + np.outer(self.ct, gc) + b
                L -= np.logaddexp.reduce(L, axis=1, keepdims=True)
                ll = float(self.n @ (self.p * L).sum(1))
                R = self.n[:, None] * (np.exp(L) - self.p)
                gW = self.Z.T @ R + lam * W_
                return (-ll + lam / 2 * (W_ * W_).sum(),
                        np.concatenate([(gW * D[:, None]).ravel(), self.ct @ R, R.sum(0)]))
        r = minimize(fun, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=2 * maxiter, gtol=1e-6, ftol=1e-11, maxcor=20))
        if r.nit >= maxiter:
            raise RuntimeError(f"{self.kind} λ={lam:g}: not converged in {maxiter} iterations")
        x = r.x
        if self.kind == "ord":
            self.par = [x[:d] * D, float(x[d]), x[d + 1:]]
            return dict(w=self.V @ self.par[0], g=self.par[1], th=to_th(self.par[2]))
        self.par = [x[:d * K].reshape(d, K) * D[:, None], x[d * K:d * K + K], x[d * K + K:]]
        return dict(W=self.V @ self.par[0], g=self.par[1], b=self.par[2])


def logp(kind, m, X, ct):
    if kind == "ord":
        return ord_logp(X @ m["w"] + m["g"] * ct, m["th"])
    L = X @ m["W"] + np.outer(ct, m["g"]) + m["b"]
    return L - np.logaddexp.reduce(L, axis=1, keepdims=True)


def cv_fit(kind, X, ct, p, n, rng):
    """5-fold CV over the λ grid (shared folds per call seed), then the full-train fit at the chosen λ."""
    fold = rng.permutation(len(X)) % 5
    score = np.zeros(len(LAMS))
    for f in range(5):
        tr, va = fold != f, fold == f
        M = Model(kind, X[tr], ct[tr], p[tr], n[tr])
        for j, lam in enumerate(LAMS):
            m = M.fit(lam)
            score[j] += float(n[va] @ (p[va] * logp(kind, m, X[va], ct[va])).sum(1))
    score /= n.sum()
    j = int(np.argmax(score))
    if j == len(LAMS) - 1:
        raise RuntimeError(f"{kind}: CV picked the smallest λ of the grid; extend it")
    M = Model(kind, X, ct, p, n)
    for lam in LAMS[:j + 1]:
        m = M.fit(lam)
    return m, {"lambda": float(LAMS[j]), "at_largest_lambda": j == 0, "cv_ll": score.tolist()}


# --- the test statistic and reading -------------------------------------------------------------------------------
def reading(lls, n, rng):
    """lls: dict base/ord/free → per-galaxy Σ_k p log p̂. The four readings, first match applies."""
    N = len(n)
    w = np.empty((N_BOOT, N))
    for b in range(N_BOOT):
        w[b] = np.bincount(rng.integers(0, N, N), minlength=N)
    wn = w * n
    tot = wn.sum(1)
    B = {k: wn @ v / tot for k, v in lls.items()}
    P = {k: float(n @ v / n.sum()) for k, v in lls.items()}
    gap = B["free"] - B["base"]
    gp = P["free"] - P["base"]
    lo, hi = np.percentile(gap, [2.5, 97.5])
    share_p = (P["ord"] - P["base"]) / gp
    ok = gap > 0
    share = (B["ord"][ok] - B["base"][ok]) / gap[ok]
    slo, shi = (np.percentile(share, [2.5, 97.5]) if ok.sum() >= N_BOOT // 2 else (np.nan, np.nan))
    r = ("NO SIGNAL" if lo <= 0 else "ONE AXIS" if slo >= 0.9 else "MOSTLY ONE AXIS" if share_p >= 0.75 else "SEVERAL")
    return {"LL": P, "gap_free_base": gp, "gap_ci95": [float(lo), float(hi)], "gap_ord_base": P["ord"] - P["base"],
            "share": float(share_p), "share_ci95": [float(slo), float(shi)], "share_undefined_resamples": int((~ok).sum()),
            "reading": r}


def analyse(X, ct, p, n, tr, te, seed, want_models=False):
    """One embedding matrix, one question. X raw (N×384) rows over tr ∪ te; returns the record (and fitted models)."""
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    c = (ct - ct[tr].mean()) / (ct[tr].std() or 1.0)
    mods, rec = {}, {}
    for kind in ("ord", "free"):
        mods[kind], rec[kind] = cv_fit(kind, Z[tr], c[tr], p[tr], n[tr], np.random.default_rng(seed))
    marg = n[tr] @ p[tr] / n[tr].sum()
    lls = {k: (p[te] * logp(k, mods[k], Z[te], c[te])).sum(1) for k in ("ord", "free")}
    lls["base"] = p[te] @ np.log(np.clip(marg, 1e-12, None))
    rec.update(reading(lls, n[te], np.random.default_rng(seed + 1)))
    rec["marginal"] = marg.tolist()
    if not want_models:
        return rec, None
    return rec, {"mods": mods, "mu": mu, "sd": sd, "Zte": Z[te], "Z": Z}


# --- self-checks -----------------------------------------------------------------------------------------------------
def step_test() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)); ct = rng.normal(size=60); n = rng.integers(1, 30, 60).astype(float)
    p = rng.dirichlet(np.ones(4), 60)
    for kind in ("ord", "free"):
        M = Model(kind, X, ct, p, n)
        # finite-difference the objective through the same code the optimiser sees
        import scipy.optimize as so
        captured = {}
        orig = so.minimize

        def grab(fun, x0, **kw):
            captured["f"], captured["x"] = fun, x0
            x = x0 + 0.1 * rng.normal(size=len(x0))
            v, g = fun(x)
            num = np.array([(fun(x + 1e-6 * e)[0] - fun(x - 1e-6 * e)[0]) / 2e-6 for e in np.eye(len(x))])
            captured["err"] = float(np.max(np.abs(num - g)))
            return orig(fun, x0, **kw)
        so.minimize = grab
        M.fit(0.1)
        so.minimize = orig
        print(kind, "max |grad − finite difference|", captured["err"])
        assert captured["err"] < 1e-6
    th = np.array([-1.0, 0.2, 1.3]); lp = ord_logp(rng.normal(size=20) * 3, th)
    assert np.allclose(np.exp(lp).sum(1), 1)
    print("cumulative-logit probabilities sum to 1")


# --- the data ---------------------------------------------------------------------------------------------------------
def load():
    import almagest_axes as AXS
    return AXS, AXS.Data(untrained=True)


def question_rows(AXS, d, name):
    """Rows (eligible, complete nuisances, ≥ 1 vote on the ordered answers), soft targets, vote counts, covariate."""
    Q = QUESTIONS[name]
    f0 = f"{Q['q']}_{Q['cats'][0]}"
    tr, te = d.rows(f0)
    rows = np.concatenate([tr, te])
    ids = d.union[rows]
    fr = np.column_stack([np.asarray(d.lab.vote_fraction(f"{Q['q']}_{c}", ids), float) for c in Q["cats"]])
    n = sum(np.nan_to_num(d.lab._column(ids, f"{Q['q']}_{c}_count")) for c in Q["cats"])
    ct = (np.asarray(d.lab.vote_fraction(f"{Q['q']}_{Q['cov']}", ids), float) if Q["cov"] else np.zeros(len(ids)))
    tot = fr.sum(1)
    ok = np.isfinite(tot) & (tot > 0) & (n > 0) & np.isfinite(ct)
    is_tr = np.r_[np.ones(len(tr), bool), np.zeros(len(te), bool)]
    return dict(rows=rows[ok], p=fr[ok] / tot[ok, None], n=n[ok], ct=ct[ok], tr=is_tr[ok], te=~is_tr[ok],
                dropped=int((~ok).sum()), ids=ids[ok])


# --- plants (D28) ---------------------------------------------------------------------------------------------------
def step_plants() -> None:
    AXS, d = load()
    r = question_rows(AXS, d, "arm count")
    X = d.clean["A1"][r["rows"]]
    tr, te, n = r["tr"], r["te"], r["n"]
    ct0 = np.zeros(len(n))
    Z = (X - X[tr].mean(0)) / X[tr].std(0)
    ev, V = np.linalg.eigh(Z[tr].T @ Z[tr] / tr.sum())
    U = (Z @ V[:, ::-1][:, :40]); U /= U[tr].std(0)
    rng = np.random.default_rng(SEED)

    def direction(k):
        a = rng.normal(size=(40, k)); u = U @ a
        return u / u[tr].std(0)

    def draw(P):
        # P: (N, K); multinomial counts with the galaxy's real n, then the same fractions path
        c = np.array([rng.multinomial(int(m), q / q.sum()) for m, q in zip(n, P)], float)
        return c / c.sum(1, keepdims=True)

    def softmax(L):
        return np.exp(L - np.logaddexp.reduce(L, axis=1, keepdims=True))
    th = np.array([-2.0, -0.5, 0.7, 2.0])
    dirs = direction(5)
    out = {}

    def run(name, P):
        t0 = time.perf_counter()
        rec, _ = analyse(X, ct0, draw(P), n, tr, te, SEED)
        out[name] = {k: rec[k] for k in ("LL", "share", "share_ci95", "gap_free_base", "gap_ci95", "reading")}
        out[name]["lambda"] = {k: rec[k]["lambda"] for k in ("ord", "free")}
        out[name]["seconds"] = time.perf_counter() - t0
        log(f"plant {name}: {rec['reading']} share {rec['share']:.3f} {np.round(rec['share_ci95'], 3)} ({out[name]['seconds']:.0f}s)")

    run("ordinal (true one axis)", np.exp(ord_logp(2.0 * dirs[:, 0], th)))
    run("independent directions (true several)", softmax(1.2 * dirs[:, :5]))
    for b in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
        base = np.outer(2.0 * dirs[:, 0], np.linspace(-1, 1, 5))
        run(f"one axis + independent directions, b={b}", softmax(base + b * dirs[:, 1:5] @ np.eye(4, 5)))
    run("stereotype: one axis, uneven spacing φ=(0,.1,1,1.1,1.2)", softmax(np.outer(3.0 * dirs[:, 0], [0, 0.1, 1.0, 1.1, 1.2])))
    for amp in (0.15, 0.3, 0.6):  # weak true one axis: can ONE AXIS still be reached when the gap is small?
        run(f"weak ordinal, amplitude {amp}", np.exp(ord_logp(amp * dirs[:, 0], th)))
    # a 1 % prevalence top category, like "dominant"/"more than 4": thresholds skewed, amplitude 1
    run("ordinal, rare top categories", np.exp(ord_logp(1.0 * dirs[:, 0], np.array([-1.5, 1.2, 3.0, 4.2]))))
    marg = np.exp(ord_logp(2.0 * dirs[:, 0], th)).mean(0)
    run("null (no signal in X)", np.repeat(marg[None], len(n), 0))
    res = {"note": "synthetic soft votes through the identical code path on A1's cleaned arm-count rows and real vote counts; no "
                   "real vote fraction enters. D28: every reading must be reached.", "plants": out,
           "readings_reached": sorted({v["reading"] for v in out.values()})}
    (DOC / "a_plants.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: (v["reading"], round(v["share"], 3)) for k, v in out.items()}, indent=1))


# --- the real run -----------------------------------------------------------------------------------------------------
def kendall_deciles(score, untr, ecat, res, rng):
    """τ(score, expected category) per decile of petroRad_r/psfWidth_r; A1 with a bootstrap CI, untrained as points."""
    from scipy.stats import kendalltau
    edges = np.quantile(res, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, res, side="right") - 1, 0, 9)
    out = []
    for k in range(10):
        m = np.flatnonzero(b == k)
        bs = [kendalltau(score[i], ecat[i])[0] for i in (rng.choice(m, m.size) for _ in range(200))]
        u = [float(kendalltau(v[m], ecat[m])[0]) for v in untr.values()]
        lo, hi = (float(x) for x in np.nanpercentile(bs, [2.5, 97.5]))
        out.append({"decile": k + 1, "ratio_range": [float(edges[k]), float(edges[k + 1])], "n": int(m.size),
                    "tau": float(kendalltau(score[m], ecat[m])[0]), "ci95": [lo, hi], "untrained": u,
                    "above_untrained_max": bool(lo > max(u))})
    h = sum(e["above_untrained_max"] for e in out)
    return {"deciles": out, "n_holding": h, "control": "HOLDS" if h >= 6 else "DISAPPEARS" if h <= 2 else "PARTIAL",
            "tau_all": float(kendalltau(score, ecat)[0]),
            "tau_all_untrained": [float(kendalltau(v, ecat)[0]) for v in untr.values()]}


def step_run() -> None:
    assert_hashed()
    AXS, d = load()
    size = np.asarray(d.lab.nuisance_value("size", d.union), float)
    psf = np.asarray(d.lab.nuisance_value("psf", d.union), float)
    names = ["A1"] + [k for k in d.mats if k != "A1"]
    axes37 = json.loads((AXS.DOCS / "a1_axes.json").read_text())["order"]
    path = DOC / "a_ordinal.json"
    out = json.loads(path.read_text()) if path.exists() else {
        "prereg_sha1": PREREG_SHA1, "brief_sha1": "cfaed32917e90f11ef3bebdf4cdcc1c56bc82ed6",
        "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(), "questions": {}}
    for qi, name in enumerate(QUESTIONS):
        if name in out["questions"]:
            continue
        t0 = time.perf_counter()
        Q, r = QUESTIONS[name], question_rows(AXS, d, name)
        rows, p, n, ct, tr, te = (r[k] for k in ("rows", "p", "n", "ct", "tr", "te"))
        K = p.shape[1]
        ecat = p @ np.arange(1, K + 1)
        rec = {"question": Q["q"], "categories": Q["cats"], "covariate": Q["cov"], "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "dropped_no_votes_on_target": r["dropped"], "median_votes": float(np.median(n)), "embeddings": {}}
        keep = {}
        for kind in ("cleaned", "raw"):
            e, score = {}, {}
            for enc in names:
                X = (d.clean if kind == "cleaned" else d.mats)[enc][rows]
                a, ex = analyse(X, ct, p, n, tr, te, SEED + qi, want_models=True)
                e[enc] = a
                score[enc] = ex["Zte"] @ ex["mods"]["ord"]["w"]
                if enc == "A1":
                    keep[kind] = ex
                    if Q["cov"]:
                        e["A1, no covariate (sensitivity)"] = analyse(X, np.zeros_like(ct), p, n, tr, te, SEED + qi)[0]
                log(f"{name} {kind} {enc}: {a['reading']} share {a['share']:.3f} {np.round(a['share_ci95'], 3)} gap {a['gap_free_base']:.4f}")
            rng = np.random.default_rng(SEED + 10 + qi)
            e["visibility"] = kendall_deciles(score["A1"], {k: score[k] for k in names[1:]}, ecat[te], size[rows][te] / psf[rows][te], rng)
            rec["embeddings"][kind] = e
            # geometry (test scores)
            ex = keep[kind]
            W = ex["mods"]["free"]["W"]
            s_ord = score["A1"]
            ctr = {f"{Q['cats'][k + 1]} − {Q['cats'][k]}": float(np.corrcoef(s_ord, ex["Zte"] @ (W[:, k + 1] - W[:, k]))[0, 1]) for k in range(K - 1)}
            sv = np.linalg.svd(W - W.mean(1, keepdims=True), compute_uv=False) ** 2
            geo = {"r_ordinal_vs_free_adjacent_contrasts": ctr, "free_contrast_singular_energy_share": (sv / sv.sum()).tolist()}
            S_ = np.column_stack([ex["Zte"] @ (W[:, k + 1] - W[:, k]) for k in range(K - 1)])
            geo["free_contrast_score_corr"] = np.corrcoef(S_.T).tolist()
            g37 = {}
            for f in axes37:
                if kind == "cleaned":
                    z = np.load(AXS.AX / f"{f}.npz")
                    s = d.clean["A1"][rows][te] @ z["we"] + float(z["be"])
                else:  # no raw probes were stored by the 37-axis run: refit the identical probe on raw rows, own question only
                    if not f.startswith(Q["q"]):
                        continue
                    ftr, _ = d.rows(f)
                    y, _ = d.yv(f, ftr)
                    s = AXS.Probe(d.mats["A1"][ftr], y, d.c).d(d.mats["A1"][rows][te])
                g37[f] = float(np.corrcoef(s_ord, s)[0, 1])
            geo["r_ordinal_vs_37_axis_probes"] = g37
            rec[f"geometry_{kind}"] = geo
        # strip picks (cleaned): 11 ordinal scores across train's 5th–95th percentile, 5 nearest test galaxies
        ex = keep["cleaned"]
        w = ex["mods"]["ord"]["w"]
        s_all = ex["Z"] @ w
        s_tr, s_te = s_all[tr], s_all[te]
        ids_te = r["ids"][te]
        picks = []
        for t in np.linspace(0, 1, 11):
            v = np.percentile(s_tr, 5 + 90 * t)
            i = np.argsort(np.abs(s_te - v))[:5]
            picks.append({"score": float(v), "objid": [int(o) for o in ids_te[i]], "expected_category": ecat[te][i].round(2).tolist()})
        rec["strip"] = {"along": "cleaned ordinal score w·x (covariate term excluded), train 5th–95th percentile, 11 steps, nearest 5 test galaxies", "picks": picks}
        np.savez(OUT / f"a_{Q['q']}.npz", w_cleaned=keep["cleaned"]["mods"]["ord"]["w"], mu=keep["cleaned"]["mu"], sd=keep["cleaned"]["sd"],
                 th=keep["cleaned"]["mods"]["ord"]["th"], w_raw=keep["raw"]["mods"]["ord"]["w"])
        rec["seconds"] = time.perf_counter() - t0
        out["questions"][name] = rec
        path.write_text(json.dumps(out, indent=1, default=float))
    log("→ docs/axes/a_ordinal.json")


def step_sheet() -> None:
    from PIL import Image, ImageDraw, ImageFont
    import j4_spread_controls as J
    _, cache = J.check(verbose=False)
    out = json.loads((DOC / "a_ordinal.json").read_text())["questions"]
    T, HEAD, G = 72, 46, 16
    im = Image.new("RGB", (11 * T + 2 * G, len(out) * (5 * T + HEAD + G) + G), (250, 249, 246))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    for k, (name, r) in enumerate(out.items()):
        y0 = G + k * (5 * T + HEAD + G)
        c, rw = r["embeddings"]["cleaned"]["A1"], r["embeddings"]["raw"]["A1"]
        dr.text((G, y0), f"{name}: {' → '.join(x.split('_', 1)[1] for x in r['categories'])}", fill=(29, 27, 25), font=font)
        dr.text((G, y0 + 16), f"real galaxies (test), cleaned ordinal score low → high, 5 nearest per slice. Reading: cleaned {c['reading']}, raw {rw['reading']}",
                fill=(90, 85, 80), font=small)
        for j, pk in enumerate(r["strip"]["picks"]):
            dr.text((G + j * T + 4, y0 + 30), f"E={np.mean(pk['expected_category']):.1f}", fill=(90, 85, 80), font=small)
            for i, oid in enumerate(pk["objid"]):
                x = np.asarray(cache.get(int(oid)), dtype=np.float32)
                rgb = np.stack([x[2], x[1], x[0]], -1)
                lo, hi = np.percentile(rgb, [1, 99.7])
                a = (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)[64:192, 64:192] * 255).astype(np.uint8)
                im.paste(Image.fromarray(a).resize((T, T), Image.LANCZOS), (G + j * T, y0 + HEAD + i * T))
    im.save(DOC / "a_strips.png", optimize=True)


def main() -> None:
    DOC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    {"test": step_test, "plants": step_plants, "run": step_run, "sheet": step_sheet}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()


if __name__ == "__main__":
    main()
