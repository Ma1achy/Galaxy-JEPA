"""Part F (user, 24 Nov 2026): is it the labels or the encoder? Frozen embeddings only; Part A's fits imported unchanged. Descriptive.

    .venv/bin/python artifacts/axes_f.py f1     # Part A's fits in bins of the galaxy's vote count on the question
    .venv/bin/python artifacts/axes_f.py f2     # GZ DECaLS votes as the target, SDSS A1 embeddings as input, same galaxies with GZ2 votes

Fixed before any F number was computed:
- Vote count = Part A's n (votes on the ordered answers; "can't tell" is not one). Bins 1–4, 5–8, 9–16, 17+; a bin with train or test
  below 500 is merged into its smaller neighbour. Arm count is reported with Part A's marginal baseline and, beside it, A-bis's matched one.
- Per bin: (i) fitted and tested in the bin; (ii) the all-train fit tested in the bin; (iii) the 17+ bin (or the top bin left) with its
  votes thinned to 4, in-bin fit and test: if the labels' noise alone drives the low bins, (iii) should drop to the 1–4 bin's level.
- τ = Kendall τ of w·x with the expected category over the bin's test rows, with a 200-resample CI for A1 and the untrained as points.
- F2 target, per galaxy and question: GZD-5 if it has a vote on the ordered answers, else GZD-1/2, else GZD-8. Same galaxies, GZ2 target.
  Bulge has no shared answer set (aa_findings.md: the 3-way collapse agrees on 47%), so it is native per campaign: GZD-5's five answers on
  GZD-5 rows, GZD-1/2's three on GZD-1/2 rows; GZ2's native four on the same rows. No covariate on either side (GZD-1/2 has no "can't tell").

Investigation code: terse, excluded from lint/CI. Writes docs/axes/f1.json, f2.json; large outputs to runs/almagest/axes_phys/f/.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import kendalltau

import axes_a_bis as AB
import axes_a_ordinal as A

BINS = [(1, 4), (5, 8), (9, 16), (17, 10**6)]
MIN = 500
SEED = A.SEED + 60
ENC = ("A1", "untrained-s0", "untrained-s1", "untrained-s2")


def merge_bins(n, tr, te):
    b = [list(x) for x in BINS]
    count = lambda x: (int(((n >= x[0]) & (n <= x[1]) & tr).sum()), int(((n >= x[0]) & (n <= x[1]) & te).sum()))  # noqa: E731
    merged = []
    while len(b) > 1:
        small = [k for k in range(len(b)) if min(count(b[k])) < MIN]
        if not small:
            break
        k = small[0]
        j = k + 1 if k == 0 else k - 1 if k == len(b) - 1 else min((k - 1, k + 1), key=lambda q: min(count(b[q])))
        lo, hi = sorted((k, j))
        merged.append(f"{b[lo]}+{b[hi]}")
        b[lo:hi + 1] = [[b[lo][0], b[hi][1]]]
    return [tuple(x) for x in b], merged


def tau_ci(s, e, rng):
    t = float(kendalltau(s, e)[0])
    bs = [kendalltau(s[i], e[i])[0] for i in (rng.integers(0, len(s), len(s)) for _ in range(200))]
    return t, [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]


def fit_test(X, ct, p, n, tr, te, seed, cov):
    """Part A's analyse (same code) plus A-bis's matched baseline where there is a covariate. Returns the record and the test score."""
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    c = (ct - ct[tr].mean()) / (ct[tr].std() or 1.0)
    m, rec, lls = {}, {}, {}
    for kind in ("ord", "free"):
        m[kind], rec[kind] = A.cv_fit(kind, Z[tr], c[tr], p[tr], n[tr], np.random.default_rng(seed))
        lls[kind] = (p[te] * A.logp(kind, m[kind], Z[te], c[te])).sum(1)
    marg = n[tr] @ p[tr] / n[tr].sum()
    lls["base"] = p[te] @ np.log(np.clip(marg, 1e-12, None))
    out = A.reading(lls, n[te], np.random.default_rng(seed + 1))
    out["lambda"] = {k: rec[k]["lambda"] for k in rec}
    if cov:
        for kind in ("ord", "free"):
            m0 = AB.cov_only(kind, c[tr], p[tr], n[tr])
            lls[kind + "cov"] = (p[te] * A.logp(kind, m0, np.zeros((te.sum(), 1)), c[te])).sum(1)
        r2 = AB.reading2(lls, n[te], np.random.default_rng(seed + 1))
        out["matched_baseline"] = {k: r2[k] for k in ("gap_free", "gap_free_ci95", "share", "share_ci95", "reading")}
    return out, (Z, m, c, marg)


def slim(a):
    keep = ("gap_free_base", "gap_ci95", "gap_ord_base", "share", "share_ci95", "reading", "share_undefined_resamples", "lambda", "matched_baseline", "LL")
    return {k: a[k] for k in keep if k in a}


def lls_at(Z, c, m, p, marg, sel):
    return {**{k: (p[sel] * A.logp(k, m[k], Z[sel], c[sel])).sum(1) for k in ("ord", "free")}, "base": p[sel] @ np.log(np.clip(marg, 1e-12, None))}


def thin(p, n, k, rng):
    """Multinomial draw of min(n, k) votes from each galaxy's own counts (without replacement), as new soft targets."""
    cts = np.round(p * n[:, None]).astype(int)
    out = np.array([rng.multivariate_hypergeometric(c, min(k, c.sum())) if c.sum() else c for c in cts], float)
    tot = out.sum(1)
    return out / tot[:, None], tot


def step_f1(only=None):
    AXS, d = A.load()
    path = A.DOC / "f1.json"
    out = json.loads(path.read_text()) if path.exists() else {"questions": {}}
    for qi, name in enumerate(A.QUESTIONS):
        if name in out["questions"]:
            continue
        r = A.question_rows(AXS, d, name)
        rows, p, n, ct, tr, te = (r[k] for k in ("rows", "p", "n", "ct", "tr", "te"))
        cov = A.QUESTIONS[name]["cov"] is not None
        K = p.shape[1]
        ecat = p @ np.arange(1, K + 1)
        bins, merged = merge_bins(n, tr, te)
        rec = {"merged": merged, "bins": [{"range": list(b), "n_train": int(((n >= b[0]) & (n <= b[1]) & tr).sum()),
                                           "n_test": int(((n >= b[0]) & (n <= b[1]) & te).sum())} for b in bins], "cleaned": {}, "raw": {}}
        size = np.asarray(d.lab.nuisance_value("size", d.union), float)[rows] / np.asarray(d.lab.nuisance_value("psf", d.union), float)[rows]
        for b, e in zip(bins, rec["bins"]):
            m = (n >= b[0]) & (n <= b[1]) & te
            e["median_size_over_psf"] = float(np.median(size[m]))
            e["median_votes_test"] = float(np.median(n[m]))
            e["ecat_sd_test"] = float(ecat[m].std())
            e["binomial_noise_note"] = "ecat SD includes vote noise; not corrected"
        for kind in ("cleaned", "raw"):
            for enc in ENC:
                X = (d.clean if kind == "cleaned" else d.mats)[enc][rows]
                t0 = time.perf_counter()
                allfit, (Z, m, c, marg) = fit_test(X, ct, p, n, tr, te, SEED + qi, cov)
                entry = {"all_galaxies": slim(allfit), "bins": []}
                s_all = Z @ m["ord"]["w"]
                for b in bins:
                    inb = (n >= b[0]) & (n <= b[1])
                    a, (Zb, mb, cb, _) = fit_test(X, ct, p, n, tr & inb, te & inb, SEED + qi, cov)
                    sel = te & inb
                    be = {"range": list(b), "in_bin": slim(a)}
                    rng = np.random.default_rng(SEED + 7)
                    sb = (Zb @ mb["ord"]["w"])[sel]
                    if enc == "A1":
                        be["tau_in_bin"], be["tau_in_bin_ci"] = tau_ci(sb, ecat[sel], rng)
                        be["tau_allfit"], be["tau_allfit_ci"] = tau_ci(s_all[sel], ecat[sel], rng)
                    else:
                        be["tau_in_bin"] = float(kendalltau(sb, ecat[sel])[0])
                        be["tau_allfit"] = float(kendalltau(s_all[sel], ecat[sel])[0])
                    be["allfit"] = {k: v for k, v in A.reading(lls_at(Z, c, m, p, marg, sel), n[sel], np.random.default_rng(SEED + 8)).items()
                                    if k in ("gap_free_base", "gap_ci95", "share", "share_ci95")}
                    entry["bins"].append(be)
                top = bins[-1]  # thinning control on the top bin: labels reduced to 4 votes
                inb = (n >= top[0]) & (n <= top[1])
                pt, nt = thin(p[inb], n[inb], 4, np.random.default_rng(SEED + 9))
                P_, N_ = p.copy(), n.copy()
                P_[inb], N_[inb] = pt, nt
                sel = te & inb
                a, (Zb, mb, _, _) = fit_test(X, ct, P_, N_, tr & inb, sel, SEED + qi, cov)
                e4 = P_ @ np.arange(1, K + 1)
                sb = (Zb @ mb["ord"]["w"])[sel]
                entry["thinned_top_bin"] = {**slim(a), "tau": tau_ci(sb, e4[sel], np.random.default_rng(SEED + 7))[0] if enc == "A1" else float(kendalltau(sb, e4[sel])[0]),
                                            "tau_vs_full_labels": float(kendalltau(sb, ecat[sel])[0]), "votes": 4}
                rec[kind][enc] = entry
                A.log(f"F1 {name} {kind} {enc} {time.perf_counter() - t0:.0f}s gaps " + " ".join(f"{e['range'][0]}:{e['in_bin']['gap_free_base']:.4f}/τ{e['tau_in_bin']:.3f}" for e in entry["bins"])
                      + f" thin {entry['thinned_top_bin']['gap_free_base']:.4f}/τ{entry['thinned_top_bin']['tau']:.3f}")
        out["questions"][name] = rec
        path.write_text(json.dumps(out, indent=1, default=float))


# --- F2 ------------------------------------------------------------------------------------------------------------------
DEC_ARM = ["1", "2", "3", "4", "more-than-4"]
F2Q = {  # GZ2 question → DECaLS column stem and ordered answers (native sets)
    "arm count": ("spiral-arm-count", DEC_ARM),
    "winding": ("spiral-winding", ["tight", "medium", "loose"]),
}
BULGE = {"gzd5": ("bulge-size", ["none", "small", "moderate", "large", "dominant"]), "gzd12": ("bulge-size", ["none", "obvious", "dominant"])}


def decals_counts(setup):
    """object_id → per-campaign DECaLS counts for our union (the Brief AA match: nearest within 3″, one row per galaxy)."""
    import aa2_decals as A2
    ext = A.REPO / "artifacts/out/ext"  # Almagest's environment moves R.OUT; the Brief AA files are read where they were verified
    A2.CAMPAIGNS = {c: ext / p.name for c, p in A2.CAMPAIGNS.items()}
    t = A2.ours(setup)
    dec = A2.load_decals()
    camp = {}
    for c, df in dec.items():
        m = A2.crossmatch(t, df)
        camp[c] = (t.object_id.astype(np.int64).to_numpy()[m.row.to_numpy()], df.iloc[m.src.to_numpy()].reset_index(drop=True))
    return camp


def step_f2():
    AXS, d = A.load()
    camp = decals_counts(d.setup)
    ids_u = d.union.astype(np.int64)
    out = {"questions": {}}

    def counts(c, stem, answers):
        oid, df = camp[c]
        cols = [f"{stem}_{a}" for a in answers]
        if not all(x in df.columns for x in cols):
            return None
        M = df[cols].apply(lambda s: s.astype(float)).fillna(0).to_numpy()
        dd = dict(zip(oid.tolist(), M))
        return np.array([dd.get(int(i), np.full(len(answers), np.nan)) for i in ids_u])
    jobs = []
    for name, (stem, ans) in F2Q.items():
        per = {c: counts(c, stem, ans) for c in ("gzd5", "gzd12", "gzd8")}
        jobs.append((name, "DECaLS pooled (GZD-5 where it has a vote)", per, ("gzd5", "gzd12", "gzd8")))
        jobs.append((name, "GZD-5 only", per, ("gzd5",)))
        jobs.append((name, "GZD-1/2 only", per, ("gzd12",)))
    for c in ("gzd5", "gzd12"):
        jobs.append(("bulge prominence", f"{c} native answers", {c: counts(c, *BULGE[c])}, (c,)))
    for ji, (name, arm, per, order) in enumerate(jobs):
        r = A.question_rows(AXS, d, name)  # GZ2 side: eligible, complete nuisances, ≥ 1 GZ2 vote on the ordered answers
        rows = r["rows"]
        C = np.full((len(rows), next(iter(per.values())).shape[1]), np.nan)
        src = np.full(len(rows), "", dtype=object)
        for c in order:
            if per[c] is None:
                continue
            M = per[c][rows]
            take = (src == "") & np.isfinite(M).all(1) & (np.nan_to_num(M).sum(1) > 0)
            C[take], src[take] = M[take], c
        ok = src != ""
        rows = rows[ok]
        nd = C[ok].sum(1)
        pd_ = C[ok] / nd[:, None]
        tr, te = r["tr"][ok], r["te"][ok]
        Kd, Kg = pd_.shape[1], r["p"].shape[1]
        ed, eg = pd_ @ np.arange(1, Kd + 1), r["p"][ok] @ np.arange(1, Kg + 1)
        q = lambda v: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "frac_ge_10": float((v >= 10).mean())}  # noqa: E731
        rec = {"arm": arm, "n_train": int(tr.sum()), "n_test": int(te.sum()), "campaigns": {c: int((src[ok] == c).sum()) for c in order},
               "votes_decals": q(nd), "votes_gz2": q(r["n"][ok]), "categories_decals": Kd, "categories_gz2": Kg,
               "label_agreement_spearman_ecat": float(__import__("scipy.stats").stats.spearmanr(ed[te], eg[te])[0]), "enc": {}}
        ct = np.zeros(len(rows))
        for kind in ("cleaned", "raw"):
            rec["enc"][kind] = {}
            for enc in ENC:
                X = (d.clean if kind == "cleaned" else d.mats)[enc][rows]
                e = {}
                for tgt, P, N, ec in (("decals", pd_, nd, ed), ("gz2", r["p"][ok], r["n"][ok], eg)):
                    a, (Z, m, _, _) = fit_test(X, ct, P, N, tr, te, SEED + 100 + ji, False)
                    s = (Z @ m["ord"]["w"])[te]
                    e[tgt] = {**slim(a), "tau_own": float(kendalltau(s, ec[te])[0]),
                              "tau_other": float(kendalltau(s, (eg if tgt == "decals" else ed)[te])[0])}
                    if enc == "A1":
                        e[tgt]["tau_own_ci"] = tau_ci(s, ec[te], np.random.default_rng(SEED))[1]
                        e[tgt]["tau_other_ci"] = tau_ci(s, (eg if tgt == "decals" else ed)[te], np.random.default_rng(SEED))[1]
                rec["enc"][kind][enc] = e
                A.log(f"F2 {name} [{arm}] {kind} {enc} n={tr.sum()}/{te.sum()} decals gap {e['decals']['gap_free_base']:.4f} share {e['decals']['share']:.3f} τ {e['decals']['tau_own']:.3f} | "
                      f"gz2 gap {e['gz2']['gap_free_base']:.4f} share {e['gz2']['share']:.3f} τ {e['gz2']['tau_own']:.3f}")
        out["questions"][f"{name} | {arm}"] = rec
        (A.DOC / "f2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    globals()["step_" + sys.argv[1]]()
