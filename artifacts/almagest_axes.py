"""Almagest geometry — A1, all 37 answers through the spiral pipeline (steps 2–4), with a descriptive label each.

DESCRIPTIVE, for the Plate (user brief, 20 Nov 2026, branch `almagest-geometry`): no claim, nothing in the science
record. Every answer as `almagest_spiral.py` did spiral, with its accepted deviations: the same 20-column cleaner
(fitted once on the ladder union's train rows — answer-independent, so reused, not refitted), the cleaned probe's
score d, γ(d) through equal-count bin means of d, s spanning the first to last bin mean, bend against γ's own
chord and Σw. No eligibility gate: every answer gets an axis and a label (NOISE / PROXY of X / WEAK / STRONG).

    .venv/bin/python artifacts/almagest_axes.py fit        # per answer → runs/almagest/a1/axes/<answer>.{json,npz}
    .venv/bin/python artifacts/almagest_axes.py assemble   # labels, proxies + residuals, Σw matrix, table, Plate
    .venv/bin/python artifacts/almagest_axes.py sheet      # one strip per answer, in one contact sheet

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import almagest_spiral as S  # noqa: E402  (sets A1's ladder environment and the read-only bank links)

import numpy as np  # noqa: E402

R, REPO, ALM, DOCS = S.R, S.REPO, S.ALM, S.DOCS
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.extract import feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, _fit  # noqa: E402

AX = ALM / "axes"
SEED = 20261120
N_SHUFFLE = 5
PROXY_COS = 0.8
PLANE_COS = 0.3
SLICE_HALF = 0.05
PLATE_OUT = ALM / "plate_axes.bin"  # the spiral-only v3 file (plate.bin) is left as it is


def n_bins(n: int) -> int:
    """20 equal-count bins if every one keeps ≥ 500; else as many as keep ≥ 100 (max 20), at least 10; 0 = THIN."""
    if n // 20 >= 500:
        return 20
    k = min(20, n // 100)
    return k if k >= 10 else 0


class GammaN(S.Gamma):
    """S.Gamma with the bin count as a parameter (the spiral used 20)."""

    def __init__(self, Z, dsc, nb: int) -> None:  # the body of S.Gamma.__init__, N_DBINS → nb
        from scipy.interpolate import make_smoothing_spline
        self.nb = nb
        q = np.quantile(dsc, np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(q, dsc, side="right") - 1, 0, nb - 1)
        self.counts = np.bincount(b, minlength=nb)
        assert (self.counts > 0).all()
        self.d_ranges = np.array([[dsc[b == k].min(), dsc[b == k].max()] for k in range(nb)])
        M = np.array([Z[b == k].mean(0) for k in range(nb)])
        self.dbar = np.array([dsc[b == k].mean() for k in range(nb)])
        self.centre, self.P, self.var = S.top_k(M)
        self.k = self.P.shape[0]
        self.Y = (M - self.centre) @ self.P.T
        self.spl = [make_smoothing_spline(self.dbar, self.Y[:, j]) for j in range(self.k)]
        self.g0 = (self.Y[1] - self.Y[0]) / (self.dbar[1] - self.dbar[0])
        self.g1 = (self.Y[-1] - self.Y[-2]) / (self.dbar[-1] - self.dbar[-2])
        self.lo, self.hi = float(self.dbar[0]), float(self.dbar[-1])
        self.train_d_range = (float(dsc.min()), float(dsc.max()))
        self.grid = np.linspace(self.lo, self.hi, S.D_GRID)
        step = np.linalg.norm(np.diff(self.at(self.grid), axis=0), axis=1)
        assert (step > 0).all(), "γ stalls: s would not be strictly monotone in d"
        self.arc = np.concatenate([[0], np.cumsum(step)])
        self.length = float(self.arc[-1])
        self.speed0, self.speed1 = float(np.linalg.norm(self.g0)), float(np.linalg.norm(self.g1))

    def describe(self) -> dict:
        out = super().describe()
        out["n_bins"] = self.nb
        return out


def gamma_or_none(Z, d, nb):
    if not nb:
        return None, "THIN"
    try:
        return GammaN(Z, d, nb), None
    except AssertionError as e:
        return None, f"γ failed: {e}"


class Probe:
    """The ladder's L2 logistic probe (its own standardisation), read in embedding units as well."""

    def __init__(self, X, y, c) -> None:
        self.sc, clf = _fit(Embeddings(X, y, np.zeros(len(y))), c=c)
        self.w, self.b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
        self.we = self.w / self.sc.scale_  # filter in embedding units: d = X·we + be
        self.be = self.b - float(self.sc.mean_ @ self.we)
        self.sigma_e = np.cov(X.T) @ self.we  # pattern Σw in embedding units (the mean moves along it as d grows)

    def d(self, X):
        return X @ self.we + self.be

    def z(self, X):
        return self.sc.transform(X)


def unit(v):
    return v / np.linalg.norm(v)


def cosv(a, b) -> float:
    return float(unit(a) @ unit(b))


class Data:
    """A1's R footing over all features: union, cleaned matrices for A1 and the three untrained seeds."""

    def __init__(self, untrained: bool = True) -> None:
        t0 = time.perf_counter()
        self.setup = R.prepare("runs/m_v2/encoder.pt", R.MAX_TRAIN, label="ALM", sources=1)
        self.ctx = R.Ctx(self.setup, load_frozen_encoder(self.setup.ckpt), 0, dry=True)
        self.lab = self.ctx.labels["full"]
        self.feats = list(self.ctx.features)
        self.c = self.setup.pc.c
        real = self.ctx.real
        self.union = np.asarray(real.object_ids)
        self.row = {int(o): i for i, o in enumerate(self.union)}
        self.N = np.column_stack([self.lab.nuisance_value(n, self.union) for n in S.NUIS])
        self.ok = np.isfinite(self.N).all(1)
        trm = np.isin(self.union, np.asarray(self.setup.train_ids))
        self.mats = {"A1": real.x.astype(np.float64)}
        if untrained:
            for u in self.ctx.untrained:
                assert np.array_equal(np.asarray(u.object_ids), self.union)
                self.mats[u.encoder_name] = u.x.astype(np.float64)
        self.clean, self.cleaner = {}, {}
        for name, X in self.mats.items():  # one cleaner per encoder, fitted on union train rows: answer-independent
            cl = S.Cleaner(X[self.ok & trm], self.N[self.ok & trm])
            Xc = np.full_like(X, np.nan)
            Xc[self.ok] = cl(X[self.ok], self.N[self.ok])
            self.clean[name], self.cleaner[name] = Xc, cl
        saved = np.load(ALM / "a1_cleaned.npy")
        ok = self.ok
        self.reuse_check = float(np.max(np.abs(saved[ok] - self.clean["A1"][ok])))
        assert self.reuse_check < 1e-3, f"the A1 cleaner differs from the spiral run's ({self.reuse_check})"
        print(f"data {time.perf_counter() - t0:.0f}s, cleaner reuse max|Δ| {self.reuse_check:.2e}", file=sys.stderr)

    def rows(self, f):
        tr = np.array([self.row[int(i)] for i in feature_ids(self.ctx.real, self.lab, f, self.setup.train_ids)])
        te = np.array([self.row[int(i)] for i in feature_ids(self.ctx.real, self.lab, f, self.setup.test_ids)])
        return tr[self.ok[tr]], te[self.ok[te]]

    def yv(self, f, rows):
        v = np.asarray(self.lab.vote_fraction(f, self.union[rows]), dtype=np.float64)
        return (v >= 0.5).astype(int), v

    def reach(self, f, rows):
        ids = self.union[rows]
        return sum(np.asarray(self.lab._column(ids, c), dtype=np.float64)
                   for c in self.lab.scheme.by_name[f].reach_count_cols())


def two_class(*ys) -> bool:
    return all(len(np.unique(y)) == 2 for y in ys)


def split_half(X, y, c, rng):
    """Two disjoint random halves of train: cos of the two w's and of the two Σw's (embedding units); the same for
    N_SHUFFLE label shuffles over the same galaxies and halves."""
    half = rng.permutation(len(y)) < len(y) // 2
    out = {"n_halves": [int(half.sum()), int((~half).sum())]}
    if not two_class(y[half], y[~half]):
        return {**out, "cos_w": None, "reason": "a half has one class"}
    a, b = Probe(X[half], y[half], c), Probe(X[~half], y[~half], c)
    out["cos_w"], out["cos_sigma_w"] = cosv(a.we, b.we), cosv(a.sigma_e, b.sigma_e)
    sw, ss = [], []
    for _ in range(N_SHUFFLE):
        yp = rng.permutation(y)
        if not two_class(yp[half], yp[~half]):
            continue
        a, b = Probe(X[half], yp[half], c), Probe(X[~half], yp[~half], c)
        sw.append(cosv(a.we, b.we)); ss.append(cosv(a.sigma_e, b.sigma_e))
    out["shuffled_cos_w"], out["shuffled_cos_sigma_w"] = sw, ss
    out["shuffled_max_cos_w"] = max(sw) if sw else None
    out["shuffled_max_cos_sigma_w"] = max(ss) if ss else None
    out["stable"] = bool(sw) and out["cos_w"] > max(sw)
    return out


def bend_of(g, P: Probe, Z) -> dict:
    G = g.sampled()
    sig_z = np.cov(Z.T) @ P.w
    bd = S.bend(G, P.w, sig_z, g.length)
    return {k: v for k, v in bd.items() if not k.startswith("_")}, bd


# --- fit ---------------------------------------------------------------------------------------------
def plate_inputs(d: Data):
    import plate_export as PE
    h, A = PE.read_plate(S.PLATE_OUT)  # the spiral v3 file: v2 sections + spiral_*; carried byte-identical
    gid = A["objid"].astype(np.int64)
    bank = np.load(S.LADDER_A1 / "c2_a1_embeddings.npz", allow_pickle=False)
    pos = {int(o): i for i, o in enumerate(bank["ids"])}
    Xp = bank["x"][[pos[int(o)] for o in gid]].astype(np.float64)
    Np = np.column_stack([d.lab.nuisance_value(n, list(gid)) for n in S.NUIS])
    okp = np.isfinite(Np).all(1)
    Xpc = np.full_like(Xp, np.nan)
    Xpc[okp] = d.cleaner["A1"](Xp[okp], Np[okp])
    return h, A, gid, Xp, Xpc, okp


def fit_answer(d: Data, j: int, f: str, plate) -> tuple[dict, dict]:
    h, A, gid, Xp, Xpc, okp = plate
    tr, te = d.rows(f)
    y_tr, v_tr = d.yv(f, tr); y_te, v_te = d.yv(f, te)
    nb = n_bins(len(tr))
    rec = {"answer": f, "j": j, "question": h["answers"][j]["question"], "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_pos_train": int(y_tr.sum()), "n_pos_test": int(y_te.sum()), "n_bins": nb, "thin": nb == 0}
    arr: dict = {}
    Xc, X = d.clean["A1"], d.mats["A1"]
    # cleaned probe, d, γ
    P = Probe(Xc[tr], y_tr, d.c)
    Z_tr, Z_te = P.z(Xc[tr]), P.z(Xc[te])
    d_tr, d_te = Z_tr @ P.w + P.b, Z_te @ P.w + P.b
    rec["auc_cleaned"] = S.auc(y_te, d_te)
    g, why = gamma_or_none(Z_tr, d_tr, nb)
    rec["gamma"] = why or "fitted"
    if g is not None:
        s_te = g.s(d_te)
        o = np.argsort(d_te, kind="stable")
        assert (np.diff(s_te[o]) >= 0).all() and S.auc(y_te, s_te) == rec["auc_cleaned"]
        rec["curve"] = {k: v for k, v in g.describe().items() if k not in ("d_bin_ranges", "bin_mean_d", "counts")}
        rec["curve"]["min_bin_count"] = int(g.counts.min())
        rec["bend"], bd = bend_of(g, P, Z_tr)
        rec["test_s_outside_0_1"] = int(((s_te < 0) | (s_te > 1)).sum())
        rec["extrapolation"] = {"ds_dd_below": g.speed0 / g.length, "ds_dd_above": g.speed1 / g.length}
    # raw probe: AUC, raw–cleaned cosine, uncleaned bend
    Pr = Probe(X[tr], y_tr, d.c)
    rec["auc_raw"] = S.auc(y_te, Pr.d(X[te]))
    rec["auc_drop"] = rec["auc_raw"] - rec["auc_cleaned"]
    rec["cos_raw_cleaned_w"] = cosv(Pr.we, P.we)
    Zr = Pr.z(X[tr])
    gr, why_r = gamma_or_none(Zr, Zr @ Pr.w + Pr.b, nb)
    rec["bend_uncleaned"] = bend_of(gr, Pr, Zr)[0] if gr is not None else why_r
    # untrained seeds (cleaned with their own cleaner, same rows)
    un = {}
    for name in [k for k in d.mats if k != "A1"]:
        Pu = Probe(d.clean[name][tr], y_tr, d.c)
        Zu = Pu.z(d.clean[name][tr])
        e = {"auc_cleaned": S.auc(y_te, Pu.d(d.clean[name][te]))}
        gu, wu = gamma_or_none(Zu, Zu @ Pu.w + Pu.b, nb)
        e["bend"] = bend_of(gu, Pu, Zu)[0] if gu is not None else wu
        un[name] = e
    rec["untrained"] = un
    aucs = [e["auc_cleaned"] for e in un.values()]
    rec["untrained_auc_range"] = [min(aucs), max(aucs)]
    ub = [e["bend"] for e in un.values() if isinstance(e["bend"], dict)]
    if ub:
        rec["untrained_bend_range"] = {k: [min(b[k] for b in ub), max(b[k] for b in ub)]
                                       for k in ("chord_dev_max", "bend_1_minus_chord_over_arc", "cos_chord_sigma_w", "cos_chord_w")}
    # poorly-voted tail: γ on each reach half against the full γ (same z space, same d), the spiral's mean-gap bar
    if g is not None:
        reach = d.reach(f, tr); hi = reach >= np.nanmedian(reach)
        rng = np.random.default_rng(SEED); rnd = rng.permutation(len(tr)) < len(tr) // 2
        halves = {}
        for name, m in (("well_voted", hi), ("other_half", ~hi), ("random_a", rnd), ("random_b", ~rnd)):
            halves[name] = gamma_or_none(Z_tr[m], d_tr[m], n_bins(int(m.sum())))
        cf = {"median_reach": float(np.nanmedian(reach)), "n_well_voted": int(hi.sum()), "n_other_half": int((~hi).sum())}
        pairs = {"other_half_vs_full": ("other_half", None), "well_voted_vs_full": ("well_voted", None),
                 "well_voted_vs_other_half": ("well_voted", "other_half"), "random_a_vs_random_b": ("random_a", "random_b")}
        for key, (a, b) in pairs.items():
            ga, gb = halves[a][0], (g if b is None else halves[b][0])
            if ga is None or gb is None:
                cf[key] = halves[a][1] if ga is None else halves[b][1]
                continue
            c = S.compare(ga, gb, g, g.length)
            c["tail_ratio_mean_gap"] = c[f"mean_gap_over_length_s>={S.TAIL_S}"] / c["mean_gap_over_length"]
            cf[key] = c
        tested = [k for k in ("other_half_vs_full", "well_voted_vs_other_half") if isinstance(cf[k], dict)]
        cf["flag"] = (None if not tested else
                      any(cf[k]["tail_ratio_mean_gap"] > S.TAIL_RATIO for k in tested))
        rec["poorly_voted_tail"] = cf
    # stability (cleaned, train)
    rec["stability"] = split_half(Xc[tr], y_tr, d.c, np.random.default_rng(SEED + j))
    # linearity, from the ladder's A1 R1 record (not recomputed)
    r1 = json.loads((S.LADDER_A1 / "r_r1.json").read_text()).get(f"full:{f}", {})
    rec["linearity"] = {k: r1.get(k) for k in ("width", "linear_auc", "mlp_auc", "headroom", "headroom_lo", "headroom_hi", "verdict")}
    rec["linearity"]["mlp_beats_linear_beyond_ci"] = bool(r1.get("headroom_lo") is not None and r1["headroom_lo"] > 0)
    # strip picks (test): s slices ±0.05 nearest γ(s); THIN: 11 d values (train 5th–95th percentile), nearest in d
    picks = {}
    for t in np.linspace(0, 1, 11):
        if g is not None:
            cand = np.flatnonzero(np.abs(s_te - t) <= SLICE_HALF)
            dist = np.linalg.norm(Z_te[cand] - g.full(g.d_of_s([t]))[0], axis=1)
            pick = cand[np.argsort(dist)[:5]]
            key = f"s={t:.1f}"
        else:
            dt = float(np.percentile(d_tr, 5 + 90 * t))
            pick = np.argsort(np.abs(d_te - dt))[:5]; cand = pick
            key = f"d={dt:.2f}"
        picks[key] = {"objid": [int(i) for i in d.union[te][pick]], "vote_fraction": v_te[pick].round(3).tolist(),
                      "n_in_slice": int(cand.size)}
    rec["strip"] = {"along": "s" if g is not None else "d (THIN: train 5th–95th percentile, 11 steps)", "picks": picks}
    # Plate: d, s, |r| for its galaxies; γ, tangents, w, Σw in its 80-D coordinates; old QR axis vs w[j]
    test = (A["flags"] & 1).astype(bool)
    assert not np.isin(gid[test], d.union[tr]).any(), "a Plate test galaxy is in this answer's train rows"
    Zp = np.full_like(Xpc, np.nan); Zp[okp] = P.z(Xpc[okp])
    dp = Zp @ P.w + P.b
    d_lo, d_hi = (float(x) for x in np.nanpercentile(dp, [0.1, 99.9]))
    pl = {"d_lo": d_lo, "d_hi": d_hi}
    arr["d8"] = S.u8(dp, d_lo, d_hi)

    def to80(zk, vector=False):
        x = zk * P.sc.scale_ + (0 if vector else P.sc.mean_)
        st = (x - (0 if vector else A["std_mean"])) / A["std_scale"]
        return st @ A["basis"].T - (0 if vector else A["centre"])

    if g is not None:
        sp = g.s(dp)
        rp = np.linalg.norm(Zp - g.full(np.where(okp, dp, 0.0)), axis=1); rp[~okp] = np.nan
        r_hi = float(np.nanmax(rp))
        arr["s8"], arr["r8"] = S.u8(sp, 0.0, 1.0), S.u8(rp, 0.0, r_hi)
        pl["r_scale"] = r_hi / 254
        knots = np.linspace(0, 1, S.CURVE_POINTS)
        dk = g.d_of_s(knots)
        arr["gamma80"], arr["d_knots"] = to80(g.full(dk)).astype(np.float32), dk.astype(np.float32)
        t0 = (-g.g0 @ g.P) / g.speed0 * g.length; t1 = (g.g1 @ g.P) / g.speed1 * g.length
        arr["tangents80"] = np.stack([to80(t0[None], True)[0], to80(t1[None], True)[0]]).astype(np.float32)
        pl["ds_dd"] = [g.speed0 / g.length, g.speed1 / g.length]
        pl["s_outside_0_1"] = int(((sp < 0) | (sp > 1))[okp].sum())
    else:
        arr["s8"] = arr["r8"] = np.full(len(gid), 255, np.uint8)
    sig_z = np.cov(Z_tr.T) @ P.w
    w_st = (P.w / P.sc.scale_) * A["std_scale"]
    w80r = A["basis"] @ w_st
    p80r = to80(sig_z[None], True)[0]
    p_st = sig_z * P.sc.scale_ / A["std_scale"]
    pl["retention"] = {"filter_w": float(np.linalg.norm(w80r) / np.linalg.norm(w_st)),
                       "pattern_sigma_w": float(np.linalg.norm(p80r) / np.linalg.norm(p_st))}
    arr["w80"], arr["p80"] = unit(w80r).astype(np.float32), unit(p80r).astype(np.float32)
    q = h["answers"][j]["q"]
    yp = A["votes"][j] >= 128
    m = test & (A["counts"][q] >= 1) & okp
    Zq = A["coords"].astype(np.float64) * np.asarray(h["coords"]["scale"])[:, None]
    pb = h["probe_basis"]
    if two_class(yp[m]):
        pl["auc"] = {"n_plate_test_eligible": int(m.sum()), "old_qr_coordinate": S.auc(yp[m], Zq[j][m]),
                     "probe_basis_w_j": S.auc(yp[m], (np.asarray(pb["w"][j]) @ Zq + pb["b"][j])[m]),
                     "cleaned_d": S.auc(yp[m], dp[m])}
    else:
        pl["auc"] = {"n_plate_test_eligible": int(m.sum()), "note": "one class among the Plate's eligible test galaxies"}
    rec["plate"] = pl
    arr.update(we=P.we, be=P.be, sigma_e=P.sigma_e)
    return rec, arr


def step_fit() -> None:
    AX.mkdir(parents=True, exist_ok=True)
    d = Data()
    plate = plate_inputs(d)
    assert [a["id"] for a in plate[0]["answers"]] == d.feats, "Plate answer order ≠ the label scheme's"
    for j, f in enumerate(d.feats):
        if (AX / f"{f}.json").exists():
            continue
        t0 = time.perf_counter()
        rec, arr = fit_answer(d, j, f, plate)
        rec["seconds"] = time.perf_counter() - t0
        np.savez(AX / f"{f}.npz", **arr)
        (AX / f"{f}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(f"[{j + 1}/37] {f}: n {rec['n_train']}, bins {rec['n_bins']}, {rec['seconds']:.0f}s, "
              f"load {os.getloadavg()[0]:.1f}", file=sys.stderr, flush=True)


# --- assemble ------------------------------------------------------------------------------------------
def load_recs():
    feats = json.loads((AX / "_feats.json").read_text()) if (AX / "_feats.json").exists() else None
    recs = {}
    for p in sorted(AX.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        recs[r["answer"]] = r
    order = sorted(recs, key=lambda f: recs[f]["j"])
    return order if feats is None else feats, recs


def step_assemble() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    import plate_export as PE
    order, recs = load_recs()
    assert len(order) == 37, len(order)
    arrs = {f: np.load(AX / f"{f}.npz") for f in order}
    Q = [recs[f]["question"] for f in order]
    SW = np.array([unit(arrs[f]["sigma_e"]) for f in order])
    C = SW @ SW.T
    # base label: NOISE / WEAK / STRONG (stability, then strength); PROXY overrides WEAK/STRONG afterwards
    for f in order:
        r = recs[f]
        st = r["stability"]
        if not st.get("stable"):
            r["label"] = "NOISE"
        else:
            r["label"] = "STRONG" if r["auc_cleaned"] > r["untrained_auc_range"][1] else "WEAK"
        r["base_label"] = r["label"]
    for i, f in enumerate(order):
        r = recs[f]
        others = [(abs(C[i, k]), C[i, k], order[k]) for k in range(37) if Q[k] != Q[i]]
        top = max(others)
        r["proxy"] = {"top_match": top[2], "cos": top[1]}
        strong = [o for o in others if recs[o[2]]["base_label"] == "STRONG" and o[0] >= PROXY_COS]
        if r["base_label"] != "NOISE" and strong:
            x = max(strong)
            r["label"] = f"PROXY of {x[2]}"
            r["proxy"]["of"] = x[2]; r["proxy"]["of_cos"] = x[1]
    # residuals: refit after projecting X's Σw out of the cleaned embeddings
    prox = [f for f in order if "of" in recs[f]["proxy"]]
    if prox:
        d = Data(untrained=False)
        for f in prox:
            u = unit(arrs[recs[f]["proxy"]["of"]]["sigma_e"])
            Xc = d.clean["A1"]
            tr, te = d.rows(f)
            y_tr, _ = d.yv(f, tr); y_te, _ = d.yv(f, te)
            Xtr = Xc[tr] - np.outer(Xc[tr] @ u, u); Xte = Xc[te] - np.outer(Xc[te] @ u, u)
            Pp = Probe(Xtr, y_tr, d.c)
            sh = split_half(Xtr, y_tr, d.c, np.random.default_rng(SEED + recs[f]["j"]))
            recs[f]["proxy"]["residual"] = {"auc": S.auc(y_te, Pp.d(Xte)), "split_half_cos_w": sh.get("cos_w"),
                                            "shuffled_max_cos_w": sh.get("shuffled_max_cos_w")}
    # exclusivity within questions: GZ2 questions are single-choice; vote fractions within one sum to 1
    qs: dict = {}
    for f in order:
        qs.setdefault(recs[f]["question"], []).append(f)
    excl = {q: ("complements (two answers; fractions sum to 1)" if len(a) == 2 else f"{len(a)} exclusive alternatives")
            for q, a in qs.items()}
    # Σw matrix: clustered on 1 − |cos| (complements are anti-parallel and belong together), average linkage
    D = 1 - np.abs(C); np.fill_diagonal(D, 0)
    lk = linkage(squareform(D, checks=False), "average")
    leaf = leaves_list(lk)
    import re
    short = [re.sub(r"^(t\d+)_.*?_a\d+_(.+)$", r"\1:\2", f) for f in order]
    lab1 = {"STRONG": "S", "WEAK": "W", "NOISE": "N"}
    tag = [lab1.get(recs[f]["label"], "P") for f in order]
    fig, ax = plt.subplots(figsize=(13, 11.5))
    im = ax.imshow(C[np.ix_(leaf, leaf)], cmap="RdBu_r", vmin=-1, vmax=1)
    names = [f"{short[i]} [{tag[i]}]" for i in leaf]
    ax.set_xticks(range(37)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(37)); ax.set_yticklabels(names, fontsize=7)
    for a_ in range(37):
        for b_ in range(37):
            ax.text(b_, a_, f"{C[leaf[a_], leaf[b_]]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center",
                    va="center", fontsize=4.2, color="white" if abs(C[leaf[a_], leaf[b_]]) > 0.6 else "#333")
    fig.colorbar(im, ax=ax, shrink=0.7, label="cos(Σw_a, Σw_b), cleaned embedding units")
    ax.set_title("A1, cleaned Σw (pattern) for all 37 answers, clustered on 1 − |cos| (average linkage); "
                 "[S]TRONG [W]EAK [N]OISE [P]ROXY", fontsize=9)
    fig.tight_layout(); fig.savefig(DOCS / "a1_axes_sigma_w_cos.png", dpi=140); plt.close(fig)
    strong = [i for i, f in enumerate(order) if recs[f]["label"] == "STRONG"]
    planes = sorted([(abs(C[a, b]), order[a], order[b], float(C[a, b]), Q[a] == Q[b])
                     for x, a in enumerate(strong) for b in strong[x + 1:] if abs(C[a, b]) < PLANE_COS])
    out = {"note": "descriptive, for Almagest; not in the science record", "order": order,
           "labels": {f: recs[f]["label"] for f in order}, "exclusivity": excl,
           "sigma_w_cos": C.round(4).tolist(), "cluster_order": [order[i] for i in leaf],
           "candidate_planes": [{"a": a, "b": b, "cos": c, "same_question": s} for _, a, b, c, s in planes],
           "rules": {"bins": "20 equal-count if each ≥ 500; else n//100 (max 20) if ≥ 10; else THIN (no γ)",
                     "NOISE": f"split-half cos(w_a, w_b) ≤ max over {N_SHUFFLE} label shuffles (same galaxies, same halves)",
                     "PROXY": f"|cos(Σw, Σw_X)| ≥ {PROXY_COS}, X STRONG (base label) in another question; first match after NOISE",
                     "WEAK/STRONG": "stable; cleaned test AUC within/below vs above the 3 untrained seeds' max (cleaned the same way)",
                     "tail_flag": f"γ on the poorly-voted half (reach < median) vs the full γ, or the two halves: tail (s ≥ {S.TAIL_S}) "
                                  f"mean gap at matched d > {S.TAIL_RATIO}× overall",
                     "planes": f"STRONG pairs with |cos(Σw)| < {PLANE_COS}"},
           "answers": recs}
    (DOCS / "a1_axes.json").write_text(json.dumps(out, indent=1, default=float))
    for f in order:  # labels back onto the per-answer records
        (AX / f"{f}.json").write_text(json.dumps(recs[f], indent=1, default=float))
    write_plate(order, recs, arrs, PE)
    write_table(order, recs, excl, planes)
    print("→ docs/almagest/a1_axes.json, a1_axes_table.md, a1_axes_sigma_w_cos.png;", PLATE_OUT, file=sys.stderr)


def write_plate(order, recs, arrs, PE) -> None:
    h, A = PE.read_plate(S.PLATE_OUT)
    n = len(A["objid"])
    secs = [(sec["name"], A[sec["name"]], sec["meaning"]) for sec in h["sections"]]
    nan80 = np.full((S.CURVE_POINTS, 80), np.nan, np.float32)
    stack = lambda k, empty: np.stack([arrs[f][k] if k in arrs[f] else empty for f in order])  # noqa: E731
    secs += [("axes_d", stack("d8", None), "[37, n] cleaned probe decision per answer, header.axes_all[j].d"),
             ("axes_s", stack("s8", None), "[37, n] γ arc-length coordinate s per answer (255 = missing / THIN)"),
             ("axes_r", stack("r8", None), "[37, n] off-axis |z − γ(d)| per answer (255 = missing / THIN)"),
             ("axes_gamma", stack("gamma80", nan80), "[37, 101, 80] γ at s = 0, 0.01, …, 1 in the 80-D coords (NaN: THIN)"),
             ("axes_d_knots", stack("d_knots", np.full(S.CURVE_POINTS, np.nan, np.float32)), "[37, 101] d at those s"),
             ("axes_tangents", stack("tangents80", np.full((2, 80), np.nan, np.float32)),
              "[37, 2, 80] outward end tangents at s = 0 and s = 1, one unit of s = γ's whole length"),
             ("axes_filter_w", stack("w80", None), "[37, 80] unit filter w (READ a position)"),
             ("axes_pattern", stack("p80", None), "[37, 80] unit pattern Σw (WALK)")]
    for name, a, _ in secs[-8:]:
        assert a.shape[0] == 37 and a.shape[-1] in (n, 80, S.CURVE_POINTS), (name, a.shape)
    h2 = {k: v for k, v in h.items() if k != "sections"}
    h2["axes_all"] = [{"id": f, "j": recs[f]["j"], "label": recs[f]["label"], "proxy_of": recs[f]["proxy"].get("of"),
                       "thin": recs[f]["thin"], "n_bins": recs[f]["n_bins"],
                       "d": {"decode": "d = lo + v*(hi − lo)/254; 255 missing", "lo": recs[f]["plate"]["d_lo"],
                             "hi": recs[f]["plate"]["d_hi"]},
                       "r_scale": recs[f]["plate"].get("r_scale"), "ds_dd_below_above": recs[f]["plate"].get("ds_dd"),
                       "retention": recs[f]["plate"]["retention"]} for f in order]
    h2["axes_all_note"] = ("one cleaned axis per answer (docs/almagest/a1_axes_report.md). Labels are descriptive: NOISE "
                           "(the probe direction does not repeat across train halves above shuffled labels), PROXY of X "
                           "(its pattern Σw lies within |cos| ≥ 0.8 of a STRONG answer's in another question), WEAK "
                           "(stable but no better than an untrained encoder), STRONG. s, |r|, γ and tangents are missing "
                           "for THIN answers (too few galaxies for 10 bins of 100): read them along d only. The old QR "
                           "coordinates and probe_basis are untouched; header.spiral and spiral_* are the spiral-only "
                           "construction and equal row 7 here")
    PE.write_plate(PLATE_OUT, h2, secs)
    h3, A3 = PE.read_plate(PLATE_OUT)
    assert all(np.array_equal(A3[k], A[k]) for k in A), "v2/v3 sections changed"
    assert np.array_equal(A3["axes_d"][7], A["spiral_d"]), "row 7 differs from the spiral run's spiral_d"


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def write_table(order, recs, excl, planes) -> None:
    L = ["| answer | label | raw AUC | cleaned | drop | cos raw–cl | QR / w[j] (Plate, raw) | bins | max off chord A1 (untr.) "
         "| 1−chord/arc A1 (untr.) | uncleaned 1−c/a | cos(chord,Σw) | cos(chord,w) | tail flag | retained w / Σw "
         "| split-half cos w (shuf. max) | MLP>lin |",
         "|" + "---|" * 17]
    cur = None
    for f in order:
        r = recs[f]
        if r["question"] != cur:
            cur = r["question"]
            L.append(f"| **{cur}** — {excl[cur]} |" + " |" * 16)
        b, ub, unb = r.get("bend"), r.get("untrained_bend_range"), r.get("bend_uncleaned")
        rng = lambda k: f"{fmt(b[k])} ({fmt(ub[k][0])}–{fmt(ub[k][1])})" if b and ub else (fmt(b[k]) if b else "—")  # noqa: E731
        pa = r["plate"]["auc"]
        tf = r.get("poorly_voted_tail", {}).get("flag")
        st = r["stability"]
        L.append(" | ".join([
            f"| {f.split('_', 2)[2] if f.count('_') >= 2 else f}", r["label"], fmt(r["auc_raw"]), fmt(r["auc_cleaned"]),
            fmt(r["auc_drop"]), fmt(r["cos_raw_cleaned_w"]),
            f"{fmt(pa.get('old_qr_coordinate'))} / {fmt(pa.get('probe_basis_w_j'))}",
            "THIN" if r["thin"] else str(r["n_bins"]), rng("chord_dev_max"), rng("bend_1_minus_chord_over_arc"),
            fmt(unb["bend_1_minus_chord_over_arc"]) if isinstance(unb, dict) else "—",
            fmt(b["cos_chord_sigma_w"]) if b else "—", fmt(b["cos_chord_w"]) if b else "—",
            {True: "**FLAG**", False: "—", None: "n/a"}[tf],
            f"{fmt(r['plate']['retention']['filter_w'], 2)} / {fmt(r['plate']['retention']['pattern_sigma_w'], 2)}",
            f"{fmt(st.get('cos_w'))} ({fmt(st.get('shuffled_max_cos_w'))})",
            "**YES**" if r["linearity"]["mlp_beats_linear_beyond_ci"] else "no"]) + " |")
    P = ["| a | b | cos(Σw) | same question |", "|---|---|---|---|"]
    P += [f"| {a} | {b} | {c:+.3f} | {'yes' if s else 'no'} |" for _, a, b, c, s in planes]
    X = ["| answer | label | top match outside its question | cos | residual AUC | residual split-half cos w |",
         "|---|---|---|---|---|---|"]
    for f in order:
        p = recs[f]["proxy"]; rs = p.get("residual", {})
        X.append(f"| {f} | {recs[f]['label']} | {p['top_match']} | {p['cos']:+.3f} | {fmt(rs.get('auc'))} | "
                 f"{fmt(rs.get('split_half_cos_w'))} |")
    (DOCS / "a1_axes_table.md").write_text("## Per answer\n\n" + "\n".join(L) + "\n\n## Candidate two-concept planes "
                                           "(STRONG pairs, |cos Σw| < 0.3)\n\n" + "\n".join(P) +
                                           "\n\n## Proxy readings\n\n" + "\n".join(X) + "\n")


# --- sheet ---------------------------------------------------------------------------------------------
def step_sheet() -> None:
    from PIL import Image, ImageDraw, ImageFont
    import j4_spread_controls as J
    order, recs = load_recs()
    _, cache = J.check(verbose=False)
    T, COLS, GAP, HEAD = 56, 3, 14, 30
    sw, sh = 11 * T, 5 * T + HEAD
    rows = -(-len(order) // COLS)
    sheet = Image.new("RGB", (COLS * sw + (COLS + 1) * GAP, rows * (sh + GAP) + GAP + 40), (250, 249, 246))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except OSError:
        font = small = ImageFont.load_default()
    dr.text((GAP, 10), "A1, all 37 answers: test galaxies along each cleaned axis (s = 0 … 1, ±0.05 slices, 5 nearest γ(s); "
                       "THIN: along d), central 128 px of the 256 px stamp, i r g, shown at 56 px. Label per strip.",
            fill=(29, 27, 25), font=small)
    colour = {"STRONG": (29, 27, 25), "WEAK": (150, 110, 20), "NOISE": (192, 57, 43)}
    for n, f in enumerate(order):
        r = recs[f]
        x0, y0 = GAP + (n % COLS) * (sw + GAP), 40 + GAP + (n // COLS) * (sh + GAP)
        dr.text((x0, y0 + 2), f"{f}", fill=(29, 27, 25), font=small)
        dr.text((x0, y0 + 15), f"{r['label']}" + ("  (THIN, along d)" if r["thin"] else ""),
                fill=colour.get(r["label"], (40, 90, 160)), font=font)
        for c, (key, p) in enumerate(r["strip"]["picks"].items()):
            for k, oid in enumerate(p["objid"]):
                im = np.asarray(cache.get(int(oid)), dtype=np.float32)
                rgb = np.stack([im[2], im[1], im[0]], -1)
                lo, hi = np.percentile(rgb, [1, 99.7])
                a = (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)[64:192, 64:192] * 255).astype(np.uint8)
                sheet.paste(Image.fromarray(a).resize((T, T), Image.LANCZOS), (x0 + c * T, y0 + HEAD + k * T))
    out = DOCS / "a1_axes_contact_sheet.jpg"
    sheet.convert("RGB").save(out, quality=90, optimize=True)
    print(f"→ {out} {out.stat().st_size / 1e6:.1f} MB", file=sys.stderr)


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"fit": step_fit, "assemble": step_assemble, "sheet": step_sheet}.get(step, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()


if __name__ == "__main__":
    main()
