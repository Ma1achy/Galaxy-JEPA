"""Brief Y3 — the encoder against machine-measured pitch angle (Hayes primary; Hart and Yu & Ho
replications).

  --bank      embed Hart / Yu & Ho galaxies in P2's train partition outside the union (M + 3
              untrained draws, the same constructors as R's bank) → out/y3_extra_bank.npz
  --planted3  D28 checks through Y3's code → out/y3_planted.json
  --y3        T1–T5 → out/y3_pitch.json

Embeddings are z-scored with the union's per-dimension mean and sd (as ``Ctx.z``), so the extra
rows sit in the same coordinates as the banked ones. Val is never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402
import u3_graded as U3  # noqa: E402
import v2_independent as V2  # noqa: E402
import w2_name_pcs as W  # noqa: E402
import x1_handedness as X  # noqa: E402
import y_pitch as Y  # noqa: E402

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_direction  # noqa: E402

EXTRA = R.OUT / "y3_extra_bank.npz"
OUT = R.OUT / "y3_pitch.json"
PLANTED = R.OUT / "y3_planted.json"
FLOOR = U3.FLOOR  # t10 reach ≥ 21, as U3
WIND = ("a28_tight", "a29_medium", "a30_loose")
ENCODERS = ("M", "untrained-s0", "untrained-s1", "untrained-s2")


def num(t, c) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(t[c], errors="coerce").to_numpy(float)


def winding(t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w_avg (Masters et al. 2019 Eq. 1), plurality category (0 tight, 1 medium, 2 loose, −1
    none/tie) and reach, from the raw GZ2 t10 columns."""
    frac = np.column_stack([num(t, f"t10_arms_winding_{a}_fraction") for a in WIND])
    cnt = np.column_stack([num(t, f"t10_arms_winding_{a}_count") for a in WIND])
    reach = cnt.sum(1)
    w = 0.5 * frac[:, 1] + 1.0 * frac[:, 0]
    top = np.argmax(np.nan_to_num(frac, nan=-1), 1)
    tied = (frac == np.nanmax(frac, 1, keepdims=True)).sum(1) > 1
    cat = np.where(np.isfinite(frac).all(1) & ~tied & (reach >= FLOOR), top, -1)
    w = np.where(reach >= FLOOR, w, np.nan)
    return w, cat, reach


# ------------------------------------------------------------------ embeddings


class Emb:
    """Every encoder's z-scored embedding for any id in the union or the extra bank."""

    def __init__(self, ctx) -> None:
        mats = [ctx.real, *ctx.untrained]
        self.pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
        self.x = {}
        self.raw = {}
        extra = np.load(EXTRA) if EXTRA.exists() else None
        self.xpos = {int(o): i for i, o in enumerate(extra["ids"])} if extra is not None else {}
        for name, m in zip(ENCODERS, mats, strict=True):
            x = m.x.astype(np.float64)
            mu, sd = x.mean(0), x.std(0) + 1e-8
            e = extra[name].astype(np.float64) if extra is not None else np.zeros((0, x.shape[1]))
            self.raw[name] = np.vstack([x, e])
            self.x[name] = (self.raw[name] - mu) / sd
        self.n_union = len(self.pos)

    def rows(self, ids) -> np.ndarray:
        return np.array([self.pos[o] if o in self.pos else self.n_union + self.xpos[o]
                         for o in ids])

    def has(self, o: int) -> bool:
        return o in self.pos or o in self.xpos


# ------------------------------------------------------------------ tests


def agreement(w: np.ndarray, p: np.ndarray, vis: np.ndarray, seed: int) -> dict:
    ok = np.isfinite(w) & np.isfinite(p) & np.isfinite(vis)
    out = {"n": int(ok.sum())}
    if ok.sum() < Y.MIN_N:
        return out
    out["rho"], out["p"] = W.spearman_perm(w[ok], p[ok], seed=seed)
    out["partial"], out["p_partial"] = U2.partial_perm(w[ok], p[ok], vis[ok][:, None], seed=seed + 1)
    out["ci"] = V2.boot_ci(w[ok], p[ok], seed)
    return out


def agreement_state(c: dict, sig: bool) -> str:
    """Expected sign negative: tighter votes (higher w_avg) ↔ smaller pitch."""
    if c["n"] < Y.MIN_N:
        return "INSUFFICIENT"
    if not sig:
        return "NO AGREEMENT"
    if c["rho"] > 0:
        return "CONTRARY"
    return "AGREE" if c["rho"] <= -0.3 and c["partial"] <= -0.3 else "WEAK AGREEMENT"


def decode_all(E: Emb, tr: np.ndarray, te: np.ndarray, ytr, yte) -> dict:
    """Ridge on each encoder (V2.ridge_predict); predictions on te per encoder."""
    return {n: V2.ridge_predict(E.x[n][tr], ytr, E.x[n][te])[0] for n in ENCODERS}


def two_by_two(E, tr, te, P, Wv, vis, seed: int) -> dict:
    """V2's Experiment D with visibility added to both partials (the brief: control it in every
    comparison)."""
    pp = decode_all(E, tr, te, P[tr], P[te])
    pw = decode_all(E, tr, te, Wv[tr], Wv[te])
    Pt, Wt, Vt = P[te], Wv[te], vis[te]
    rec: dict = {"n_train": int(tr.size), "n_test": int(te.size)}
    for key, pred, y in (("decode_bt", pp, Pt), ("decode_bavg", pw, Wt)):
        rho, p = W.spearman_perm(pred["M"], y, seed=seed + len(key))
        unt = [V2.spearman(pred[n], y) for n in ENCODERS[1:]]
        rec[key] = {"rho": rho, "p": p, "untrained": unt, "margin": [rho - max(unt), rho - min(unt)],
                    "partial_vis": U2.partial_perm(pred["M"], y, Vt[:, None], seed=0, perm=False)[0]}
    # Controls are the OTHER decoder's prediction (the encoder's own, noise-free reading of the
    # other target), not the noisy target: controlling a noisy proxy leaves a shared quantity
    # half-removed, and V2's form reads BOTH on two noisy copies of one thing (planted, Y3).
    for key, pred, y, other_pred, other in (("a_m", pp, Pt, pw, Wt), ("a_v", pw, Wt, pp, Pt)):
        cov = np.column_stack([other_pred["M"], Vt])
        r, p = U2.partial_perm(pred["M"], y, cov, seed=seed + 7 + len(key))
        unt = [U2.partial_perm(pred[n], y, np.column_stack([other_pred[n], Vt]), seed=0,
                               perm=False)[0] for n in ENCODERS[1:]]
        v2_form = U2.partial_perm(pred["M"], y, np.column_stack([other, Vt]), seed=0,
                                  perm=False)[0]
        rec[key] = {"rho": r, "p": p, "untrained": unt, "v2_form_descriptive": v2_form}
    rec["pred_pitch_M"] = pp["M"]
    return rec


# Z2's D28 rate check: at 50 draws and the 95th per leg, a planted shared quantity fired a leg in
# 17 of 100 draws (10% and 7% per leg). 200 draws steady the percentile, and the 97.5th per leg
# is Bonferroni over the two legs. Y3's T2 verdict was hashed at 50 / 95th and holds under both.
N_SHARED = 200


def partials_m(E, tr, te, a, b, vis) -> tuple[float, float]:
    """M's two cross-controlled partials only (no permutation) — the shared null's statistic."""
    pa = V2.ridge_predict(E.x["M"][tr], a[tr], E.x["M"][te])[0]
    pb = V2.ridge_predict(E.x["M"][tr], b[tr], E.x["M"][te])[0]
    v = vis[te]
    am = U2.partial_perm(pa, a[te], np.column_stack([pb, v]), seed=0, perm=False)[0]
    av = U2.partial_perm(pb, b[te], np.column_stack([pa, v]), seed=0, perm=False)[0]
    return am, av


def shared_null(E, tr, te, target, rho_a: float, rho_b: float, vis, seed: int,
                latent: str = "random") -> dict:
    """What the partials read if measurement and votes were two noisy copies of ONE quantity the
    encoder sees, each copy's noise set so its decode ρ matches the observed one.

    latent="random" (primary, from Z): each draw puts the shared latent on a fresh random
    direction of M's embedding. latent="ridge" (Y3's hashed form): the latent is M's ridge fit to
    ``target``. Z2's D28 calibration found the ridge form biased low, because a ridge fit lives in
    the easily decoded subspace: planted shared quantities on random directions exceeded its 97.5th
    in 4–22% of draws, depending on the direction."""
    from sklearn.linear_model import RidgeCV

    rows = np.r_[tr, te]
    rng = np.random.default_rng(seed)
    zM = E.x["M"]

    def standard(v):
        out = np.full(len(target), np.nan)
        out[rows] = (v - v.mean()) / v.std()
        return out

    if latent == "ridge":
        fixed = standard(RidgeCV(alphas=V2.ALPHAS).fit(zM[tr], target[tr]).predict(zM[rows]))
    sa = np.sqrt(max(1 / max(rho_a, 1e-3) ** 2 - 1, 0))
    sb = np.sqrt(max(1 / max(rho_b, 1e-3) ** 2 - 1, 0))
    am, av = [], []
    for _ in range(N_SHARED):
        lat = fixed if latent == "ridge" else standard(zM[rows] @ rng.normal(size=zM.shape[1]))
        a, b = lat.copy(), lat.copy()
        a[rows] += sa * rng.normal(size=rows.size)
        b[rows] += sb * rng.normal(size=rows.size)
        x, y = partials_m(E, tr, te, a, b, vis)
        am.append(x)
        av.append(y)
    return {"a_m_95": float(np.percentile(am, 95)), "a_v_95": float(np.percentile(av, 95)),
            "a_m_975": float(np.percentile(am, 97.5)), "a_v_975": float(np.percentile(av, 97.5)),
            "a_m_median": float(np.median(am)), "a_v_median": float(np.median(av)),
            "noise_sd": [float(sa), float(sb)], "k": N_SHARED, "latent": latent}


def shared_calibration(E, tr, te, vis, null: dict, rng, n: int = 100) -> dict:
    """D28 as a rate: fresh planted shared quantities (random directions, noise 0.5) against
    ``null``. The either-leg rate is the chance a shared quantity reads as a one-sided verdict."""
    n_all = len(vis)
    zM = E.x["M"]
    am, av = [], []
    for _ in range(n):
        s1 = zM @ rng.normal(size=zM.shape[1])
        s1 = (s1 - s1.mean()) / s1.std()
        x, y = partials_m(E, tr, te, s1 + 0.5 * rng.normal(size=n_all),
                          s1 + 0.5 * rng.normal(size=n_all), vis)
        am.append(x)
        av.append(y)
    am, av = np.array(am), np.array(av)
    return {q: {"leg": [float(np.mean(am > null[f"a_m_{q}"])), float(np.mean(av > null[f"a_v_{q}"]))],
                "either": float(np.mean((am > null[f"a_m_{q}"]) | (av > null[f"a_v_{q}"])))}
            for q in ("95", "975")} | {"n": n, "plant_median": [float(np.median(am)),
                                                                 float(np.median(av))]}


def state_2x2(b: dict, sig: dict, q: str = "975") -> str:
    """V2's states, with one more condition on each leg: the partial must exceed the shared-quantity
    null's percentile ``q`` (97.5th from Z: Bonferroni over the two legs; Y3 hashed the 95th)."""
    def leg(k):
        r = b[k]
        if sig[k] and r["rho"] < 0:
            return "INVERTED"
        return bool(sig[k] and r["rho"] >= V2.NEGLIGIBLE and r["rho"] > max(r["untrained"])
                    and r["rho"] > b["shared_null"][f"{k}_{q}"])

    m, v = leg("a_m"), leg("a_v")
    if "INVERTED" in (m, v):
        return f"INVERTED (A_m {m}, A_v {v})"
    if m and v:
        return "BOTH"
    if m:
        return "MEASUREMENT BEYOND VOTES"
    if v:
        return "VOTES BEYOND MEASUREMENT"
    decoded = all(b[k]["state"] == "DECODED" for k in ("decode_bt", "decode_bavg"))
    return "SHARED ONLY" if decoded else "NEITHER"


def spread(p: np.ndarray, cat: np.ndarray, vis: np.ndarray, seed: int) -> dict:
    """U3-C: is medium's pitch spread wider than both tight's and loose's? Pitch is residualised
    on visibility (rank-linear); spread is the MAD; the null permutes categories within
    visibility quintiles."""
    ok = np.isfinite(p) & (cat >= 0) & np.isfinite(vis)
    p, g, v = p[ok], cat[ok], vis[ok]
    counts = np.bincount(g, minlength=3)
    out = {"n": counts.tolist()}
    if counts.min() < Y.MIN_N:
        return out
    rv = rankdata(v)
    a = np.column_stack([np.ones(v.size), rv])
    res = p - a @ np.linalg.lstsq(a, p, rcond=None)[0]

    def mad(x):
        return float(np.median(np.abs(x - np.median(x))))

    def stat(gg):
        m = [mad(res[gg == k]) for k in range(3)]
        return m[1] - max(m[0], m[2]), m

    d, m = stat(g)
    strata = np.minimum((rankdata(v) - 1) * 5 // v.size, 4).astype(int)
    rng = np.random.default_rng(seed)
    null = np.empty(W.N_PERM)
    for i in range(W.N_PERM):
        gg = g.copy()
        for s in range(5):
            ix = np.flatnonzero(strata == s)
            gg[ix] = rng.permutation(gg[ix])
        null[i] = stat(gg)[0]
    out.update(D=d, mad=m, median_pitch=[float(np.median(p[g == k])) for k in range(3)],
               p_wider=float((1 + np.sum(null >= d)) / (1 + W.N_PERM)),
               p_narrower=float((1 + np.sum(null <= d)) / (1 + W.N_PERM)))
    out["p"] = min(1.0, 2 * min(out["p_wider"], out["p_narrower"]))
    return out


def spread_state(c: dict, sig: bool) -> str:
    if min(c["n"]) < Y.MIN_N:
        return "INSUFFICIENT"
    if not sig:
        return "NOT WIDER"
    return "WIDER" if c["D"] > 0 else "NARROWER"


def ordering(p: np.ndarray, cat: np.ndarray, vis: np.ndarray, seed: int) -> dict:
    """Kendall τ of winding category against pitch, raw and within visibility quintiles
    (stratified permutation); expected positive (loose = larger pitch)."""
    ok = np.isfinite(p) & (cat >= 0) & np.isfinite(vis)
    p, g, v = p[ok], cat[ok], vis[ok]
    out = {"n": int(ok.sum())}
    if ok.sum() < Y.MIN_N:
        return out
    strata = np.minimum((rankdata(v) - 1) * 5 // v.size, 4).astype(int)

    def strat_tau(gg):
        return float(np.mean([kendalltau(gg[strata == s], p[strata == s]).statistic
                              for s in range(5)]))

    out["tau_raw"] = float(kendalltau(g, p).statistic)
    obs = strat_tau(g)
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(2000):  # kendalltau is slow; 2,000 resolves BY at m = 4 (rank-1 bar 6e-3)
        gg = g.copy()
        for s in range(5):
            ix = np.flatnonzero(strata == s)
            gg[ix] = rng.permutation(gg[ix])
        hits += abs(strat_tau(gg)) >= abs(obs) - 1e-12
    out.update(tau_within_vis=obs, p=(1 + hits) / 2001)
    return out


def ordering_state(c: dict, sig: bool) -> str:
    if c["n"] < Y.MIN_N:
        return "INSUFFICIENT"
    if not sig:
        return "NOT ORDERED"
    if c["tau_within_vis"] < 0:
        return "REVERSED"
    return ("ORDERED BEYOND VISIBILITY" if c["tau_within_vis"] >= 0.5 * c["tau_raw"]
            else "ORDERED, MOSTLY VISIBILITY")


def axis_vs_pitch(proj: dict, P: np.ndarray, vis: np.ndarray, seed: int) -> dict:
    """U3's winding axis (fitted on A's tight/loose extremes) against measured pitch on B."""
    raw, praw = W.spearman_perm(proj["M"], P, seed=seed)
    part, ppart = U2.partial_perm(proj["M"], P, vis[:, None], seed=seed + 1)
    unt = [U2.partial_perm(proj[n], P, vis[:, None], seed=0, perm=False)[0] for n in ENCODERS[1:]]
    return {"n": int(P.size), "raw": raw, "p_raw": praw, "partial": part, "p": ppart,
            "untrained_partial": unt, "retention": part / raw if raw else float("nan")}


def axis_state(c: dict, sig: bool) -> str:
    """Existence from the raw test; the rest from retention and the untrained partials. The
    tight-positive axis should go with smaller pitch, so the sign is negative; read by
    magnitude, with the sign stated."""
    if c["n"] < Y.MIN_N:
        return "INSUFFICIENT"
    if not sig:
        return "NONE"
    # A reversal must be a real partial, not a zero whose sign is a coin flip: Z2's D28 check
    # read the pure-visibility plant (partial −0.025) as REVERSED. Amended before Z2 was hashed.
    if (np.sign(c["partial"]) != np.sign(c["raw"]) and c["p"] < W.ALPHA
            and abs(c["partial"]) >= V2.NEGLIGIBLE):
        return "REVERSED UNDER CONTROL"
    if c["retention"] < 0.5:
        return "MOSTLY VISIBILITY"
    if abs(c["partial"]) <= max(abs(u) for u in c["untrained_partial"]):
        return "NOT ABOVE UNTRAINED"
    return ("WINDING BEYOND VISIBILITY" if abs(c["partial"]) >= 0.1
            else "BEYOND VISIBILITY, NEGLIGIBLE")


def by(p: dict, m: int) -> dict:
    return nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli", n_tests=m)


# ------------------------------------------------------------------ assembly


def load(setup):
    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    t = Y.table(setup)
    t["oid"] = t.object_id.astype("int64")
    return ctx, t


def winding_axis(ctx, E: Emb) -> dict[str, np.ndarray]:
    """U3's arms_winding direction per encoder: logistic on A's tight vs loose (raw embeddings)."""
    lab = ctx.labels["full"]
    ids, g = U3.categories(lab, "t10_arms_winding", WIND, ctx.setup.train_ids)
    ends = (g == 0) | (g == 2)
    ids = [o for o, k in zip(ids, ends, strict=True) if k]
    y = (g[ends] == 0).astype(np.int64)  # + = tight
    rows = E.rows([int(o) for o in ids])
    return {n: probe_direction(Embeddings(E.raw[n][rows], y, y.astype(float)), name="winding",
                               c=ctx.pc.c).w_unit for n in ENCODERS}


def bank(setup) -> None:
    frozen = load_frozen_encoder(setup.ckpt)
    t = Y.table(setup)
    want = (t.hart_pa.notna() | t.yuho_pa.notna()) & (t.part == "train")
    ids = sorted(int(o) for o in t.object_id[want])
    ds = StampDataset(setup.ds.cache, setup.rows, ids)
    blob = {"ids": np.array(ids)}
    blob["M"] = extract_matrix(frozen, ds, device=setup.device).x
    for s in range(3):
        blob[f"untrained-s{s}"] = ctl.untrained_encoder_matrix(frozen.config, ds,
                                                               device=setup.device, seed=s).x
        R._release(setup.device)
    np.savez(EXTRA, **blob)
    print(f"Y3 bank: {len(ids)} extra galaxies", file=sys.stderr)


def y3(setup) -> dict:
    ctx, t = load(setup)
    E = Emb(ctx)
    oid = t.oid.to_numpy()
    P = num(t, f"H_{Y.PRIMARY}")
    Wv, cat, _ = winding(t)
    rec: dict = {}

    # T1 — agreement (no embeddings; all matched rows in the 230k)
    t1, p1 = {}, {}
    for k, col in (("hayes", f"H_{Y.PRIMARY}"), ("hart", "hart_pa"), ("yuho", "yuho_pa")):
        m = np.isfinite(num(t, col)) & np.isfinite(Wv)
        vis = np.full(len(t), np.nan)
        vis[m] = Y.visibility(t[m])
        t1[k] = agreement(Wv, num(t, col), vis, seed=len(k))
        if "p" in t1[k]:
            p1[k] = t1[k]["p"]
    rec["T1"] = finish(t1, p1, agreement_state)

    # T2 — the 2×2 on Hayes, A → B
    hay = np.isfinite(P) & np.isfinite(Wv)
    vis_h = np.full(len(t), np.nan)
    vis_h[hay] = Y.visibility(t[hay])
    ok = hay & np.isfinite(vis_h)
    trm, tem = ok & (t.part == "A").to_numpy(), ok & (t.part == "B").to_numpy()
    rows_tr, rows_te = E.rows(oid[trm]), E.rows(oid[tem])
    ab_m = ok & t.part.isin(["A", "B"]).to_numpy()
    xP = np.full(E.n_union + len(E.xpos), np.nan)
    xW, xV = xP.copy(), xP.copy()
    r_ab = E.rows(oid[ab_m])
    xP[r_ab], xW[r_ab], xV[r_ab] = P[ab_m], Wv[ab_m], vis_h[ab_m]
    b = two_by_two(E, rows_tr, rows_te, xP, xW, xV, seed=101)
    pred_hayes = b.pop("pred_pitch_M")
    p2 = {k: b[k]["p"] for k in ("decode_bt", "decode_bavg", "a_m", "a_v")}
    fam = len(p2)
    nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=fam)
    sig2 = nulls_mod.family_significant(p2, alpha=W.ALPHA, method="benjamini_yekutieli",
                                        n_tests=fam)
    for k in ("decode_bt", "decode_bavg"):
        b[k]["state"] = V2.decode_state(b[k], sig2[k])
    b["shared_null"] = shared_null(E, rows_tr, rows_te, xP, b["decode_bt"]["rho"],
                                   b["decode_bavg"]["rho"], xV, seed=77)
    b["state"] = state_2x2(b, sig2)
    b["shared_null_ridge"] = shared_null(E, rows_tr, rows_te, xP, b["decode_bt"]["rho"],
                                         b["decode_bavg"]["rho"], xV, seed=77, latent="ridge")
    b["state_y3_rule"] = state_2x2({**b, "shared_null": b["shared_null_ridge"]}, sig2, q="95")
    rec["T2"] = {"hayes": b, "labels": {"decode_bt": "pitch (Hayes)", "decode_bavg": "w_avg",
                                        "a_m": "A_m: pitch beyond votes", "a_v":
                                        "A_v: votes beyond pitch"}}

    # T2 replication: the Hayes-fitted decoder transferred, and 5-fold ridge within each reference
    rep, prep = {}, {}
    for k, col in (("hart", "hart_pa"), ("yuho", "yuho_pa")):
        ref = num(t, col)
        ev = np.isfinite(ref) & ~trm & t.part.isin(["A", "B", "train"]).to_numpy() & np.array(
            [E.has(int(o)) for o in oid])
        vis = np.full(len(t), np.nan)
        vis[ev] = Y.visibility(t[ev])
        ev &= np.isfinite(vis)
        er = E.rows(oid[ev])
        r: dict = {"n": int(ev.sum())}
        tr_fit = rows_tr
        for n in ENCODERS:
            pred = V2.ridge_predict(E.x[n][tr_fit], xP[tr_fit], E.x[n][er])[0]
            cv = cv_pred(E.x[n][er], ref[ev])
            if n == "M":
                r["transfer"], r["p_transfer"] = U2.partial_perm(pred, ref[ev], vis[ev][:, None],
                                                                 seed=len(k) + 5)
                r["transfer_raw"] = V2.spearman(pred, ref[ev])
                r["cv"], r["p_cv"] = U2.partial_perm(cv, ref[ev], vis[ev][:, None], seed=len(k) + 9)
                r["cv_raw"] = V2.spearman(cv, ref[ev])
            else:
                r.setdefault("transfer_untrained", []).append(
                    U2.partial_perm(pred, ref[ev], vis[ev][:, None], seed=0, perm=False)[0])
                r.setdefault("cv_untrained", []).append(
                    U2.partial_perm(cv, ref[ev], vis[ev][:, None], seed=0, perm=False)[0])
        rep[k] = r
        prep[f"{k}:transfer"], prep[f"{k}:cv"] = r["p_transfer"], r["p_cv"]
    sigr = nulls_mod.family_significant(prep, alpha=W.ALPHA, method="benjamini_yekutieli",
                                        n_tests=len(prep))
    for k, r in rep.items():
        for leg in ("transfer", "cv"):
            r[f"{leg}_state"] = rep_state(r[leg], sigr[f"{k}:{leg}"], r[f"{leg}_untrained"])
    rec["T2_replication"] = rep

    # T3 — U3-C spread, and T4a — ordering, on every matched row (no embeddings)
    t3, p3, t4, p4 = {}, {}, {}, {}
    for k, col in (("hayes", f"H_{Y.PRIMARY}"), ("hart", "hart_pa"), ("yuho", "yuho_pa")):
        ref = num(t, col)
        m = np.isfinite(ref) & (cat >= 0)
        vis = np.full(len(t), np.nan)
        vis[m] = Y.visibility(t[m])
        t3[k] = spread(ref, cat, vis, seed=200 + len(k))
        t4[k] = ordering(ref, cat, vis, seed=300 + len(k))
        for d, pp in ((t3, p3), (t4, p4)):
            if "p" in d[k]:
                pp[k] = d[k]["p"]
    rec["T3"] = finish(t3, p3, spread_state)

    # T4b — U3's winding axis against measured pitch, visibility controlled (Hayes, B)
    axis = winding_axis(ctx, E)
    proj = {n: E.raw[n][rows_te] @ axis[n] for n in ENCODERS}
    ab = axis_vs_pitch(proj, P[tem], vis_h[tem], seed=401)
    p4["axis"] = ab["p_raw"]
    nulls_mod.assert_null_resolution(2000, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=len(p4))
    sig4 = nulls_mod.family_significant(p4, alpha=W.ALPHA, method="benjamini_yekutieli",
                                        n_tests=len(p4))
    for k, c in t4.items():
        c["state"] = ordering_state(c, sig4.get(k, False))
    ab["state"] = axis_state(ab, sig4["axis"])
    rec["T4"] = {"ordering": t4, "axis": ab,
                 "decoded_pitch_partial_vis_B": b["decode_bt"]["partial_vis"],
                 "pitch_vs_visibility_B": V2.spearman(P[tem], vis_h[tem])}

    # T5 — exploratory: handedness decodable? arc counts vs votes and visibility
    rec["T5"] = exploratory(t, E, oid, trm, tem, vis_h, pred_hayes)
    return rec


def cv_pred(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import KFold

    out = np.empty(y.size)
    for a, b in KFold(5, shuffle=True, random_state=0).split(x):
        out[b] = V2.ridge_predict(x[a], y[a], x[b])[0]
    return out


def rep_state(r: float, sig: bool, unt: list[float]) -> str:
    if not sig:
        return "NOT REPLICATED"
    if r <= 0:
        return "INVERTED"
    return "REPLICATED" if r > max(unt) else "REPLICATED, NOT ABOVE UNTRAINED"


def finish(d: dict, p: dict, state) -> dict:
    fam = len(p)
    if fam:
        nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                         n_tests=fam)
    sig = nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli",
                                       n_tests=fam) if fam else {}
    for k, c in d.items():
        c["state"] = state(c, sig.get(k, False))
    return d


def exploratory(t, E, oid, trm, tem, vis_h, pred_hayes) -> dict:
    """Labelled exploratory; no verdict states."""
    from sklearn.linear_model import LogisticRegression

    out: dict = {"label": "EXPLORATORY"}
    hs = np.array([Y.sz(v) for v in t.H_chirality_alenWtd])
    for name, lab in (("hayes_chirality", hs),):
        tr, te = trm & np.isfinite(lab), tem & np.isfinite(lab)
        r: dict = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        for n in ENCODERS:
            clf = LogisticRegression(max_iter=3000, C=0.1).fit(E.x[n][E.rows(oid[tr])], lab[tr])
            auc = X._auc(lab[te] > 0, clf.decision_function(E.x[n][E.rows(oid[te])]))
            r[n] = auc
        out[name] = r
    arcs = {}
    arm = np.column_stack([num(t, f"t11_arms_number_{a}_fraction") for a in
                           ("a31_1", "a32_2", "a33_3", "a34_4", "a36_more_than_4")])
    arm_mean = arm @ np.array([1, 2, 3, 4, 5.0]) / np.where(arm.sum(1) > 0, arm.sum(1), np.nan)
    hay = np.isfinite(num(t, f"H_{Y.PRIMARY}"))
    for col in ("H_numDcoArcsGE000", "H_numDcoArcsGE040", "H_numDcoArcsGE100"):
        a = num(t, col)
        m = hay & np.isfinite(a) & np.isfinite(arm_mean)
        vis = Y.visibility(t[m])
        arcs[col[2:]] = {"n": int(m.sum()), "vs_visibility": V2.spearman(a[m], vis),
                         "vs_gz2_arm_mean": V2.spearman(a[m], arm_mean[m]),
                         "vs_gz2_arm_mean_partial_vis": U2.partial_perm(
                             a[m], arm_mean[m], vis[:, None], seed=0, perm=False)[0],
                         "gz2_arm_mean_vs_visibility": V2.spearman(arm_mean[m], vis)}
    out["arcs"] = arcs
    return out


# ------------------------------------------------------------------ planted (D28)


def planted_2x2(E, rtr, rte, oid, ab_m, vis, rng) -> dict:
    """BOTH / SHARED ONLY / NEITHER through the identical two_by_two + shared null + state_2x2.
    Second consumer: Z3 (V2's B/T rows)."""
    n_all = E.n_union + len(E.xpos)
    zM = E.x["M"]
    u1, u2 = rng.normal(size=zM.shape[1]), rng.normal(size=zM.shape[1])
    s1, s2 = zM @ u1, zM @ u2
    s1, s2 = (s1 - s1.mean()) / s1.std(), (s2 - s2.mean()) / s2.std()
    xv = np.full(n_all, np.nan)
    xv[E.rows(oid[ab_m])] = vis[ab_m]
    out = {}
    for name, (a, bb) in {"both": (s1 + 0.5 * rng.normal(size=n_all), s2 + 0.5 * rng.normal(size=n_all)),
                          "shared": (s1 + 0.5 * rng.normal(size=n_all), s1 + 0.5 * rng.normal(size=n_all)),
                          "neither": (rng.normal(size=n_all), rng.normal(size=n_all))}.items():
        b = two_by_two(E, rtr, rte, a, bb, xv, seed=11)
        b.pop("pred_pitch_M")
        p = {k: b[k]["p"] for k in ("decode_bt", "decode_bavg", "a_m", "a_v")}
        sig = by(p, 4)
        for k in ("decode_bt", "decode_bavg"):
            b[k]["state"] = V2.decode_state(b[k], sig[k])
        b["shared_null"] = shared_null(E, rtr, rte, a, b["decode_bt"]["rho"],
                                       b["decode_bavg"]["rho"], xv, seed=78)
        ridge = shared_null(E, rtr, rte, a, b["decode_bt"]["rho"], b["decode_bavg"]["rho"], xv,
                            seed=78, latent="ridge")
        out[f"T2_{name}"] = {"state": state_2x2(b, sig),
                             "state_y3_rule": state_2x2({**b, "shared_null": ridge}, sig, q="95"),
                             "shared_null": b["shared_null"],
                             "a_m": b["a_m"]["rho"], "a_v": b["a_v"]["rho"],
                             "v2_form": [b["a_m"]["v2_form_descriptive"], b["a_v"]["v2_form_descriptive"]],
                             "decode": [b["decode_bt"]["rho"], b["decode_bavg"]["rho"]]}
        if name == "shared":
            out["T2_calibration"] = {"random": shared_calibration(E, rtr, rte, xv, b["shared_null"],
                                                                  np.random.default_rng(31)),
                                     "ridge": shared_calibration(E, rtr, rte, xv, ridge,
                                                                 np.random.default_rng(31))}
    return out


def planted3(setup) -> dict:
    ctx, t = load(setup)
    E = Emb(ctx)
    oid = t.oid.to_numpy()
    rng = np.random.default_rng(9)
    P = num(t, f"H_{Y.PRIMARY}")
    Wv, cat, _ = winding(t)
    hay = np.isfinite(P) & np.isfinite(Wv)
    vis = np.full(len(t), np.nan)
    vis[hay] = Y.visibility(t[hay])
    ok = hay & np.isfinite(vis)
    trm, tem = ok & (t.part == "A").to_numpy(), ok & (t.part == "B").to_numpy()
    rtr, rte = E.rows(oid[trm]), E.rows(oid[tem])
    out: dict = {}

    # T1: agreement states
    w = Wv[ok]
    v = vis[ok]
    for name, y in (("agree", -w + 0.3 * w.std() * rng.normal(size=w.size)),
                    ("none", rng.permutation(w)), ("contrary", w + w.std() * rng.normal(size=w.size))):
        c = agreement(w, y, v, seed=1)
        out[f"T1_{name}"] = {**{k: c[k] for k in ("rho", "partial", "p")},
                             "state": agreement_state(c, by({"x": c["p"]}, 3)["x"])}

    # T2: planted targets on M's own embedding (untrained cannot share them)
    out.update(planted_2x2(E, rtr, rte, oid, ok & t.part.isin(["A", "B"]).to_numpy(), vis, rng))

    # T3: spread states
    m = np.isfinite(P) & (cat >= 0) & np.isfinite(vis)
    pm, gm, vm = P[m], cat[m], vis[m]
    lo, hi = pm[gm == 0], pm[gm == 2]
    for name in ("wider", "narrower", "null"):
        pp = pm.copy()
        med = np.flatnonzero(gm == 1)
        if name == "wider":
            pp[med] = np.where(rng.random(med.size) < 0.5, rng.choice(lo, med.size), rng.choice(hi, med.size))
        elif name == "narrower":
            pp[med] = np.median(pm) + 0.2 * (pm[med] - np.median(pm[med]))
        else:
            pp = rng.permutation(pp)
        c = spread(pp, gm, vm, seed=3)
        out[f"T3_{name}"] = {"D": c["D"], "p": c["p"], "state": spread_state(c, by({"x": c["p"]}, 3)["x"])}

    # T4b: axis states — target built from visibility, or from the axis itself
    axis = winding_axis(ctx, E)
    proj = {n: E.raw[n][rte] @ axis[n] for n in ENCODERS}
    pr = proj["M"]
    vt = vis[tem]
    for name, y in (("beyond", (pr - pr.mean()) / pr.std() + 1.0 * rng.normal(size=pr.size)),
                    ("visibility", vt + 0.3 * rng.normal(size=pr.size)),
                    ("none", rng.normal(size=pr.size))):
        if name == "visibility":
            # a projection that tracks visibility only: correlate the axis score with vis
            proj_v = {n: vt + 0.5 * rng.normal(size=vt.size) if n == "M" else rng.normal(size=vt.size)
                      for n in ENCODERS}
            c = axis_vs_pitch(proj_v, y, vt, seed=5)
        else:
            c = axis_vs_pitch(proj, y, vt, seed=5)
        out[f"T4_{name}"] = {"raw": c["raw"], "partial": c["partial"], "retention": c["retention"],
                             "state": axis_state(c, by({"x": c["p_raw"]}, 4)["x"])}
    # REVERSED (added in Z2): the axis carries visibility plus a part that runs against the target
    e = rng.normal(size=vt.size)
    proj_r = {n: vt + 0.5 * e if n == "M" else rng.normal(size=vt.size) for n in ENCODERS}
    c = axis_vs_pitch(proj_r, vt - 0.5 * e + 0.3 * rng.normal(size=vt.size), vt, seed=5)
    out["T4_reversed"] = {"raw": c["raw"], "partial": c["partial"], "retention": c["retention"],
                          "state": axis_state(c, by({"x": c["p_raw"]}, 4)["x"])}
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--y3"
    global OUT, PLANTED
    if "--z2" in sys.argv:  # Brief Z2: Hayes joined on its DR7 OBJID (46,882 rows, not 37,381)
        Y.HAYES_KEY = "dr7objid"
        OUT, PLANTED = OUT.with_name("y3_pitch_z2.json"), PLANTED.with_name("y3_planted_z2.json")
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="Y3", sources=1)
    if mode == "--bank":
        bank(setup)
        return
    out, path = (planted3(setup), PLANTED) if mode == "--planted3" else (y3(setup), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:8000], file=sys.stderr)


if __name__ == "__main__":
    main()
