"""Brief V2 — measurements independent of the votes: axis ratio (2a) and bulge-to-total (2c).

Pre-registered in artifacts/v_findings.md §V2 (hashed before this ran). 2b (pitch angle) is
blocked on Hart et al.'s table and is not run here.

  decode    : ridge (α by GCV on train) from the z-scored embedding to the measurement; test
              Spearman, bootstrap CI, permutation p; the same on R's three untrained draws.
  direction : (2a) the ladder's logistic probe on a vote label vs on a base-rate-matched axis-ratio
              label, same galaxies; split-half disattenuated cosine; the CAV beside it.
  2×2       : (2c) partial Spearman of each ridge prediction with its target given the other
              target — what the encoder carries of B/T beyond B_avg, and of B_avg beyond B/T.

    uv run python artifacts/v2_independent.py  -> artifacts/out/v2_independent.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402

from galaxy_jepa.data.metadata import GZ2_TREE, vote_column  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_direction  # noqa: E402

ALPHAS = np.logspace(-2, 4, 13)
N_BOOT = 2_000
NEGLIGIBLE = 0.10
BAND = 2.576 / np.sqrt(384)
SIMARD = R.OUT / "ext" / "simard2011_table1.tsv"
PAIRS = (("t02_edgeon_a04_yes", "low"), ("t07_rounded_a18_cigar_shaped", "low"),
         ("t07_rounded_a16_completely_round", "high"))
B_AVG = {"a11_just_noticeable": 0.2, "a12_obvious": 0.8, "a13_dominant": 1.0}
OUT = R.OUT / "v2_independent.json"


# ------------------------------------------------------------------ inputs


def metadata(union: np.ndarray) -> dict[str, np.ndarray]:
    csv.field_size_limit(1 << 24)
    pos = {int(o): i for i, o in enumerate(union)}
    cols = {k: np.full(union.size, np.nan) for k in ("expAB_r", "deVAB_r", "petroRad_r",
                                                      "modelMag_r")}
    dr7 = np.zeros(union.size, dtype=np.int64)
    with open(Path("data/probe/metadata.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            i = pos.get(int(row["object_id"]))
            if i is None:
                continue
            dr7[i] = int(row["dr7objid"])
            for k in cols:
                if row.get(k):
                    cols[k][i] = float(row[k])
    return {**cols, "dr7objid": dr7}


def simard(dr7: np.ndarray) -> dict[str, np.ndarray]:
    want = {int(d): i for i, d in enumerate(dr7)}
    out = {k: np.full(dr7.size, np.nan) for k in ("bt", "e_bt", "pps")}
    with open(SIMARD) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if not p[0].strip().isdigit():
                continue
            i = want.get(int(p[0]))
            if i is None:
                continue
            for k, j in (("bt", 1), ("e_bt", 2), ("pps", 3)):
                if p[j].strip():
                    out[k][i] = float(p[j])
    return out


# ------------------------------------------------------------------ statistics


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot_ci(a: np.ndarray, b: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = [spearman(a[i], b[i]) for i in (rng.integers(0, a.size, a.size) for _ in range(N_BOOT))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ridge_predict(xtr, ytr, xte) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import RidgeCV

    m = RidgeCV(alphas=ALPHAS).fit(xtr, ytr)
    return m.predict(xte), float(m.alpha_)


def decode(ctx, rows_tr, rows_te, y_tr, y_te, seed: int) -> dict:
    """Ridge on M and the three untrained draws; permutation p and CI on M only."""
    out: dict = {"n_train": int(rows_tr.size), "n_test": int(rows_te.size)}
    for m in [ctx.real, *ctx.untrained]:
        z = ctx.z(m).x
        pred, alpha = ridge_predict(z[rows_tr], y_tr, z[rows_te])
        if m is ctx.real:
            rho, p = U2.spearman_perm(pred, y_te, seed=seed)
            out.update(rho=rho, p=p, ci=boot_ci(pred, y_te, seed), alpha=alpha, pred=pred)
        else:
            out.setdefault("untrained", []).append(spearman(pred, y_te))
    u = out["untrained"]
    out["margin"] = [out["rho"] - max(u), out["rho"] - min(u)]
    return out


def decode_state(d: dict, sig: bool) -> str:
    if not sig:
        return "NOT DECODED"
    if d["rho"] < NEGLIGIBLE:
        return "NEGLIGIBLE"
    return "DECODED" if d["margin"][0] > 0 else "NOT ABOVE UNTRAINED"


# ------------------------------------------------------------------ 2a


def direction_pairs(ctx, ab: np.ndarray) -> dict:
    s, lab = ctx.setup, ctx.labels["full"]
    pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
    rng = np.random.default_rng(s.pc.seed)
    zr = ctx.z(ctx.real).x
    out = {}
    for f, side in PAIRS:
        ids = feature_ids(ctx.real, lab, f, s.train_ids)
        emb = feature_embeddings(ctx.real, lab, f, s.train_ids)
        a = np.array([ab[pos[int(o)]] for o in ids])
        ok = np.isfinite(a)
        x, yv, a = emb.x[ok], emb.y[ok], a[ok]
        rate = float(yv.mean())
        tau = np.quantile(a, rate if side == "low" else 1 - rate)
        ym = (a < tau if side == "low" else a > tau).astype(np.int64)
        rows = np.array([pos[int(o)] for o, k in zip(ids, ok, strict=True) if k])

        def logit(sel, y):
            return probe_direction(Embeddings(x[sel], y[sel], y[sel].astype(float)), name=f,
                                   c=s.pc.c).w_unit

        def cav(sel, y):
            d = zr[rows[sel]][y[sel] == 1].mean(0) - zr[rows[sel]][y[sel] == 0].mean(0)
            return d / np.linalg.norm(d)

        half = rng.permutation(x.shape[0]) % 2 == 0
        rec = {"n": int(x.shape[0]), "base_rate": rate, "tau": float(tau),
               "label_agreement": float((ym == yv).mean())}
        for est, fit in (("logistic", logit), ("cav", cav)):
            va, vb, ma, mb = fit(half, yv), fit(~half, yv), fit(half, ym), fit(~half, ym)
            rel_v, rel_m = float(va @ vb), float(ma @ mb)
            r_raw = float((va @ mb + vb @ ma) / 2)
            full = float(fit(np.ones_like(half), yv) @ fit(np.ones_like(half), ym))
            r_d = r_raw / np.sqrt(rel_v * rel_m) if min(rel_v, rel_m) > 0 else float("nan")
            if min(rel_v, rel_m) < 0.3:
                v = "UNRELIABLE"
            elif r_raw < -BAND:
                v = "OPPOSED"
            elif abs(r_raw) <= BAND:
                v = "UNRELATED"
            elif r_d >= 0.8:
                v = "SAME DIRECTION"
            elif r_d >= 0.3:
                v = "RELATED"
            else:
                v = "DISTINCT"
            rec[est] = {"reliability_vote": rel_v, "reliability_measure": rel_m,
                        "r_raw": r_raw, "r_d": float(r_d), "cos_full": full, "verdict": v}
        out[f] = rec
    return out


# ------------------------------------------------------------------ 2c


def bulge_block(ctx, rows_tr, rows_te, bt, bavg, seed: int) -> dict:
    d_bt = decode(ctx, rows_tr, rows_te, bt[rows_tr], bt[rows_te], seed)
    d_bv = decode(ctx, rows_tr, rows_te, bavg[rows_tr], bavg[rows_te], seed + 1)
    bt_te, bv_te = bt[rows_te], bavg[rows_te]
    a_m = U2.partial_perm(d_bt["pred"], bt_te, bv_te[:, None], seed=seed + 2)
    a_v = U2.partial_perm(d_bv["pred"], bv_te, bt_te[:, None], seed=seed + 3)
    untr = {"a_m": [], "a_v": []}
    for m in ctx.untrained:
        z = ctx.z(m).x
        p_bt = ridge_predict(z[rows_tr], bt[rows_tr], z[rows_te])[0]
        p_bv = ridge_predict(z[rows_tr], bavg[rows_tr], z[rows_te])[0]
        untr["a_m"].append(U2.partial_perm(p_bt, bt_te, bv_te[:, None], seed=0, perm=False)[0])
        untr["a_v"].append(U2.partial_perm(p_bv, bv_te, bt_te[:, None], seed=0, perm=False)[0])
    for d in (d_bt, d_bv):
        d.pop("pred")
    return {"agreement": {"rho": spearman(bavg[np.r_[rows_tr, rows_te]],
                                          bt[np.r_[rows_tr, rows_te]]),
                          "ci": boot_ci(bavg[np.r_[rows_tr, rows_te]],
                                        bt[np.r_[rows_tr, rows_te]], seed)},
            "decode_bt": d_bt, "decode_bavg": d_bv,
            "a_m": {"rho": a_m[0], "p": a_m[1], "untrained": untr["a_m"]},
            "a_v": {"rho": a_v[0], "p": a_v[1], "untrained": untr["a_v"]}}


def two_by_two(b: dict, sig: dict, prefix: str) -> str:
    def leg(k):
        r = b[k]
        if sig[f"{prefix}{k}"] and r["rho"] < 0:
            return "INVERTED"
        return bool(sig[f"{prefix}{k}"] and r["rho"] >= NEGLIGIBLE
                    and r["rho"] > max(r["untrained"]))

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


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="V2", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    union = ctx.real.object_ids
    md = metadata(union)
    pos = {int(o): i for i, o in enumerate(union)}
    tr = np.array([pos[int(o)] for o in setup.train_ids])
    te = np.array([pos[int(o)] for o in setup.test_ids])
    rec: dict = {}

    # 2a — decoding
    for col in ("expAB_r", "deVAB_r"):
        v = md[col]
        ok = np.isfinite(v) & (v > 0) & (v <= 1)
        rec[f"decode_{col}"] = decode(ctx, tr[ok[tr]], te[ok[te]], v[tr[ok[tr]]], v[te[ok[te]]],
                                      seed=setup.pc.seed)
        rec[f"decode_{col}"].pop("pred")
        print(f"V2 2a decode {col}: rho {rec[f'decode_{col}']['rho']:+.3f} "
              f"untrained {rec[f'decode_{col}']['untrained']}", file=sys.stderr)
    rec["directions"] = direction_pairs(ctx, md["expAB_r"])
    for f, r in rec["directions"].items():
        print(f"V2 2a {f}: logistic r_d {r['logistic']['r_d']:+.2f} -> {r['logistic']['verdict']}"
              f"; cav r_d {r['cav']['r_d']:+.2f} -> {r['cav']['verdict']}", file=sys.stderr)
    R._save(OUT, rec)

    # 2c — B/T against bulge prominence
    sm = simard(md["dr7objid"])
    lab = setup.labels
    answers = GZ2_TREE["t05_bulge_prominence"]
    uids = union.tolist()
    frac = np.column_stack([lab._column(uids, vote_column("t05_bulge_prominence", a))
                            for a in answers])
    reach = sum(lab._column(uids, vote_column("t05_bulge_prominence", a, "count"))
                for a in answers)
    bavg = sum(w * frac[:, answers.index(a)] for a, w in B_AVG.items())
    bt = sm["bt"]
    base = np.isfinite(bt) & np.isfinite(bavg) & (reach >= 21)
    flag = ((md["petroRad_r"] < 3) | (md["modelMag_r"] > 17) | (sm["pps"] > 0.32)
            | (sm["e_bt"] > 0.1))
    rec["bulge"] = {"n_flagged": int((base & flag).sum()), "n": int(base.sum())}
    for name, sel in (("primary", base), ("unflagged", base & ~flag)):
        rec["bulge"][name] = bulge_block(ctx, tr[sel[tr]], te[sel[te]], bt, bavg,
                                         seed=setup.pc.seed + 10)
        print(f"V2 2c {name}: {json.dumps(rec['bulge'][name], default=float)[:400]}",
              file=sys.stderr)
    everyone = np.isfinite(bt)
    d = decode(ctx, tr[everyone[tr]], te[everyone[te]], bt[tr[everyone[tr]]],
               bt[te[everyone[te]]], seed=setup.pc.seed + 20)
    d.pop("pred")
    rec["bulge"]["decode_bt_all_union"] = d
    top = np.argmax(np.nan_to_num(frac, nan=-1), axis=1)
    rec["bulge"]["bt_by_plurality"] = {
        a: {"n": int((base & (top == j)).sum()),
            "q25_50_75": np.nanpercentile(bt[base & (top == j)], [25, 50, 75]).tolist()}
        for j, a in enumerate(answers) if (base & (top == j)).sum() > 0}
    R._save(OUT, rec)

    # BY across the confirmatory family (m = 5), then the states
    p = {"2a": rec["decode_expAB_r"]["p"],
         "bt": rec["bulge"]["primary"]["decode_bt"]["p"],
         "bavg": rec["bulge"]["primary"]["decode_bavg"]["p"],
         "a_m": rec["bulge"]["primary"]["a_m"]["p"],
         "a_v": rec["bulge"]["primary"]["a_v"]["p"]}
    nulls_mod.assert_null_resolution(U2.N_PERM, alpha=0.05, method="benjamini_yekutieli",
                                     n_tests=len(p))
    sig = nulls_mod.family_significant(p, alpha=0.05, method="benjamini_yekutieli",
                                       n_tests=len(p))
    rec["decode_expAB_r"]["state"] = decode_state(rec["decode_expAB_r"], sig["2a"])
    rec["decode_deVAB_r"]["state"] = decode_state(rec["decode_deVAB_r"],
                                                  rec["decode_deVAB_r"]["p"] < 0.05)
    for name in ("primary", "unflagged"):
        b = rec["bulge"][name]
        s = sig if name == "primary" else {
            "bt": b["decode_bt"]["p"] < 0.05, "bavg": b["decode_bavg"]["p"] < 0.05,
            "a_m": b["a_m"]["p"] < 0.05, "a_v": b["a_v"]["p"] < 0.05}
        b["decode_bt"]["state"] = decode_state(b["decode_bt"], s["bt"])
        b["decode_bavg"]["state"] = decode_state(b["decode_bavg"], s["bavg"])
        b["verdict"] = two_by_two(b, {"a_m": s["a_m"], "a_v": s["a_v"]}, "")
    if rec["bulge"]["primary"]["verdict"] != rec["bulge"]["unflagged"]["verdict"]:
        rec["bulge"]["flag_sensitive"] = True
    rec["significant"] = {k: bool(v) for k, v in sig.items()}
    R._save(OUT, rec)
    print(f"V2 2c verdict: {rec['bulge']['primary']['verdict']} "
          f"(unflagged: {rec['bulge']['unflagged']['verdict']})", file=sys.stderr)


if __name__ == "__main__":
    main()
