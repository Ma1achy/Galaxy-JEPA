"""Brief Z3 — V2's B/T 2×2 rerun with Y3's corrected design.

V2's Experiment D read BOTH (A_m 0.36, A_v 0.47). But Y3's D28 check showed that V2's form
controls a noisy copy of the other target and reads BOTH on two noisy copies of ONE quantity. The
rerun uses V2's rows (Simard B/T; the GZ2 bulge-prominence average, reach ≥ 21; A → B) with
Y3's `two_by_two`:
- each leg controls the OTHER decoder's prediction plus V1 visibility;
- each leg must beat the matched shared-quantity null;
- the result is read by `state_2x2`.
The states and family are in `z_findings.md` §Z3.

  --planted   D28 — BOTH / SHARED ONLY / NEITHER planted on V2's primary rows
  --z3        the test (primary; unflagged as the sensitivity; no-visibility form descriptive)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402
import v2_independent as V2  # noqa: E402
import w2_name_pcs as W  # noqa: E402
import y3_science as Y3  # noqa: E402
import y_pitch as Y  # noqa: E402

from galaxy_jepa.data.metadata import GZ2_TREE, vote_column  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402

OUT = R.OUT / "z3_bt.json"
PLANTED = R.OUT / "z3_planted.json"
FAMILY = ("decode_bt", "decode_bavg", "a_m", "a_v")


def rows(setup):
    """V2's B/T rows, in union order, plus V1 visibility for every union galaxy that has it."""
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    union = ctx.real.object_ids
    md = V2.metadata(union)
    sm = V2.simard(md["dr7objid"])
    lab = setup.labels
    answers = GZ2_TREE["t05_bulge_prominence"]
    uids = union.tolist()
    frac = np.column_stack([lab._column(uids, vote_column("t05_bulge_prominence", a))
                            for a in answers])
    reach = sum(lab._column(uids, vote_column("t05_bulge_prominence", a, "count"))
                for a in answers)
    bavg = sum(w * frac[:, answers.index(a)] for a, w in V2.B_AVG.items())
    bt = sm["bt"]
    base = np.isfinite(bt) & np.isfinite(bavg) & (reach >= 21)
    flag = ((md["petroRad_r"] < 3) | (md["modelMag_r"] > 17) | (sm["pps"] > 0.32)
            | (sm["e_bt"] > 0.1))
    meta = pd.read_csv(Path(setup.cfg.paths.probe_dir) / "metadata.csv",
                       usecols=["object_id", *Y.VIS_COLS], dtype={"object_id": str},
                       low_memory=False)
    meta = meta.set_index(meta.object_id.astype("int64")).reindex([int(o) for o in union])
    vcols = meta[list(Y.VIS_COLS)].to_numpy(float)
    pos = {int(o): i for i, o in enumerate(union)}
    tr = np.array([pos[int(o)] for o in setup.train_ids])
    te = np.array([pos[int(o)] for o in setup.test_ids])
    return ctx, np.asarray(union).astype(np.int64), bt, bavg, base, flag, vcols, tr, te


def visibility_on(vcols, sel) -> np.ndarray:
    """V1's composite computed on the analysed rows (as Y3 does), NaN elsewhere."""
    v = np.full(len(vcols), np.nan)
    ok = sel & np.isfinite(vcols).all(1)
    v[ok] = Y.visibility(pd.DataFrame(vcols[ok], columns=list(Y.VIS_COLS)))
    return v


def block(E, tr, te, bt, bavg, sel, vcols, seed: int) -> dict:
    vis = visibility_on(vcols, sel)
    ok = sel & np.isfinite(vis)
    rtr, rte = tr[ok[tr]], te[ok[te]]
    b = Y3.two_by_two(E, rtr, rte, bt, bavg, vis, seed=seed)
    pred_bt = b.pop("pred_pitch_M")
    pred_bv = V2.ridge_predict(E.x["M"][rtr], bavg[rtr], E.x["M"][rte])[0]
    b["shared_null"] = Y3.shared_null(E, rtr, rte, bt, b["decode_bt"]["rho"],
                                      b["decode_bavg"]["rho"], vis, seed=seed + 50)
    b["shared_null_ridge"] = Y3.shared_null(E, rtr, rte, bt, b["decode_bt"]["rho"],
                                            b["decode_bavg"]["rho"], vis, seed=seed + 50,
                                            latent="ridge")
    # Descriptive: the corrected controls without visibility (V2 had none)
    b["no_visibility"] = {
        "a_m": U2.partial_perm(pred_bt, bt[rte], pred_bv[:, None], seed=0, perm=False)[0],
        "a_v": U2.partial_perm(pred_bv, bavg[rte], pred_bt[:, None], seed=0, perm=False)[0],
        "v2_original_a_m": U2.partial_perm(pred_bt, bt[rte], bavg[rte][:, None], seed=0,
                                           perm=False)[0],
        "v2_original_a_v": U2.partial_perm(pred_bv, bavg[rte], bt[rte][:, None], seed=0,
                                           perm=False)[0]}
    b["agreement"] = {"rho": V2.spearman(bavg[np.r_[rtr, rte]], bt[np.r_[rtr, rte]]),
                      "partial_vis": U2.partial_perm(bavg[np.r_[rtr, rte]], bt[np.r_[rtr, rte]],
                                                     vis[np.r_[rtr, rte]][:, None], seed=0,
                                                     perm=False)[0]}
    return b


def settle(b: dict, sig: dict) -> None:
    for k in ("decode_bt", "decode_bavg"):
        b[k]["state"] = V2.decode_state(b[k], sig[k])
    b["significant"] = {k: bool(v) for k, v in sig.items()}
    b["state"] = Y3.state_2x2(b, sig)
    b["marginal"] = [k for k in ("a_m", "a_v")
                     if abs(b[k]["rho"] - b["shared_null"][f"{k}_975"]) < 0.02]
    b["state_y3_rule"] = Y3.state_2x2({**b, "shared_null": b["shared_null_ridge"]}, sig, q="95")


def z3(setup) -> dict:
    ctx, union, bt, bavg, base, flag, vcols, tr, te = rows(setup)
    E = Y3.Emb(ctx)
    assert E.n_union == union.size
    rec: dict = {"n_base": int(base.sum()), "n_flagged": int((base & flag).sum())}
    for name, sel in (("primary", base), ("unflagged", base & ~flag)):
        rec[name] = block(E, tr, te, bt, bavg, sel, vcols, seed=setup.pc.seed + 10)
        print(f"Z3 {name}: A_m {rec[name]['a_m']['rho']:+.3f} A_v {rec[name]['a_v']['rho']:+.3f} "
              f"null95 {rec[name]['shared_null']['a_m_95']:+.3f}/"
              f"{rec[name]['shared_null']['a_v_95']:+.3f}", file=sys.stderr)
    p = {k: rec["primary"][k]["p"] for k in FAMILY}
    nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=len(p))
    settle(rec["primary"], Y3.by(p, len(p)))
    # Unflagged is the sensitivity, unadjusted, as V2 read it
    settle(rec["unflagged"], {k: rec["unflagged"][k]["p"] < W.ALPHA for k in FAMILY})
    rec["flag_sensitive"] = rec["primary"]["state"] != rec["unflagged"]["state"]
    return rec


def planted(setup) -> dict:
    ctx, union, _, _, base, _, vcols, tr, te = rows(setup)
    E = Y3.Emb(ctx)
    vis = visibility_on(vcols, base)
    ok = base & np.isfinite(vis)
    rtr, rte = tr[ok[tr]], te[ok[te]]
    ab = np.zeros(union.size, bool)
    ab[np.r_[rtr, rte]] = True
    return Y3.planted_2x2(E, rtr, rte, union, ab, vis, np.random.default_rng(9))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--z3"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="Z3", sources=1)
    out, path = (planted(setup), PLANTED) if mode == "--planted" else (z3(setup), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000], file=sys.stderr)


if __name__ == "__main__":
    main()
