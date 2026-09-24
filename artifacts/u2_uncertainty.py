"""Brief U2 — uncertainty geometry (Fig 2): on-axis tracking, off-axis ambiguity, visibility.

Pre-registered in artifacts/u_findings.md §U2 (hashed before any correlation was computed).
M's 4-epoch checkpoint, P2's split; answers with a linear direction (R1/R2 in T1's ladder), both
populations. A local vote floor on the QUESTION's reach (D8 scoping): 21 headline, 10 and 37 as
sensitivity, with a counts-only fallback to 10 fixed before any correlation exists.

  ON-AXIS  : axis = logistic on TRAIN consensus extremes (f ≥ 0.8 vs f ≤ 0.2; firewall asserted);
             project the TEST ambiguous middle (0.2 < f < 0.8); Spearman(projection, f).
  OFF-AXIS : primary — S1's bend coordinate (2nd eigenvector of R2's cross-fitted path covariance,
             fitted on TRAIN at all f, z-scored embedding), scored on every TEST galaxy;
             secondary — distance from the line through the two TRAIN-extreme class centroids
             (firewalled: extremes only). Each against AMBIGUITY a = 1 − 2|f − 0.5| (Spearman-
             identical to the answer's binary vote entropy).
  VISIBILITY: partial Spearman of off-axis vs ambiguity controlling magnitude, SNR, size and
             redshift together (rank-residualised; Freedman–Lane permutation), plus the separation.

    uv run python artifacts/u2_uncertainty.py --counts   -> out/u2_counts.json   (no correlations)
    uv run python artifacts/u2_uncertainty.py            -> out/u2_uncertainty.json
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
from s1_spiral_inclination import bend_direction  # noqa: E402

from galaxy_jepa.data.splits import assert_uncertainty_firewall  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import LabelProvider, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_direction  # noqa: E402

LOW, HIGH = 0.2, 0.8
HEADLINE, FALLBACK = 21, 10
FLOORS = (10, 21, 37)
N_PERM = 10_000
BATCH = 250
ALPHA = 0.05
MIN_EXTREME = 50  # per class, train
VISIBILITY = ("magnitude", "snr", "size", "redshift")
FLAGGED = {"t08_odd_feature_a24_merger": "merger", "t04_spiral_a08_spiral": "spiral",
           "t06_odd_a14_yes": "anything odd"}
COUNTS = R.OUT / "u2_counts.json"
RECORD = R.OUT / "u2_uncertainty.json"


def n_min(rho: float = 0.20, power: float = 0.80, m: int = 37) -> int:
    """Middle-band N to detect Spearman ρ at BY's rank-1 threshold with the stated power."""
    from scipy.stats import norm

    alpha1 = ALPHA / (m * sum(1.0 / k for k in range(1, m + 1)))
    z = norm.ppf(1 - alpha1 / 2) + norm.ppf(power)
    return int(math.ceil((z / math.atanh(rho)) ** 2 + 3))


N_MIN = n_min()


def with_floor(lab: LabelProvider, floor: int) -> LabelProvider:
    return LabelProvider(lab.rows, feature_cols=lab.feature_cols, nuisance_cols=lab.nuisance_cols,
                         nuisance_flag_cols=lab.nuisance_flag_cols, threshold=lab.threshold,
                         scheme=lab.scheme, population=lab.population, vote_count_min=floor,
                         consensus_gate=lab.consensus_gate, copy_rows=False)


def gated_features() -> dict[str, list[str]]:
    t1 = json.loads((R.OUT / "t1_ladder.json").read_text())
    return {pop: [r["feature"] for r in t1[pop] if r["rung"] in ("R1", "R2")]
            for pop in ("full", "conditional")}


def split_counts(real, lab, f, train_ids, test_ids) -> dict:
    ftr = lab.vote_fraction(f, feature_ids(real, lab, f, train_ids))
    fte = lab.vote_fraction(f, feature_ids(real, lab, f, test_ids))
    return {"train_pos": int((ftr >= HIGH).sum()), "train_neg": int((ftr <= LOW).sum()),
            "test_middle": int(((fte > LOW) & (fte < HIGH)).sum()), "test_all": int(fte.size)}


def usable(c: dict) -> bool:
    return c["test_middle"] >= N_MIN and min(c["train_pos"], c["train_neg"]) >= MIN_EXTREME


def counts(ctx) -> dict:
    s, out = ctx.setup, {"n_min": N_MIN, "min_extreme": MIN_EXTREME}
    for pop, feats in gated_features().items():
        for f in feats:
            per = {str(fl): split_counts(ctx.real, with_floor(ctx.labels[pop], fl), f,
                                         s.train_ids, s.test_ids) for fl in FLOORS}
            chosen = (HEADLINE if usable(per[str(HEADLINE)]) else
                      FALLBACK if usable(per[str(FALLBACK)]) else None)
            out[f"{pop}:{f}"] = {"population": pop, "feature": f, "counts": per,
                                 "headline_floor": chosen}
    COUNTS.write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ statistics


def _ranks(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v).astype(np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def spearman_perm(x: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[float, float]:
    """(ρ, two-sided add-one permutation p), shuffling ``y`` — the vote-side quantity."""
    rx, ry = _unit(_ranks(x)), _unit(_ranks(y))
    obs = float(rx @ ry)
    rng, hits = np.random.default_rng(seed), 0
    for start in range(0, N_PERM, BATCH):
        b = min(BATCH, N_PERM - start)
        perm = rng.permuted(np.broadcast_to(ry, (b, ry.size)).copy(), axis=1)
        hits += int(np.sum(np.abs(perm @ rx) >= abs(obs) - 1e-12))
    return obs, (1 + hits) / (1 + N_PERM)


def partial_perm(x: np.ndarray, y: np.ndarray, covs: np.ndarray, *, seed: int,
                 perm: bool = True) -> tuple[float, float]:
    """Partial Spearman of x and y given ``covs`` (rank-residualised), Freedman–Lane p."""
    rx, ry = _ranks(x), _ranks(y)
    design = np.column_stack([np.ones(x.size)] + [_ranks(c) for c in covs.T])
    q, _ = np.linalg.qr(design)
    ex = rx - q @ (q.T @ rx)
    ey = ry - q @ (q.T @ ry)
    ex /= np.linalg.norm(ex)
    obs = float(ex @ ey / np.linalg.norm(ey))
    if not perm:
        return obs, float("nan")
    rng, hits = np.random.default_rng(seed), 0
    for start in range(0, N_PERM, BATCH):
        b = min(BATCH, N_PERM - start)
        e = rng.permuted(np.broadcast_to(ey, (b, ey.size)).copy(), axis=1)
        proj = e @ q
        norm = np.sqrt(np.maximum(np.einsum("ij,ij->i", e, e) - np.einsum("ij,ij->i", proj, proj),
                                  1e-300))
        hits += int(np.sum(np.abs((e @ ex) / norm) >= abs(obs) - 1e-12))
    return obs, (1 + hits) / (1 + N_PERM)


def line_distance(x: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    u = (c1 - c0) / np.linalg.norm(c1 - c0)
    d = x - c0
    return np.linalg.norm(d - np.outer(d @ u, u), axis=1)


# ------------------------------------------------------------------ one answer, one floor


def measure(ctx, pop: str, f: str, floor: int, *, seed: int, perm: bool = True) -> dict:
    s, lab = ctx.setup, with_floor(ctx.labels[pop], floor)
    ids_tr = feature_ids(ctx.real, lab, f, s.train_ids)
    ids_te = feature_ids(ctx.real, lab, f, s.test_ids)
    f_tr, f_te = lab.vote_fraction(f, ids_tr), lab.vote_fraction(f, ids_te)
    ext = (f_tr <= LOW) | (f_tr >= HIGH)
    mid = (f_te > LOW) & (f_te < HIGH)
    assert_uncertainty_firewall(f_tr[ext], low=LOW, high=HIGH)
    amb = 1.0 - 2.0 * np.abs(f_te - 0.5)
    vis = np.column_stack([np.where(np.asarray(lab.nuisance_valid(n, ids_te), bool),
                                    lab.nuisance_value(n, ids_te), np.nan) for n in VISIBILITY])
    vok = np.all(np.isfinite(vis), axis=1)
    out: dict = {"n_fit": int(ext.sum()), "n_middle": int(mid.sum()), "n_test": int(f_te.size),
                 "n_visibility": int(vok.sum()), "matrices": {}}
    for m in [ctx.real, *ctx.untrained]:
        is_real = m is ctx.real
        # on-axis: the locked protocol (raw embedding, the ladder's probe)
        rtr, rte = m.rows_for(ids_tr), m.rows_for(ids_te)
        fit = Embeddings(m.x[rtr][ext], (f_tr[ext] >= HIGH).astype(np.int64), f_tr[ext])
        w = probe_direction(fit, name=f, c=ctx.pc.c).w_unit
        proj = m.x[rte][mid] @ w
        on = spearman_perm(proj, f_te[mid], seed=seed) if is_real and perm else \
            (float(np.corrcoef(_ranks(proj), _ranks(f_te[mid]))[0, 1]), float("nan"))
        rec = {"on_axis": {"rho": on[0], "p": on[1]}}
        # off-axis, z-scored embedding
        z = ctx.z(m)
        ztr, zte = z.x[z.rows_for(ids_tr)], z.x[z.rows_for(ids_te)]
        try:
            d = bend_direction(ztr, f_tr)
            bend = zte @ d
        except (ValueError, IndexError, np.linalg.LinAlgError):
            bend = None
        c0 = ztr[f_tr <= LOW].mean(axis=0)
        c1 = ztr[f_tr >= HIGH].mean(axis=0)
        dist = line_distance(zte, c0, c1)
        for name, v in (("bend", bend), ("line", dist)):
            if v is None:
                rec[name] = None
                continue
            do_perm = is_real and perm
            raw = spearman_perm(v, amb, seed=seed + 1) if do_perm else \
                (float(np.corrcoef(_ranks(v), _ranks(amb))[0, 1]), float("nan"))
            part = partial_perm(v[vok], amb[vok], vis[vok], seed=seed + 2, perm=do_perm)
            r = {"raw": raw[0], "p_raw": raw[1], "partial": part[0], "p_partial": part[1]}
            if is_real:
                raw_vok = float(np.corrcoef(_ranks(v[vok]), _ranks(amb[vok]))[0, 1])
                r["raw_on_visibility_rows"] = raw_vok
                r["separation"] = {}
                for j, nz in enumerate(VISIBILITY):
                    others = np.delete(vis[vok], j, axis=1)
                    r["separation"][nz] = {
                        # which visibility variable the displacement carries, the other three held
                        "disp_vs_var_given_others": partial_perm(v[vok], vis[vok][:, j], others,
                                                                 seed=0, perm=False)[0],
                        # ambiguity association left after holding this one variable alone
                        "disp_vs_amb_given_var": partial_perm(v[vok], amb[vok], vis[vok][:, [j]],
                                                              seed=0, perm=False)[0],
                    }
            rec[name] = r
        out["matrices"]["real" if is_real else m.encoder_name] = rec
    return out


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="U2", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    cts = json.loads(COUNTS.read_text()) if COUNTS.exists() else counts(ctx)
    r2 = json.loads((R.OUT / "r_r2.json").read_text())["full"]["features"]
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    feats = gated_features()
    for pop in ("full", "conditional"):
        m = sum(1 for f in feats[pop] if cts[f"{pop}:{f}"]["headline_floor"] is not None)
        nulls_mod.assert_null_resolution(N_PERM, alpha=ALPHA, method="benjamini_yekutieli",
                                         n_tests=m)
    for pop in ("full", "conditional"):
        for i, f in enumerate(feats[pop]):
            c = cts[f"{pop}:{f}"]
            for floor in FLOORS:
                key = f"{pop}:{f}:{floor}"
                if key in rec:
                    continue
                t0 = time.perf_counter()
                cc = c["counts"][str(floor)]
                if min(cc["train_pos"], cc["train_neg"]) < 2 or cc["test_middle"] < 3:
                    rec[key] = {"skipped": "too few galaxies to fit or rank", "counts": cc}
                else:
                    rec[key] = measure(ctx, pop, f, floor, seed=setup.pc.seed + i)
                    rec[key]["counts"] = cc
                rec[key]["headline"] = floor == c["headline_floor"]
                if pop == "full" and f in r2:
                    rec[key]["readout_ceiling"] = r2[f].get("readout")
                    rec[key]["r2_verdict"] = r2[f].get("verdict")
                R._save(RECORD, rec)
                print(f"U2 {pop:<11s} {f[:38]:<38s} floor {floor:>2d} "
                      f"{time.perf_counter() - t0:5.0f}s", file=sys.stderr)
    verdicts(rec, cts, feats)


def verdicts(rec: dict, cts: dict, feats: dict) -> None:
    """BY within each (population, measurement) family at the headline floor; then the reads."""
    out = {}
    for pop in ("full", "conditional"):
        rows = {f: rec[f"{pop}:{f}:{cts[f'{pop}:{f}']['headline_floor']}"]
                for f in feats[pop] if cts[f"{pop}:{f}"]["headline_floor"] is not None}
        fam = {}
        for meas, get in (("on_axis", lambda r: r["on_axis"]["p"]),
                          ("bend_raw", lambda r: r["bend"]["p_raw"] if r["bend"] else 1.0),
                          ("bend_partial", lambda r: r["bend"]["p_partial"] if r["bend"] else 1.0),
                          ("line_raw", lambda r: r["line"]["p_raw"]),
                          ("line_partial", lambda r: r["line"]["p_partial"])):
            p = {f: get(r["matrices"]["real"]) for f, r in rows.items()}
            fam[meas] = nulls_mod.family_significant(p, alpha=ALPHA, method="benjamini_yekutieli",
                                                     n_tests=len(p))
        for f, r in rows.items():
            real = r["matrices"]["real"]
            on = real["on_axis"]["rho"]
            v = {"on_axis": ("TRACKS" if fam["on_axis"][f] and on > 0 else
                             "CONTRARY" if fam["on_axis"][f] else "NO EVIDENCE")}
            for kind in ("bend", "line"):
                o = real[kind]
                if o is None:
                    v[kind] = "UNCHARACTERISED"
                    continue
                raw_sig = fam[f"{kind}_raw"][f] and o["raw"] > 0
                part_sig = fam[f"{kind}_partial"][f] and o["partial"] > 0
                if not raw_sig:
                    v[kind] = "NONE"
                elif part_sig:
                    v[kind] = "BEYOND VISIBILITY"
                elif abs(o["partial"]) <= 0.5 * abs(o["raw_on_visibility_rows"]):
                    v[kind] = "RESOLUTION-LIMITED"
                else:
                    v[kind] = "UNRESOLVED"
            out[f"{pop}:{f}"] = v
    rec["verdicts"] = out
    R._save(RECORD, rec)


if __name__ == "__main__":
    if sys.argv[1:] == ["--counts"]:
        setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="U2", sources=1)
        c = counts(R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True))
        print(f"U2 counts: N_MIN {N_MIN} middle test galaxies, >= {MIN_EXTREME} per train class",
              file=sys.stderr)
        for k, v in c.items():
            if not isinstance(v, dict):
                continue
            cc = v["counts"]
            print(f"  {k[:52]:<52s} " + "  ".join(
                f"{fl}:{cc[str(fl)]['test_middle']:>6d}/{min(cc[str(fl)]['train_pos'], cc[str(fl)]['train_neg']):>5d}"
                for fl in FLOORS) + f"  -> {v['headline_floor']}", file=sys.stderr)
    else:
        main()
