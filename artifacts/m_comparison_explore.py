"""Exploratory readings for the M comparison write-up (after the hash; no state) → docs/m_comparison/explore.json.

    .venv/bin/python artifacts/m_comparison_explore.py

User, 19 Nov, write-up items 3–4. Everything here is post-hash and exploratory: it adds no state and changes
no reading of the record (`docs/m_comparison/m_comparison_score.json`). It reuses the scoring code's own z
(`m_comparison.score`'s construction: (AUC − bank mean) / √(bank sd² + real_se²)), its BY existence family
(`by_reject`, α 0.05, m 37, t with K − 1 df) and its drift / centred-noise / A-bound functions, so the
numbers sit on the same footing as the record's.

- **Distance from the existence bar.** Per run, the BY step-up rejects the k smallest p; the bar a cell must
  clear is the p threshold at that k, i.e. a critical t, z_crit = t.isf(k·α/(m·H_m), K − 1). Distance is
  z − z_crit (positive: above the bar), and in AUC terms (z − z_crit) × the cell's z denominator.
- **Run-level noise in AUC units.** The drift (mean over non-VOID cells of Δ1 − Δ2, Δ = z_v2 − z_v1) and the
  A-bound are in z units; z = ΔAUC / denominator and the denominator shrinks as √n grows, so the same AUC
  offset is a larger z on a well-sampled answer. Converted cell by cell (raw AUC: (A1 − M) − (A2 − O2)) and
  by the median denominator.

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m_comparison as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTF = REPO / "docs/m_comparison/explore.json"
SCORE = REPO / "docs/m_comparison/m_comparison_score.json"


def main() -> None:
    rec = {r: json.loads(Path(p).read_text()) for r, p in C.RECORDS.items()}
    bnk = {s: json.loads(Path(p).read_text()) for s, p in C.BANKS.items()}
    score = json.loads(SCORE.read_text())
    H = float(np.sum(1.0 / np.arange(1, C.N_FEATURES + 1)))
    out: dict = {"note": "post-hash, exploratory, no state (user, 19 Nov)", "populations": {}}
    for pop in C.POPS:
        feats = [row["feature"] for row in rec["M"][pop]]
        rows = {r: {row["feature"]: row for row in rec[r][pop]} for r in C.RUNS}
        mats = {s: C.bank_matrix(bnk[s], feats) for s in ("v1", "v2")}
        mC = {s: m.mean(0) for s, m in mats.items()}
        sdC = {s: m.std(0, ddof=1) for s, m in mats.items()}
        num = lambda r, k: np.array([float(rows[r][f][k]) if C.p2_row(rows[r][f]) else np.nan for f in feats])  # noqa: E731
        auc = {r: num(r, "auc") for r in C.RUNS}
        se = {r: num(r, "real_se") for r in C.RUNS}
        dd = {r: np.sqrt(sdC[C.SIDE[r]] ** 2 + se[r] ** 2) for r in C.RUNS}
        z = {r: (auc[r] - mC[C.SIDE[r]]) / dd[r] for r in C.RUNS}
        cells = {c["feature"]: c for c in score["populations"][pop]["cells"]}
        for r in C.RUNS:  # same z as the record
            assert np.allclose(np.nan_to_num(z[r]), [cells[f]["z"][r] for f in feats]), (pop, r)
        crit, dist, dist_auc = {}, {}, {}
        for r in C.RUNS:
            p = student_t.sf(np.nan_to_num(z[r]), C.K_BANK - 1)
            E = C.by_reject(p)
            assert (E == np.array([bool(rows[r][f]["exceeds_null"]) for f in feats])).all(), (pop, r)
            k = int(E.sum())
            crit[r] = float(student_t.isf(max(k, 1) * C.ALPHA_BY / (C.N_FEATURES * H), C.K_BANK - 1))
            dist[r] = z[r] - crit[r]
            dist_auc[r] = dist[r] * dd[r]
        scoreable = [f for f in feats if cells[f]["state"] not in ("VOID", "THIN")]
        j = [feats.index(f) for f in scoreable]
        per_cell = {f: {"state": cells[f]["state"],
                        "z_minus_crit": {r: round(float(dist[r][i]), 2) for r in C.RUNS},
                        "auc_minus_bar_at_crit": {r: round(float(dist_auc[r][i]), 4) for r in C.RUNS},
                        "existence": {r: bool(rows[r][f]["exceeds_null"]) for r in C.RUNS}}
                    for i, f in enumerate(feats)}
        dmin = np.array([min(dist[r][i] for r in C.RUNS) for i in j])
        dmin_auc = np.array([min(dist_auc[r][i] for r in C.RUNS) for i in j])
        nearest = sorted(scoreable, key=lambda f: min(per_cell[f]["z_minus_crit"].values()))[:5]
        # run-level noise, in z (as recorded) and in AUC
        ok = np.array([cells[f]["state"] != "VOID" for f in feats])
        d1, d2 = z["A1"] - z["M"], z["A2"] - z["O2"]
        a1 = (auc["A1"] - auc["M"]) - (auc["A2"] - auc["O2"])
        med_dd = float(np.nanmedian(np.concatenate([dd[r][ok] for r in C.RUNS])))
        a = score["populations"][pop]["aggregate"]["run_noise"]["a_bound"]
        r2 = np.nan_to_num(np.mean([(se[r] / dd[r]) ** 2 for r in C.RUNS], 0), nan=1.0)  # the record's ν
        nu = np.where(ok, r2 / max(r2[ok].mean(), 1e-12), 1.0)
        out["populations"][pop] = {
            "critical_z": {r: round(crit[r], 3) for r in C.RUNS},
            "n_scoreable": len(scoreable),
            "min_distance_over_runs": {
                "z": {"min": round(float(dmin.min()), 2), "median": round(float(np.median(dmin)), 2),
                      "n_below_2": int((dmin < 2).sum()), "n_below_1": int((dmin < 1).sum()),
                      "n_negative": int((dmin < 0).sum())},
                "auc": {"min": round(float(dmin_auc.min()), 4), "median": round(float(np.median(dmin_auc)), 4)}},
            "nearest_five": {f: per_cell[f] for f in nearest},
            "absent_everywhere": [f for f in scoreable if not any(per_cell[f]["existence"].values())],
            "run_means_auc_nonvoid": {r: round(float(np.nanmean(auc[r][ok])), 4) for r in C.RUNS},
            "drift": {"z_mean_recorded": score["populations"][pop]["aggregate"]["drift"]["mean"],
                      "z_mean_recomputed": round(float(np.mean((d1 - d2)[ok])), 3),
                      "auc_mean": round(float(np.nanmean(a1[ok])), 4),
                      "auc_se": round(float(np.nanstd(a1[ok], ddof=1) / np.sqrt(ok.sum())), 4),
                      "median_z_denominator_auc": round(med_dd, 5),
                      "label": "one draw of the run-level noise; not an estimate of its size"},
            "a_bound": {"ratio_recorded": a["omega_up_x_seed_noise"],
                        "sigma_c_z": round(float(np.sqrt(C.seed_noise_centred(d1, d2, nu, ok))), 3),
                        "omega_up_z": round(float(a["omega_up_x_seed_noise"] * np.sqrt(C.seed_noise_centred(d1, d2, nu, ok))), 3),
                        "omega_up_auc_at_median_denominator":
                            round(float(a["omega_up_x_seed_noise"] * np.sqrt(C.seed_noise_centred(d1, d2, nu, ok)) * med_dd), 4)},
            "cells": per_cell,
        }
    OUTF.write_text(json.dumps(out, indent=1))
    print(f"explore → {OUTF}", file=sys.stderr)


if __name__ == "__main__":
    main()
