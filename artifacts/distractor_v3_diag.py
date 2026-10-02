"""Diagnosis of the two v3 plant failures (distractor_plants.json, 2026-10-02). Measures, changes nothing.

1. The null plant is one permutation draw, and its CI is the test-split bootstrap: it holds the probe
   fixed. v3's draw sat at z ≈ 2.05 (NI 0.0025, CI lo 0.00016). Is that the ~5% miss a 95% CI makes by
   construction, or a bias? 20 independent permutation draws; the coverage rate answers it.
2. The secondary index's score-0 plant read SEEDS DISAGREE 2 of 20. The bar is 2 s_m, with
   s_m = |NI_M1 − NI_M2| on the secondary set: one difference that can cancel across variables. 200
   realisations, recording d_e, so the rate under s_m and under the pooled spread S̄ both come out.

    artifacts/_heavy.sh diag uv run python artifacts/distractor_v3_diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import distractor_plants as dp  # noqa: E402

NULL_DRAWS, SCORE_REPS = 20, 200


def main() -> None:
    d = dp.setup()
    d["r_flux"] = dp._r_flux_pread(d)
    d["sky"], _ = dp._sky()
    rng = np.random.default_rng(dp.SEED + 1)  # draws independent of the plant run's
    w, _ = dp.weights(d["test"], b=dp.N_BOOT)
    x = {t: (dp._load(t, d["train"]), dp._load(t, d["test"])) for t in ("m1", "m2")}
    out: dict = {"null": []}
    for _ in range(NULL_DRAWS):
        pt, pe = rng.permutation(len(d["train"])), rng.permutation(len(d["test"]))
        r = dp.readout(dp.probe(d, x["m1"][0][pt], x["m1"][1][pe]), w)
        out["null"].append({"NI": r["NI"], "NI_ci": r["NI_ci"],
                            "labels": {n: p["label_1a"] for n, p in r["per"].items()}})
        print(f"  null {len(out['null'])}: NI {r['NI']:+.4f} CI {r['NI_ci']}", flush=True)
    nl = out["null"]
    out["null_summary"] = {
        "draws": len(nl), "ci_contains_0": sum(a["NI_ci"][0] <= 0 <= a["NI_ci"][1] for a in nl),
        "all_near_chance": sum(all(v == "NEAR CHANCE" for v in a["labels"].values()) for a in nl),
        "both": sum(a["NI_ci"][0] <= 0 <= a["NI_ci"][1] and all(v == "NEAR CHANCE" for v in a["labels"].values())
                    for a in nl),
        "NI_mean": float(np.mean([a["NI"] for a in nl])), "NI_sd": float(np.std([a["NI"] for a in nl], ddof=1))}
    del x

    sc = {t: dp.probe(d, dp._load(t, d["train"]), dp._load(t, d["test"])) for t in ("m1", "m2")}
    rm = tuple(dp.readout(sc[t], w) for t in ("m1", "m2"))
    sig = {n: float(abs(rm[0]["per"][n]["auc"] - rm[1]["per"][n]["auc"]) / np.sqrt(2)) for n in dp.NUISANCE}
    sec = dp.SECONDARY
    ni_m = [dp._ni(r, sec) for r in rm]
    s_m = abs(ni_m[0] - ni_m[1])
    s_bar = float(np.mean([abs(rm[0]["per"][n]["auc"] - rm[1]["per"][n]["auc"]) for n in sec]))
    de = []
    for _ in range(SCORE_REPS):
        rb = [dp.readout(dp._score_plant(sc[t], 0.0, sig, rng), w) for t in ("m1", "m2")]
        de.append([dp._ni(r, sec) - sum(ni_m) / 2 for r in rb])
    de = np.array(de)
    opp = de[:, 0] * de[:, 1] < 0
    mn = np.abs(de).min(1)
    out["secondary_score0"] = {"reps": SCORE_REPS, "s_m": s_m, "S_bar": s_bar, "d_e_sd": float(de.std()),
                               "disagree_rate_bar_2s_m": float((opp & (mn > 2 * s_m)).mean()),
                               "disagree_rate_bar_2S_bar": float((opp & (mn > 2 * s_bar)).mean())}
    p = Path(__file__).parent / "out" / "distractor_v3_diag.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "null"}, indent=1))


if __name__ == "__main__":
    main()
