"""aligned_comparison.md v4, criterion 3: calibrate the seed-range floor F (D28, seed-noise null).

Rule (every seed-range rule in criterion 3): a direction is called only if BOTH aligned seeds lie
beyond BOTH M seeds by more than F, i.e. min(A) − max(M) > F (up) or min(M) − max(A) > F (down).
Null: the four seeds are exchangeable, each ~ N(μ, σ_q²), σ_q = |M1 − M2| / √2 (the observed spread).
All four are location-scale, so F_q = k·σ_q with one k; k is the 95th percentile of
max(up gap, down gap)/σ over 10⁶ exchangeable draws. The plant is the user's 20-realisation null at
F_q per quantity, plus the rate over 10⁵. Power: aligned seeds ~ N(μ + Δ, σ_q²), Δ named in advance.

    uv run python artifacts/aligned_c3_floor.py          # -> artifacts/out/c3_floor.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aligned_c13 import seed_range  # noqa: E402  the rule the real scoring uses, for the plant

OUT = Path(__file__).resolve().parent / "out"
SEED, N_REF, N_PLANT, N_BIG, ALPHA = 20260927, 1_000_000, 20, 100_000, 0.05
# Effect sizes named before this was run (2026-09-27), in each quantity's own units and direction.
EFFECTS = {"N": (-0.10, -0.25), "PR": (3.0, 6.0), "flag": (-0.10, -0.25), "Mo": (0.05, 0.10)}


def gaps(a: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return a.min(1) - m.max(1), m.min(1) - a.max(1)  # (up, down)


def calls(a, m, F):
    up, down = gaps(a, m)
    return up > F, down > F


def main() -> None:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((N_REF, 4))
    up, down = gaps(z[:, :2], z[:, 2:])
    k = float(np.quantile(np.maximum(up, down), 1 - ALPHA))
    rate_k0 = float((np.maximum(up, down) > 0).mean())
    c3 = json.loads((OUT / "c13_plants.json").read_text())["c3"]
    vals = {q: (c3["M1"][q], c3["M2"][q]) for q in ("N", "Mo", "PR")}
    for q, v in json.loads(sys.argv[1]).items() if len(sys.argv) > 1 else []:
        vals[q] = tuple(v)  # the flag fraction, once M2's SAE exists
    out = {"rule": "call iff min(A)-max(M) > F (up) or min(M)-max(A) > F (down)",
           "k_95": k, "false_call_rate_at_F0_exchangeable": rate_k0, "seed": SEED, "quantities": {}}
    for q, (m1, m2) in vals.items():
        mu, sig = (m1 + m2) / 2, abs(m1 - m2) / np.sqrt(2)
        F = k * sig
        r = {"M1": m1, "M2": m2, "sigma": sig, "F": F}
        x = mu + sig * rng.standard_normal((N_PLANT, 4))  # the D28 plant, through seed_range itself
        st = [seed_range(tuple(v[:2]), tuple(v[2:]), "HI", "SAME", "LO", F) for v in x]
        r["plant_20_states"] = st
        r["plant_20_false_calls"] = sum(t != "SAME" for t in st)
        r["plant_20_false_call_rate"] = r["plant_20_false_calls"] / N_PLANT
        x = mu + sig * rng.standard_normal((N_BIG, 4))  # the same rule, vectorised, for the rate
        u, d = calls(x[:, :2], x[:, 2:], F)
        r["null_1e5_false_call_rate"] = float((u | d).mean())
        for eff in EFFECTS[q]:
            x = mu + sig * rng.standard_normal((N_BIG, 4))
            x[:, :2] += eff
            u, d = calls(x[:, :2], x[:, 2:], F)
            right = u if eff > 0 else d
            r[f"power_at_{eff:+g}"] = float(right.mean())
            r[f"power_at_{eff:+g}_20"] = sum(seed_range(tuple(v[:2]), tuple(v[2:]), "HI", "SAME", "LO", F)
                                             == ("HI" if eff > 0 else "LO") for v in x[:N_PLANT])
        out["quantities"][q] = r
    out["plant_pass"] = all(r["plant_20_false_calls"] <= 1 for r in out["quantities"].values())
    (OUT / "c3_floor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
