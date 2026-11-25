"""Part B review addition (user, 25 Nov): partial Spearman given the five nuisances for A1 and every untrained seed, and A1's gain over the best untrained seed on it.

    .venv/bin/python artifacts/axes_b_partial.py      # → docs/axes/b_partial.json

Fixed before computing: axes_b's own ridge (GCV α, same rows, same winsorising, same halves for statmorph), refitted per encoder; partial ρ = correlation of the nuisance-residualised
ranks (axes_b.partial). Gain = A1's partial ρ minus the best untrained seed's (the seed chosen on the point estimate), with a 500-resample paired bootstrap over test galaxies of the
residualised ranks (residualised once on the full test set, as axes_b's own partial CI). Raw and cleaned embeddings; size and magnitude raw only, as before.
"""
import json
import time

import numpy as np

import axes_b as Bm

AXS, d = Bm.A.load()
z = np.load(Bm.B / "measures.npz")
assert np.array_equal(z["ids"], d.union)
names = [k for k in z.files if k != "ids"]
out = {"note": __doc__, "targets": {}}
for j, n in enumerate(names):
    t0 = time.perf_counter()
    y = z[n]
    tr, te = Bm.rows_for(d, y, n)
    lo, hi = np.percentile(y[tr], [0.5, 99.5])
    yc = np.clip(y, lo, hi)
    ry = Bm.resid_ranks(yc[te], d.N[te])
    rec = {"n_test": len(te), "nuisance": n in Bm.NUISANCE}
    for kind in ("raw",) if n in Bm.NUISANCE else ("raw", "cleaned"):
        R = {}
        for e in Bm.ENC:
            X = (d.clean if kind == "cleaned" else d.mats)[e]
            m = Bm.Gram(X[tr]).fit(yc[tr])
            R[e] = Bm.resid_ranks(X[te] @ m["we"] + m["b"], d.N[te])
        p = {e: float(np.corrcoef(R[e], ry)[0, 1]) for e in R}
        best = max(Bm.ENC[1:], key=lambda e: p[e])
        rng = np.random.default_rng(Bm.SEED + 2000 + j)
        dv = []
        for _ in range(500):
            i = rng.integers(0, len(te), len(te))
            dv.append(np.corrcoef(R["A1"][i], ry[i])[0, 1] - np.corrcoef(R[best][i], ry[i])[0, 1])
        rec[kind] = {"partial": p, "best_untrained": best, "gain": p["A1"] - p[best], "gain_ci": np.percentile(dv, [2.5, 97.5]).tolist()}
    out["targets"][n] = rec
    (Bm.DOC / "b_partial.json").write_text(json.dumps(out, indent=1, default=float))
    k = rec.get("cleaned", rec["raw"])
    Bm.A.log(f"partial {n}: A1 {k['partial']['A1']:.3f} best untrained {k['partial'][k['best_untrained']]:.3f} gain {k['gain']:+.3f} {np.round(k['gain_ci'], 3)} {time.perf_counter() - t0:.0f}s")
