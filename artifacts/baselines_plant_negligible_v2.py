"""Negligible plant, strengthened (baselines_comparison.md, Plant results): rho 0.9, delta 0.004.

The rule is unchanged; only the plant. No real result is read.

    .venv/bin/python artifacts/baselines_plant_negligible_v2.py
"""
import json, sys
sys.path.insert(0, "artifacts")
import numpy as np
import baselines_comparison as B
from aligned_c2 import weights
answers, ys, rows, test = B._labels()
wt, pos = weights(test)
cols = {f: [pos[int(o)] for o in rows[f]] for f in answers}
w = B._w(); J = len(answers)
rng = np.random.default_rng(20261024)
base = rng.uniform(0.70, 0.95, J)
rng = np.random.default_rng(20261025)
on_x = {"A1": 0.0, "A2": 0.0, "X0": 0.004, "X1": 0.004}
res = []
for _ in range(10):
    A, Bt = B._realise(rng, answers, ys, cols, wt, base, tau=0.0003, sig=0.0005, delta=on_x, rho=0.9)
    r = B.decide(A, Bt, answers, w)
    res.append({"state": r["state"], "D": r["D"], "ci": r["ci"], "bar": r["bar"]})
    print(r["state"], round(r["D"], 5), [round(x, 5) for x in r["ci"]], round(r["bar"], 5), flush=True)
json.dump({"plant": "negligible_v2", "tau": 0.0003, "sig": 0.0005, "delta": 0.004, "rho": 0.9,
           "base_seed": 20261024, "noise_seed": 20261025, "runs": res},
          open("artifacts/out/baselines_plants_negligible_v2.json", "w"), indent=1)
