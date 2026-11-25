"""F5 top-up (f_report.md): the saved top-20 PCA / ICA component scores of cleaned A1 against every Part B measurement (catalogue and statmorph). Same rule as F5:
Spearman on test rows with a finite target. Writes docs/axes/f5_topup.json; f5.json is not touched."""
import json
import numpy as np
from scipy.stats import spearmanr
import axes_a_ordinal as A
AXS, d = A.load()
z = np.load(A.OUT / "f/f5_scores.npz")
assert np.array_equal(z["ids"], d.union)
m = np.load(A.OUT / "b/measures.npz")
te = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
out = {"n_test": int(te.sum()), "measures": [k for k in m.files if k != "ids"], "components": {}}
for kind in ("pca", "ica"):
    for k in range(20):
        s = z[kind][:, k].astype(float)
        row = {}
        for n in out["measures"]:
            y = m[n].astype(float)
            o = te & np.isfinite(y)
            row[n] = {"rho": float(spearmanr(s[o], y[o])[0]), "n": int(o.sum())}
        top = sorted(row, key=lambda n: -abs(row[n]["rho"]))[:3]
        out["components"][f"{kind}{k + 1}"] = {"targets": row, "top3": [(n, round(row[n]["rho"], 3)) for n in top]}
(A.DOC / "f5_topup.json").write_text(json.dumps(out, indent=1))
for k, v in out["components"].items():
    print(k, v["top3"])
