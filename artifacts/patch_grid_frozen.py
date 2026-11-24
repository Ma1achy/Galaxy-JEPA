"""Freeze the patch-grid evaluation set (docs/patch_grid/prereg.md §4): per-answer resolution-decile edges, per-decile class counts, and the SHA-1 of
the train and test row lists. B1's own rows and edges (petroRad_r / psfWidth_r over the answer's eligible test rows). No probe, no score.
→ docs/patch_grid/frozen_deciles.json"""
import hashlib
import json
import numpy as np
import almagest_axes as AXS
import patch_grid_score as P

d = AXS.Data(untrained=False)
res = d.lab.nuisance_value("size", d.union).astype(float) / d.lab.nuisance_value("psf", d.union).astype(float)
sha = lambda ids: hashlib.sha1("\n".join(str(int(i)) for i in ids).encode()).hexdigest()  # noqa: E731
out = {"definition": "decile edges = quantiles of petroRad_r / psfWidth_r over the answer's eligible test rows; SHA-1 of newline-joined objIDs in row order", "answers": {}}
for name, a in P.KEY.items():
    tr, te = d.rows(a)
    y_te, _ = d.yv(a, te)
    fin = np.isfinite(res[te]); te, y = te[fin], y_te[fin]
    e = np.quantile(res[te], np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(e, res[te], side="right") - 1, 0, 9)
    out["answers"][name] = {"key": a, "n_train": len(tr), "n_test": len(te), "edges": e.tolist(), "pos_neg": [[int(y[b == k].sum()), int((1 - y[b == k]).sum())] for k in range(10)],
                            "train_ids_sha1": sha(d.union[tr]), "test_ids_sha1": sha(d.union[te]),
                            "min_class_deciles_1_5": int(min(min(int(y[b == k].sum()), int((1 - y[b == k]).sum())) for k in range(5)))}
(P.DOC / "frozen_deciles.json").write_text(json.dumps(out, indent=1))
f = P.DOC / "frozen_deciles.json"
print(hashlib.sha1(f.read_bytes()).hexdigest(), len(out["answers"]))
sm = sorted((v["min_class_deciles_1_5"], n) for n, v in out["answers"].items())
print(sm[:5])
