"""Part C, post hoc (24 Nov 2026): the ARI comparison at matched k. The stated rule compares each encoder at its own silhouette k
(A1 7 or 3, untrained 2); a 2-way split cannot score high against 8 groups. Same subsampling, every k, every seed. Adds `matched_k` to c_star.json."""
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from threadpoolctl import threadpool_limits
import axes_c_star as C

ids, _, _ = C.sample()
df, F = C.table()
sid = [str(i) for i in ids]
g, fl = C.groups(df.loc[sid], F.loc[sid]), F.loc[sid]
sat, edge = fl.SATURATED.to_numpy(int), fl.EDGE.to_numpy(int)
emb, ok, *_ = C.footing(ids)
U = [f"untrained-s{s}" for s in range(3)]
out = {}
with threadpool_limits(3):
    for sp in ("raw", "cleaned"):
        rng = np.random.default_rng(C.SEED)
        ix = [rng.choice(len(ids), int(.8 * len(ids)), replace=False) for _ in range(C.N_SUB)]
        out[sp] = {}
        for k in C.KS:
            L = {n: [C.km(emb[sp, n][s], k, seed=b, n_init=5).labels_ for b, s in enumerate(ix)] for n in ("A1", *U)}
            r = {}
            for nm, t in (("pattern", g), ("SATURATED", sat), ("EDGE", edge)):
                A = {n: np.array([ari(t[s], l) for s, l in zip(ix, L[n])]) for n in L}
                best = max(U, key=lambda n: A[n].mean())
                d = A["A1"] - A[best]
                r[nm] = {"mean": {n: float(v.mean()) for n, v in A.items()}, "best": best, "diff": float(d.mean()),
                         "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
            out[sp][k] = r
            print(sp, k, {q: (round(v["mean"]["A1"], 3), round(v["mean"][v["best"]], 3), [round(x, 3) for x in v["ci"]]) for q, v in r.items()}, flush=True)
p = C.DOC / "c_star.json"
j = json.loads(p.read_text()); j["matched_k"] = {"status": "post hoc", **out}
p.write_text(json.dumps(j, indent=1, default=float))
