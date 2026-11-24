"""Seed spread of the low-resolution margin L at patch 16 on the Mac (JEPA = A1, A2, A3; MoCo and MAE seeds 0, 1, 2), for the power statement of
docs/patch_grid/prereg.md. Patch-16 records only: no 8×8 or 4×4 result exists. B1's own probe recipe, deciles and footing (A1's decile edges).
Untrained banks: the three patch-16 seeds, shared by all nine encoders (same architecture). → runs/almagest/patch_grid/spread.json"""
import json
import time
import numpy as np
import almagest_axes as AXS
import almagest_visibility as V
import patch_grid_score as P

d = AXS.Data(untrained=True)
ok, tr_m = d.ok, np.isin(d.union, np.asarray(d.setup.train_ids))
size, psf = d.lab.nuisance_value("size", d.union).astype(float), d.lab.nuisance_value("psf", d.union).astype(float)
res = size / psf
un = [k for k in d.mats if k != "A1"]
TRAINED = {"A1": None, "A2": "a2", "A3": "a3", **{f"{o}{s}": f"{o}_s{s}" for o in ("moco", "mae") for s in range(3)}}

ROWS = {}  # answer → (train rows, y_train, test rows, y_test, decile index)
for a in P.KEY.values():
    tr, te = d.rows(a)
    y_tr, _ = d.yv(a, tr); y_te, _ = d.yv(a, te)
    fin = np.isfinite(res[te]); te, y_te = te[fin], y_te[fin]
    edges = np.quantile(res[te], np.linspace(0, 1, 11))
    ROWS[a] = (tr, y_tr, te, y_te, np.clip(np.searchsorted(edges, res[te], side="right") - 1, 0, 9))
counts = {a: [[int(y[b == k].sum()), int((1 - y[b == k]).sum())] for k in range(10)] for a, (_, _, _, y, b) in ROWS.items()}


def curves(M, a):
    tr, y_tr, te, y_te, b = ROWS[a]
    s = AXS.Probe(M[tr], y_tr, d.c).d(M[te])
    return np.array([AXS.fast_auc(y_te[b == k], s[b == k]) for k in range(10)]), AXS.fast_auc(y_te, s)


def matrices(n):
    """(raw, cleaned) of one encoder, loaded alone: 18 GB machine, so never two trained encoders in memory at once."""
    if n == "A1" or n in un:
        return d.mats[n], d.clean[n]
    b = np.load(AXS.REPO / f"runs/ladder_v2/{TRAINED[n]}/o1_embeddings.npz", allow_pickle=False)
    assert np.array_equal(b["ids"], d.union), n
    x = b["real"].astype(np.float64)
    c = np.full_like(x, np.nan)
    c[ok] = AXS.S.Cleaner(x[ok & tr_m], d.N[ok & tr_m])(x[ok], d.N[ok])
    return x, c


AUC, ALL = {}, {}
for n in [*un, *TRAINED]:
    t0 = time.perf_counter()
    for kind, M in zip(("raw", "cleaned"), matrices(n)):
        for a in ROWS:
            AUC[kind, n, a], ALL[kind, n, a] = curves(M, a)
    V.log(f"{n} {time.perf_counter() - t0:.0f}s")
out = {"note": "A1/A2/A3, moco0-2, mae0-2: patch-16 Mac seeds 0/1/2 per objective; L = mean over deciles 1-5 of (AUC - max untrained); B1 probes and deciles; probes on the frozen 39,997-galaxy train rows",
       "counts": counts, "L": {}, "auc_all": {}}
for kind in ("raw", "cleaned"):
    u = {a: np.array([AUC[kind, k, a] for k in un]) for a in ROWS}
    out["L"][kind] = {n: {a: P.low_margin(AUC[kind, n, a], u[a], counts[a]) for a in ROWS} for n in TRAINED}
    out["auc_all"][kind] = {n: {a: ALL[kind, n, a] for a in ROWS} for n in TRAINED}
    out.setdefault("auc_decile", {})[kind] = {n: {a: AUC[kind, n, a].tolist() for a in ROWS} for n in [*un, *TRAINED]}
P.OUT.mkdir(parents=True, exist_ok=True)
(P.OUT / "spread.json").write_text(json.dumps(out, indent=1, default=float))
