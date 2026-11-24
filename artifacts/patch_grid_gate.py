"""Patch grid, stage 1 → stage 2: the blind ρ gate (docs/patch_grid/prereg.md).

    python artifacts/patch_grid_gate.py auc --arm jepa_p8_s0 --emb FILE.npz --out DIR   # one encoder: raw embeddings (ids, real) → DIR/<arm>.json; prints no score
    python artifacts/patch_grid_gate.py gate --dir DIR                                   # ρ̂ per objective and the decision; nothing else leaves this program

`auc` is B1's probe recipe on the frozen rows and decile edges (frozen_deciles.json, SHA-1 checked), raw. `gate` needs no untrained bank: the untrained
maximum is the same for every seed and the seed mean is taken out. It never forms a difference between patch sizes; ρ̂ is unchanged by any shift of one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_grid_score as P  # noqa: E402

FROZEN_SHA1 = "f245f70a19d435a6e0b619e194f0a423c3df48d4"


def frozen() -> dict:
    f = P.DOC / "frozen_deciles.json"
    if hashlib.sha1(f.read_bytes()).hexdigest() != FROZEN_SHA1:
        raise SystemExit("frozen_deciles.json is not the frozen file")
    return json.loads(f.read_text())["answers"]


def auc(arm: str, emb: str, out: Path) -> None:
    import almagest_axes as AXS
    fr, d = frozen(), AXS.Data(untrained=False)
    b = np.load(emb, allow_pickle=False)
    if not np.array_equal(b["ids"], d.union):
        raise SystemExit("the embedding file's ids are not the frozen union")
    M = b["real"].astype(np.float64)
    res = d.lab.nuisance_value("size", d.union).astype(float) / d.lab.nuisance_value("psf", d.union).astype(float)
    sha = lambda ids: hashlib.sha1("\n".join(str(int(i)) for i in ids).encode()).hexdigest()  # noqa: E731
    rec = {}
    for name, f in fr.items():
        tr, te = d.rows(f["key"])
        te = te[np.isfinite(res[te])]
        if sha(d.union[tr]) != f["train_ids_sha1"] or sha(d.union[te]) != f["test_ids_sha1"]:
            raise SystemExit(f"{name}: rows are not the frozen rows")
        y_tr, _ = d.yv(f["key"], tr)
        y, _ = d.yv(f["key"], te)
        k = np.clip(np.searchsorted(f["edges"], res[te], side="right") - 1, 0, 9)
        s = AXS.Probe(M[tr], y_tr, d.c).d(M[te])
        rec[f["key"]] = [AXS.fast_auc(y[k == j], s[k == j]) for j in range(10)]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}.json").write_text(json.dumps({"arm": arm, "embedding": str(emb), "auc_decile_raw": rec}, default=float))
    print(f"{arm}: written")


def low_matrix(recs: dict, o: str, p: int, seeds: list) -> np.ndarray:
    """(seeds, 30) deciles-1–5 mean AUC; NaN where an answer is undefined by the frozen class counts."""
    fr = frozen()
    return np.array([[np.mean(recs[o, p, s][f["key"]][: P.LOWDEC]) if min(min(c) for c in f["pos_neg"][: P.LOWDEC]) >= P.MIN_CLASS else np.nan for f in fr.values()] for s in seeds])


def gate(dir_: Path) -> dict:
    recs = {}
    for f in dir_.glob("*.json"):
        o, p, s = f.stem.split("_")
        recs[o, int(p[1:]), int(s[1:])] = json.loads(f.read_text())["auc_decile_raw"]
    out = {}
    for o in P.OBJ:
        ss = [s for s in P.seeds_of(recs, o, 16) if s in P.seeds_of(recs, o, 8)]
        r = P.rho_hat(low_matrix(recs, o, 16, ss), low_matrix(recs, o, 8, ss)) if ss else float("nan")
        out[o] = {"rho_hat": r, "seeds": ss, "decision": P.gate_decision(o, r)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["auc", "gate"])
    ap.add_argument("--arm")
    ap.add_argument("--emb")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    a = ap.parse_args()
    if a.mode == "auc":
        auc(a.arm, a.emb, Path(a.out))
        return
    for o, r in gate(Path(a.dir)).items():
        print(f"{o}: rho-hat {r['rho_hat']:.4f} (seeds {r['seeds']}): {r['decision']}")


if __name__ == "__main__":
    main()
