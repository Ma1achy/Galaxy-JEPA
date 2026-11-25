"""Patch grid, after training: embeddings → per-decile AUCs → the registered readings (docs/patch_grid/prereg.md §4–§7). Run from the repo root.

    python artifacts/patch_grid_auc.py auc   --name jepa_p8_s0 --emb FILE.npz --out DIR   # one encoder (trained or untrained) → DIR/<name>.json; prints no score
    python artifacts/patch_grid_auc.py valid --runs DIR                                   # VOID arms by collapse/halt and the divergence rule (training traces only)
    python artifacts/patch_grid_auc.py plants                                             # D28 for the divergence rule
    python artifacts/patch_grid_auc.py manifest --dir DIR                                 # DIR/outputs.sha1 over every AUC record: written before scoring
    python artifacts/patch_grid_auc.py final --dir DIR [--void NAME ...] --out FILE.json  # every reading, once all 27 arms and 9 untrained banks are scored

`auc` is B1's probe recipe (spread.json's) on the frozen rows and decile edges (frozen_deciles.json, SHA-1 checked), raw (primary) and cleaned.
`final` is the only program that forms a difference between patch sizes. It refuses to run on a partial grid (VOID arms must be named), without
outputs.sha1, or if any record has changed since it was written. Every mode but `plants` and `valid` first checks the pinned code of record.
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
SPREAD_SHA1 = "d4d05a2b072f91d2188524d9527a0235f53f1c15"
KINDS = ("raw", "cleaned")
UNTRAINED_SEEDS = (0, 1, 2)
STOP = 101_308
DIVERGE, WINDOW = 0.5, 500  # F7's rule (artifacts/baselines.md): final 500-step mean more than 50% above the lowest 500-step rolling mean


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(dir_: Path) -> None:
    f = dir_ / "outputs.sha1"
    if f.exists():
        raise SystemExit(f"{f} exists: it is written once, before scoring")
    f.write_text("".join(f"{sha1(p)}  {p.name}\n" for p in sorted(dir_.glob("*.json"))))
    print(f"{f}: {len(f.read_text().splitlines())} records")


def check_manifest(dir_: Path) -> None:
    f = dir_ / "outputs.sha1"
    if not f.exists():
        raise SystemExit("final: outputs.sha1 does not exist (patch_grid_auc.py manifest, before scoring)")
    want = {ln.split()[1]: ln.split()[0] for ln in f.read_text().splitlines()}
    have = {p.name: sha1(p) for p in dir_.glob("*.json")}
    if want != have:
        raise SystemExit("final: the records differ from outputs.sha1: " + ", ".join(sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))[:5]))


def frozen() -> dict:
    f = P.DOC / "frozen_deciles.json"
    if hashlib.sha1(f.read_bytes()).hexdigest() != FROZEN_SHA1:
        raise SystemExit("frozen_deciles.json is not the frozen file")
    return json.loads(f.read_text())["answers"]


# --- one encoder ------------------------------------------------------------------------------------------------------
def auc(name: str, emb: str, out: Path) -> None:
    import almagest_axes as AXS
    fr, d = frozen(), AXS.Data(untrained=False)
    b = np.load(emb, allow_pickle=False)
    if "name" not in b.files or str(b["name"]) != name:
        raise SystemExit(f"{emb} says it is {str(b['name']) if 'name' in b.files else 'unnamed'}, not {name}")
    if not np.array_equal(b["ids"], d.union):
        raise SystemExit("the embedding file's ids are not the frozen union")
    x = b["real"].astype(np.float64)
    tr_m = np.isin(d.union, np.asarray(d.setup.train_ids))
    cl = np.full_like(x, np.nan)
    cl[d.ok] = AXS.S.Cleaner(x[d.ok & tr_m], d.N[d.ok & tr_m])(x[d.ok], d.N[d.ok])
    res = d.lab.nuisance_value("size", d.union).astype(float) / d.lab.nuisance_value("psf", d.union).astype(float)
    sha = lambda ids: hashlib.sha1("\n".join(str(int(i)) for i in ids).encode()).hexdigest()  # noqa: E731
    rec = {k: {"decile": {}, "all": {}} for k in KINDS}
    for nm, f in fr.items():
        tr, te = d.rows(f["key"])
        te = te[np.isfinite(res[te])]
        if sha(d.union[tr]) != f["train_ids_sha1"] or sha(d.union[te]) != f["test_ids_sha1"]:
            raise SystemExit(f"{nm}: rows are not the frozen rows")
        y_tr, _ = d.yv(f["key"], tr)
        y, _ = d.yv(f["key"], te)
        k = np.clip(np.searchsorted(f["edges"], res[te], side="right") - 1, 0, 9)
        for kind, M in (("raw", x), ("cleaned", cl)):
            s = AXS.Probe(M[tr], y_tr, d.c).d(M[te])
            rec[kind]["decile"][f["key"]] = [AXS.fast_auc(y[k == j], s[k == j]) for j in range(10)]
            rec[kind]["all"][f["key"]] = AXS.fast_auc(y, s)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"name": name, "embedding": str(emb), "embedding_sha1": sha1(Path(emb)), **rec}, default=float))
    print(f"{name}: written")


# --- validity: training traces only -------------------------------------------------------------------------------------
def diverged(losses) -> dict:
    """F7's rule on a run's per-step losses: non-finite anywhere, or the final 500-step mean > 1.5 × the lowest 500-step rolling mean."""
    x = np.asarray(losses, float)
    if len(x) < WINDOW or not np.isfinite(x).all():
        return {"diverged": True, "why": "non-finite loss" if len(x) >= WINDOW else "fewer than 500 steps"}
    roll = np.convolve(x, np.ones(WINDOW) / WINDOW, "valid")
    f, low = float(x[-WINDOW:].mean()), float(roll.min())
    return {"diverged": f > low * (1 + DIVERGE), "final_500_mean": f, "lowest_rolling_500": low, "rise_over_low": f / low - 1}


def valid(runs: Path) -> dict:
    out = {}
    for t in sorted(runs.glob("*/traces.json")):
        r = json.loads(t.read_text())
        d = diverged(r["losses"])  # a non-finite loss halts the trainer before it is recorded, so `halted` covers it too
        h, n = bool(r.get("halted", False)), len(r["losses"])
        out[t.parent.name] = {**d, "halted": h, "steps": n, "VOID": d["diverged"] or h, "incomplete": not (d["diverged"] or h) and n != STOP}
    return out


def plants_divergence() -> dict:
    """D28 for the divergence rule: it fires on planted divergences and on a non-finite loss, and not on the nine Mac traces or a rise of 49%."""
    n = 2000
    low = np.full(n, 1.0)
    rise = lambda r: np.concatenate([low, np.linspace(1.0, 1.0 + r, 500), np.full(1000, 1.0 + r)])  # noqa: E731
    cases = {"rise 51%": rise(0.51), "rise 60%": rise(0.6), "rise 49%": rise(0.49), "early dip then flat": np.concatenate([np.linspace(1.3, 0.1, 1000), np.full(3000, 0.13)]),
             "non-finite": np.concatenate([low, [np.nan]]), "too short": np.ones(100)}
    out = {k: diverged(v) for k, v in cases.items()}
    out["mac"] = {m: diverged(json.loads((P.REPO / f"runs/{m}/traces.json").read_text())["losses"]) for m in
                  ("m_v2", "m_v2_s1", "m_v2_s2", "moco_s0", "moco_s1", "moco_s2", "mae_s0", "mae_s1", "mae_s2")}
    (P.DOC / "plants_divergence.json").write_text(json.dumps(out, indent=1, default=float))
    want = {"rise 51%": True, "rise 60%": True, "rise 49%": False, "early dip then flat": False, "non-finite": True, "too short": True}
    got = {k: out[k]["diverged"] for k in want}
    assert got == want, got
    assert not any(v["diverged"] for v in out["mac"].values())
    print(got, "| fires on none of the nine Mac runs")
    return out


# --- the readings -------------------------------------------------------------------------------------------------------
def tables(dir_: Path, kind: str, void=()):
    recs = {f.stem: json.loads(f.read_text()) for f in dir_.glob("*.json")}
    auc_, un = {}, {}
    missing = []
    for o in P.OBJ:
        for p in P.PATCH:
            for s in (0, 1, 2):
                n = f"{o}_p{p}_s{s}"
                if n in void:
                    continue
                if n not in recs:
                    missing.append(n)
                    continue
                auc_[o, p, s] = {a: np.array(recs[n][kind]["decile"][a]) for a in P.KEY.values()}
    for p in P.PATCH:
        us = []
        for s in UNTRAINED_SEEDS:
            n = f"untrained_p{p}_s{s}"
            if n not in recs:
                missing.append(n)
            else:
                us.append(recs[n][kind]["decile"])
        for o in P.OBJ:
            un[o, p] = {a: np.array([u[a] for u in us]) for a in P.KEY.values()} if us else None
    if missing:
        raise SystemExit("final: missing (name an arm with --void if it is VOID; an untrained bank can't be): " + ", ".join(missing))
    return auc_, un


def device_control(dir_: Path, void) -> dict:
    """GPU patch-16 arm against the Mac run with the same train_seed: full-sample raw AUC per answer (spread.json is the Mac record; SHA-1 kept)."""
    f = P.OUT / "spread.json"
    if sha1(f) != SPREAD_SHA1:
        raise SystemExit("spread.json is not the Mac record of the pre-registration")
    sp = json.loads(f.read_text())
    mac_names = {"jepa": ["A1", "A2", "A3"], "moco": ["moco0", "moco1", "moco2"], "mae": ["mae0", "mae1", "mae2"]}
    m = sp["auc_all"]["raw"]
    gpu, mac = {}, {}
    for o, ns in mac_names.items():
        for s, n in enumerate(ns):
            nm = f"{o}_p16_s{s}"
            if nm in void or not (dir_ / f"{nm}.json").exists():
                continue
            gpu[o, s] = json.loads((dir_ / f"{nm}.json").read_text())["raw"]["all"]
            mac[o, s] = m[n]
    return {"spread_sha1": hashlib.sha1(f.read_bytes()).hexdigest(), **P.device(gpu, mac, {s: m[n] for s, n in enumerate(mac_names["jepa"])})}


def final(dir_: Path, void, out: Path) -> None:
    if out.resolve().parent == dir_.resolve():
        raise SystemExit("final: --out must not be inside the records directory")
    check_manifest(dir_)
    P.load_K()
    fr = frozen()
    counts = {f["key"]: np.array(f["pos_neg"]) for f in fr.values()}
    nm = {v: k for k, v in P.KEY.items()}
    rep = {"void": list(void), "kinds": {}}
    for kind in KINDS:  # raw is the registered reading, cleaned is beside it
        A, un = tables(dir_, kind, void)
        short = {e: {nm[a]: v for a, v in r.items()} for e, r in A.items()}
        un_s = {k: {nm[a]: v for a, v in r.items()} for k, r in un.items()}
        L = P.all_L(short, un_s, {nm[a]: c for a, c in counts.items()})
        r = P.score_L(L)
        rep["kinds"][kind] = {"L": {"_".join(map(str, e)): v for e, v in L.items()}, "steps": {f"{o}/{g}": v for (o, g), v in r["steps"].items()},
                              "readings": {f"{o}/{g}": v for (o, g), v in r["readings"].items()},
                              "objective_pairs": {"/".join(k): v for k, v in r["objective_pairs"].items()},
                              "rho_hat_descriptive": {o: {"16->8": P.rho_descriptive(L, o, 16, 8), "8->4": P.rho_descriptive(L, o, 8, 4)} for o in P.OBJ}}
    rep["device_control"] = device_control(dir_, void)
    rep["rho_hat_planted_bias_and_sd"] = json.loads((P.DOC / "plants_rho.json").read_text())["2_bias"]
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"final: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["auc", "manifest", "valid", "plants", "final"])
    for f in ("--name", "--emb", "--out", "--dir", "--runs"):
        ap.add_argument(f)
    ap.add_argument("--void", nargs="*", default=[])
    a = ap.parse_args()
    if a.mode in ("auc", "manifest", "final"):
        import patch_grid_pins
        patch_grid_pins.verify()
    if a.mode == "manifest":
        manifest(Path(a.dir))
    elif a.mode == "auc":
        auc(a.name, a.emb, Path(a.out))
    elif a.mode == "plants":
        plants_divergence()
    elif a.mode == "valid":
        for k, v in valid(Path(a.runs)).items():
            print(k, json.dumps(v))
    else:
        final(Path(a.dir), tuple(a.void), Path(a.out))


if __name__ == "__main__":
    main()
