"""Brief AA1 — label-efficiency curve: how well does the frozen encoder classify with few labels?

The ladder's probe (standardised L2 logistic, C = the pipeline's), on M's frozen embeddings and the
three untrained draws, all 37 answers, full population (readout: no vote floor, D8 superseded).
Train on n ∈ {100, 300, 1k, 3k, 10k, 30k, full} galaxies subsampled from A, stratified on the
label, 5 subsets per n (the SAME subsets for every encoder); always scored on all of B.
States and the prediction: `aa_findings.md` §AA1.

  --planted   D28 — the state logic on planted curves, and one planted label through the fits
  --aa1       the curves
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_auc  # noqa: E402

NS = (100, 300, 1_000, 3_000, 10_000, 30_000)
K = 5
SMALL = 1_000
OUT = R.OUT / "aa1_label_efficiency.json"
PLANTED = R.OUT / "aa1_planted.json"


def subsets(y: np.ndarray, n: int, seed: int) -> np.ndarray | None:
    """Stratified: each class keeps its share, at least one of each; None if n can't hold both."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k1 = int(round(n * pos.size / y.size))
    k1 = min(max(k1, 1), pos.size)
    k0 = n - k1
    if k0 < 1 or k0 > neg.size:
        return None
    return np.sort(np.r_[rng.choice(pos, k1, replace=False), rng.choice(neg, k0, replace=False)])


def curve(mats: dict[str, EmbeddingMatrix], labels, feature: str, tr_ids, te_ids, c: float) -> dict:
    train = {k: feature_embeddings(m, labels, feature, tr_ids) for k, m in mats.items()}
    test = {k: feature_embeddings(m, labels, feature, te_ids) for k, m in mats.items()}
    y = train["M"].y
    n_full = int(y.size)
    rec: dict = {"n_full": n_full, "base_rate": float(y.mean()), "n_test": int(test["M"].y.size),
                 "auc": {}}
    if len(np.unique(test["M"].y)) < 2 or len(np.unique(y)) < 2:
        rec["skipped"] = "single class"
        return rec
    for n in (*[n for n in NS if n < n_full], n_full):
        draws = [None] if n == n_full else [subsets(y, n, s) for s in range(K)]
        if any(d is None for d in draws) and n != n_full:
            rec["auc"][str(n)] = "below class floor"
            continue
        per = {}
        for k in mats:
            vals = []
            for d in draws:
                tr = train[k] if d is None else Embeddings(train[k].x[d], train[k].y[d],
                                                           train[k].fraction[d])
                vals.append(probe_auc(tr, test[k], c=c))
            per[k] = vals
        rec["auc"]["full" if n == n_full else str(n)] = per
    return rec


def summarise(rec: dict) -> dict:
    """Mean curves, the paired margin (M − mean untrained, per subset) and n90/n95."""
    ns, m, u, mar, se = [], [], [], [], []
    for key, per in rec["auc"].items():
        if not isinstance(per, dict):
            continue
        mm = np.array(per["M"])
        uu = np.mean([per[k] for k in per if k != "M"], 0)
        d = mm - uu
        ns.append(rec["n_full"] if key == "full" else int(key))
        m.append(float(mm.mean()))
        u.append(float(uu.mean()))
        mar.append(float(d.mean()))
        se.append(float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0)
    o = np.argsort(ns)
    out = {"n": [ns[i] for i in o], "auc_M": [m[i] for i in o], "auc_untrained": [u[i] for i in o],
           "margin": [mar[i] for i in o], "margin_se": [se[i] for i in o]}
    if ns:
        full = out["auc_M"][-1]
        for q in (0.90, 0.95):
            lit = [n for n, a in zip(out["n"], out["auc_M"], strict=True) if a >= q * full]
            abv = [n for n, a in zip(out["n"], out["auc_M"], strict=True)
                   if a - 0.5 >= q * (full - 0.5)]
            out[f"n{int(q * 100)}_literal"] = lit[0] if lit else None
            out[f"n{int(q * 100)}_above_chance"] = abv[0] if abv else None
    return out


def margin_state(s: dict) -> str:
    """First match applies (aa_findings.md §AA1)."""
    n, mar, se = np.array(s.get("n", [])), np.array(s.get("margin", [])), np.array(s.get("margin_se", []))
    if n.size < 3:
        return "INSUFFICIENT"
    if not np.any(mar - 1.96 * np.maximum(se, 1e-9) > 0):
        return "NOT ABOVE UNTRAINED"
    small, large = n <= SMALL, n > SMALL
    if not small.any() or not large.any():
        return "INSUFFICIENT"
    i_s = np.flatnonzero(small)[np.argmax(mar[small])]
    i_l = np.flatnonzero(large)[np.argmax(mar[large])]
    gap = mar[i_s] - mar[i_l]
    noise = 2 * np.hypot(se[i_s], se[i_l])
    if gap > noise:
        return "LARGEST AT SMALL N"
    if -gap > noise:
        return "LARGEST AT LARGE N"
    return "FLAT"


def catalogue_state(states: list[str]) -> str:
    """The prediction, across evaluable features (those with a margin)."""
    ev = [s for s in states if s in ("LARGEST AT SMALL N", "LARGEST AT LARGE N", "FLAT")]
    if not ev:
        return "NOT EVALUABLE"
    small = sum(s == "LARGEST AT SMALL N" for s in ev) / len(ev)
    large = sum(s == "LARGEST AT LARGE N" for s in ev) / len(ev)
    if small > 0.5:
        return "SUPPORTED"
    if large > 0.5:
        return "CONTRADICTED"
    return "MIXED"


def context(setup):
    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    mats = {"M": ctx.real, **{f"untrained-s{i}": m for i, m in enumerate(ctx.untrained)}}
    return ctx, mats


def planted(setup) -> dict:
    out: dict = {"state_logic": {}}
    ns = [100, 300, 1000, 3000, 10000, 30000, 40000]
    se = [0.02, 0.015, 0.01, 0.006, 0.004, 0.003, 0.0]
    shapes = {"small": [0.20, 0.18, 0.15, 0.10, 0.06, 0.05, 0.05],
              "large": [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.12],
              "flat": [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
              "none": [0.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0],
              "thin": [0.1, 0.1]}
    for k, mar in shapes.items():
        s = {"n": ns[: len(mar)], "margin": mar, "margin_se": se[: len(mar)]}
        out["state_logic"][k] = margin_state(s)
    out["catalogue"] = {"3 small 1 flat": catalogue_state(["LARGEST AT SMALL N"] * 3 + ["FLAT"]),
                        "3 large 1 flat": catalogue_state(["LARGEST AT LARGE N"] * 3 + ["FLAT"]),
                        "split": catalogue_state(["LARGEST AT SMALL N", "LARGEST AT LARGE N", "FLAT"]),
                        "none": catalogue_state(["NOT ABOVE UNTRAINED"] * 3)}

    # One planted label through the identical fits: a label that is linear in M's embedding
    ctx, mats = context(setup)
    x = ctx.real.x.astype(np.float64)
    w = np.random.default_rng(0).normal(size=x.shape[1])
    s = (x - x.mean(0)) / (x.std(0) + 1e-8) @ w
    s = (s - s.mean()) / s.std() + 1.0 * np.random.default_rng(1).normal(size=s.size)
    ylab = (s > np.quantile(s, 0.7)).astype(np.int64)

    class Planted:
        features = ("planted",)

        def eligible(self, f, ids):
            return list(ids)

        def binary_label(self, f, ids):
            pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
            return ylab[[pos[int(o)] for o in ids]]

        def vote_fraction(self, f, ids):
            return self.binary_label(f, ids).astype(float)

    rec = curve(mats, Planted(), "planted", setup.train_ids, setup.test_ids, c=ctx.pc.c)
    sm = summarise(rec)
    out["planted_label"] = {**sm, "state": margin_state(sm)}
    return out


def aa1(setup) -> dict:
    ctx, mats = context(setup)
    labels = ctx.labels["full"]
    rec: dict = {"c": ctx.pc.c, "features": {}}
    for f in ctx.features:
        r = curve(mats, labels, f, setup.train_ids, setup.test_ids, c=ctx.pc.c)
        if "skipped" not in r:
            r["summary"] = summarise(r)
            r["state"] = margin_state(r["summary"])
        rec["features"][f] = r
        R._save(OUT, rec)
        print(f"AA1 {f}: {r.get('state')} {[round(v, 3) for v in r.get('summary', {}).get('margin', [])]}",
              file=sys.stderr, flush=True)
    rec["catalogue"] = catalogue_state([r.get("state", "INSUFFICIENT") for r in rec["features"].values()])
    return rec


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--aa1"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="AA1", sources=1)
    out, path = (planted(setup), PLANTED) if mode == "--planted" else (aa1(setup), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
