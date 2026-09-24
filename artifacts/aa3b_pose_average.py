"""Brief AA3b — does the pose code cost anything?

Every union galaxy (74,829) embedded by M under the 8 dihedral transforms; the 8 are averaged.
A 180° rotation sends the band-offset vector (AA3a) to its negative, so the average should cancel
it. Then, on the averaged M against M as trained, three readouts:
  existence   the ladder's probe AUC, all 37 answers, full population, A → B, paired bootstrap
  salience    V3's held-out recovery test (one PC of 50 aligned on A, frozen, AUC on B)
  winding     Y3's T4b: U3's winding axis against Hayes pitch, visibility controlled
Untrained references are the recorded ones (not averaged) — declared in `aa_findings.md` §AA3b.

  --bank      the 8 transforms in one pass per stamp → out/aa3b_d4_<name>.npy (float16 memmaps)
  --planted   D28 — PC1/PC2 collapse; the state logic; a pose label and an invariant label
              through the identical existence path
  --aa3b      the test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import v3_own_directions as V3  # noqa: E402
import x1_handedness as X  # noqa: E402
import y3_science as Y3  # noqa: E402
import y_pitch as Y  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import (EmbeddingMatrix, extract_matrix,  # noqa: E402
                                         feature_embeddings, feature_ids)
from galaxy_jepa.probing.logistic import (_fit, paired_auc_bootstrap,  # noqa: E402
                                          probe_direction)

D4 = [f"r{k}{'m' if m else ''}" for m in (0, 1) for k in range(4)]  # r0 … r3, r0m … r3m
OUT = R.OUT / "aa3b_pose_average.json"
PLANTED = R.OUT / "aa3b_planted.json"
MATERIAL = 0.01      # |ΔAUC| a change must reach, with its paired 95% CI excluding 0
COUNT = 2            # answers moving one way before the catalogue calls it
COLLAPSE = 0.10      # averaged PC1/PC2 variance must fall below this share of the original
RANK = {"NO AXIS": 0, "INVERTED": 0, "NOT ABOVE UNTRAINED": 1, "WEAK AXIS": 2, "PARTIAL AXIS": 3,
        "ENCODER AXIS": 4}


def d4(img: torch.Tensor, name: str) -> torch.Tensor:
    k, mirror = int(name[1]), name.endswith("m")
    if mirror:
        img = torch.flip(img, dims=(-1,))
    return torch.rot90(img, k, dims=(-2, -1))


class D4Set(Dataset):
    def __init__(self, base, index, name: str) -> None:
        self.base, self.index, self.name = base, index, name

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[int(self.index[i])])
        item["image"] = d4(item["image"], self.name)
        return item


def path(name: str) -> Path:
    return R.OUT / f"aa3b_d4_{name}.npy"


def bank(setup) -> None:
    """One read per stamp, eight forward passes on it. The disk is the bottleneck: memmap page
    faults in dataset order ran at ~0.2 stamps/s. Stamps are read with one explicit pread each,
    in cache-row order so the disk streams; the bytes are exactly what StampDataset returns
    (checked on the first batch). Each transform's embedding lands at the galaxy's union row in a
    float16 memmap. Resumable by batch."""
    import os

    from galaxy_jepa.core.encoder import assert_frozen

    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    ids = np.asarray(ctx.real.object_ids)
    np.save(R.OUT / "aa3b_d4_ids.npy", ids)
    cache = setup.ds.cache
    crow = np.array([cache.row_of(int(o)) for o in ids])
    order = np.argsort(crow)
    shape, nb = tuple(cache.index.shape), int(np.prod(cache.index.shape)) * np.dtype(cache.index.dtype).itemsize
    fd = os.open(cache.cache_dir / "stamps.f16", os.O_RDONLY)

    def read(r: int) -> np.ndarray:
        return np.frombuffer(os.pread(fd, nb, int(crow[r]) * nb), dtype=cache.index.dtype).reshape(shape)

    idx = X.ds_index(setup, ids)
    for r in order[:4]:
        assert np.array_equal(read(r), setup.ds[int(idx[r])]["image"].numpy()), "pread != dataset"
    n, dim = ids.size, ctx.real.x.shape[1]
    mm = {k: np.lib.format.open_memmap(path(k), mode="r+" if path(k).exists() else "w+",
                                       dtype=np.float16, shape=(n, dim)) for k in D4}
    prog = R.OUT / "aa3b_d4_progress.json"
    done = json.loads(prog.read_text())["done"] if prog.exists() else 0
    assert_frozen(frozen)
    frozen.to(setup.device).eval()
    bs = 128
    with torch.no_grad():
        for s0 in range(done, n, bs):
            rows = order[s0:s0 + bs]
            imgs = torch.from_numpy(np.stack([read(r) for r in rows])).float()
            for k in D4:
                e = frozen.encode(d4(imgs, k).to(setup.device)).cpu().numpy()
                mm[k][rows] = e.astype(np.float16)
            if (s0 // bs) % 20 == 0 or s0 + bs >= n:
                for k in D4:
                    mm[k].flush()
                prog.write_text(json.dumps({"done": min(s0 + bs, n)}))
                print(f"AA3b bank {min(s0 + bs, n)}/{n} {__import__('time').strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
    os.close(fd)
    prog.write_text(json.dumps({"done": n, "complete": True}))


def averaged(ctx) -> EmbeddingMatrix:
    ids = np.load(R.OUT / "aa3b_d4_ids.npy")
    assert np.array_equal(ids, np.asarray(ctx.real.object_ids))
    assert json.loads((R.OUT / "aa3b_d4_progress.json").read_text()).get("complete"), "bank incomplete"
    acc = np.zeros(ctx.real.x.shape, np.float64)
    for name in D4:
        acc += np.load(path(name)).astype(np.float64)
    return EmbeddingMatrix(ctx.real.object_ids, (acc / len(D4)).astype(np.float32), "M-d4avg")


def collapse(ctx, setup, avg: EmbeddingMatrix) -> dict:
    """Variance of the averaged embedding on M's own PC1/PC2 (A's basis), as a share of M's."""
    b = X.basis(ctx, setup)[0]
    pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
    A = np.array([pos[int(o)] for o in setup.train_ids])
    z0 = (ctx.real.x[A].astype(np.float64) - b["mu"]) @ b["v"][:, :10]
    z1 = (avg.x[A].astype(np.float64) - b["mu"]) @ b["v"][:, :10]
    share = (z1.var(0) / z0.var(0)).tolist()
    r0 = np.load(path("r0"))[A].astype(np.float64)
    return {"var_share_pc1_10": share, "orig_matches_bank_r0": float(np.corrcoef(
        (r0 - b["mu"]) @ b["v"][:, 0], z0[:, 0])[0, 1]),
            "collapsed": bool(share[0] <= COLLAPSE and share[1] <= COLLAPSE)}


# ------------------------------------------------------------------ readouts


def existence(mats: dict[str, EmbeddingMatrix], labels, feature, tr, te, c: float, seed: int) -> dict:
    """The ladder's probe on each matrix; paired bootstrap of AUC(avg) − AUC(orig) on B."""
    scores, y = {}, None
    for k, m in mats.items():
        train = feature_embeddings(m, labels, feature, tr)
        test = feature_embeddings(m, labels, feature, te)
        if len(np.unique(train.y)) < 2 or len(np.unique(test.y)) < 2:
            return {"skipped": "single class"}
        scaler, clf = _fit(train, c=c)
        scores[k] = clf.predict_proba(scaler.transform(test.x))[:, 1]
        y = test.y
    pt, lo, hi = paired_auc_bootstrap([y, y], [scores["avg"], scores["orig"]], [1, -1],
                                      n_boot=2000, seed=seed)
    from sklearn.metrics import roc_auc_score

    rec = {"auc_orig": float(roc_auc_score(y, scores["orig"])),
           "auc_avg": float(roc_auc_score(y, scores["avg"])), "delta": pt, "ci": [lo, hi],
           "n_test": int(y.size)}
    rec["change"] = change(rec)
    return rec


def change(r: dict) -> str:
    if r["delta"] >= MATERIAL and r["ci"][0] > 0:
        return "IMPROVED"
    if r["delta"] <= -MATERIAL and r["ci"][1] < 0:
        return "WORSENED"
    return "UNCHANGED"


def held_out(m: EmbeddingMatrix, ctx, setup, recorded: dict, rng) -> dict:
    """V3's held-out recovery on ``m``: align one of 50 PCs per feature on A by |Spearman| with
    the vote fraction, freeze, AUC on B against the probe's; verdict by V3's rule against the
    RECORDED untrained recoveries."""
    lab = setup.labels
    pos = {int(o): i for i, o in enumerate(m.object_ids)}
    A = np.array([pos[int(o)] for o in setup.train_ids])
    x = m.x[A].astype(np.float64)
    mu, _, v = V3.pca(x)
    sa = (x - mu) @ v[:, : V3.K_POOL]
    rows_pos = {p: i for i, p in enumerate(A)}
    out, p = {}, {}
    for f, rec0 in recorded.items():
        if "p" not in rec0:
            continue
        ids = feature_ids(m, lab, f, setup.train_ids)
        idx = np.array([rows_pos[pos[int(o)]] for o in ids])
        fr = lab.vote_fraction(f, ids)
        rho = np.array([V3.sp(sa[idx, j], fr) for j in range(V3.K_POOL)])
        j = int(np.nanargmax(np.abs(rho)))
        te = feature_embeddings(m, lab, f, setup.test_ids)
        y = te.y.astype(int)
        score = (te.x - mu) @ v[:, j] * np.sign(rho[j])
        a, pv = V3.auc_perm(score, y, rng)
        tr = feature_embeddings(m, lab, f, setup.train_ids)
        d = probe_direction(tr, name=f, c=setup.pc.c)
        ap = V3.auc(te.x @ d.w_raw, y)
        out[f] = {"pc": j + 1, "auc_pc": a, "auc_probe": ap, "recovery": (a - 0.5) / (ap - 0.5),
                  "untrained_recovery_recorded": [u["recovery"] for u in rec0["untrained"]]}
        p[f] = pv
    sig = nulls_mod.family_significant(p, alpha=0.05, method="benjamini_yekutieli",
                                       n_tests=len(recorded))
    for f, r in out.items():
        umax = max(r["untrained_recovery_recorded"])
        if sig[f] and r["auc_pc"] < 0.5:
            v_ = "INVERTED"
        elif not sig[f]:
            v_ = "NO AXIS"
        elif r["recovery"] <= umax:
            v_ = "NOT ABOVE UNTRAINED"
        elif r["recovery"] >= 0.8:
            v_ = "ENCODER AXIS"
        elif r["recovery"] >= 0.5:
            v_ = "PARTIAL AXIS"
        else:
            v_ = "WEAK AXIS"
        r["verdict"] = v_
    return out


def winding(setup, avg: EmbeddingMatrix | None) -> dict:
    """Y3's T4b on the Y1 join, with M's raw rows optionally replaced by the averaged ones."""
    ctx, t = Y3.load(setup)
    E = Y3.Emb(ctx)
    if avg is not None:
        n = E.n_union
        raw = E.raw["M"].copy()
        raw[:n] = avg.x.astype(np.float64)
        E.raw["M"] = raw
        mu, sd = raw[:n].mean(0), raw[:n].std(0) + 1e-8
        E.x["M"] = (raw - mu) / sd
    oid = t.oid.to_numpy()
    P = Y3.num(t, f"H_{Y.PRIMARY}")
    Wv, _, _ = Y3.winding(t)
    hay = np.isfinite(P) & np.isfinite(Wv)
    vis = np.full(len(t), np.nan)
    vis[hay] = Y.visibility(t[hay])
    ok = hay & np.isfinite(vis)
    tem = ok & (t.part == "B").to_numpy()
    rows_te = E.rows(oid[tem])
    axis = Y3.winding_axis(ctx, E)
    proj = {k: E.raw[k][rows_te] @ axis[k] for k in Y3.ENCODERS}
    c = Y3.axis_vs_pitch(proj, P[tem], vis[tem], seed=401)
    return c


def verdict(ex: dict, ho_orig: dict, ho_avg: dict, col: dict) -> dict:
    """First match applies (aa_findings.md §AA3b)."""
    imp = [f for f, r in ex.items() if r.get("change") == "IMPROVED"]
    wor = [f for f, r in ex.items() if r.get("change") == "WORSENED"]
    up = [f for f in ho_avg if RANK.get(ho_avg[f]["verdict"], 0)
          > RANK.get(ho_orig[f]["verdict"].split(" — ")[0], 0)]
    down = [f for f in ho_avg if RANK.get(ho_avg[f]["verdict"], 0)
            < RANK.get(ho_orig[f]["verdict"].split(" — ")[0], 0)]
    better, worse = len(imp) + len(up), len(wor) + len(down)
    if not col["collapsed"]:
        s = "NOT INTERPRETABLE"
    elif better >= COUNT and worse >= COUNT:
        s = "MIXED"
    elif better >= COUNT:
        s = "POSE WAS COSTING"
    elif worse >= COUNT:
        s = "POSE WAS HELPING"
    else:
        s = "HARMLESS"
    return {"state": s, "existence_improved": imp, "existence_worsened": wor,
            "salience_up": up, "salience_down": down}


def context(setup):
    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    return ctx, averaged(ctx)


def planted(setup) -> dict:
    ctx, avg = context(setup)
    out: dict = {"collapse": collapse(ctx, setup, avg)}
    col = {"collapsed": True}
    ex = lambda chs: {f"f{i}": {"change": c} for i, c in enumerate(chs)}  # noqa: E731
    ho = {"a": {"verdict": "WEAK AXIS"}}
    out["state_logic"] = {
        "costing": verdict(ex(["IMPROVED", "IMPROVED", "UNCHANGED"]), ho, ho, col)["state"],
        "helping": verdict(ex(["WORSENED", "WORSENED"]), ho, ho, col)["state"],
        "mixed": verdict(ex(["IMPROVED", "IMPROVED", "WORSENED", "WORSENED"]), ho, ho, col)["state"],
        "harmless": verdict(ex(["IMPROVED", "UNCHANGED"]), ho, ho, col)["state"],
        "salience_up": verdict(ex([]), {"a": {"verdict": "WEAK AXIS"}, "b": {"verdict": "NO AXIS"}},
                               {"a": {"verdict": "PARTIAL AXIS"}, "b": {"verdict": "WEAK AXIS"}},
                               col)["state"],
        "no_collapse": verdict(ex(["IMPROVED"] * 3), ho, ho, {"collapsed": False})["state"]}

    # Two planted labels through the identical existence path: one on M's PC1 (pose — averaging
    # must destroy it) and one invariant under the dihedral group (a function of the averaged
    # embedding's own top PC — averaging must keep it)
    b = X.basis(ctx, setup)[0]
    rng = np.random.default_rng(2)
    pc1 = (ctx.real.x.astype(np.float64) - b["mu"]) @ b["v"][:, 0]
    xa = avg.x.astype(np.float64)
    mu_a, _, va = V3.pca(xa)
    inv = (xa - mu_a) @ va[:, 2]

    class Label:
        features = ("pose", "invariant")

        def __init__(self, s):
            s = (s - s.mean()) / s.std() + 0.5 * rng.normal(size=s.size)
            self.y = (s > np.median(s)).astype(np.int64)
            self.pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}

        def eligible(self, f, ids):
            return list(ids)

        def binary_label(self, f, ids):
            return self.y[[self.pos[int(o)] for o in ids]]

        def vote_fraction(self, f, ids):
            return self.binary_label(f, ids).astype(float)

    mats = {"orig": ctx.real, "avg": avg}
    for name, s in (("pose", pc1), ("invariant", inv)):
        out[f"label_{name}"] = existence(mats, Label(s), name, setup.train_ids, setup.test_ids,
                                         c=ctx.pc.c, seed=0)
    return out


def aa3b(setup) -> dict:
    ctx, avg = context(setup)
    rec: dict = {"collapse": collapse(ctx, setup, avg)}
    mats = {"orig": ctx.real, "avg": avg}
    labels = ctx.labels["full"]
    ex = {}
    for i, f in enumerate(ctx.features):
        ex[f] = existence(mats, labels, f, setup.train_ids, setup.test_ids, c=ctx.pc.c, seed=i)
        print(f"AA3b {f}: {ex[f].get('change')} {ex[f].get('delta', 0):+.4f}", file=sys.stderr,
              flush=True)
    rec["existence"] = ex
    R._save(OUT, rec)
    recorded = json.loads(V3.OUT.read_text())["held_out"]
    rng = np.random.default_rng(setup.pc.seed)
    rec["salience_avg"] = held_out(avg, ctx, setup, recorded, rng)
    rec["salience_orig_recorded"] = {f: {k: r.get(k) for k in ("pc", "auc_pc", "auc_probe",
                                                               "recovery", "verdict")}
                                     for f, r in recorded.items()}
    R._save(OUT, rec)
    rec["winding_orig"] = winding(setup, None)
    rec["winding_avg"] = winding(setup, avg)
    for k in ("winding_orig", "winding_avg"):
        rec[k]["state"] = Y3.axis_state(rec[k], rec[k]["p_raw"] < 0.05)
    rec["verdict"] = verdict(ex, recorded, rec["salience_avg"], rec["collapse"])
    return rec


def ensemble(setup) -> dict:
    """Exploratory, post hoc (not in the hashed pre-registration): is AA3b's gain pose removal or
    view ensembling? Arms that remove the pose code with 0, 2, 4 and 8 views averaged."""
    ctx, avg8 = context(setup)
    b = X.basis(ctx, setup)[0]
    x = ctx.real.x.astype(np.float64)
    v = b["v"][:, :2]
    proj = x - ((x - b["mu"]) @ v) @ v.T

    def mean_of(names):
        return sum(np.load(path(k)).astype(np.float64) for k in names) / len(names)

    arms = {"project_pc12 (0 views)": proj, "r0+r2 (2 views)": mean_of(["r0", "r2"]),
            "rotations (4 views)": mean_of(["r0", "r1", "r2", "r3"]), "D4 (8 views)": avg8.x}
    labels = ctx.labels["full"]
    out: dict = {}
    for name, xa in arms.items():
        m = EmbeddingMatrix(ctx.real.object_ids, np.asarray(xa, np.float32), name)
        share = [float(((np.asarray(xa, np.float64) - b["mu"]) @ b["v"][:, i]).var()
                       / ((x - b["mu"]) @ b["v"][:, i]).var()) for i in (0, 1)]
        ex = {f: existence({"orig": ctx.real, "avg": m}, labels, f, setup.train_ids, setup.test_ids,
                           c=ctx.pc.c, seed=i) for i, f in enumerate(ctx.features)}
        d = np.array([r["delta"] for r in ex.values() if "delta" in r])
        out[name] = {"pc12_share": share, "median_delta": float(np.median(d)),
                     "range": [float(d.min()), float(d.max())],
                     "improved": sum(r.get("change") == "IMPROVED" for r in ex.values()),
                     "worsened": sum(r.get("change") == "WORSENED" for r in ex.values()),
                     "per_answer": {f: r.get("delta") for f, r in ex.items()}}
        print(f"AA3b ensemble {name}: share {share} median {np.median(d):+.4f} "
              f"improved {out[name]['improved']}", file=sys.stderr, flush=True)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--aa3b"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="AA3b", sources=1)
    if mode == "--bank":
        bank(setup)
        return
    if mode == "--ensemble":
        out, p = ensemble(setup), R.OUT / "aa3b_ensemble.json"
    else:
        out, p = (planted(setup), PLANTED) if mode == "--planted" else (aa3b(setup), OUT)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:8000])


if __name__ == "__main__":
    main()
