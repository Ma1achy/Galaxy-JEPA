"""Precompute JEPA's masks for one train seed and grid (patch-grid pre-registration; user, 24 Nov 2026).

    python artifacts/patch_grid_masks.py build --seed 0 --grid 64 --workers 16 --cache-base DIR --out DIR   # full 101,308 steps at batch 32
    python artifacts/patch_grid_masks.py check --seed 0 --grid 64 --cache-base DIR --out DIR                # 1,024 galaxies against MultiBlockMasker.sample

Draws use `MultiBlockMasker`'s own generator calls (masking/precomputed.draw) for step s with seed `train_seed + s`, on the batch the
training loop will see (`ResumableShuffle(train_seed)`), so the result is a pure function of (seed, grid, cache). Steps are independent:
shards run in processes and the output does not depend on the worker count. CPU only; no GPU, no pixel is read (the petroRad scalars are).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
os.environ.setdefault("GJ_CONFIG", "configs/pretrain_v2.yaml")
os.environ.setdefault("F0_NORM_PREFIX", "246d8de6ba9a")
STOP = 101_308
_G: dict = {}


def context(seed: int, grid: int, batch: int) -> dict:
    """Everything the draws depend on: the split's petro scalars in order, the mask config, the weight-map recipe (Jepa's own method)."""
    from f0_preconditions import check
    from i2_sigreg_run import _split_ids
    from galaxy_jepa.data.dataset import StampDataset
    from galaxy_jepa.objectives.jepa import JepaConfig
    cfg, cache = check(verbose=False)
    train_ids, _, _ = _split_ids(cfg)
    ds = StampDataset(cache, {}, train_ids, scalars=cache.scalars)
    jc: JepaConfig = cfg.to_jepa_config()
    return {"petro": np.asarray(ds._petro, float), "pscale": ds.pixel_scale, "n": len(ds), "jcfg": jc, "stamp_px": int(cache.index.height), "seed": seed, "grid": grid, "batch": batch}


def maps_for(c: dict, step: int) -> np.ndarray:
    from galaxy_jepa.data.dataset import ResumableShuffle
    from galaxy_jepa.objectives.jepa import Jepa
    sh = _G.setdefault("sampler", ResumableShuffle(c["n"], seed=c["seed"]))
    idx = []
    for i in range(step * c["batch"], (step + 1) * c["batch"]):
        e, off = divmod(i, c["n"])
        idx.append(int(sh.permutation(e)[off]))
    ns = SimpleNamespace(config=c["jcfg"], stamp_px=c["stamp_px"], encoder=SimpleNamespace(grid_size=c["grid"]), _global_half=c["jcfg"].global_box_frac * c["stamp_px"])
    return Jepa.weight_maps(ns, c["petro"][idx], np.full(len(idx), c["pscale"]))


def _init(c: dict) -> None:
    logging.getLogger("galaxy_jepa.data.bbox").setLevel(logging.ERROR)
    _G["c"] = c


def _shard(steps: range):
    from galaxy_jepa.masking.blocks import MultiBlockMasker
    from galaxy_jepa.masking.precomputed import draw, maps_digest
    c = _G["c"]
    m = MultiBlockMasker(c["grid"], c["jcfg"].mask)
    dims, tl, dg = [], [], []
    for s in steps:
        w = maps_for(c, s)
        d, t = draw(m, w, seed=c["seed"] + s)
        dims.append(d)
        tl.append(t)
        dg.append(maps_digest(w))
    return steps.start, np.stack(dims), np.stack(tl), np.asarray(dg, dtype=np.uint64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STOP)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32, help="check: batches compared (32 batches of 32 = 1,024 galaxies)")
    a = ap.parse_args()
    if a.cache_base:
        os.environ["F0_CACHE_BASE"] = a.cache_base
    logging.getLogger("galaxy_jepa.data.bbox").setLevel(logging.ERROR)
    from galaxy_jepa.masking.precomputed import SCHEME, PrecomputedMasker, _cfg_json
    out = Path(a.out)
    c = context(a.seed, a.grid, a.batch)
    _init(c)
    if a.mode == "build":
        out.mkdir(parents=True, exist_ok=True)
        shards = [range(s, min(s + 64, a.steps)) for s in range(0, a.steps, 64)]
        dims, tl, dg = np.empty((a.steps, 5, 2), np.uint8), np.empty((a.steps, a.batch, 5, 2), np.uint8), np.empty(a.steps, np.uint64)
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(c,)) as pool:
            for k, (s0, d, t, g) in enumerate(pool.imap_unordered(_shard, shards)):
                dims[s0 : s0 + len(g)], tl[s0 : s0 + len(g)], dg[s0 : s0 + len(g)] = d, t, g
                if k % 50 == 0:
                    print(f"{k + 1}/{len(shards)} shards, {time.perf_counter() - t0:.0f} s", flush=True)
        for n, x in (("dims", dims), ("tl", tl), ("digest", dg)):
            np.save(out / f"{n}.npy", x)
        (out / "meta.json").write_text(json.dumps({"scheme": SCHEME, "grid_size": a.grid, "train_seed": a.seed, "batch": a.batch, "steps": a.steps,
                                                   "n_train": c["n"], "mask_config": _cfg_json(c["jcfg"].mask), "petro_k": c["jcfg"].petro_k,
                                                   "global_box_frac": c["jcfg"].global_box_frac, "seconds": time.perf_counter() - t0}))
        print(f"built {a.steps:,} steps in {time.perf_counter() - t0:.0f} s ({sum(f.stat().st_size for f in out.glob('*.npy')) / 1e6:.0f} MB)")
        return
    from galaxy_jepa.masking.blocks import MultiBlockMasker
    import torch
    pm = PrecomputedMasker(out, grid_size=a.grid, config=c["jcfg"].mask, train_seed=a.seed)
    ref = MultiBlockMasker(a.grid, c["jcfg"].mask)
    total = len(pm.digest)
    steps = sorted(np.random.default_rng(20261124).choice(total, size=min(a.n, total), replace=False).tolist())
    bad, n = 0, 0
    for s in steps:
        w = maps_for(c, s)
        x, y = pm.sample(w, seed=a.seed + s), ref.sample(w, seed=a.seed + s)
        n += len(w)
        bad += not (torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))
    print(f"check: {n} galaxies in {len(steps)} steps of {total:,}; {bad} batches differ from MultiBlockMasker.sample")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
