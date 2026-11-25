"""Patch grid: embed the frozen union with a trained arm, or with an untrained encoder (docs/patch_grid/prereg.md §4). Runs on the rental box, or anywhere with the cache.

    python artifacts/patch_grid_embed.py trained   --arm jepa_p8_s0 --ckpt runs/patch_grid/jepa_p8_s0/encoder.pt --cache-base DIR --out DIR
    python artifacts/patch_grid_embed.py untrained --patch 8 --seed 0 --cache-base DIR --out DIR            # seeds 0, 1, 2 per patch size

The ladder's recipe (j4_spread_controls.prepare → probing.extract.extract_matrix, probing.controls.untrained_encoder_matrix): a frozen encoder,
fp32, no autocast, the penultimate block mean-pooled, the 74,825 union ids in ladder order (docs/patch_grid/union_ids.npy, SHA-1 pinned), the fp16 stamps.
Patch 16 is embedded here too, so every arm and every bank comes from one code path and device. No label is read.
Output <name>.npz: ids, real, name, patch, checkpoint_sha1 (trained) or config (untrained), device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
os.environ.setdefault("GJ_CONFIG", "configs/pretrain_v2.yaml")
os.environ.setdefault("F0_NORM_PREFIX", "246d8de6ba9a")
UNION = REPO / "docs/patch_grid/union_ids.npy"
UNION_SHA1 = "d5cba4b7debce7693a8181b69688fc1d3957260b"
N_UNION = 74_825
BATCH = {16: 128, 8: 64, 4: 32}


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def union() -> np.ndarray:
    if sha1(UNION) != UNION_SHA1:
        raise SystemExit("patch_grid_embed: union_ids.npy is not the frozen file")
    ids = np.load(UNION)
    assert ids.shape == (N_UNION,) and ids.dtype == np.int64 and (np.diff(ids) > 0).all()
    return ids


def dataset(limit: int | None):
    from f0_preconditions import check
    from galaxy_jepa.data.dataset import StampDataset
    cfg, cache = check(verbose=False)
    ids = union()[:limit] if limit else union()
    ds = StampDataset(cache, {}, ids.tolist(), scalars=cache.scalars)
    if not np.array_equal(np.asarray(ds.object_ids, dtype=np.int64), ids):
        raise SystemExit("patch_grid_embed: the cache does not hold every union stamp in order")
    return cfg, ds, ids


def write(out: Path, name: str, ids: np.ndarray, x: np.ndarray, **meta) -> None:
    if not np.isfinite(x).all() or x.shape != (len(ids), 384):
        raise SystemExit(f"{name}: embedding is {x.shape} with {int((~np.isfinite(x)).sum())} non-finite values")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", ids=ids, real=x.astype(np.float32), name=name, **meta)
    print(f"{name}: {x.shape[0]:,} x {x.shape[1]} written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trained", "untrained"])
    ap.add_argument("--arm")
    ap.add_argument("--ckpt")
    ap.add_argument("--patch", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cache-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--limit", type=int, help="first N union stamps (smokes only; the file says so)")
    a = ap.parse_args()
    if a.cache_base:
        os.environ["F0_CACHE_BASE"] = a.cache_base
    if not a.limit:
        import patch_grid_pins
        patch_grid_pins.verify()
    import torch
    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing.controls import untrained_encoder_matrix
    from galaxy_jepa.probing.extract import extract_matrix
    from patch_grid_run import PATCHES, parse
    cfg, ds, ids = dataset(a.limit)
    if not a.limit:
        patch_grid_pins.verify_cache(ds.cache.cache_dir, ds.cache.cache_dir.name)
    dev = a.device or cfg.runtime.resolved_device()
    model = {"img_size": int(ds.cache.index.height), **cfg.model.model_kwargs()}
    t0 = time.perf_counter()
    if a.mode == "trained":
        o, patch, _ = parse(a.arm)
        enc = load_frozen_encoder(a.ckpt)
        want = {**model, "patch_size": patch}
        got = {k: enc.config[k] for k in want}
        if got != want:
            raise SystemExit(f"{a.arm}: the checkpoint's encoder is {got}, the arm is {want}")
        x = extract_matrix(enc, ds, device=dev, batch_size=a.batch or BATCH[patch])
        name, meta = a.arm, {"checkpoint_sha1": sha1(Path(a.ckpt))}
    else:
        if a.patch not in PATCHES or a.seed not in (0, 1, 2):
            raise SystemExit("patch_grid_embed: untrained needs --patch 16|8|4 and --seed 0|1|2")
        patch = a.patch
        x = untrained_encoder_matrix({**model, "patch_size": patch}, ds, device=dev, batch_size=a.batch or BATCH[patch], seed=a.seed)
        name, meta = f"untrained_p{patch}_s{a.seed}", {"config": json.dumps({**model, "patch_size": patch}, sort_keys=True)}
    if not np.array_equal(x.object_ids, ids):
        raise SystemExit("patch_grid_embed: rows came back in another order")
    write(Path(a.out), name + ("_smoke" if a.limit else ""), ids, x.x, patch=patch, device=dev, torch_version=torch.__version__, **meta)
    print(f"{time.perf_counter() - t0:.0f}s on {dev}")


if __name__ == "__main__":
    main()
