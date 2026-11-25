"""Patch-grid arms (docs/patch_grid/prereg.md): JEPA, MoCo, MAE at patch 16 / 8 / 4, seeds 0–2, bf16, one GPU.

    python artifacts/patch_grid_run.py jepa_p8_s0 --cache-base DIR [--masks DIR] [--plan]
    python artifacts/patch_grid_run.py mae_p4_s1 --cache-base DIR --smoke 3 --batch 2     # plumbing only; own scratch dir

The machinery is the baselines' (baseline_run.py): `train_objective`, `ResumableShuffle(train_seed)`, `seed_init(train_seed)`, the
same F0 checks, the split, the 101,308-step stop inside a 253,270-step schedule. Two things are this grid's:

* a config is A1's (or MoCo's / MAE's Mac config) with the overlay of configs/patch_grid/<objective>_p<patch>.yaml, and `matched` refuses
  any determining difference outside the allowed set (patch size, autocast, MoCo's window, MAE's grouped block, the device);
* JEPA at patch 4 reads precomputed masks (artifacts/patch_grid_masks.py): the same masks, not drawn in the training process.

Nothing here reads a probe, an AUC or an embedding: probes run afterwards, under the hash. Investigation code: terse.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
os.environ.setdefault("GJ_CONFIG", "configs/pretrain_v2.yaml")
os.environ.setdefault("F0_NORM_PREFIX", "246d8de6ba9a")

OBJ = ("jepa", "moco", "mae")
PATCHES = (16, 8, 4)
STOP = 101_308
#: determining fields allowed to differ from the base config, as dotted paths
ALLOWED = {"jepa": {"model.patch_size", "autocast"}, "moco": {"model.patch_size", "autocast", "objective.window_min", "objective.window_max"},
           "mae": {"model.patch_size", "autocast", "objective.name", "objective.mask_group_px"}}


def parse(arm: str) -> tuple[str, int, int]:
    o, p, s = arm.split("_")
    o, p, s = o, int(p[1:]), int(s[1:])
    if o not in OBJ or p not in PATCHES or s not in (0, 1, 2):
        raise SystemExit(f"patch_grid_run: arm must be <jepa|moco|mae>_p<16|8|4>_s<0|1|2>, not {arm!r}")
    return o, p, s


def merged(arm: str) -> tuple[dict, dict]:
    """(base config dict, overlay applied) for an arm."""
    o, p, _ = parse(arm)
    ov = yaml.safe_load((REPO / f"configs/patch_grid/{o}_p{p}.yaml").read_text())
    base = yaml.safe_load((REPO / ov.pop("base")).read_text())
    out = copy.deepcopy(base)
    for k, v in ov.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
    return base, out


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


def config(arm: str, *, steps_smoke: int | None = None, batch: int | None = None):
    """(RunConfig, A1-style base RunConfig) with every determining difference checked against ALLOWED."""
    from galaxy_jepa.harness import GroupedMaeRunConfig, HarnessConfig, MaeRunConfig, MocoRunConfig
    o, p, _ = parse(arm)
    base, out = merged(arm)
    cls = {"jepa": HarnessConfig, "moco": MocoRunConfig, "mae": GroupedMaeRunConfig}[o]
    ref = {"jepa": HarnessConfig, "moco": MocoRunConfig, "mae": MaeRunConfig}[o](**base).with_resolved_device()
    cfg = cls(**out).with_resolved_device()
    a, b = flat(ref.determining_dump()), flat(cfg.determining_dump())
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if diff - ALLOWED[o]:
        raise SystemExit(f"patch_grid_run: {arm} differs from its base outside the allowed set: {sorted(diff - ALLOWED[o])}")
    if batch is not None:
        cfg = cfg.model_copy(update={"objective": cfg.objective.model_copy(update={"batch_size": batch})})
    if steps_smoke is not None and not cfg.smoke:
        raise SystemExit("a smoke needs smoke: true in the config")
    return cfg, ref, sorted(diff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--cache-base", help="directory holding the v2 fp16 cache (its pipeline-hash folder); default $F0_CACHE_BASE")
    ap.add_argument("--masks", help="precomputed JEPA masks for this seed and grid (required for jepa_p4)")
    ap.add_argument("--out", help="out_dir (default runs/patch_grid/<arm>)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", type=int, help="train this many steps in a scratch dir; plumbing only")
    ap.add_argument("--batch", type=int, help="batch size, smokes only")
    ap.add_argument("--device", help="override the device (smokes only)")
    args = ap.parse_args()
    if args.cache_base:
        os.environ["F0_CACHE_BASE"] = args.cache_base
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s", stream=sys.stderr, force=True)
    logging.getLogger("galaxy_jepa.data.bbox").setLevel(logging.ERROR)
    o, patch, seed = parse(args.arm)
    if args.batch and not args.smoke:
        raise SystemExit("--batch is for smokes: a run's batch is the recipe's")
    if o == "jepa" and patch == 4 and not args.masks:
        raise SystemExit("jepa_p4 needs --masks (precomputed masks; artifacts/patch_grid_masks.py)")

    if not args.smoke and not args.plan:  # a real run only with the pinned code of record
        import patch_grid_pins
        patch_grid_pins.verify()
    import torch
    from torch.utils.data import DataLoader

    from f0_preconditions import check
    from i2_sigreg_run import _split_ids
    from m2_long_run import MONITOR_BATCH, _merge
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
    from galaxy_jepa.core.config import config_hash
    from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset
    from galaxy_jepa.harness import build_objective, seed_init
    from galaxy_jepa.objectives.loop import _to_device, learning_rate, train_objective

    cfg, ref, diff = config(args.arm, steps_smoke=args.smoke, batch=args.batch)
    a1_cfg, cache = check(verbose=False)  # A1's config: the freeze, the cache and its pipeline hash, or a refusal
    if not args.smoke and not args.plan:
        patch_grid_pins.verify_cache(cache.cache_dir, cache.cache_dir.name)
    if args.device:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": args.device})})
    obj = cfg.objective
    out = Path(args.out or (REPO / "runs/patch_grid" / (f"smoke_{args.arm}" if args.smoke else args.arm)))
    stop = args.smoke or STOP
    train_ids, monitor_ids, _ = _split_ids(cfg)
    per_epoch = len(train_ids) // obj.batch_size
    ocfg = getattr(obj, f"to_{o}_config")(seed=seed)
    ch = config_hash(cfg.determining_dump())
    print(f"{args.arm}: config_hash v2:{ch[:16]}; differs from its base only in {diff}")
    print(f"  stop {stop:,} of a {obj.steps:,}-step schedule; batch {obj.batch_size}; lr at the stop {learning_rate(stop, ocfg):.3e}; autocast {cfg.autocast}; device {cfg.runtime.device}")
    print(f"  train stamps {len(train_ids):,} ({per_epoch:,} steps/epoch); encoder patch {cfg.model.patch_size}; out {out}")
    if args.plan:
        return
    if out.exists() and any(out.iterdir()) and not (out / "checkpoints").exists():
        raise SystemExit(f"{out} is not empty and holds no checkpoint: refusing to write into it")
    tr = out / "traces.json"
    if tr.exists():  # a finished run would train one step past its stop, and a halted one would resume past its halt
        raise SystemExit(f"{tr} exists: this run has ended. A VOID arm's rerun goes to a fresh --out")
    device = cfg.runtime.resolved_device()
    out.mkdir(parents=True, exist_ok=True)
    ds = StampDataset(cache, {}, train_ids, scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=seed)
    loader = DataLoader(ds, batch_size=obj.batch_size, sampler=sampler, drop_last=True, num_workers=args.workers, pin_memory=device.startswith("cuda"))
    objective = build_objective(ocfg, seed_init(seed, cache.index.height, cfg.model.model_kwargs()))
    if o == "jepa" and args.masks:
        from galaxy_jepa.masking.precomputed import PrecomputedMasker
        objective.masker = PrecomputedMasker(args.masks, grid_size=objective.encoder.grid_size, config=ocfg.mask, train_seed=seed)
    every = args.smoke or int(os.environ.get("GJ_CHECKPOINT_EVERY") or obj.checkpoint_every)
    ck = TrainCheckpointer(out / "checkpoints", every=every, keep=STOP // every + 3, config_hash=ch, normalisation_hash=cfg.normalisation.content_hash,
                           schedule={"steps": ocfg.steps, "lr": ocfg.lr, "lr_final": ocfg.lr_final, "warmup_steps": ocfg.warmup_steps,
                                     "batch_size": ocfg.batch_size, "objective": o, "brief": "patch grid"})
    n_mon = MONITOR_BATCH if not args.batch else 2  # only a small-batch plumbing smoke shrinks it: two stamps have effective rank 1, so a long smoke would halt
    mon = _to_device(next(iter(DataLoader(StampDataset(cache, {}, monitor_ids[:n_mon], scalars=cache.scalars), batch_size=n_mon))), device)
    written = sorted(int(f.stem.split("_")[1]) for f in (out / "checkpoints").glob("ckpt_*.pt"))
    done = written[-1] if written else 0
    t0 = time.perf_counter()
    r = train_objective(objective, loader, device=device, monitor_batch=mon, checkpoint_path=out / "encoder.pt", autocast_dtype=cfg.autocast_dtype(),
                        checkpointer=ck, sampler=sampler, collapse_floor=cfg.collapse_floor, resume=done > 0, stop_after=stop - done)
    secs = time.perf_counter() - t0
    (out / "traces.json").write_text(json.dumps({"collapse_trace": r.collapse_trace, "losses": _merge([], r.losses), "objective": o, "arm": args.arm,
                                                 "config_hash": f"v2:{ch}", "halted": r.halted, "steps": r.steps_completed, "seconds": secs}))
    print(f"{args.arm}: {r.steps_completed} steps in {secs:.1f} s ({r.steps_completed / secs:.2f}/s){' HALTED' if r.halted else ''}; last loss {r.losses[-1] if r.losses else float('nan'):.4f}; "
          f"grid {objective.encoder.grid_size}x{objective.encoder.grid_size}; {'peak cuda %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9) if device.startswith('cuda') else 'mps/cpu'}")


if __name__ == "__main__":
    main()
