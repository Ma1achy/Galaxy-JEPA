"""D12's baselines — MoCo and MAE on M's data, backbone, budget and data order (baselines.md §9 item 8).

M's driver with the objective swapped and the stopping rule removed. m2_long_run's machinery is
imported, not copied — `Run`, the segment schedule, the trace merge, the k2 probe subprocess — and
the loop is `train_objective`, the one that trained M (build 2a). `m2_long_run.main` itself is not
called: it is JEPA-shaped throughout (SIGReg, EMA schedule, the M4 rule) and it is the driver of
record for M, O2 and A1–A3, so it stays untouched.

**Matched to M, checked rather than trusted (§1).** The baseline config (configs/moco.yaml or
mae.yaml) must equal M's configs/pretrain.yaml on every determining field except `objective`; F0
runs against M's config, so the cache, freeze, effect floor and splits are M's by construction, and
the k2 probe (which reads M's config) lands on the identical held-out galaxies. Training seed 0
pairs with M, seed 1 with O2; the split seed is the config's 0 for both (§1, Seeds). The encoder is
`seed_init(train_seed)` built first, so its init is byte-identical to M's; the data order is
`ResumableShuffle(train_seed)`, identical to M's.

**A fixed stop, no rule (§1.1).** Every arm stops at step 101,308 (4.00 epochs) with `steps` left at
253,270, so the LR sits at 65.9% of its own peak as M's did. Probes at 0.5 / 1 / 2 / 4 epochs are
trajectories only: reported, never used to choose a checkpoint (spec 1C).

**F7 checks** (§5.3 rows 2–3) go through the same path: `--lr` and `--stop` make a different recipe
(lr is determining, so the hash moves) in its own directory, with no probe. Judged on pretext loss.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/baseline_run.py moco0 --plan
    uv run python artifacts/baseline_run.py moco0                     # MoCo seed 0 (pairs with M)
    uv run python artifacts/baseline_run.py moco0 --lr 6.25e-6 --stop 5000   # an F7 arm
"""

from __future__ import annotations

import os

# M's v1 config and cache are the reference; F0 reads these at import time. A baseline launched
# under the v2 variables would be matched to A1, not M — refuse rather than silently re-target.
for _k, _want in (("GJ_CONFIG", "configs/pretrain.yaml"),):
    if os.environ.setdefault(_k, _want) != _want:
        raise SystemExit(f"baseline_run: {_k}={os.environ[_k]!r}; the baselines are matched to M "
                         f"({_want}) and F0 must check M's config")
for _k in ("F0_CACHE_BASE", "F0_NORM_PREFIX"):
    if _k in os.environ:
        raise SystemExit(f"baseline_run: {_k} is set; the baselines train on M's v1 cache (§1)")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402
from i2_sigreg_run import _split_ids  # noqa: E402
from m2_long_run import MONITOR_BATCH, OUT, PROBE, Run, _merge, _schedule  # noqa: E402

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer  # noqa: E402
from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import MaeRunConfig, MocoRunConfig, build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.loop import _to_device, learning_rate, train_objective  # noqa: E402

STOP = 101_308  # M's and O2's stop: round(4.0 × 25,327)
PROBE_EPOCHS = (0.5, 1.0, 2.0, 4.0)  # M's epoch points; trajectories only (§1.1)


@dataclasses.dataclass(frozen=True)
class Arm:
    kind: str  # "moco" | "mae"
    run: Run


def _arm(kind: str, seed: int) -> Arm:
    tag = f"{kind}_s{seed}"
    return Arm(kind, Run(label=f"{kind.upper()}{seed}", tag=tag, probe_epochs=PROBE_EPOCHS,
                         train_seed=seed, out_dir=f"runs/{tag}"))


# §5.3's order: MoCo s0, MAE s0, MoCo s1, MAE s1
ARMS = {f"{k}{s}": _arm(k, s) for s in (0, 1) for k in ("moco", "mae")}
CONFIGS = {"moco": ("configs/moco.yaml", MocoRunConfig), "mae": ("configs/mae.yaml", MaeRunConfig)}


def _matched(base, m) -> None:
    """Every determining field but the objective is M's (§1), on the resolved device."""
    bd, md = base.determining_dump(), m.determining_dump()
    bd.pop("objective"), md.pop("objective")
    if bd != md:
        diff = sorted(k for k in set(bd) | set(md) if bd.get(k) != md.get(k))
        raise SystemExit(f"baseline_run: the config differs from M's outside `objective`: {diff}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--plan", action="store_true", help="print the budget and schedule, run nothing")
    ap.add_argument("--no-probe", action="store_true", help="train only; skip the probes")
    ap.add_argument("--lr", type=float, help="F7 check: a different peak LR (a different recipe)")
    ap.add_argument("--stop", type=int, help="F7 check / rehearsal: one segment to this step, no "
                                             "probe, in its own directory")
    ap.add_argument("--out", help="override the out_dir (required with --lr or --stop)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S", stream=sys.stderr, force=True)
    arm = ARMS[args.arm]
    run, L = arm.run, arm.run.label
    log = logging.getLogger(run.tag)

    m_cfg, cache = check(verbose=False)  # M's config, resolved device; F0 raises on any failure
    path, cls = CONFIGS[arm.kind]
    cfg = cls(**yaml.safe_load((REPO / path).read_text())).with_resolved_device()
    _matched(cfg, m_cfg)
    if not cfg.smoke:
        raise SystemExit(f"{L}: smoke is False — a baseline is not a headline (§1, F1)")
    obj = cfg.objective
    variant = args.lr is not None or args.stop is not None
    if args.lr is not None:
        obj = obj.model_copy(update={"lr": args.lr, "lr_final": args.lr * 1e-3})  # M's shape
        cfg = cfg.model_copy(update={"objective": obj})
    if variant and not args.out:
        raise SystemExit(f"{L}: --lr / --stop make a different run; name its --out directory")
    out_dir = args.out or run.out_dir
    if variant and Path(out_dir).name == Path(run.out_dir).name:
        raise SystemExit(f"{L}: a variant may not write to the arm's own directory {run.out_dir}")
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"out_dir": out_dir})})
    out = Path(cfg.paths.out_dir)

    train_seed, split_seed = run.train_seed, cfg.seed
    train_ids, monitor_ids, _ = _split_ids(cfg)
    per_epoch = len(train_ids) // obj.batch_size
    total = obj.steps
    if args.stop is not None:
        points, args.no_probe = [args.stop], True
    else:
        points = _schedule(per_epoch, total, run.probe_epochs)
        if points[-1] != STOP:
            raise SystemExit(f"{L}: the last probe point is {points[-1]}, not M's stop {STOP}")
    every = int(os.environ.get("GJ_CHECKPOINT_EVERY") or obj.checkpoint_every)
    keep = total // every + len(points) + 2
    ocfg = getattr(obj, f"to_{arm.kind}_config")(seed=train_seed)
    ch = config_hash(cfg.determining_dump())

    print(f"\n{L} objective   : {arm.kind} from {path}; config_hash v2:{ch[:16]}"
          f"{'  (VARIANT: ' + ', '.join(f'{k}={v}' for k, v in (('lr', args.lr), ('stop', args.stop)) if v is not None) + ')' if variant else ''}")
    print(f"{L} matched     : every determining field but `objective` equals configs/pretrain.yaml "
          f"(device {cfg.runtime.device})")
    print(f"{L} budget      : schedule {total:,} steps; stop at {points[-1]:,} "
          f"({points[-1] / per_epoch:.2f} ep, lr {learning_rate(points[-1], ocfg):.3e} = "
          f"{learning_rate(points[-1], ocfg) / ocfg.lr:.1%} of peak)")
    print(f"{L} seeds       : train={train_seed} (weights, data order, views/masks)  split={split_seed}")
    print(f"{L} out_dir     : {out}")
    print(f"{L} checkpoints : every {every:,}, keep={keep}")
    print(f"{L} probes      : {'none' if args.no_probe else ', '.join(f'{p:,}' for p in points)} "
          "(trajectories only, never a checkpoint choice)")
    if arm.kind == "moco":
        print(f"{L} views       : gain ×U(1±{obj.gain_global}) global, ×U(1±{obj.gain_band}) per band; "
              f"noise u×σ_sky {obj.sky_sigma}, u~U(0,{obj.noise_u_max}); blur p {obj.blur_p}")
    if args.plan:
        print(f"\n{L} --plan: nothing launched")
        return

    device = cfg.runtime.resolved_device()
    out.mkdir(parents=True, exist_ok=True)
    ds = StampDataset(cache, {}, train_ids, scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=train_seed)  # M's order for the same seed
    loader = DataLoader(ds, batch_size=obj.batch_size, sampler=sampler, drop_last=True)
    objective = build_objective(ocfg, seed_init(train_seed, cache.index.height,
                                                cfg.model.model_kwargs()))
    schedule = {"steps": ocfg.steps, "lr": ocfg.lr, "lr_final": ocfg.lr_final,
                "warmup_steps": ocfg.warmup_steps, "batch_size": ocfg.batch_size,
                "objective": arm.kind, "brief": "D12 baseline"}
    if arm.kind == "moco":
        schedule["momentum"] = ocfg.momentum
    checkpointer = TrainCheckpointer(out / "checkpoints", every=every, keep=keep, config_hash=ch,
                                     normalisation_hash=cfg.normalisation.content_hash,
                                     schedule=schedule)
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=cache.scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)

    written = sorted(int(f.stem.split("_")[1]) for f in (out / "checkpoints").glob("ckpt_*.pt"))
    done = written[-1] if written else 0
    if done:
        log.warning(f"{L} resuming from step %d (%.2f ep)", done, done / per_epoch)
    losses: list[float] = []
    collapse: dict[str, list[float]] = {}
    curve: list[dict] = []
    if (out / "traces.json").exists():
        prior = json.loads((out / "traces.json").read_text())
        losses, collapse = prior["losses"], prior["collapse_trace"]
    curve_path = OUT / f"{run.tag}_curve.json" if not variant else out / "curve.json"
    if curve_path.exists():
        curve = [r for r in json.loads(curve_path.read_text())["curve"] if r["step"] <= done]
    t_start = time.perf_counter()

    for i, target in enumerate(points, 1):
        if target <= done:
            log.info(f"{L} segment %d/%d: done at step %d", i, len(points), target)
            continue
        delta = target - done
        gc.collect()
        if device.startswith("mps"):
            torch.mps.empty_cache()
        log.info(f"{L} segment %d/%d: +%d steps -> %d (%.2f ep)", i, len(points), delta, target,
                 target / per_epoch)
        t0 = time.perf_counter()
        result = train_objective(objective, loader, device=device, monitor_batch=mon_batch,
                                 checkpoint_path=out / "encoder.pt",
                                 autocast_dtype=cfg.autocast_dtype(), checkpointer=checkpointer,
                                 sampler=sampler, collapse_floor=cfg.collapse_floor,
                                 resume=done > 0, stop_after=delta)
        secs = time.perf_counter() - t0
        done = result.steps_completed
        losses = _merge(losses, result.losses)
        for k, v in result.collapse_trace.items():
            collapse.setdefault(k, [])
            collapse[k] = collapse[k] + [x for j, x in enumerate(v) if j >= len(collapse[k])]
        (out / "traces.json").write_text(json.dumps({
            "collapse_trace": collapse, "losses": losses, "prediction_losses": losses,
            "sigreg_losses": [float("nan")] * len(losses), "objective": arm.kind,
        }))
        extra = f"  top1 {objective.last_top1:.3f}" if arm.kind == "moco" else ""
        log.info(f"{L} segment %d done: step %d, %.2f h, %.4f steps/s%s%s", i, done, secs / 3600,
                 delta / secs, extra, "  HALTED" if result.halted else "")
        if result.halted:
            log.error(f"{L} halted at step %d — the collapse monitor fired. Stopping.", done)
            break
        if done != target:
            log.warning(f"{L} landed at %d, not %d", done, target)
        if args.no_probe:
            continue

        tag = f"{run.tag}{i}"
        proc = subprocess.run(
            [sys.executable, str(PROBE), "--run", str(out), "--steps", str(done), "--tag", tag],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log.error(f"{L} probe at step %d failed:\n%s", done, proc.stderr[-2000:])
            break
        rec = json.loads((OUT / f"{tag}_trajectory.json").read_text())["checkpoints"][-1]
        rec["epoch"] = done / per_epoch
        curve.append(rec)
        log.info(f"{L} PROBE step %d (%.2f ep): consensus %.4f [%.4f, %.4f]  all %.4f  amb %.4f",
                 done, rec["epoch"], rec["auc"], rec["auc_lo"], rec["auc_hi"], rec["auc_all"],
                 rec["auc_ambiguous"])
        curve_path.write_text(json.dumps({
            "objective": arm.kind, "budget_steps": total, "stop": STOP, "probe_points": points,
            "stopping_rule": None, "smoke": True, "config_hash": f"v2:{ch}",
            "train_seed": train_seed, "split_seed": split_seed, "curve": curve,
        }, indent=2))

    log.info(f"{L} finished: %d steps, %.2f h", done, (time.perf_counter() - t_start) / 3600)


if __name__ == "__main__":
    main()
