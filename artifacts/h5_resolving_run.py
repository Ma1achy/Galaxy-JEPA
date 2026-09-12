"""H5 — the resolving run. Two arms at 3,000 steps, then probe both frozen encoders.

H2 settled the *causal* question at 500 steps: the learning-rate schedule drives the effective
rank down, monotonically over a 64x range, replicated by two mechanisms, on a comparison
controlled to bit-identity. It did **not** settle which schedule is better, because effective
rank is a collapse diagnostic and the objective is AUC. This run answers that, with the decision
rule pre-registered in `artifacts/h5_decision_rule.md` before any arm was launched.

**Every H2 control is unchanged**: one seed through `harness.seed_init` so both arms start from
identical weights, one `ResumableShuffle` seed so they see the same data in the same order, the
same per-step mask seeds, the same monitor batch, and `steps=50000` held constant in both so the
EMA momentum ramp cannot differ. The only thing that varies is the (lr, weight_decay) sequence.

**What this run can and cannot test, stated before the numbers.** The proposal's cosine decay is
the *real* 50,000-step one, not a compressed proxy — which means it is very nearly inert inside
3,000 steps: the LR is still ~99.7% of peak at step 3,000. So this run tests the **peak and the
warmup**, and says nothing about the decay. The decay is carried on the reference recipe's
authority, and that limit is reported rather than glossed.

    uv run python artifacts/h5_resolving_run.py arm   --name baseline
    uv run python artifacts/h5_resolving_run.py probe --name baseline
    uv run python artifacts/h5_resolving_run.py read
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from f1_loader_bench import OUT, _split_ids  # noqa: E402

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer  # noqa: E402
from galaxy_jepa.callbacks.collapse import CollapseMonitor  # noqa: E402
from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.models.vit import save_encoder  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, ema_momentum  # noqa: E402

STEPS = 3000
MONITOR_EVERY = 25  # H2's cadence, so the first 500 steps are directly comparable
MONITOR_BATCH = 64
CHECKPOINT_EVERY = 1000
RUNS = Path(__file__).resolve().parent.parent / "runs" / "h5"
POINTS = OUT / "h5_arms.jsonl"
PROBES = OUT / "h5_probes.jsonl"

REF_LR, REF_BATCH = 1e-3, 2048  # I-JEPA's published peak and batch
REF_DECAY_RATIO = 1e-3 / 1e-6  # the reference decays peak -> peak/1000
REF_WARMUP_FRAC = 15 / 600  # 15 of 600 epochs = 2.50% of the schedule


@dataclasses.dataclass(frozen=True)
class Arm:
    """One (lr, wd) schedule over the REAL 50,000-step horizon — never a compressed proxy."""

    name: str
    peak_lr: float
    warmup: int
    decay: bool
    why: str

    def at(self, step: int, total: int, base_wd: float) -> tuple[float, float]:
        """The (lr, weight_decay) at 0-based ``step`` of a ``total``-step schedule."""
        lr = self.peak_lr * min(1.0, (step + 1) / max(self.warmup, 1))
        if self.decay and step + 1 > self.warmup:
            floor = self.peak_lr / REF_DECAY_RATIO
            t = (step + 1 - self.warmup) / max(total - self.warmup, 1)
            lr = floor + (self.peak_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        return lr, base_wd  # weight decay is deliberately constant in BOTH arms


def arms(batch_size: int, total_steps: int) -> dict[str, Arm]:
    sqrt_lr = REF_LR * math.sqrt(batch_size / REF_BATCH)
    return {
        a.name: a
        for a in (
            Arm("baseline", REF_LR, 100, False, "the configured recipe, unchanged"),
            Arm(
                "proposal",
                sqrt_lr,
                round(REF_WARMUP_FRAC * total_steps),
                True,
                f"reference recipe adapted: sqrt-scaled peak {sqrt_lr:.4g}, the reference's "
                f"relative 2.50% warmup, cosine to peak/1000. WD unchanged at the configured value",
            ),
        )
    }


def run_arm(name: str) -> dict:
    cfg, cache = check(verbose=False)
    cfg = cfg.model_copy(update={"smoke": True})  # never readable as a headline result
    device = cfg.runtime.resolved_device()
    jcfg = cfg.to_jepa_config()
    arm = arms(jcfg.batch_size, jcfg.steps)[name]
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    # the baseline arm must BE the real recipe, not a paraphrase of it
    if name == "baseline":
        for s in (0, 1, 50, 99, 100, 1000, 2999):
            mine, _ = arm.at(s, jcfg.steps, jcfg.weight_decay)
            theirs = jcfg.lr * min(1.0, (s + 1) / max(jcfg.warmup_steps, 1))
            assert abs(mine - theirs) < 1e-18, f"baseline diverges from train_jepa at step {s}"

    train, monitor_ids, _ = _split_ids(cfg)
    scalars = cache.scalars
    ds = StampDataset(cache, {}, train, scalars=scalars)
    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)

    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    jepa.to(device)
    opt = torch.optim.AdamW(
        [*jepa.encoder.parameters(), *jepa.predictor.parameters()],
        lr=arm.peak_lr,
        weight_decay=jcfg.weight_decay,
    )
    ch = config_hash(cfg.determining_dump())
    checkpointer = TrainCheckpointer(
        out / "checkpoints",
        every=CHECKPOINT_EVERY,
        config_hash=ch,
        normalisation_hash=cfg.normalisation.content_hash,
        schedule={
            "steps": jcfg.steps, "lr": arm.peak_lr, "warmup_steps": arm.warmup,
            "ema_start": jcfg.ema_start, "ema_end": jcfg.ema_end, "batch_size": jcfg.batch_size,
            "decay": "cosine" if arm.decay else "none", "arm": name,
        },
    )
    monitor = CollapseMonitor(floor=cfg.collapse_floor, total_steps=jcfg.steps)
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)
    gc.collect()
    if device.startswith("mps"):
        torch.mps.empty_cache()  # the H2 lesson: release the retained pool, 8.03 -> ~4 GB

    losses: list[float] = []
    trace: list[dict] = []
    t0 = time.perf_counter()
    it = iter(loader)
    for step in range(STEPS):
        batch = _to_device(next(it), device)
        lr, wd = arm.at(step, jcfg.steps, jcfg.weight_decay)
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = lr, wd
        opt.zero_grad(set_to_none=True)
        loss = jepa.loss_step(batch, seed=jcfg.seed + step)  # same masks in both arms
        loss.backward()
        opt.step()
        jepa.ema_update(ema_momentum(step, jcfg.steps, jcfg.ema_start, jcfg.ema_end))
        losses.append(float(loss.item()))
        if step % MONITOR_EVERY == 0:
            with torch.no_grad():
                sig = monitor.update(step, jepa.encoder.encode(mon_batch["image"].float()))
            halt = monitor.should_halt(sig)
            if device.startswith("mps"):
                torch.mps.empty_cache()
            driver = int(torch.mps.driver_allocated_memory()) if device.startswith("mps") else 0
            trace.append(
                {
                    "step": step, "loss": losses[-1], "lr": lr, "weight_decay": wd,
                    "std": sig.std, "effective_rank": sig.effective_rank,
                    "mean_cosine": sig.mean_cosine, "would_halt": halt, "driver_bytes": driver,
                }
            )
            print(f"    {name:<9} step {step:>5}/{STEPS}  lr {lr:.3e}  loss {losses[-1]:.4f}  "
                  f"erank {sig.effective_rank:6.2f}  std {sig.std:7.3f}  cos {sig.mean_cosine:+.3f}"
                  f"  halt={halt}  driver {driver/1e9:.2f} GB", file=sys.stderr, flush=True)
        if (step + 1) % CHECKPOINT_EVERY == 0:
            checkpointer.save(step=step + 1, jepa=jepa, optimiser=opt,
                              losses=losses, collapse_history=monitor.history)

    # the frozen-encoder export the probe reads. `jepa.encoder` (the online context encoder) is
    # what `train_jepa` itself exports and what the pilot probed, so the comparison is like-for-like.
    save_encoder(jepa.encoder, out / "encoder.pt",
                 extra={"steps": STEPS, "arm": name, "smoke": True, "halted": False})
    return {
        "arm": name, "why": arm.why, "peak_lr": arm.peak_lr, "warmup": arm.warmup,
        "decay": "cosine" if arm.decay else "none", "weight_decay": jcfg.weight_decay,
        "batch_size": jcfg.batch_size, "steps_run": STEPS, "config_steps": jcfg.steps,
        "seconds": time.perf_counter() - t0, "device": device, "config_hash": ch,
        "losses": [round(v, 6) for v in losses], "trace": trace,
        "halt_reason": monitor.halt_reason, "checkpoint": str(out / "encoder.pt"),
        "checkpoint_writes": checkpointer.writes,
        "erank_final": trace[-1]["effective_rank"],
        "erank_min": min(r["effective_rank"] for r in trace),
        "std_final": trace[-1]["std"], "std_max": max(r["std"] for r in trace),
        "loss_last50_mean": float(np.mean(losses[-50:])),
        "loss_finite": bool(np.isfinite(losses).all()),
    }


def probe_arm(name: str) -> dict:
    """Probe one arm's frozen encoder. The deciding measurement — AUC, not effective rank."""
    from galaxy_jepa.harness import evaluate_probe

    cfg, _ = check(verbose=False)
    out = RUNS / name
    # `evaluate_probe` opens the baked cache under its out_dir; point a symlink at the real one
    # rather than copying 415.8 GB, and give each arm its own out_dir so the reports never collide.
    link = out / "cache"
    if not link.exists():
        link.symlink_to(Path(cfg.paths.out_dir).resolve() / "cache")
    probe_cfg = cfg.model_copy(update={"smoke": True}).model_copy(
        update={"paths": cfg.paths.model_copy(update={"out_dir": str(out)})}
    )
    t0 = time.perf_counter()
    res = evaluate_probe(probe_cfg, checkpoint=out / "encoder.pt")
    rec = {
        "arm": name, "auc": res.auc, "auc_lo": res.auc_lo, "auc_hi": res.auc_hi,
        "n_train": res.n_train, "n_test": res.n_test, "seconds": time.perf_counter() - t0,
        "checkpoint": str(out / "encoder.pt"), "smoke": True,
    }
    with PROBES.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["arm", "probe", "plan"])
    ap.add_argument("--name", default="baseline")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan":
        cfg, _ = check(verbose=False)
        j = cfg.to_jepa_config()
        print(f"batch {j.batch_size}  config steps {j.steps}  wd {j.weight_decay} (both arms)")
        for a in arms(j.batch_size, j.steps).values():
            lr0, _ = a.at(0, j.steps, j.weight_decay)
            lrw, _ = a.at(a.warmup - 1, j.steps, j.weight_decay)
            lre, _ = a.at(STEPS - 1, j.steps, j.weight_decay)
            print(f"  {a.name:<9} peak {a.peak_lr:.4e}  warmup {a.warmup:>5}  decay {a.decay}")
            print(f"            lr@0 {lr0:.3e}   lr@warmup-end {lrw:.3e}   lr@{STEPS} {lre:.3e}"
                  f"  ({lre / a.peak_lr:.1%} of peak)")
            print(f"            {a.why}")
        return

    if args.mode == "arm":
        rec = run_arm(args.name)
        with POINTS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("RESULT " + json.dumps({k: v for k, v in rec.items() if k != "losses"}))
        return

    print("PROBE " + json.dumps(probe_arm(args.name)))


if __name__ == "__main__":
    main()
