"""Brief I — the SIGReg arms. Three at 3,000 steps, then probe every checkpoint.

The rule was committed before any of this ran (`artifacts/i_decision_rule.md`; `git log` is the
proof), as H5's was.

**Every H5 control is unchanged**: one seed through `harness.seed_init` so all three arms start
from identical weights, one `ResumableShuffle` seed so they see the same data in the same order,
the same per-step mask seeds, the same monitor batch, and `steps=50000` held constant so the EMA
momentum ramp cannot differ between arms. The only thing that varies is `sigreg_lambda`.

**Unlike H5, the loop is the production one.** H5 carried its own `(lr, wd)` schedule class
because D17 did not exist yet; it does now, so these arms go through `train_jepa` and read the
schedule from `configs/pretrain.yaml`. That makes the `d17` arm a free replication check — it is
the same recipe H5 called `proposal`, so it must reproduce H5's numbers, and if it does not, the
production path has drifted from what H5 measured and *that* is the finding.

    uv run python artifacts/i2_sigreg_run.py plan
    uv run python artifacts/i2_sigreg_run.py arm --name d17
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
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
from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, learning_rate, train_jepa  # noqa: E402

STEPS = 3000
MONITOR_EVERY = 25  # H5's cadence, so the traces are directly comparable
MONITOR_BATCH = 64
CHECKPOINT_EVERY = 500  # six per arm — the Q2 loss-usability sweep needs the points
RUNS = Path(__file__).resolve().parent.parent / "runs" / "i2"
POINTS = OUT / "i2_arms.jsonl"

#: What the `d17` arm must reproduce: H5's `proposal` arm at its LAST MONITOR READING, step
#: 2,975 — not step 3,000. H5's headline "loss @3000" of 0.3168 is that reading, and a per-step
#: latent MSE swings enough between neighbouring steps to make the distinction matter. Not a
#: tolerance chosen to pass: the two runs share seed, data order, masks and schedule, so the
#: only licensed difference is float non-determinism in the MPS kernels over 3,000 steps.
H5_PROPOSAL = {"loss": 0.3168, "erank": 11.77, "std": 4.03, "cos": 0.286}


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    lam: float
    why: str


ARMS = {
    a.name: a
    for a in (
        Arm("d17", 0.0, "the adopted recipe, no penalty computed — the comparator"),
        Arm("sigreg_050", 0.05, "LeJEPA §6.1 verbatim: the recommended lambda, at eight views"),
        Arm(
            "sigreg_006",
            0.00625,
            "0.05 x (1/8): §6.1 says the optimum scales with the view count, and this project "
            "has one context view against their eight. Weaker provenance, labelled so",
        ),
    )
}


def run_arm(name: str) -> dict:
    arm = ARMS[name]
    cfg, cache = check(verbose=False)
    obj = cfg.objective.model_copy(
        update={
            "sigreg_lambda": arm.lam,
            "monitor_every": MONITOR_EVERY,
            "checkpoint_every": CHECKPOINT_EVERY,
        }
    )
    cfg = cfg.model_copy(update={"smoke": True, "objective": obj})  # never a headline result
    device = cfg.runtime.resolved_device()
    jcfg = cfg.to_jepa_config()
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    train, monitor_ids, _ = _split_ids(cfg)
    scalars = cache.scalars
    ds = StampDataset(cache, {}, train, scalars=scalars)
    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)

    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    ch = config_hash(cfg.determining_dump())
    checkpointer = TrainCheckpointer(
        out / "checkpoints",
        every=CHECKPOINT_EVERY,
        # the default keep=3 prunes the older ones, and question 2 needs every checkpoint the
        # run wrote — the point of the sweep is the trajectory, not the endpoint
        keep=STEPS // CHECKPOINT_EVERY + 2,
        config_hash=ch,
        normalisation_hash=cfg.normalisation.content_hash,
        schedule={
            "steps": jcfg.steps,
            "lr": jcfg.lr,
            "lr_final": jcfg.lr_final,
            "warmup_steps": jcfg.warmup_steps,
            "ema_start": jcfg.ema_start,
            "ema_end": jcfg.ema_end,
            "batch_size": jcfg.batch_size,
            "sigreg_lambda": arm.lam,
            "arm": name,
        },
    )
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)
    gc.collect()
    if device.startswith("mps"):
        torch.mps.empty_cache()  # the H2 lesson: release the retained pool before the loop

    t0 = time.perf_counter()
    result = train_jepa(
        jepa,
        loader,
        device=device,
        monitor_batch=mon_batch,
        checkpoint_path=out / "encoder.pt",
        autocast_dtype=cfg.autocast_dtype(),
        checkpointer=checkpointer,
        sampler=sampler,
        collapse_floor=cfg.collapse_floor,
        resume=False,  # each arm is a fresh run; a resume would silently continue the last one
        stop_after=STEPS,
    )
    seconds = time.perf_counter() - t0

    # the monitor trace, rejoined with the per-step terms at the steps it was read
    trace = []
    ct = result.collapse_trace
    for i, step in enumerate(int(s) for s in ct["step"]):
        trace.append(
            {
                "step": step,
                "lr": learning_rate(step, jcfg),
                "loss": result.losses[step],
                "loss_prediction": result.prediction_losses[step],
                "loss_sigreg": _finite(result.sigreg_losses[step]),
                "std": ct["std"][i],
                "effective_rank": ct["effective_rank"][i],
                "mean_cosine": ct["mean_cosine"][i],
            }
        )
    losses = result.losses
    return {
        "arm": name,
        "why": arm.why,
        "sigreg_lambda": arm.lam,
        "sigreg_slices": jcfg.sigreg_slices,
        "lr": jcfg.lr,
        "lr_final": jcfg.lr_final,
        "warmup_steps": jcfg.warmup_steps,
        "weight_decay": jcfg.weight_decay,
        "batch_size": jcfg.batch_size,
        "steps_run": result.steps_completed,
        "config_steps": jcfg.steps,
        "seconds": seconds,
        "steps_per_s": result.steps_completed / seconds,
        "device": device,
        "config_hash": ch,
        "halted": result.halted,
        "checkpoint": str(out / "encoder.pt"),
        "checkpoint_writes": checkpointer.writes,
        "losses": [round(v, 6) for v in losses],
        "losses_prediction": [round(v, 6) for v in result.prediction_losses],
        "losses_sigreg": [_finite(v) for v in result.sigreg_losses],
        "trace": trace,
        "loss_final": losses[-1],
        "loss_deepest": float(np.min(losses)),
        "loss_deepest_step": int(np.argmin(losses)),
        "loss_last50_mean": float(np.mean(losses[-50:])),
        "loss_finite": bool(np.isfinite(losses).all()),
        "erank_final": trace[-1]["effective_rank"],
        "erank_min": min(r["effective_rank"] for r in trace),
        "erank_max": max(r["effective_rank"] for r in trace),
        "std_final": trace[-1]["std"],
        "std_max": max(r["std"] for r in trace),
        "mean_cosine_final": trace[-1]["mean_cosine"],
        "sigreg_first": trace[0]["loss_sigreg"],
        "sigreg_final": trace[-1]["loss_sigreg"],
    }


def _finite(v: float) -> float | None:
    """NaN means the penalty was not computed — say ``null``, never a measured zero."""
    return None if v != v else round(float(v), 6)


def replication(rec: dict) -> dict:
    """How far the `d17` arm sits from H5's `proposal`. Reported whether or not it passes."""
    got = {
        "loss": rec["trace"][-1]["loss"],  # the same monitor reading H5 quoted, not step 3,000
        "erank": rec["erank_final"],
        "std": rec["std_final"],
        "cos": rec["mean_cosine_final"],
    }
    return {
        k: {
            "h5": H5_PROPOSAL[k],
            "now": round(got[k], 4),
            "rel": round(abs(got[k] - H5_PROPOSAL[k]) / abs(H5_PROPOSAL[k]), 4),
        }
        for k in H5_PROPOSAL
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["arm", "plan"])
    ap.add_argument("--name", default="d17")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan":
        cfg, _ = check(verbose=False)
        j = cfg.to_jepa_config()
        print(f"batch {j.batch_size}  config steps {j.steps}  run {STEPS}  wd {j.weight_decay}")
        print(f"lr {j.lr:.4e} -> {j.lr_final:.4e}  warmup {j.warmup_steps}  "
              f"lr@{STEPS} {learning_rate(STEPS - 1, j):.4e}")
        print(f"slices {j.sigreg_slices}  quad {j.sigreg_quad_points}  domain +-{j.sigreg_domain}")
        for a in ARMS.values():
            print(f"  {a.name:<11} lambda {a.lam:<9} {a.why}")
        return

    rec = run_arm(args.name)
    with POINTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    slim = {k: v for k, v in rec.items() if not k.startswith("losses")}
    if args.name == "d17":
        slim["replicates_h5_proposal"] = replication(rec)
    print("RESULT " + json.dumps(slim))


if __name__ == "__main__":
    main()
