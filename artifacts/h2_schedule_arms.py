"""Brief H2 — is the learning-rate schedule what makes the effective rank fall?

G4 established that ``train_jepa`` applies **warmup only**: no cosine decay, no weight-decay ramp,
against a reference recipe (I-JEPA, Assran et al. 2023) that uses both. The configured 1e-3 is a
*batch-2048* peak learning rate being applied at batch 32 — 64x above the linearly-scaled
equivalent, 8x above the square-root one — and it stays there for the whole run. That is a concrete,
testable candidate for the smoke's effective rank falling 22.6 -> 4.1 by step 175 and flattening.

**What makes this a controlled comparison.** Every arm shares the same initial weights
(``harness.seed_init`` — this experiment was not reliably runnable before that fix, because nothing
in the package called ``torch.manual_seed``), the same data order (one ``ResumableShuffle`` seed),
the same per-step mask seeds (``loss_step(seed=cfg.seed + step)``), the same monitor batch, and the
same ``steps=50000`` so the EMA cosine ramp is bit-identical across arms. The **only** thing that
varies is the ``(lr, weight_decay)`` sequence. The baseline arm's schedule is asserted equal to
``train_jepa``'s own line, so "baseline" means the real recipe and not a paraphrase of it.

**Two honest limits, stated before the numbers.** (1) The *real* cosine decay is analytically inert
in a 500-step window — at step 500 of 50,000 the LR is still 99.98% of peak — so the decay arm has
to be a **compressed proxy** over the window, and it answers "does annealing arrest the fall", not
"what does the real schedule do". Even compressed it is still at 91.6% of peak at step 175, where
the fall has already happened. (2) A lower peak LR trivially shows higher rank at a fixed step count
because less has happened. So erank is read **alongside the loss**, and the arms are additionally
compared at *matched loss* rather than only at matched step.

    uv run python artifacts/h2_schedule_arms.py all          # every arm, one subprocess each
    uv run python artifacts/h2_schedule_arms.py arm --name sqrt
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from f1_loader_bench import OUT, _machine, _split_ids  # noqa: E402

from galaxy_jepa.callbacks.collapse import CollapseMonitor  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, ema_momentum  # noqa: E402

STEPS = 500
MONITOR_EVERY = 25  # the smoke's cadence, so the traces are directly comparable
MONITOR_BATCH = 64
POINTS = OUT / "h2_arms.jsonl"

REF_LR, REF_BATCH = 1e-3, 2048  # I-JEPA's published peak and batch
LR_MIN = 1e-6  # I-JEPA's published cosine floor


@dataclasses.dataclass(frozen=True)
class Arm:
    """One (lr, wd) schedule. ``steps`` is never varied — it would move the EMA ramp."""

    name: str
    peak_lr: float
    warmup: int
    decay: str  # "none" | "cosine_window"
    wd_ramp_to: float | None  # None = constant at the configured value
    why: str

    def lr(self, step: int, base_wd: float) -> tuple[float, float]:
        """The (lr, weight_decay) this arm applies at ``step``. ``step`` is 0-based."""
        # warmup, byte-identical in form to `train_jepa`'s own line
        lr = self.peak_lr * min(1.0, (step + 1) / max(self.warmup, 1))
        if self.decay == "cosine_window" and step + 1 > self.warmup:
            t = (step + 1 - self.warmup) / max(STEPS - self.warmup, 1)
            lr = LR_MIN + (self.peak_lr - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        wd = base_wd
        if self.wd_ramp_to is not None:
            wd = base_wd + (self.wd_ramp_to - base_wd) * min((step + 1) / STEPS, 1.0)
        return lr, wd


def arms(batch_size: int) -> dict[str, Arm]:
    linear = REF_LR * batch_size / REF_BATCH
    sqrt = REF_LR * math.sqrt(batch_size / REF_BATCH)
    return {
        a.name: a
        for a in (
            Arm("baseline", REF_LR, 100, "none", None, "the configured recipe, as the reference"),
            Arm("linear", linear, 100, "none", None, f"linear scaling of I-JEPA's peak: {linear:.4g}"),
            Arm("sqrt", sqrt, 100, "none", None, f"sqrt scaling of I-JEPA's peak: {sqrt:.4g}"),
            Arm(
                "cosine",
                REF_LR,
                100,
                "cosine_window",
                None,
                "baseline peak + cosine to 1e-6 COMPRESSED into the 500-step window — isolates "
                "decay from peak, but see the limit: the real 50,000-step cosine is inert here",
            ),
            Arm(
                "warmup1250",
                REF_LR,
                1250,
                "none",
                None,
                "I-JEPA warms up over 2.5% of its schedule (15 of 600 epochs); ours is 0.20% "
                "(100 of 50,000). 1,250 steps is the reference's RELATIVE warmup, and unlike "
                "decay it acts inside the window where the rank falls",
            ),
            Arm(
                "wd_ramp",
                REF_LR,
                100,
                "none",
                0.4,
                "baseline + I-JEPA's 0.04 -> 0.4 weight-decay ramp, compressed into the window",
            ),
        )
    }


def run_arm(name: str) -> dict:
    cfg, cache = check(verbose=False)
    cfg = cfg.model_copy(update={"smoke": True})
    device = cfg.runtime.resolved_device()
    jcfg = cfg.to_jepa_config()
    arm = arms(jcfg.batch_size)[name]

    # the baseline arm must BE the real recipe, not a paraphrase of it
    if name == "baseline":
        for s in (0, 1, 50, 99, 100, 250, 499):
            mine, _ = arm.lr(s, jcfg.weight_decay)
            theirs = jcfg.lr * min(1.0, (s + 1) / max(jcfg.warmup_steps, 1))
            assert abs(mine - theirs) < 1e-18, f"baseline diverges from train_jepa at step {s}"

    train, monitor_ids, _ = _split_ids(cfg)
    scalars = cache.scalars
    ds = StampDataset(cache, {}, train, scalars=scalars)
    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)

    # identical initial weights across arms — the fix that makes this experiment possible
    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    jepa.to(device)
    opt = torch.optim.AdamW(
        [*jepa.encoder.parameters(), *jepa.predictor.parameters()],
        lr=arm.peak_lr,
        weight_decay=jcfg.weight_decay,
    )
    monitor = CollapseMonitor(floor=cfg.collapse_floor, total_steps=jcfg.steps)
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)
    gc.collect()

    losses: list[float] = []
    trace: list[dict] = []
    t0 = time.perf_counter()
    it = iter(loader)
    for step in range(STEPS):
        batch = _to_device(next(it), device)
        lr, wd = arm.lr(step, jcfg.weight_decay)
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = lr, wd
        opt.zero_grad(set_to_none=True)
        loss = jepa.loss_step(batch, seed=jcfg.seed + step)  # same masks in every arm
        loss.backward()
        opt.step()
        jepa.ema_update(ema_momentum(step, jcfg.steps, jcfg.ema_start, jcfg.ema_end))
        losses.append(float(loss.item()))
        if step % MONITOR_EVERY == 0:
            with torch.no_grad():
                sig = monitor.update(step, jepa.encoder.encode(mon_batch["image"].float()))
            halt = monitor.should_halt(sig)
            trace.append(
                {
                    "step": step, "loss": losses[-1], "lr": lr, "weight_decay": wd,
                    "std": sig.std, "effective_rank": sig.effective_rank,
                    "mean_cosine": sig.mean_cosine, "would_halt": halt,
                    "ema_momentum": ema_momentum(step, jcfg.steps, jcfg.ema_start, jcfg.ema_end),
                }
            )
            print(f"    {name:<11} step {step:>4}  lr {lr:.3e}  loss {losses[-1]:.4f}  "
                  f"erank {sig.effective_rank:6.2f}  cos {sig.mean_cosine:+.3f}  halt={halt}",
                  file=sys.stderr)
    return {
        "arm": name, "why": arm.why, "peak_lr": arm.peak_lr, "warmup": arm.warmup,
        "decay": arm.decay, "wd_ramp_to": arm.wd_ramp_to, "batch_size": jcfg.batch_size,
        "steps_run": STEPS, "config_steps": jcfg.steps, "seconds": time.perf_counter() - t0,
        "device": device, "losses": [round(v, 6) for v in losses], "trace": trace,
        "halt_reason": monitor.halt_reason,
        "erank_final": trace[-1]["effective_rank"], "erank_min": min(r["effective_rank"] for r in trace),
        "loss_last50_mean": float(np.mean(losses[-50:])), "loss_finite": bool(np.isfinite(losses).all()),
    }


def _spawn(name: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", __file__, "arm", "--name", name], capture_output=True, text=True
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-10:]
        rec = {"arm": name, "failed": True, "returncode": proc.returncode, "tail": tail}
    else:
        rec = json.loads(line[len("RESULT "):])
    with POINTS.open("a") as fh:  # on disk before the next arm starts (the G3 lesson)
        fh.write(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "arm"])
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    if args.mode == "arm":
        print("RESULT " + json.dumps(run_arm(args.name)))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    cfg, cache = check()
    bs = cfg.objective.batch_size
    table = arms(bs)
    del cache
    gc.collect()
    print(f"\nH2  {len(table)} arms x {STEPS} steps at batch {bs}, identical seed/order/masks")
    for a in table.values():
        print(f"  {a.name:<11} peak {a.peak_lr:.4g}  warmup {a.warmup:<5} decay {a.decay:<14} "
              f"wd->{a.wd_ramp_to}")
    results = []
    for name in (args.only or list(table)):
        r = _spawn(name)
        results.append(r)
        if r.get("failed"):
            print(f"  {name:<11} FAILED rc={r['returncode']}: {(r.get('tail') or [''])[-1][:140]}")
        else:
            print(f"  {name:<11} erank {r['erank_final']:6.2f} (min {r['erank_min']:.2f})  "
                  f"loss {r['loss_last50_mean']:.4f}  {r['seconds']:.0f}s")
    path = OUT / "h2_arms.json"
    path.write_text(json.dumps({"machine": _machine(), "steps": STEPS, "arms": results}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
