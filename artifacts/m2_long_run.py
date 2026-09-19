"""Brief M2 — the long λ=0 baseline run, segmented so the stopping rule can actually fire.

**The budget is 10 epochs: 253,270 steps** (810,491 train stamps // batch 32 = 25,327 per epoch),
~46 h at Brief L's measured 1.515 steps/s. Every run before this one was diagnostic — 3,000 steps
is 0.12 epochs, 10,500 is 0.42, J's 50,000 was 1.97, and the reference trains for 600.

Why segmented, rather than one invocation. M4 pre-registers a stopping rule, and a rule can only
fire if the probe runs *while there is still budget to save*. But probing alongside training is
the one lesson this project has already paid for twice: a CPU sweep beside a training loop costs
2.8x because the loop's dataloader starves, and the machine OOM'd once. So training stops, the
probe runs with the machine to itself, and training resumes:

    segment 1 -> step  12,663 (0.5 ep)  probe
    segment 2 -> step  25,327 (1 ep)    probe
    segment 3 -> step  50,654 (2 ep)    probe   <- earliest the rule can fire, ~9.3 h in
    segments 4-7 -> 4 / 6 / 8 / 10 epochs

`stop_after` "ends *this invocation*, leaving `config.steps` — and so the LR and EMA schedules —
untouched" (`objectives/jepa.py:319`), and it is counted from this invocation's start
(`done - start_step`), so each segment passes its own delta. **`config.steps` stays 253,270
throughout**: segmenting changes when the process exits, never the recipe. The loop also lands a
checkpoint exactly where it stops, so probe points sit on exact epoch boundaries rather than on
the nearest scheduled checkpoint.

Nothing is reimplemented. Training is `train_jepa`, the pre-flight is `m1_preflight`, and the
probe is `k2_trajectory_probe.py` run as a subprocess against this run's directory — the same
held-out split and probe config as K2/L2, so the numbers sit alongside 0.9358 / 0.9470 / 0.9554 /
0.9278. A subprocess rather than an import, deliberately: it gives the probe's memory back to the
machine before training resumes.

**Not a results run.** `smoke: true`, `effect_floor` unset, so no ladder verdict can be reached
from it; no rung assignments, and no AUC-based checkpoint selection (1C stands).

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/m2_long_run.py --plan     # budget and schedule; launches nothing
    uv run python artifacts/m2_long_run.py 2>&1 | tee runs/m2.log
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402
from i2_sigreg_run import _split_ids  # noqa: E402
from m1_preflight import main as preflight  # noqa: E402

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer  # noqa: E402
from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, learning_rate, train_jepa  # noqa: E402

MONITOR_BATCH = 64

#: Probe points, in epochs. Dense early because that is where every previous run lived and where
#: a turn would be missed, sparse late because a plateau does not need resolution.
PROBE_EPOCHS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0)

#: M4, stated before the numbers. Stop when consensus AUC improves by less than this across two
#: CONSECUTIVE probe intervals, and the trend is not still rising within intervals.
FLAT_DELTA = 0.002

OUT = REPO / "artifacts" / "out"
PROBE = Path(__file__).resolve().parent / "k2_trajectory_probe.py"


def _schedule(steps_per_epoch: int, total: int) -> list[int]:
    """Absolute step for each probe point, clipped to the budget and de-duplicated."""
    pts = sorted({min(total, round(e * steps_per_epoch)) for e in PROBE_EPOCHS})
    return pts


def _merge(into: list[float], new: list[float]) -> list[float]:
    """Fold one invocation's traces into the cumulative ones.

    ``train_jepa`` returns full-length lists in which the steps restored from a checkpoint come
    back as NaN rather than being dropped — the lists stay aligned and the gap is visible instead
    of implied. So a merge is "take whatever this invocation actually measured", and the result is
    indexed by ABSOLUTE step, which is what the probe reads.
    """
    if len(new) > len(into):
        into = into + [math.nan] * (len(new) - len(into))
    for i, v in enumerate(new):
        if not math.isnan(v):
            into[i] = v
    return into


def _read_rule(curve: list[dict]) -> tuple[bool, str]:
    """M4's stopping rule, applied to the consensus AUC. Returns (stop, why).

    The rule as briefed is a raw ΔAUC per probe interval. The intervals are UNEQUAL by design
    (0.5, 1, 2, 4, 6, 8, 10 epochs), so a raw delta favours stopping late, when intervals are
    widest — a 2-epoch gap has four times the room to improve that a 0.5-epoch gap has. The slope
    per epoch is therefore reported alongside, and if the two disagree the run CONTINUES: the rule
    may stop the run early, it may never extend it.
    """
    if len(curve) < 3:
        return False, f"{len(curve)} probe point(s) — the rule needs 3"
    (a, b, c) = curve[-3:]
    d1, d2 = b["auc"] - a["auc"], c["auc"] - b["auc"]
    s1 = d1 / (b["epoch"] - a["epoch"])
    s2 = d2 / (c["epoch"] - b["epoch"])
    flat = d1 < FLAT_DELTA and d2 < FLAT_DELTA
    rising = d2 > d1  # still accelerating within intervals — not a plateau
    detail = (
        f"ΔAUC {d1:+.4f} then {d2:+.4f} (rule: both < {FLAT_DELTA}); "
        f"slope/epoch {s1:+.4f} then {s2:+.4f}"
    )
    if flat and not rising:
        return True, f"FLAT — {detail}"
    if flat and rising:
        return False, f"flat by ΔAUC but still rising within intervals, so continuing — {detail}"
    return False, f"still improving — {detail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="print the budget and schedule, run nothing")
    ap.add_argument("--skip-preflight", action="store_true", help="only for a resume after a crash")
    ap.add_argument("--test", action="store_true",
                    help="PLUMBING ONLY: a 60-step run in a scratch dir, to prove segmenting, "
                         "resume and the trace merge before 46 h is committed. Changes `steps`, "
                         "so it is a different recipe and can never be a result.")
    ap.add_argument("--no-probe", action="store_true", help="train only; skip the probe between "
                                                            "segments (used by --test)")
    ap.add_argument("--test-points", default="20,40,60",
                    help="--test only: the segment boundaries. Run it once as '20,40,60' and once "
                         "as '60' and the traces must agree step for step -- that is the proof "
                         "that segmenting changes WHEN the process exits and not the recipe.")
    ap.add_argument("--test-out", default="runs/m_test", help="--test only: where to write")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    log = logging.getLogger("m2")

    cfg, cache = check(verbose=False)
    obj = cfg.objective
    if not cfg.smoke:
        raise SystemExit("M2: smoke is False — this run is not a result and must say so")
    if obj.sigreg_lambda != 0.0:
        raise SystemExit(f"M2: sigreg_lambda is {obj.sigreg_lambda}; D21 runs the baseline at 0")

    train_ids, monitor_ids, _ = _split_ids(cfg)
    per_epoch = len(train_ids) // obj.batch_size
    total = obj.steps
    points = _schedule(per_epoch, total)
    if args.test:
        # Deliberately a DIFFERENT recipe -- `steps` is determining, so this hashes apart and can
        # never be mistaken for the real run. It exists to prove the machinery, not to measure.
        obj = obj.model_copy(update={"steps": 60, "checkpoint_every": 20, "monitor_every": 10})
        cfg = cfg.model_copy(update={"objective": obj, "paths": cfg.paths.model_copy(
            update={"out_dir": args.test_out})})
        points = [int(x) for x in args.test_points.split(",")]
        total = points[-1]
        args.no_probe, args.skip_preflight = True, True
        log.warning("M2 --test: 60 steps to runs/m_test. PLUMBING ONLY, not a measurement.")
    if total % per_epoch:
        log.warning("budget is %.4f epochs, not a whole number", total / per_epoch)

    out = Path(cfg.paths.out_dir)
    scheduled = total // obj.checkpoint_every
    # `keep` must cover the scheduled checkpoints AND the one each segment lands where it stops,
    # or the earliest would be pruned — which is exactly how `keep=3` destroyed a trajectory
    # mid-measurement once. The harness derives `scheduled + 2`, which is short by one per segment.
    keep = scheduled + len(points) + 2

    print(f"\nM2 budget      : {total:,} steps = {total / per_epoch:.2f} epochs "
          f"({len(train_ids):,} train // batch {obj.batch_size} = {per_epoch:,}/epoch)")
    print(f"M2 wall-clock  : ~{total / 1.515 / 3600:.1f} h at 1.515 steps/s (Brief L, this path)")
    print(f"M2 out_dir     : {out}")
    print(f"M2 checkpoints : every {obj.checkpoint_every:,} -> {scheduled} scheduled + "
          f"{len(points)} segment-end, keep={keep} ({keep * 381 / 1024:.1f} GiB)")
    print(f"M2 stopping    : ΔAUC < {FLAT_DELTA} across two consecutive intervals, "
          f"not still rising within them")
    print("M2 segments    :")
    prev = 0
    for i, s in enumerate(points, 1):
        print(f"   {i}. +{s - prev:>7,} -> step {s:>7,}  ({s / per_epoch:>5.2f} ep, "
              f"lr {learning_rate(s, cfg.to_jepa_config()):.3e}, "
              f"~{(s - prev) / 1.515 / 3600:>4.1f} h)")
        prev = s
    if args.plan:
        print("\nM2 --plan: nothing launched")
        return

    if not args.skip_preflight:
        preflight()  # raises on the first unchecked fact

    device = cfg.runtime.resolved_device()
    jcfg = cfg.to_jepa_config()
    out.mkdir(parents=True, exist_ok=True)
    scalars = cache.scalars
    ds = StampDataset(cache, {}, train_ids, scalars=scalars)
    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)
    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    ch = config_hash(cfg.determining_dump())
    checkpointer = TrainCheckpointer(
        out / "checkpoints",
        every=obj.checkpoint_every,
        keep=keep,
        config_hash=ch,
        normalisation_hash=cfg.normalisation.content_hash,  # F0 refuses a config without it
        schedule={
            "steps": jcfg.steps,  # 253,270 in EVERY segment — the recipe does not move
            "lr": jcfg.lr,
            "lr_final": jcfg.lr_final,
            "warmup_steps": jcfg.warmup_steps,
            "ema_start": jcfg.ema_start,
            "ema_end": jcfg.ema_end,
            "batch_size": jcfg.batch_size,
            "sigreg_lambda": jcfg.sigreg_lambda,
            "brief": "M",
            # `probe_points` is deliberately NOT here. The checkpointer asserts this dict matches
            # on resume, because the LR warmup and the EMA ramp are functions of it -- so putting
            # the probe schedule in it made a run refuse to continue itself merely because we had
            # changed our mind about WHEN TO LOOK. Observing a run is not part of its recipe, the
            # same distinction `monitor_every` needed. Caught by the 60-step resume test.
        },
    )
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)

    # Pick up whatever is already on disk. A 46 h run WILL be interrupted -- Brief M's first
    # attempt was stopped an hour in so the SSD could be unplugged -- and a driver that always
    # starts segment 1 fresh throws that away. What matters is the newest checkpoint, not which
    # segment the process thinks it is on.
    written = sorted(int(f.stem.split("_")[1]) for f in (out / "checkpoints").glob("ckpt_*.pt"))
    done = written[-1] if written else 0
    if done:
        log.warning("M2 resuming from step %d (%.2f ep) — %d checkpoint(s) on disk",
                    done, done / per_epoch, len(written))
    losses: list[float] = []
    pred: list[float] = []
    sig: list[float] = []
    collapse: dict[str, list[float]] = {}
    curve: list[dict] = []
    if (out / "traces.json").exists():
        prior = json.loads((out / "traces.json").read_text())
        losses, pred, sig = prior["losses"], prior["prediction_losses"], prior["sigreg_losses"]
        collapse = prior["collapse_trace"]
    if (OUT / "m_curve.json").exists():
        curve = json.loads((OUT / "m_curve.json").read_text())["curve"]
        curve = [r for r in curve if r["step"] <= done]  # never carry a probe of a lost step
    t_start = time.perf_counter()

    for i, target in enumerate(points, 1):
        if target <= done:
            # Skipping the TRAINING is right; skipping its PROBE POINT is not. The stopping rule
            # reads the curve, so a resume that quietly drops the first point would need four
            # segments to do what three should, and would misreport the slope. The probe wrote
            # its own file, so recover from that rather than re-extracting for 20 minutes.
            prior = OUT / f"m{i}_trajectory.json"
            if any(r["step"] == target for r in curve):
                log.info("M2 segment %d/%d: done at step %d, probe already in the curve",
                         i, len(points), target)
            elif prior.exists():
                rec = json.loads(prior.read_text())["checkpoints"][-1]
                if rec["step"] == target:
                    rec["epoch"] = target / per_epoch
                    curve.append(rec)
                    log.info("M2 segment %d/%d: done at step %d, probe recovered from %s "
                             "(consensus %.4f)", i, len(points), target, prior.name, rec["auc"])
                else:
                    log.warning("M2 segment %d: %s is step %d, not %d — not using it",
                                i, prior.name, rec["step"], target)
            else:
                log.warning("M2 segment %d/%d: done at step %d but NO probe on disk — the curve "
                            "is short by one and the stopping rule needs a further point",
                            i, len(points), target)
            continue
        delta = target - done
        gc.collect()
        if device.startswith("mps"):
            torch.mps.empty_cache()  # H2: release the retained pool before the loop
        log.info("M2 segment %d/%d: +%d steps -> %d (%.2f ep)",
                 i, len(points), delta, target, target / per_epoch)
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
            # Resume whenever anything is on disk -- NOT "whenever this is not segment 1". The
            # two differ exactly when a run was interrupted inside segment 1, which is the case
            # that costs the most to get wrong.
            resume=done > 0,
            stop_after=delta,
        )
        secs = time.perf_counter() - t0
        done = result.steps_completed
        losses, pred, sig = (
            _merge(losses, result.losses),
            _merge(pred, result.prediction_losses),
            _merge(sig, result.sigreg_losses),
        )
        for k, v in result.collapse_trace.items():
            collapse.setdefault(k, [])
            collapse[k] = collapse[k][: len(collapse[k])] + [
                x for j, x in enumerate(v) if j >= len(collapse[k])
            ]
        (out / "traces.json").write_text(json.dumps({
            "collapse_trace": collapse, "losses": losses,
            "prediction_losses": pred, "sigreg_losses": sig,
        }))
        log.info("M2 segment %d done: step %d, %.2f h, %.4f steps/s%s",
                 i, done, secs / 3600, delta / secs, "  HALTED" if result.halted else "")
        if result.halted:
            log.error("M2 halted at step %d — the collapse monitor fired. Stopping.", done)
            break
        if done != target:
            log.warning("M2 landed at %d, not %d", done, target)

        # --- probe, with the machine to itself -------------------------------------------
        if args.no_probe:
            log.info("M2 segment %d: probe skipped (--no-probe)", i)
            continue
        tag = f"m{i}"
        proc = subprocess.run(
            [sys.executable, str(PROBE), "--run", str(out), "--steps", str(done), "--tag", tag],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log.error("M2 probe at step %d failed:\n%s", done, proc.stderr[-2000:])
            break
        rec = json.loads((OUT / f"{tag}_trajectory.json").read_text())["checkpoints"][-1]
        rec["epoch"] = done / per_epoch
        rec["hours"] = (time.perf_counter() - t_start) / 3600
        curve.append(rec)
        log.info("M2 PROBE step %d (%.2f ep): consensus %.4f [%.4f, %.4f]  all %.4f  amb %.4f",
                 done, rec["epoch"], rec["auc"], rec["auc_lo"], rec["auc_hi"],
                 rec["auc_all"], rec["auc_ambiguous"])

        stop, why = _read_rule(curve)
        log.info("M2 stopping rule: %s", why)
        (OUT / "m_curve.json").write_text(json.dumps({
            "budget_steps": total, "steps_per_epoch": per_epoch, "probe_points": points,
            "flat_delta": FLAT_DELTA, "smoke": True, "config_hash": f"v2:{ch}",
            "stopped_early": stop, "stop_reason": why, "curve": curve,
        }, indent=2))
        if stop:
            log.info("M2 STOPPING EARLY at %d of %d steps (%.2f of %.2f epochs) — %s",
                     done, total, done / per_epoch, total / per_epoch, why)
            break

    wall = (time.perf_counter() - t_start) / 3600
    first, last = (curve[0], curve[-1]) if curve else ({}, {})
    log.info("M2 finished: %d steps, %.2f h. AUC %.4f -> %.4f over %.2f epochs",
             done, wall, first.get("auc", float("nan")), last.get("auc", float("nan")),
             last.get("epoch", 0.0))
    if len(curve) >= 2:
        slope = (curve[-1]["auc"] - curve[-2]["auc"]) / (curve[-1]["epoch"] - curve[-2]["epoch"])
        log.info("M2 final slope: %+.4f AUC/epoch — the number a rental case is argued from",
                 slope)


if __name__ == "__main__":
    main()
