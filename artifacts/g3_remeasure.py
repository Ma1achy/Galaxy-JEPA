"""Brief G3 — re-measure after G2, and do not assume the win.

Three questions, in the order they matter. Does the array-backed dataset move the loader's
stamps/s, or only its memory? Does the worker curve that previously *refused* now run, and where
does it saturate? And is the model still compute-bound at ~41.5 stamps/s — in which case the loader
work bought headroom for a faster machine rather than speed here, which is the right outcome but a
different one, and must be said plainly.

G1's checkpoint write cost comes out of the same pass, because the honest denominator for "what
does a checkpoint cost" is the throughput of the run it interrupts.

Both layers are measured through F1's and F2's own functions wherever they exist — the before/after
table is only a comparison if both halves ran the same code. What is new here is the dataset under
them (:func:`build_array_dataset`, what ``harness._prepare`` now builds) and, in the model phase,
the checkpointer and the pre-registered collapse floor.

One subprocess per point. The machine has 18 GB and Brief F's in-process sweep took it down.

    uv run python artifacts/g3_remeasure.py loader      # F1's pipeline layer + the worker curve
    uv run python artifacts/g3_remeasure.py throughput   # F2's model + the checkpoint cost
    uv run python artifacts/g3_remeasure.py point --workers 4 --access shuffled --seconds 55
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from f1_loader_bench import (  # noqa: E402
    BYTES_PER_STAMP,
    OUT,
    RSS_GUARD,
    WINDOW,
    _machine,
    _memsize,
    _split_ids,
    current_rss_bytes,
    rss_bytes,
)
from f2_model_smoke import (  # noqa: E402
    MEASURE_CAP_S,
    MEASURE_STEPS,
    MONITOR_BATCH,
    MONITOR_EVERY,
    WARMUP,
    _driver_mem,
    _instrument,
    _sync,
    build_jepa,
)

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer  # noqa: E402
from galaxy_jepa.callbacks.collapse import CollapseMonitor  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, ema_momentum  # noqa: E402

WORKER_SWEEP = (0, 2, 4, 8)
HEADLINE_SECONDS = 130  # F1's own headline duration, so the two numbers are comparable
SWEEP_SECONDS = 55
# Compressed from the config's 1,500 so several writes land inside a 300-step run. The reported
# overhead is then rescaled to the configured interval, which is the figure the run plan needs.
CHECKPOINT_EVERY = 25


def build_array_dataset(cfg, cache):
    """What ``harness._prepare`` now builds: the aligned sidecar array, and no metadata table.

    The mirror of ``f1_loader_bench.build_production_dataset``, which is the *before* side of the
    table — that one hands ``StampDataset`` both corpora's full row dicts (4.07 GB resident on this
    corpus) so ``__getitem__`` can read one key out of them.
    """
    t0 = time.perf_counter()
    train, monitor, _ = _split_ids(cfg)
    scalars = cache.scalars
    ds = StampDataset(cache, {}, train, scalars=scalars)
    gc.collect()
    return (
        ds,
        monitor,
        {
            "setup_seconds": time.perf_counter() - t0,
            "n_train": len(ds),
            "n_monitor": len(monitor),
            "sidecar_bytes": int(scalars.nbytes),
        },
    )


# --- the loader layer ----------------------------------------------------------------------


def run_point(args) -> None:
    """One (workers, access) point, in its own process. Prints a single ``RESULT`` line."""
    cfg, cache = check(verbose=False)
    ds, _monitor, meta = build_array_dataset(cfg, cache)
    gc.collect()
    retained, build_peak = current_rss_bytes(), rss_bytes()
    guard = int(RSS_GUARD * _memsize())
    projected = retained * (args.workers + 1)  # spawn: one copy of the dataset per worker
    if projected > guard:
        print("RESULT " + json.dumps({
            "num_workers": args.workers, "access": args.access, "skipped": True,
            "dataset_rss_bytes": retained, "projected_bytes": projected, "guard_bytes": guard,
            **meta}))
        return

    bs = cfg.objective.batch_size
    if args.access == "shuffled":
        # the production sampler, not `shuffle=True`: a seed-pure resumable stream
        sampler = ResumableShuffle(len(ds), seed=cfg.seed)
        loader = DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True,
                            num_workers=args.workers, persistent_workers=False)
    else:
        loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True,
                            num_workers=args.workers, persistent_workers=False)

    parent = os.getppid()
    it = iter(loader)
    t = time.perf_counter()
    first = next(it)
    startup = time.perf_counter() - t
    assert first["image"].shape[0] == bs
    windows: list[float] = []
    t0 = last = time.perf_counter()
    stamps = win = 0
    for batch in it:
        k = int(batch["image"].shape[0])
        stamps += k
        win += k
        now = time.perf_counter()
        if now - last >= WINDOW:
            windows.append(win / (now - last))
            last, win = now, 0
        if now - t0 >= args.seconds:
            break
        # If the driver is gone this process is an orphan still reading the drive, which is what
        # silently contaminated the first attempt at this sweep. Stop rather than finish.
        if os.getppid() != parent:
            sys.exit("orphaned — the driver died; abandoning this point rather than skewing "
                     "whatever runs next")
    el = time.perf_counter() - t0
    print("RESULT " + json.dumps({
        "num_workers": args.workers, "access": args.access, "skipped": False,
        "startup_to_first_batch_s": startup, "stamps": stamps, "seconds": el,
        "stamps_per_s": stamps / el, "mb_per_s": stamps * BYTES_PER_STAMP / el / 1e6,
        "dataset_rss_bytes": retained, "build_peak_rss_bytes": build_peak,
        "peak_rss_bytes": rss_bytes(), "windows_stamps_per_s": [round(x, 1) for x in windows],
        **meta}))


POINTS = OUT / "g3_points.jsonl"


def _swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def _spawn(*argv: str) -> dict:
    """One point per process, and the result is on disk before the next one starts.

    Appended rather than returned-and-collected because the watchdog kills the *parent*: the first
    attempt at this sweep lost every measurement it had taken when a SIGKILL landed before Python
    flushed its buffers. A point that has been measured should survive the point that has not.
    """
    swap0 = _swap_mb()
    proc = subprocess.run([sys.executable, "-u", __file__, *argv], capture_output=True, text=True)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
        rec = {"failed": True, "returncode": proc.returncode, "tail": tail, "argv": list(argv)}
    else:
        rec = json.loads(line[len("RESULT "):])
    rec["swap_growth_mb"] = _swap_mb() - swap0
    with POINTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def loader_sweep(only: list[str] | None = None) -> dict:
    runs: list[dict] = []
    for access in ("shuffled", "sequential"):
        if only and access not in only:
            continue
        for workers in WORKER_SWEEP:
            secs = HEADLINE_SECONDS if workers == 0 else SWEEP_SECONDS
            r = _spawn("point", "--workers", str(workers), "--access", access,
                       "--seconds", str(secs))
            runs.append(r)
            tag = f"  w={workers:<2} {access[:4]:<4}"
            if r.get("failed"):
                last = r["tail"][-1] if r["tail"] else ""
                print(f"{tag} FAILED rc={r['returncode']}: {last}")
            elif r.get("skipped"):
                print(f"{tag} SKIPPED — {r['projected_bytes']/1e9:.1f} GB projected vs "
                      f"{r['guard_bytes']/1e9:.1f} GB guard")
            else:
                print(f"{tag} {r['stamps_per_s']:7.1f} stamps/s  {r['mb_per_s']:7.1f} MB/s  "
                      f"first batch {r['startup_to_first_batch_s']:5.1f}s  "
                      f"ds {r['dataset_rss_bytes']/1e6:5.0f} MB  "
                      f"peak {r['peak_rss_bytes']/1e9:4.1f} GB  "
                      f"swap +{r['swap_growth_mb']/1024:4.1f} GB  "
                      f"win={r['windows_stamps_per_s']}")
    live = [r for r in runs if not r.get("skipped") and not r.get("failed")]
    best = max(live, key=lambda r: r["stamps_per_s"]) if live else None
    return {"runs": runs, "best": None if best is None else {
        "workers": best["num_workers"], "access": best["access"],
        "stamps_per_s": best["stamps_per_s"]}}


# --- the model layer, with the checkpointer live -------------------------------------------


def throughput(checkpoint_every: int = CHECKPOINT_EVERY) -> dict:
    """F2's throughput phase, on the array dataset, with G1's checkpointer and G5's floor live."""
    cfg, cache = check(verbose=False)
    cfg = cfg.model_copy(update={"smoke": True})
    device = cfg.runtime.resolved_device()
    ds, monitor_ids, dsmeta = build_array_dataset(cfg, cache)
    dataset_rss = current_rss_bytes()
    jepa, minfo = build_jepa(cfg, cache.index.height, device)
    jcfg = jepa.config
    bs = jcfg.batch_size

    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW([*jepa.encoder.parameters(), *jepa.predictor.parameters()],
                            lr=jcfg.lr, weight_decay=jcfg.weight_decay)
    ck = TrainCheckpointer(
        OUT / "g3_checkpoints", every=checkpoint_every, config_hash="v2:g3-remeasure",
        normalisation_hash=cache.index.normalisation_hash,
        schedule={"steps": jcfg.steps, "lr": jcfg.lr, "warmup_steps": jcfg.warmup_steps,
                  "ema_start": jcfg.ema_start, "ema_end": jcfg.ema_end},
    )
    monitor = CollapseMonitor(floor=cfg.collapse_floor, total_steps=jcfg.steps)
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=cache.scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)

    host = _instrument(jepa)
    losses: list[float] = []
    trace: list[dict] = []
    data_s = compute_s = 0.0
    peak_mem = 0
    step = 0
    it = iter(loader)
    t_start = None
    print(f"  warmup {WARMUP} steps, then up to {MEASURE_STEPS} (cap {MEASURE_CAP_S}s) at batch "
          f"{bs}, checkpointing every {checkpoint_every}", file=sys.stderr)
    while step < WARMUP + MEASURE_STEPS:
        if step == WARMUP:
            _sync(device)
            for k in host:
                host[k] = 0.0
            data_s = compute_s = 0.0
            ck.write_seconds, ck.writes = 0.0, 0
            t_start = time.perf_counter()

        t0 = time.perf_counter()
        batch = _to_device(next(it), device)
        t1 = time.perf_counter()

        for g in opt.param_groups:
            g["lr"] = jcfg.lr * min(1.0, (step + 1) / max(jcfg.warmup_steps, 1))
        opt.zero_grad(set_to_none=True)
        loss = jepa.loss_step(batch, seed=jcfg.seed + step)
        loss.backward()
        opt.step()
        jepa.ema_update(ema_momentum(step, jcfg.steps, jcfg.ema_start, jcfg.ema_end))
        loss_val = float(loss.item())
        _sync(device)
        t2 = time.perf_counter()

        if step >= WARMUP:
            data_s += t1 - t0
            compute_s += t2 - t1
            losses.append(loss_val)
            peak_mem = max(peak_mem, _driver_mem(device))

        if step % MONITOR_EVERY == 0:
            with torch.no_grad():
                sig = monitor.update(step, jepa.encoder.encode(mon_batch["image"].float()))
            halt = monitor.should_halt(sig)
            trace.append({"step": step, "loss": loss_val, "std": sig.std,
                          "effective_rank": sig.effective_rank, "mean_cosine": sig.mean_cosine,
                          "would_halt": halt})
            print(f"    step {step:>4}  loss {loss_val:.4f}  erank {sig.effective_rank:6.2f}  "
                  f"halt={halt}", file=sys.stderr)

        if (step + 1) % checkpoint_every == 0:
            ck.save(step=step + 1, jepa=jepa, optimiser=opt, losses=losses,
                    collapse_history=monitor.history)

        step += 1
        if t_start is not None and time.perf_counter() - t_start >= MEASURE_CAP_S:
            break

    measured = step - WARMUP
    # the training wall-clock *excluding* the checkpoint writes, so the overhead below is a
    # fraction of work done rather than of itself
    wall = data_s + compute_s
    steps_per_s = measured / wall
    per_write = ck.write_seconds / max(ck.writes, 1)
    interval_s = jcfg.checkpoint_every / steps_per_s
    ck_bytes = max((p.stat().st_size for p in ck.dir.glob("ckpt_*.pt")), default=0)
    fifth = max(len(losses) // 5, 1)
    hm = host["weight_maps_s"] + host["mask_sample_s"]
    return {
        **minfo, "dataset": dsmeta, "device": device, "batch_size": bs,
        "warmup_steps_excluded": WARMUP, "measured_steps": measured, "wall_s": wall,
        "steps_per_s": steps_per_s, "stamps_per_s": measured * bs / wall,
        "mb_per_s": measured * bs * BYTES_PER_STAMP / wall / 1e6,
        "data_wait_s": data_s, "compute_s": compute_s, "data_wait_fraction": data_s / wall,
        "host_mask_s": hm, "host_mask_fraction_of_compute": hm / compute_s,
        "peak_driver_bytes": peak_mem, "dataset_rss_bytes": dataset_rss,
        "peak_rss_bytes": rss_bytes(),
        "checkpoint": {
            "every_measured": checkpoint_every, "writes": ck.writes,
            "total_seconds": ck.write_seconds, "seconds_per_write": per_write,
            "bytes": ck_bytes, "configured_every": jcfg.checkpoint_every,
            "interval_minutes_at_configured": interval_s / 60,
            "overhead_fraction_at_configured": per_write / interval_s,
        },
        "loss_first": float(np.mean(losses[:fifth])), "loss_last": float(np.mean(losses[-fifth:])),
        "loss_finite": bool(np.isfinite(losses).all()),
        "collapse_trace": trace, "halt_reason": monitor.halt_reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loader", "throughput", "point", "all"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--access", choices=["shuffled", "sequential"], default="shuffled")
    ap.add_argument("--seconds", type=float, default=SWEEP_SECONDS)
    ap.add_argument("--access-only", nargs="*", default=None)
    # The control for the checkpoint cost. `overhead_fraction_at_configured` counts only the
    # write's own excluded seconds; it does not count what 380 MB of interleaved I/O does to a
    # unified-memory machine's bandwidth while the GPU is working. Setting this high enough that
    # no write lands in the measured window isolates that, on the same thermal state.
    ap.add_argument("--checkpoint-every", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "point":
        run_point(args)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"machine": _machine(), "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.mode in ("loader", "all"):
        cfg, cache = check()
        print(f"\nG3  loader layer after G2 — sidecar {cache.scalars.nbytes/1e6:.2f} MB, "
              f"batch {cfg.objective.batch_size}")
        del cfg, cache
        gc.collect()
        result["loader"] = loader_sweep(args.access_only)
    if args.mode in ("throughput", "all"):
        print("\nG3  model layer — checkpointer and collapse floor live")
        result["throughput"] = throughput(args.checkpoint_every or CHECKPOINT_EVERY)
        t = result["throughput"]
        print(f"  {t['steps_per_s']:.3f} steps/s  {t['stamps_per_s']:.1f} stamps/s  "
              f"data-wait {t['data_wait_fraction']*100:.1f}%  driver "
              f"{t['peak_driver_bytes']/1e9:.2f} GB  rss {t['peak_rss_bytes']/1e9:.2f} GB")
        c = t["checkpoint"]
        print(f"  checkpoint {c['bytes']/1e6:.0f} MB in {c['seconds_per_write']:.2f} s "
              f"({c['writes']} writes); at every={c['configured_every']} that is "
              f"{c['overhead_fraction_at_configured']*100:.3f}% of wall-clock and "
              f"{c['interval_minutes_at_configured']:.1f} min at risk")

    suffix = "" if args.checkpoint_every is None else f"_ck{args.checkpoint_every}"
    path = OUT / f"g3_{args.mode}{suffix}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
