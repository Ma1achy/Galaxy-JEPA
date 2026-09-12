"""Brief F2 — the model smoke: the real path end to end, a few hundred steps, five numbers.

Not "does it run". A green tick says nothing about a multi-day job. This measures sustained
throughput, peak memory and the largest batch that fits, whether anything is silently running on
the CPU, what share of wall-clock is spent waiting on the drive rather than computing, and
whether the collapse signals and the loss are doing anything degenerate.

The path is the real one — ``build_objective`` → ``Jepa.loss_step`` → bbox-biased
``MultiBlockMasker`` → EMA target → predictor → latent MSE → EMA update → ``CollapseMonitor``,
fed by the production ``StampDataset``/``DataLoader``. Nothing is reimplemented; ``weight_maps``
and ``masker.sample`` are wrapped with a timer so the host-side share of a step is visible, but
the wrapped callables are the objective's own.

**Each phase runs in its own subprocess.** The production dataset is 4.07 GB resident and the
model on MPS is several more; holding both plus a batch-size search in one process on 18 GB
unified memory does not fit. The batch search additionally caps the MPS allocator at the
recommended maximum so it terminates by *raising* rather than by driving the machine into swap.

The config is marked ``smoke=True``. That is a determining field, so it moves ``config_hash``,
and it is written into ``escape_hatches_used`` — a throughput measurement can never be read back
as a run.

**This does not start the real pretraining run.** It takes a bounded number of steps and writes
no encoder checkpoint.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/f2_model_smoke.py all
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
    _machine,
    build_production_dataset,
    rss_bytes,
)

from galaxy_jepa.callbacks.collapse import CollapseMonitor  # noqa: E402
from galaxy_jepa.core.config import RunStamp, write_stamp  # noqa: E402
from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.models.vit import VisionTransformer  # noqa: E402
# the objective's own device move: it downcasts the collated float64 scalars, which MPS has
# no kernel for. Reimplementing it is how this script briefly diverged from the real path.
from galaxy_jepa.objectives.jepa import _to_device, ema_momentum  # noqa: E402

WARMUP = 20
MEASURE_STEPS = 300
MEASURE_CAP_S = 420
MONITOR_EVERY = 25
MONITOR_BATCH = 64
BATCH_SEARCH = (32, 48, 64, 96, 128, 192, 256)


def _smoke_cfg():
    cfg, cache = check(verbose=False)
    return cfg.model_copy(update={"smoke": True}), cache


def _sync(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _driver_mem(device: str) -> int:
    if device.startswith("mps"):
        return int(torch.mps.driver_allocated_memory())
    if device.startswith("cuda"):
        return int(torch.cuda.max_memory_allocated())
    return 0


def build_jepa(cfg, stamp_px: int, device: str, *, quiet: bool = False):
    from galaxy_jepa.harness import build_objective

    encoder = VisionTransformer(img_size=stamp_px, **cfg.model.model_kwargs())
    jepa = build_objective(cfg.to_jepa_config(), encoder)
    jepa.to(device)
    n_enc = sum(p.numel() for p in jepa.encoder.parameters())
    n_pred = sum(p.numel() for p in jepa.predictor.parameters())
    if not quiet:
        print(f"  model: ViT img={stamp_px} grid={jepa.encoder.grid_size} "
              f"encoder {n_enc/1e6:.1f}M + predictor {n_pred/1e6:.1f}M params on {device}",
              file=sys.stderr)
    return jepa, {"encoder_params": n_enc, "predictor_params": n_pred,
                  "grid_size": jepa.encoder.grid_size}


def _instrument(jepa) -> dict:
    """Time the objective's own host-side mask work — wrapped, not reimplemented."""
    acc = {"weight_maps_s": 0.0, "mask_sample_s": 0.0}
    real_maps, real_sample = jepa.weight_maps, jepa.masker.sample

    def timed_maps(*a, **k):
        t = time.perf_counter()
        try:
            return real_maps(*a, **k)
        finally:
            acc["weight_maps_s"] += time.perf_counter() - t

    def timed_sample(*a, **k):
        t = time.perf_counter()
        try:
            return real_sample(*a, **k)
        finally:
            acc["mask_sample_s"] += time.perf_counter() - t

    jepa.weight_maps = timed_maps
    jepa.masker.sample = timed_sample
    return acc


# --- phase: throughput / data-wait / collapse / loss ---------------------------------------


def phase_throughput() -> dict:
    cfg, cache = _smoke_cfg()
    device = cfg.runtime.resolved_device()
    ds, monitor_ids, dsmeta = build_production_dataset(cfg, cache)
    dataset_rss = rss_bytes()
    jepa, minfo = build_jepa(cfg, cache.index.height, device)
    jcfg = jepa.config
    bs = jcfg.batch_size
    # exactly as harness._prepare builds it: shuffled, no workers, no drop_last
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=0)
    opt = torch.optim.AdamW([*jepa.encoder.parameters(), *jepa.predictor.parameters()],
                            lr=jcfg.lr, weight_decay=jcfg.weight_decay)
    monitor = CollapseMonitor()
    # the real held-out pretrain-monitor slice, as harness._prepare uses
    mon_ds = StampDataset(ds.cache, ds.rows, monitor_ids[:MONITOR_BATCH])
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)

    host = _instrument(jepa)
    losses: list[float] = []
    trace: list[dict] = []
    data_s = compute_s = 0.0
    peak_mem = 0
    step = 0
    it = iter(loader)
    t_start = None
    print(f"  warmup {WARMUP} steps, then up to {MEASURE_STEPS} (cap {MEASURE_CAP_S}s) "
          f"at batch {bs}", file=sys.stderr)
    while step < WARMUP + MEASURE_STEPS:
        if step == WARMUP:
            _sync(device)
            for k in host:
                host[k] = 0.0
            data_s = compute_s = 0.0
            t_start = time.perf_counter()

        t0 = time.perf_counter()
        try:
            raw = next(it)
        except StopIteration:
            it = iter(loader)
            raw = next(it)
        batch = _to_device(raw, device)
        t1 = time.perf_counter()

        for g in opt.param_groups:
            g["lr"] = jcfg.lr * min(1.0, (step + 1) / max(jcfg.warmup_steps, 1))
        opt.zero_grad(set_to_none=True)
        loss = jepa.loss_step(batch, seed=jcfg.seed + step)
        loss.backward()
        opt.step()
        jepa.ema_update(ema_momentum(step, jcfg.steps, jcfg.ema_start, jcfg.ema_end))
        loss_val = float(loss.item())  # synchronises MPS
        _sync(device)
        t2 = time.perf_counter()

        if step >= WARMUP:
            data_s += t1 - t0
            compute_s += t2 - t1
            losses.append(loss_val)
            peak_mem = max(peak_mem, _driver_mem(device))

        if step % MONITOR_EVERY == 0:
            with torch.no_grad():
                emb = jepa.encoder.encode(mon_batch["image"].float())
            sig = monitor.update(step, emb)
            trace.append({"step": step, "loss": loss_val, "std": sig.std,
                          "effective_rank": sig.effective_rank, "mean_cosine": sig.mean_cosine})
            print(f"    step {step:>4}  loss {loss_val:.4f}  std {sig.std:.4f}  "
                  f"erank {sig.effective_rank:6.1f}  cos {sig.mean_cosine:+.3f}", file=sys.stderr)
        step += 1
        if t_start is not None and time.perf_counter() - t_start >= MEASURE_CAP_S:
            break

    measured = step - WARMUP
    wall = data_s + compute_s
    fifth = max(len(losses) // 5, 1)
    hm = host["weight_maps_s"] + host["mask_sample_s"]
    return {
        **minfo, "dataset": dsmeta, "device": device,
        "batch_size": bs, "warmup_steps_excluded": WARMUP, "measured_steps": measured,
        "wall_s": wall, "steps_per_s": measured / wall, "stamps_per_s": measured * bs / wall,
        "mb_per_s": measured * bs * BYTES_PER_STAMP / wall / 1e6,
        "data_wait_s": data_s, "compute_s": compute_s, "data_wait_fraction": data_s / wall,
        "host_mask_s": hm, "host_mask_fraction_of_compute": hm / compute_s,
        "host_weight_maps_s": host["weight_maps_s"], "host_mask_sample_s": host["mask_sample_s"],
        "peak_driver_bytes": peak_mem, "dataset_rss_bytes": dataset_rss,
        "peak_rss_bytes": rss_bytes(),
        "loss_first": float(np.mean(losses[:fifth])), "loss_last": float(np.mean(losses[-fifth:])),
        "loss_min": float(min(losses)), "loss_max": float(max(losses)),
        "loss_finite": bool(np.isfinite(losses).all()),
        "collapse_trace": trace, "loss_trace": [round(v, 6) for v in losses],
    }


# --- phase: largest batch that fits --------------------------------------------------------


def phase_batch() -> dict:
    """Largest batch that fits inside the MPS recommended working set, one real train step.

    Synthetic stamps, so this measures memory and not the drive. ``set_per_process_memory_fraction
    (1.0)`` caps the allocator at the recommended maximum so the search terminates by *raising*
    rather than by driving the machine into swap — "fits" on unified memory is otherwise a soft
    boundary that degrades instead of failing.
    """
    cfg, cache = _smoke_cfg()
    device = cfg.runtime.resolved_device()
    px = cache.index.height
    del cache
    gc.collect()
    if device.startswith("mps"):
        torch.mps.set_per_process_memory_fraction(1.0)
    rec = int(torch.mps.recommended_max_memory()) if device.startswith("mps") else 0
    rows: list[dict] = []
    largest = None
    for bs in BATCH_SEARCH:
        if device.startswith("mps"):
            torch.mps.empty_cache()
        jepa, _ = build_jepa(cfg, px, device, quiet=True)
        jcfg = jepa.config
        opt = torch.optim.AdamW([*jepa.encoder.parameters(), *jepa.predictor.parameters()],
                                lr=jcfg.lr, weight_decay=jcfg.weight_decay)
        batch = {"image": torch.randn(bs, 3, px, px, dtype=torch.float16).to(device),
                 "petro_rad_arcsec": torch.full((bs,), 8.0),
                 "pixel_scale": torch.full((bs,), 0.396)}
        try:
            t0 = time.perf_counter()
            for s in range(3):
                opt.zero_grad(set_to_none=True)
                loss = jepa.loss_step(batch, seed=s)
                loss.backward()
                opt.step()
                jepa.ema_update(0.996)
                float(loss.item())
            _sync(device)
            dt = (time.perf_counter() - t0) / 3
            mem = _driver_mem(device)
            largest = bs
            rows.append({"batch_size": bs, "fits": True, "driver_bytes": mem,
                         "s_per_step": dt, "stamps_per_s": bs / dt})
            print(f"    batch {bs:>4}: fits  {mem/1e9:5.2f} GB driver  {dt:6.3f} s/step  "
                  f"{bs/dt:6.1f} stamps/s", file=sys.stderr)
        except RuntimeError as exc:
            rows.append({"batch_size": bs, "fits": False, "error": str(exc)[:200]})
            print(f"    batch {bs:>4}: OOM — {str(exc)[:110]}", file=sys.stderr)
            del jepa, opt, batch
            break
        del jepa, opt, batch
        gc.collect()
    return {"recommended_max_bytes": rec, "largest_fitting": largest,
            "search_cap": max(BATCH_SEARCH), "rows": rows,
            "capped_not_oomed": largest == max(BATCH_SEARCH)}


# --- phase: MPS placement ------------------------------------------------------------------


def phase_placement() -> dict:
    """Is anything running on the CPU that should not be?

    ``PYTORCH_ENABLE_MPS_FALLBACK`` unset is the load-bearing fact: without it an op with no MPS
    kernel *raises* rather than silently relocating to the CPU, so a run that completes has not
    silently fallen back. Recorded rather than assumed, because setting it is exactly how a
    fallback hides.
    """
    cfg, cache = _smoke_cfg()
    device = cfg.runtime.resolved_device()
    px = cache.index.height
    del cache
    gc.collect()
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    jepa, _ = build_jepa(cfg, px, device, quiet=True)
    kind = device.split(":")[0]
    off = [n for n, p in jepa.named_parameters() if p.device.type != kind]
    bufs = [n for n, b in jepa.named_buffers() if b.device.type != kind]
    batch = {"image": torch.randn(4, 3, px, px, dtype=torch.float16).to(device),
             "petro_rad_arcsec": torch.full((4,), 8.0), "pixel_scale": torch.full((4,), 0.396)}
    loss = jepa.loss_step(batch, seed=0)
    loss.backward()
    return {
        "PYTORCH_ENABLE_MPS_FALLBACK": fallback,
        "fallback_enabled": fallback not in (None, "", "0"),
        "params_off_device": off, "buffers_off_device": bufs,
        "n_params": sum(1 for _ in jepa.parameters()), "loss_device": str(loss.device),
        "backward_completed": True,
        # deliberate host-side work, not a fallback: the masking math is numpy by design
        "host_by_design": ["Jepa.weight_maps (petrosian_box + token_weight_map, numpy)",
                           "MultiBlockMasker.sample (candidate scoring, numpy)",
                           "batch['petro_rad_arcsec'].cpu().numpy() in loss_step"],
    }


PHASES = {"placement": phase_placement, "throughput": phase_throughput, "batch": phase_batch}


def _spawn(phase: str) -> dict:
    proc = subprocess.run([sys.executable, __file__, phase], capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        return {"failed": True, "returncode": proc.returncode,
                "tail": (proc.stderr or proc.stdout).strip().splitlines()[-8:]}
    return json.loads(line[len("RESULT "):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=[*PHASES, "all"])
    args = ap.parse_args()
    if args.phase != "all":
        print("RESULT " + json.dumps(PHASES[args.phase]()))
        return

    cfg, _cache = _smoke_cfg()
    device = cfg.runtime.resolved_device()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = RunStamp.create(cfg.with_resolved_device().determining_dump(),
                           data_snapshot="f2-smoke", seed=cfg.seed, device=device,
                           escape_hatches_used=["smoke"])
    result = {"machine": _machine(), "device": device, "smoke": True,
              "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "config_hash_smoke": stamp.config_hash}
    del _cache
    gc.collect()

    for phase in ("placement", "throughput", "batch"):
        print(f"\nF2  {phase}")
        result[phase] = _spawn(phase)
        r = result[phase]
        if r.get("failed"):
            print(f"  FAILED rc={r['returncode']}: {r['tail'][-1] if r['tail'] else ''}")
            continue
        if phase == "placement":
            print(f"  PYTORCH_ENABLE_MPS_FALLBACK={r['PYTORCH_ENABLE_MPS_FALLBACK']!r} | "
                  f"{len(r['params_off_device'])}/{r['n_params']} params off device | "
                  f"loss on {r['loss_device']}")
        elif phase == "throughput":
            print(f"  sustained {r['steps_per_s']:.3f} steps/s  {r['stamps_per_s']:.1f} stamps/s "
                  f"({r['measured_steps']} steps, first {WARMUP} excluded)")
            print(f"  data-wait {r['data_wait_fraction']*100:.1f}% of wall | host mask "
                  f"{r['host_mask_fraction_of_compute']*100:.1f}% of compute | peak driver "
                  f"{r['peak_driver_bytes']/1e9:.2f} GB | rss {r['peak_rss_bytes']/1e9:.2f} GB")
            print(f"  loss {r['loss_first']:.4f} -> {r['loss_last']:.4f} "
                  f"(finite={r['loss_finite']})")
        else:
            print(f"  largest fitting batch {r['largest_fitting']} "
                  f"({'search cap, not an OOM' if r['capped_not_oomed'] else 'OOM above'})")

    tag = _machine()["cpu"].replace(" ", "-").lower() or "unknown"
    path = OUT / f"f2_{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    stamp_dir = OUT / f"smoke_{tag}"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    write_stamp(stamp, stamp_dir, cfg.model_dump(mode="json"))
    print(f"\nwrote {path}  (+ stamp under {stamp_dir}, escape_hatches_used=['smoke'])")


if __name__ == "__main__":
    main()
