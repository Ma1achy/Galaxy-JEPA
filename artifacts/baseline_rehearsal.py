"""The 500-step timing rehearsal for D12's baselines (baselines.md §6), on A1's v2 cache and freeze.

Three arms, each in its own child processes, into `runs/_rehearsal_baselines/<arm>/` (scratch, on
the SSD): the reference is A1's recipe (`m2_long_run.main`, as `rehearsal_run._train` drives it:
`configs/pretrain_v2.yaml`, train seed 0, `_schedule` replaced by the one absolute stop), MoCo and
MAE are `baseline_run.main` with `--stop` (seed 0, the arms that pair with A1). Per arm:

  straight   one child to step 500; the loop is untouched but for a per-step timestamp and MPS
             memory sample, taken where `train_objective` asks its step iterator for the next step.
             Every step already ends in `loss.item()`, a device sync, so the gaps are true step
             times. Steps/s is the median over steps 100–500 (all inside the 1,250-step warm-up:
             timing and plumbing, not learning).
  resume     one child to 250, a second resuming to 500 (as a crash recovery would: nothing shared
             in memory). The losses must equal the straight run's step for step. The second child
             also times the phases — data, augment/mask, forward, backward, optimiser, EMA — by
             wrapping each at its seam with a device sync on either side. A sync changes timing,
             never values, which the identity check itself confirms; that is why phases are taken
             here and not in the straight run whose steps/s is quoted. For MAE it also logs the
             sky/galaxy split of the masked-token loss every 50 steps (F5), from the same per-patch
             losses the step uses (no second forward), galaxy = inside A1's Petrosian box.

Nothing in `src/` is edited: the wrappers replace module globals (`loop._progress`, `loop._cycle`,
`loop._to_device`) and methods for the child's lifetime only.

**The pretext-ease gate (§6), fixed before the run.** MoCo's contrastive top-1, mean over steps
450–500 of the straight run: at or above GATE_TOP1 the pretext is too easy, and the sequence stops
before F7 for the user's decision on §3.4. Below it, the rehearsal passes the gate. The InfoNCE
loss is read against log(K + 1) = 11.09 (chance for K = 65,536 negatives).

Investigation code: terse, excluded from lint/CI.

    .venv/bin/python artifacts/baseline_rehearsal.py run        # all arms, then the summary
    .venv/bin/python artifacts/baseline_rehearsal.py summary    # re-read the records only
"""

from __future__ import annotations

import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "runs" / "_rehearsal_baselines"
STOP, SPLIT, FROM = 500, 250, 100
GATE_TOP1, GATE_WINDOW = 0.95, (450, 500)
SKY_EVERY = 50
ARMS = ("ref", "moco", "mae")
ENV = {  # A1's environment (aligned_launch.sh); baseline_run holds the first three itself
    "GJ_CONFIG": "configs/pretrain_v2.yaml",
    "F0_CACHE_BASE": "/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache",
    "F0_NORM_PREFIX": "246d8de6ba9a",
    "GJ_CHECKPOINT_EVERY": "2000",
    "PYTHONUNBUFFERED": "1",
}


# --- child: one arm, one segment ------------------------------------------------------------------


def _instrument(rec: dict, phases: bool, sky: bool) -> None:
    import torch

    import galaxy_jepa.objectives.loop as LP
    from galaxy_jepa.objectives import mae as MAE
    from galaxy_jepa.objectives import moco as MOCO
    from galaxy_jepa.objectives.jepa import Jepa
    from galaxy_jepa.masking.blocks import MultiBlockMasker

    mps = torch.backends.mps.is_available()
    sync = torch.mps.synchronize if mps else (lambda: None)
    rec.update(t=[], mps_alloc=[], mps_driver=[], top1=[])
    ph: dict[str, list[float]] = {}
    step_now = [-1]

    def progress(steps, *, start=0, desc=""):
        for s in range(start, steps):
            step_now[0] = s
            rec["t"].append([s, time.perf_counter()])
            if mps:
                rec["mps_alloc"].append(torch.mps.current_allocated_memory())
                rec["mps_driver"].append(torch.mps.driver_allocated_memory())
            yield s

    LP._progress = progress

    orig_top = MOCO.Moco.loss_parts

    def moco_parts(self, batch, *, seed=None):
        out = orig_top(self, batch, seed=seed)
        rec["top1"].append([step_now[0], self.last_top1])  # computed by the step itself
        return out

    MOCO.Moco.loss_parts = moco_parts

    if sky:
        rec["sky"] = []
        import yaml

        from galaxy_jepa.data.bbox import petrosian_box
        from galaxy_jepa.masking.blocks import box_to_token_mask
        from galaxy_jepa.objectives.loop import LossParts

        mk = yaml.safe_load((REPO / ENV["GJ_CONFIG"]).read_text())["objective"]
        k, frac = mk["petro_k"], mk["global_box_frac"]

        def mae_parts(self, batch, *, seed=None):
            per_patch, masked = self.token_losses(batch, seed=seed)
            loss = (per_patch * masked).sum() / masked.sum()  # MAE.loss_parts, verbatim
            s = step_now[0]
            if s % SKY_EVERY == 0 or s == STOP - 1:
                with torch.no_grad():
                    stamp = self.encoder.grid_size * self.encoder.patch_size
                    gal = []
                    for pr, ps in zip(batch["petro_rad_arcsec"].cpu().numpy(),
                                      batch["pixel_scale"].cpu().numpy(), strict=True):
                        box = petrosian_box(float(pr) if math.isfinite(pr) else None, float(ps),
                                            k=k, stamp_px=stamp, global_half_width_px=frac * stamp)
                        gal.append(box_to_token_mask(box.half_width_px, stamp,
                                                     self.encoder.grid_size).reshape(-1))
                    g = torch.as_tensor(__import__("numpy").stack(gal), dtype=per_patch.dtype,
                                        device=per_patch.device)
                    pp, m = per_patch.detach(), masked
                    mg, ms = m * g, m * (1 - g)
                    rec["sky"].append({
                        "step": s, "loss": float(loss.detach()),
                        "galaxy_loss": float((pp * mg).sum() / mg.sum().clamp_min(1)),
                        "sky_loss": float((pp * ms).sum() / ms.sum().clamp_min(1)),
                        "sky_share_of_masked_loss": float((pp * ms).sum() / (pp * m).sum()),
                        "sky_token_frac_of_masked": float(ms.sum() / m.sum()),
                    })
            return LossParts(total=loss, prediction=loss, sigreg=None)

        MAE.Mae.loss_parts = mae_parts

    if not phases:
        return
    rec["phases"] = ph

    def timed(name, fn, *, gpu=True):
        def w(*a, **kw):
            if gpu:
                sync()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if gpu:
                sync()
            ph.setdefault(name, []).append(time.perf_counter() - t0)
            return out
        return w

    orig_cycle = LP._cycle

    def cycle(loader):
        it = orig_cycle(loader)
        while True:
            t0 = time.perf_counter()
            b = next(it)
            ph.setdefault("data_load", []).append(time.perf_counter() - t0)
            yield b

    LP._cycle = cycle
    LP._to_device = timed("data_to_device", LP._to_device)
    MOCO.Moco.views = timed("augment", MOCO.Moco.views)
    MAE.uniform_mask = timed("mask", MAE.uniform_mask, gpu=False)
    MultiBlockMasker.sample = timed("mask", MultiBlockMasker.sample, gpu=False)
    Jepa.weight_maps = timed("mask_weights", Jepa.weight_maps, gpu=False)
    for cls in (MOCO.Moco, MAE.Mae, Jepa):
        lp = cls.loss_parts
        cls.loss_parts = timed("forward_incl_augment_mask", lp)
        cls.momentum_update = timed("ema", cls.momentum_update)
        mo = cls.make_optimiser

        def make(self, _mo=mo):
            opt = _mo(self)
            opt.step = timed("optimiser", opt.step)
            return opt

        cls.make_optimiser = make
    tb = torch.Tensor.backward
    torch.Tensor.backward = timed("backward", tb)


def child(arm: str, out: str, stop: int, phases: bool, sky: bool) -> None:
    sys.path.insert(0, str(REPO / "artifacts"))
    rec: dict = {"arm": arm, "out": out, "stop": stop, "phases": phases}
    _instrument(rec, phases, sky)
    t0 = time.perf_counter()
    if arm == "ref":
        import m2_long_run as m2

        m2._schedule = lambda _pe, total, _ep: [min(stop, total)]
        sys.argv = [m2.__file__, "--skip-preflight", "--no-probe"]
        m2.main(m2.Run(label="REF", tag="_rehearsal_ref", train_seed=0, out_dir=out))
    else:
        import baseline_run

        sys.argv = [baseline_run.__file__, f"{arm}0", "--stop", str(stop), "--out", out]
        baseline_run.main()
    rec["wall_s"] = time.perf_counter() - t0
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    p = Path(out) / f"child_{stop}.json"
    p.write_text(json.dumps(rec))
    print(f"child {arm} -> {stop}: {rec['wall_s']:.0f} s, record {p}", file=sys.stderr)


# --- parent ---------------------------------------------------------------------------------------


def _spawn(arm: str, out: Path, stop: int, *, phases=False, sky=False) -> None:
    cmd = [sys.executable, __file__, "child", arm, str(out), str(stop)]
    cmd += ["--phases"] * phases + ["--sky"] * sky
    subprocess.run(cmd, cwd=REPO, env={**os.environ, **ENV}, check=True)


def run() -> None:
    if ROOT.exists():
        raise SystemExit(f"{ROOT} exists; move it aside — a rehearsal starts from nothing")
    busy = subprocess.run(["pgrep", "-fl", "sciserver_pull|bake|ladder_v2_run|baseline_run"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        raise SystemExit(f"rehearsal: another data/heavy job is running (§6):\n{busy}")
    n, k = len(ARMS) * 3, 0
    for arm in ARMS:
        for sub, stop, kw in (("straight", STOP, {}), ("resume", SPLIT, {}),
                              ("resume", STOP, {"phases": True, "sky": arm == "mae"})):
            k += 1
            print(f"=== {k}/{n} {arm}.{sub}.{stop}", flush=True)
            _spawn(arm, ROOT / arm / sub, stop, **kw)
    summary()


def _losses(d: Path) -> list[float]:
    return json.loads((d / "traces.json").read_text())["losses"]


def _arm_summary(arm: str) -> dict:
    s = json.loads((ROOT / arm / "straight" / f"child_{STOP}.json").read_text())
    r = json.loads((ROOT / arm / "resume" / f"child_{STOP}.json").read_text())
    t = dict(s["t"])
    dur = [t[i + 1] - t[i] for i in range(FROM, STOP - 1) if i in t and i + 1 in t]
    out: dict = {
        "steps_per_s_median": 1 / statistics.median(dur),
        "steps_per_s_mean": len(dur) / sum(dur),
        "n_step_gaps": len(dur),
        "straight_wall_s": s["wall_s"],
        "peak_rss_gb": s["peak_rss_bytes"] / 2**30,
        "mps_alloc_peak_gb": max(s["mps_alloc"], default=0) / 2**30,
        "mps_driver_peak_gb": max(s["mps_driver"], default=0) / 2**30,
    }
    a, b = _losses(ROOT / arm / "straight"), _losses(ROOT / arm / "resume")
    n = min(len(a), len(b))
    diff = [abs(x - y) for x, y in zip(a[:n], b[:n], strict=True)]
    out["resume_identity"] = {
        "steps_compared": n, "identical": n == STOP and all(d == 0 for d in diff),
        "max_abs_diff": max(diff, default=math.nan),
        "first_diff_step": next((i for i, d in enumerate(diff) if d != 0), None),
    }
    out["phases_ms_median"] = {k: 1e3 * statistics.median(v) for k, v in r.get("phases", {}).items()}
    tr = json.loads((ROOT / arm / "straight" / "traces.json").read_text())
    out["collapse_last"] = {k: v[-1] for k, v in tr["collapse_trace"].items() if v}
    out["loss_first_last"] = [a[0], statistics.fmean(a[-50:])]
    if arm == "moco":
        top = dict(s["top1"])
        lo, hi = GATE_WINDOW
        w = [top[i] for i in range(lo, hi) if i in top]
        out["infonce_mean_last50"] = statistics.fmean(a[-50:])
        out["log_K_plus_1"] = math.log(65_536 + 1)
        out["top1_by_100"] = {f"{i}-{i + 100}": statistics.fmean(
            [top[j] for j in range(i, i + 100) if j in top]) for i in range(0, STOP, 100)}
        out["top1_gate_mean"] = statistics.fmean(w)
        out["gate"] = "STOP: pretext too easy" if out["top1_gate_mean"] >= GATE_TOP1 else "pass"
    if arm == "mae":
        out["sky_split"] = r.get("sky", [])
    return out


def summary() -> None:
    res = {arm: _arm_summary(arm) for arm in ARMS if (ROOT / arm / "resume" / f"child_{STOP}.json").exists()}
    ref = res.get("ref", {}).get("steps_per_s_median")
    for arm, r in res.items():
        if ref:
            r["speed_vs_ref"] = r["steps_per_s_median"] / ref
        r["h_per_5000"] = 5000 / r["steps_per_s_median"] / 3600
        r["h_per_101308"] = 101_308 / r["steps_per_s_median"] / 3600
    rec = {"rehearsal": "baselines.md §6", "smoke": True, "stop": STOP, "resume_split": SPLIT,
           "median_from_step": FROM, "gate_top1": GATE_TOP1, "gate_window": GATE_WINDOW,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": res}
    (ROOT / "rehearsal.json").write_text(json.dumps(rec, indent=1))
    for arm, r in res.items():
        ri = r["resume_identity"]
        print(f"{arm:5s} {r['steps_per_s_median']:.3f} steps/s (median {FROM}–{STOP}; "
              f"x{r.get('speed_vs_ref', math.nan):.2f} ref)  5k: {r['h_per_5000']:.2f} h  "
              f"101,308: {r['h_per_101308']:.1f} h  RSS {r['peak_rss_gb']:.2f} GB  "
              f"MPS {r['mps_alloc_peak_gb']:.2f}/{r['mps_driver_peak_gb']:.2f} GB  "
              f"resume {'IDENTICAL' if ri['identical'] else 'DIFFERS at ' + str(ri['first_diff_step'])}")
        print(f"      phases ms: " + ", ".join(f"{k} {v:.1f}" for k, v in r["phases_ms_median"].items()))
        print(f"      collapse: " + ", ".join(f"{k} {v:.3g}" for k, v in r["collapse_last"].items()))
        if arm == "moco":
            print(f"      InfoNCE {r['infonce_mean_last50']:.3f} vs log(K+1) {r['log_K_plus_1']:.2f}; "
                  f"top-1 {r['top1_by_100']}; gate mean {r['top1_gate_mean']:.3f} -> {r['gate']}")
        if arm == "mae" and r["sky_split"]:
            last = r["sky_split"][-1]
            print(f"      sky/galaxy at step {last['step']}: galaxy {last['galaxy_loss']:.4f}, "
                  f"sky {last['sky_loss']:.4f}, sky share of masked loss "
                  f"{last['sky_share_of_masked_loss']:.3f} (token share {last['sky_token_frac_of_masked']:.3f})")
    print(f"record: {ROOT / 'rehearsal.json'}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), "--phases" in sys.argv, "--sky" in sys.argv)
    elif mode == "run":
        run()
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(f"mode: run | summary | child, not {mode!r}")
