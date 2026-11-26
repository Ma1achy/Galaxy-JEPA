"""Patch grid, stage 1 engineering checklist (docs/patch_grid/prereg.md, notes: outside the hash, so not pinned). Any failure stops the run before stage 2.

    python artifacts/patch_grid_stage1.py watch ARM --smoke 2000 [--masks DIR]   # launcher + 1-s GPU samples → runs/patch_grid/stage1/<arm>_smoke.json
    python artifacts/patch_grid_stage1.py watch ARM [--masks DIR]                # a real arm, the same way
    python artifacts/patch_grid_stage1.py flash ARM                             # bf16 forward/backward under a flash-only kernel context, CUDA
    python artifacts/patch_grid_stage1.py roundtrip ARM --run DIR               # checkpoint → live encoder; save, reload: identical embedding of 64 galaxies
    python artifacts/patch_grid_stage1.py judge                                 # the table; exit 1 on any failure or missing check

Bars (user, 25 Nov, after the hash; they replace the two-sided ±30% and the 85% utilisation gate): throughput at least 70% of the cost doc's lower estimate
(= the long end of its hours; faster always passes); peak memory fails only above 90% of the card; utilisation is reported and gates nothing; loss finite and
mean(steps 1,501–2,000) < mean(steps 1–500); flash active; checkpoint round trip identical on 64 galaxies.

Run the nine smokes one at a time on an idle GPU: a shared GPU confounds rate and utilisation. No probe, AUC or score is read here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
from patch_grid_run import OBJ, PATCHES, STOP, config, parse  # noqa: E402

OUT = REPO / "runs/patch_grid/stage1"
SLOW, CARD = 0.70, 0.90
#: docs/patch_ablation_cost.md, H100: hours per seed (patch 16's MoCo and MAE take the patch-8 figures, as the doc's third-seed table does)
HOURS = {("jepa", 16): (0.7, 1.1), ("moco", 16): (0.7, 1.3), ("mae", 16): (0.7, 1.2), ("jepa", 8): (0.7, 1.1), ("moco", 8): (0.7, 1.3), ("mae", 8): (0.7, 1.2),
         ("jepa", 4): (2.7, 5.3), ("moco", 4): (2.2, 4.6), ("mae", 4): (4.2, 8.3)}
#: the doc's peak GB at batch 32, for the report only. Patch 16 isn't in the doc: its memory model at a quarter of patch 8's tokens (1.5 + (p8 − 1.5)/4)
MEM = {("jepa", 8): 5.6, ("moco", 8): 5.4, ("mae", 8): 9.1, ("jepa", 4): 18.0, ("moco", 4): 17.0, ("mae", 4): 32.0}
MEM |= {(o, 16): 1.5 + (MEM[o, 8] - 1.5) / 4 for o in OBJ}


def sample(stop: threading.Event, rows: list, gpu: str) -> None:
    while not stop.is_set():
        try:
            q = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            u, m, t = q.stdout.strip().splitlines()[0].split(",")
            rows.append((time.time(), float(u), float(m), float(t)))
        except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(1.0)


def watch(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}{'_smoke' if a.smoke else ''}"
    run = REPO / "runs/patch_grid" / (f"smoke_{a.arm}" if a.smoke else a.arm)
    cmd = [sys.executable, str(REPO / "artifacts/patch_grid_run.py"), a.arm, "--workers", str(a.workers), "--out", str(run)]
    for k in ("cache_base", "masks"):
        if getattr(a, k):
            cmd += ["--" + k.replace("_", "-"), getattr(a, k)]
    cmd += ["--smoke", str(a.smoke)] if a.smoke else []
    rows, ev = [], threading.Event()
    th = threading.Thread(target=sample, args=(ev, rows, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]), daemon=True)
    th.start()
    with open(OUT / f"{tag}.log", "w") as log:
        p = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    end = time.time()
    ev.set()
    th.join()
    if p.returncode:
        raise SystemExit(f"{a.arm}: the launcher exited {p.returncode}; see {OUT / (tag + '.log')}")
    tr = json.loads((run / "traces.json").read_text())
    peak = re.search(r"peak cuda ([\d.]+) GB", (OUT / f"{tag}.log").read_text())
    secs, n = tr["seconds"], len(tr["losses"])
    lo = end - secs * (1 - 200 / max(n, 201))  # after step 200, at the run's mean rate
    u = [r[1] for r in rows if lo <= r[0] <= end - 10]
    # a resumed arm's seconds cover this invocation only, so its rate would be overstated: the judge refuses it
    resumed = "resuming at step" in (OUT / f"{tag}.log").read_text()
    rec = {"arm": a.arm, "smoke": bool(a.smoke), "steps": n, "seconds": secs, "steps_per_s": n / secs, "hours_per_seed": STOP / (n / secs) / 3600, "resumed": resumed,
           "peak_gb": float(peak.group(1)) if peak else None, "card_gb": rows[-1][3] * 1.048576e-3 if rows else None, "smi_used_gb": max(r[2] for r in rows) * 1.048576e-3 if rows else None, "util_mean": float(np.mean(u)) if u else None, "util_n": len(u),
           "halted": tr["halted"], "losses_2000": tr["losses"][:2000]}
    (OUT / f"{tag}.json").write_text(json.dumps(rec))
    print({k: v for k, v in rec.items() if k != "losses_2000"})


def objective(arm: str, dev: str):
    from galaxy_jepa.harness import build_objective, seed_init
    o, _, seed = parse(arm)
    cfg, _, _ = config(arm)
    ocfg = getattr(cfg.objective, f"to_{o}_config")(seed=seed)
    obj = build_objective(ocfg, seed_init(seed, 256, cfg.model.model_kwargs()))
    obj.to(dev)
    return cfg, ocfg, obj


def flash(a) -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.data import DataLoader
    from patch_grid_embed import dataset
    from galaxy_jepa.objectives.loop import _to_device
    if not torch.cuda.is_available():
        raise SystemExit("flash: needs CUDA (on a CPU the flash backend is another kernel)")
    cfg, ocfg, obj = objective(a.arm, "cuda")
    _, ds, _ = dataset(4)
    batch = _to_device(next(iter(DataLoader(ds, batch_size=4))), "cuda")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.autocast("cuda", dtype=cfg.autocast_dtype()):  # raises "No available kernel" if any attention can't use it
        loss = obj.loss_parts(batch, seed=0).total
        loss.backward()
    # the control: the same context must refuse fp32, or the check can't fail
    q = torch.zeros(1, 6, 16, 64, device="cuda")
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        fp32 = "accepted"
    except RuntimeError:
        fp32 = "refused"
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"arm": a.arm, "ok": bool(torch.isfinite(loss)), "fp32_under_flash_only": fp32, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (OUT / f"{a.arm}_flash.json").write_text(json.dumps(rec))
    print(rec)


def roundtrip(a) -> None:
    """Restore the run's newest checkpoint into a fresh objective; embed 64 union galaxies with the live encoder; save_encoder, load_frozen_encoder, embed again: bit-identical.
    Then the launcher's own encoder.pt against the same checkpoint."""
    import torch
    from patch_grid_embed import BATCH, dataset
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
    from galaxy_jepa.core.config import config_hash
    from galaxy_jepa.models.vit import load_frozen_encoder, save_encoder
    from galaxy_jepa.probing.extract import extract_matrix
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    o, patch, _ = parse(a.arm)
    cfg, ocfg, obj = objective(a.arm, dev)
    run = Path(a.run)
    if a.device:  # as the launcher: an overridden device is in the config hash
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": a.device})})
    if a.batch:  # a small-batch Mac smoke wrote its own hash
        cfg = cfg.model_copy(update={"objective": cfg.objective.model_copy(update={"batch_size": a.batch})})
        ocfg = getattr(cfg.objective, f"to_{o}_config")(seed=parse(a.arm)[2])
    ck = TrainCheckpointer(run / "checkpoints", every=1, keep=99, config_hash=config_hash(cfg.determining_dump()), normalisation_hash=cfg.normalisation.content_hash,
                           schedule={"steps": ocfg.steps, "lr": ocfg.lr, "lr_final": ocfg.lr_final, "warmup_steps": ocfg.warmup_steps,
                                     "batch_size": ocfg.batch_size, "objective": o, "brief": "patch grid"})
    st = ck.restore(objective=obj, optimiser=obj.make_optimiser(), map_location=dev)
    if st is None:
        raise SystemExit(f"{a.arm}: no committed checkpoint found under {run / 'checkpoints'}")
    _, ds, ids = dataset(64)
    bs = BATCH[patch]
    with tempfile.TemporaryDirectory() as t:
        save_encoder(obj.encoder, Path(t) / "e.pt")
        re_ = extract_matrix(load_frozen_encoder(Path(t) / "e.pt"), ds, device=dev, batch_size=bs).x
    for p in obj.encoder.parameters():
        p.requires_grad_(False)
    live = extract_matrix(obj.encoder.eval(), ds, device=dev, batch_size=bs).x
    own = extract_matrix(load_frozen_encoder(run / "encoder.pt"), ds, device=dev, batch_size=bs).x
    rec = {"arm": a.arm, "step": st.step, "n": len(ids), "identical": bool(np.array_equal(live, re_)), "encoder_pt_identical": bool(np.array_equal(own, live)),
           "spread": float(live.std(0).mean()), "device": dev}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.arm}_roundtrip.json").write_text(json.dumps(rec))
    print(rec)


def verdicts(arm: str, r: dict | None, fl: dict | None, rt: dict | None) -> dict[str, bool]:
    o, p, _ = parse(arm)
    hi = HOURS[o, p][1]
    r = r or {}
    x = np.asarray(r.get("losses_2000") or [np.nan], float)
    return {"throughput": bool(r) and not r["resumed"] and r["hours_per_seed"] <= hi / SLOW,
            "memory": bool(r) and r["peak_gb"] is not None and r["card_gb"] is not None and r["peak_gb"] <= CARD * r["card_gb"],
            "loss": bool(r) and len(x) == 2000 and bool(np.isfinite(x).all()) and x[1500:].mean() < x[:500].mean(),
            "flash": bool(fl) and fl["ok"] and fl["fp32_under_flash_only"] == "refused",
            "roundtrip": bool(rt) and rt["n"] == 64 and rt["identical"] and rt["encoder_pt_identical"] and rt["spread"] > 0 and rt["device"].startswith("cuda")}


def judge(a) -> None:
    load = lambda f: json.loads((OUT / f).read_text()) if (OUT / f).exists() else None  # noqa: E731
    bad = 0
    for o in OBJ:
        for p in PATCHES:
            arm = f"{o}_p{p}_s0"
            r = load(f"{arm}_smoke.json")
            v = verdicts(arm, r, load(f"{arm}_flash.json"), load(f"{arm}_roundtrip.json"))
            bad += not all(v.values())
            print(f"{arm:12s}", " ".join(f"{k}={'ok' if ok else 'FAIL'}" for k, ok in v.items()),
                  f"| {r['hours_per_seed']:.2f} h (doc {HOURS[o, p][0]}–{HOURS[o, p][1]}), {r['peak_gb']} GB (doc {MEM[o, p]:.1f}), util {r['util_mean']}" if r else "| no smoke")
    if not getattr(a, "smokes_only", False):
        for o in OBJ:
            for p in (16, 8):
                for s in (0, 1, 2):
                    arm = f"{o}_p{p}_s{s}"
                    r = load(f"{arm}.json")
                    if r:
                        v = {k: x for k, x in verdicts(arm, r, None, None).items() if k in ("throughput", "memory", "loss")}
                        bad += not all(v.values())
                        print(f"{arm:12s} real", " ".join(f"{k}={'ok' if ok else 'FAIL'}" for k, ok in v.items()), f"| {r['hours_per_seed']:.2f} h, {r['peak_gb']} GB, util {r['util_mean']}")
                    else:
                        bad += 1
                        print(f"{arm:12s} real: no record")
    if bad:
        raise SystemExit(f"stage 1 checklist: {bad} rows FAIL. Stop before stage 2 and report; nothing here is loosened on the box.")
    print("stage 1 checklist: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["watch", "flash", "roundtrip", "judge"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--smoke", type=int)
    ap.add_argument("--smokes-only", action="store_true", help="judge: check only the 9 smoke rows (Step 4, before stage-1 real arms exist)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-base")
    ap.add_argument("--masks")
    ap.add_argument("--run")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int)
    a = ap.parse_args()
    if a.mode != "judge" and not a.arm:
        raise SystemExit("name an arm")
    if a.cache_base:
        os.environ["F0_CACHE_BASE"] = a.cache_base
    globals()[a.mode](a)


if __name__ == "__main__":
    main()
