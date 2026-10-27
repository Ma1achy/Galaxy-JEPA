"""P1 (baselines_comparison.md, Common ground): the A side's epoch checkpoints, by exact resume.

A1–A3 kept a checkpoint every 2,000 steps, so the baselines' epoch points (12,664 / 25,327 / 50,654)
do not exist for them. This resumes an A run from one of its own checkpoints into a **scratch**
directory and stops at the target step. The A run's own directory is only read.

The construction is `m2_long_run.main`'s, line for line (config through F0, `_split_ids`, the
seeded init, `ResumableShuffle(train_seed)`, the checkpointer's schedule dict, `train_jepa` with
`resume` and `stop_after`), without the probe and without the curve: m2's resume path reads the run's
probe curve and logs its AUCs, and nothing here may see one. The package is the current checkout's,
where the baselines trained (build 2a's loop, proven identical to the loop A1–A3 ran by
tests/test_objective_loop.py); `--compare` is the empirical check that it reproduces A's trajectory.

    .venv/bin/python artifacts/p1_resume.py A1 --start 12000 --stop 14000 \
        --out runs/_p1/A1_identity --compare runs/m_v2/checkpoints/ckpt_000014000.pt
    .venv/bin/python artifacts/p1_resume.py A1 --start 12000 --stop 12664 --out runs/_p1/A1_e05

`--compare` requires every tensor of the training state (encoder, predictor, target encoder,
optimiser state, RNG states) and the loss history to equal the reference bit for bit; it prints
counts only and exits 3 on any difference. Writes `p1_record.json` in the scratch directory.
Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_run  # noqa: E402,F401  — A1's v2 environment (GJ_CONFIG, F0_*), before F0 reads it

import numpy as np  # noqa: E402
import torch  # noqa: E402
from aligned_train import RUNS  # noqa: E402
from f0_preconditions import check  # noqa: E402
from i2_sigreg_run import _split_ids  # noqa: E402
from m2_long_run import MONITOR_BATCH, _merge  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer  # noqa: E402
from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, train_jepa  # noqa: E402

EVERY = 2000  # A1–A3's GJ_CHECKPOINT_EVERY (aligned_launch.sh); observation, not recipe


def _sha(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _flat(x, prefix=""):
    """Every tensor and scalar in a nested payload, by path."""
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from _flat(x[k], f"{prefix}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):  # NumPy's RNG state (24 Oct: `==` crashed)
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and bool(np.array_equal(a, b)))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and bool(a == b)


def compare(got: Path, ref: Path) -> dict:
    keys = ("encoder", "predictor", "target_encoder", "optimiser", "rng", "losses", "step")
    g = torch.load(got, map_location="cpu", weights_only=False)
    r = torch.load(ref, map_location="cpu", weights_only=False)
    out = {"compared": list(keys), "n_leaves": 0, "n_differ": 0, "differ_paths": [], "missing": []}
    for k in keys:
        if k not in g or k not in r:
            out["missing"].append(k)
            continue
        fg, fr = dict(_flat(g[k], k)), dict(_flat(r[k], k))
        if set(fg) != set(fr):
            out["missing"].append(f"{k}: leaf sets differ ({len(fg)} vs {len(fr)})")
        for p in sorted(set(fg) & set(fr)):
            out["n_leaves"] += 1
            if not _same(fg[p], fr[p]):
                out["n_differ"] += 1
                if len(out["differ_paths"]) < 20:
                    out["differ_paths"].append(p)
    out["identical"] = out["n_differ"] == 0 and not out["missing"]
    return out


def amended(cmp: dict, start: Path, ref: Path) -> dict:
    """P1 as amended (user, 25 Oct): bit-identical training state (weights, optimiser, losses) and every
    random generator training draws from. A generator is shown not drawn from when the ORIGINAL run's
    state of it is unchanged from `start` to `ref` (its own checkpoints); only then may it differ."""
    s0 = torch.load(start, map_location="cpu", weights_only=False, mmap=True)["rng"]
    s1 = torch.load(ref, map_location="cpu", weights_only=False, mmap=True)["rng"]
    unused = sorted(k for k in s0 if k in s1 and dict(_flat(s0[k], f"rng/{k}")) .keys() == dict(_flat(s1[k], f"rng/{k}")).keys()
                    and all(_same(a, b) for a, b in zip(dict(_flat(s0[k])).values(), dict(_flat(s1[k])).values(), strict=True)))
    allowed = {f"rng/{k}" for k in unused}
    bad = [p for p in cmp["differ_paths"] if not any(p == a or p.startswith(a + "/") or p.startswith(a + "[") for a in allowed)]
    return {"generators_unchanged_in_original": unused, "differ_allowed": [p for p in cmp["differ_paths"] if p not in bad],
            "differ_not_allowed": bad, "differ_paths_truncated": cmp["n_differ"] > len(cmp["differ_paths"]),
            "passes": not bad and not cmp["missing"] and cmp["n_differ"] <= len(cmp["differ_paths"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=sorted(RUNS))
    ap.add_argument("--start", type=int, required=True, help="the A run's own checkpoint to resume")
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--out", required=True, help="scratch run directory (must not exist)")
    ap.add_argument("--compare", help="reference checkpoint the stop must equal bit for bit")
    ap.add_argument("--recompare", action="store_true",
                    help="no training: compare an existing scratch run's stop checkpoint (after the "
                         "24 Oct comparator crash, which came after training had finished)")
    a = ap.parse_args()
    if a.recompare:
        out, ref = Path(a.out), Path(a.compare)
        got = out / "checkpoints" / f"ckpt_{a.stop:09d}.pt"
        if not got.exists():
            raise SystemExit(f"p1: --recompare needs {got}")
        start = out / "checkpoints" / f"ckpt_{a.start:09d}.pt"
        rec = {"run": a.run, "source": RUNS[a.run].out_dir, "start": a.start, "stop": a.stop,
               "train_seed": RUNS[a.run].train_seed, "recompare": True,
               "start_sha1": _sha(start, "sha1"), "stop_sha1": _sha(got, "sha1"),
               "reference": str(ref), "reference_sha1": _sha(ref, "sha1"), "identity": compare(got, ref)}
        idt = rec["identity"]
        rec["amended"] = amended(idt, start, ref)
        print(f"p1 {a.run} amended P1: {'PASS' if rec['amended']['passes'] else 'FAIL'} "
              f"(allowed {rec['amended']['differ_allowed']}, not allowed {rec['amended']['differ_not_allowed']})", flush=True)
        print(f"p1 {a.run} identity: {idt['n_leaves']} leaves, {idt['n_differ']} differ, missing "
              f"{idt['missing']} -> {'IDENTICAL' if idt['identical'] else 'NOT IDENTICAL'}", flush=True)
        (out / "p1_record.json").write_text(json.dumps(rec, indent=1))
        sys.exit(0 if rec["amended"]["passes"] else 3)
    run = RUNS[a.run]
    src, out = Path(run.out_dir), Path(a.out)
    if out.exists():
        raise SystemExit(f"p1: {out} exists; a scratch run starts empty")
    if out.resolve().parent == src.resolve().parent and out.name == src.name:
        raise SystemExit("p1: refusing to write into the A run's own directory")
    man = json.loads((src / "checkpoints" / "checkpoints.json").read_text())
    entry = next((e for e in man["entries"] if int(e["step"]) == a.start), None)
    if entry is None:
        raise SystemExit(f"p1: {src} has no committed checkpoint at {a.start}")

    cfg, cache = check(verbose=False)
    train_seed = run.train_seed
    ch = config_hash(cfg.determining_dump())
    if ch != man["config_hash"]:
        raise SystemExit(f"p1: config_hash {ch[:16]} is not {a.run}'s {man['config_hash'][:16]}")
    jcfg = dataclasses.replace(cfg.to_jepa_config(), seed=train_seed)
    schedule = {"steps": jcfg.steps, "lr": jcfg.lr, "lr_final": jcfg.lr_final,
                "warmup_steps": jcfg.warmup_steps, "ema_start": jcfg.ema_start, "ema_end": jcfg.ema_end,
                "batch_size": jcfg.batch_size, "sigreg_lambda": jcfg.sigreg_lambda, "brief": "M"}
    if schedule != man["schedule"]:
        raise SystemExit(f"p1: schedule differs from {a.run}'s manifest")

    ck = out / "checkpoints"
    ck.mkdir(parents=True)
    shutil.copy2(src / "checkpoints" / entry["file"], ck / entry["file"])
    if _sha(ck / entry["file"], "sha256") != entry["sha256"]:
        raise SystemExit("p1: the copied checkpoint does not hash to its manifest entry")
    (ck / "checkpoints.json").write_text(json.dumps({
        "schema": man["schema"], "config_hash": man["config_hash"],
        "normalisation_hash": man["normalisation_hash"], "schedule": man["schedule"],
        "entries": [entry]}, indent=1))

    device = cfg.runtime.resolved_device()
    train_ids, monitor_ids, _ = _split_ids(cfg)
    ds = StampDataset(cache, {}, train_ids, scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=train_seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)
    jepa = build_objective(jcfg, seed_init(train_seed, cache.index.height, cfg.model.model_kwargs()))
    checkpointer = TrainCheckpointer(ck, every=EVERY, keep=10, config_hash=ch,
                                     normalisation_hash=cfg.normalisation.content_hash,
                                     schedule=schedule)
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=cache.scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)
    if device.startswith("mps"):
        torch.mps.empty_cache()
    print(f"p1 {a.run}: resume {a.start:,} -> {a.stop:,} (train seed {train_seed}, {device}) into {out}",
          flush=True)
    result = train_jepa(jepa, loader, device=device, monitor_batch=mon_batch,
                        checkpoint_path=out / "encoder.pt", autocast_dtype=cfg.autocast_dtype(),
                        checkpointer=checkpointer, sampler=sampler, collapse_floor=cfg.collapse_floor,
                        resume=True, stop_after=a.stop - a.start)
    if result.steps_completed != a.stop or result.halted:
        raise SystemExit(f"p1: landed at {result.steps_completed} (halted {result.halted}), not {a.stop}")
    (out / "traces.json").write_text(json.dumps({
        "collapse_trace": result.collapse_trace, "losses": _merge([], result.losses),
        "prediction_losses": _merge([], result.prediction_losses),
        "sigreg_losses": _merge([], result.sigreg_losses)}))
    got = ck / f"ckpt_{a.stop:09d}.pt"
    rec = {"run": a.run, "source": str(src), "start": a.start, "stop": a.stop, "train_seed": train_seed,
           "config_hash": ch, "device": device, "start_sha1": _sha(ck / entry["file"], "sha1"),
           "stop_sha1": _sha(got, "sha1")}
    code = 0
    if a.compare:
        ref = Path(a.compare)
        rec["reference"], rec["reference_sha1"] = str(ref), _sha(ref, "sha1")
        rec["identity"] = compare(got, ref)
        rec["amended"] = amended(rec["identity"], src / "checkpoints" / entry["file"], ref)
        code = 0 if rec["amended"]["passes"] else 3
        print(f"p1 {a.run} amended P1: {'PASS' if code == 0 else 'FAIL'} (allowed "
              f"{rec['amended']['differ_allowed']}, not allowed {rec['amended']['differ_not_allowed']})", flush=True)
        print(f"p1 {a.run} identity: {rec['identity']['n_leaves']} leaves, {rec['identity']['n_differ']} "
              f"differ, missing {rec['identity']['missing']} -> "
              f"{'IDENTICAL' if code == 0 else 'NOT IDENTICAL'}", flush=True)
    (out / "p1_record.json").write_text(json.dumps(rec, indent=1))
    print(f"p1 {a.run}: wrote {got.name} sha1 {rec['stop_sha1'][:12]}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
