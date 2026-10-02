"""v2 rehearsal — M's recipe, cut at an absolute step, on the 20k scratch corpus.

Not a run. It walks the v2 chain end to end (freeze -> bake -> scalars -> F0 -> train -> stop ->
resume) at a size that costs minutes, and times the loop on the new corpus, before the 827k
corpus commits days. Nothing is reimplemented: training is `m2_long_run.main` with M's recipe —
seed 0, `steps` 253,270, so the LR warmup/cosine and the EMA ramp are M's step for step — and
the only thing moved is WHERE this invocation stops. m2 derives its stop points from probe
epochs; here `_schedule` is replaced by the one absolute step asked for, which is exactly the
`--test-points` path m2 already proved (segmenting changes when the process exits, not the recipe).

M's pre-flight is skipped deliberately: it asserts M's own hashes, which the rehearsal
config (a different freeze, so a different `config_hash`) cannot and should not reproduce.

Modes (the config is GJ_CONFIG, default configs/_rehearsal.yaml; F0_CACHE_BASE and
F0_NORM_PREFIX must name the rehearsal cache and freeze for anything that trains):

    rehearsal_run.py [--stop 500]    train to the absolute step, --no-probe
    rehearsal_run.py --plan          m2's budget print; launches nothing
    rehearsal_run.py --resume-test   250 then 500 in TWO child processes, into <out_dir>/resume,
                                     then the losses must equal the straight run's step for step
    rehearsal_run.py --schedules     LR and EMA at the named steps, loaded config vs pretrain.yaml
    rehearsal_run.py --config-diff   leaf diff vs configs/pretrain.yaml + both config_hashes
    rehearsal_run.py --stop-dry-test M ONLY: copy M's last checkpoint <= 101,300 to runs/_stopdry,
                                     resume M's own config/cache, stop at 101,308, assert the
                                     stop landed; never writes into runs/m

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
M_CONFIG = "configs/pretrain.yaml"
DEFAULT_CONFIG = "configs/_rehearsal.yaml"
SCHEDULE_STEPS = (0, 100, 500, 101_300, 101_308)
RESUME_SPLIT = 250

# --stop-dry-test: M's checkpoints, and a stop that is not on the checkpoint cadence, so only the
# stop can have written it
M_RUN = REPO / "runs" / "m"
STOPDRY = "runs/_stopdry"
DRY_FROM, DRY_STOP = 101_300, 101_308
_M_ENV = ("GJ_CONFIG", "F0_CACHE_BASE", "F0_NORM_PREFIX")


def _raw(path: str) -> dict:
    return yaml.safe_load((REPO / path).read_text())


def _harness(path: str):
    from galaxy_jepa.harness import HarnessConfig

    raw = _raw(path)
    if isinstance(raw.get("normalisation"), str):
        raise SystemExit(f"{path}: normalisation is still the placeholder — paste the E5 freeze")
    return HarnessConfig(**raw).with_resolved_device()


def _flat(d, pre: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _train(label: str, tag: str, out_dir: str, stop: int) -> float:
    """One invocation of m2's loop, to absolute step `stop`. Env must already be set: f0 reads it
    at import, which is why m2 is imported here and not at the top."""
    import m2_long_run as m2

    m2._schedule = lambda _per_epoch, total, _epochs: [min(stop, total)]
    sys.argv = [str(Path(m2.__file__)), "--skip-preflight", "--no-probe"]
    t0 = time.perf_counter()
    m2.main(m2.Run(label=label, tag=tag, out_dir=out_dir))
    wall = time.perf_counter() - t0
    print(f"\n{label}: invocation wall-clock {wall:.1f} s (setup + loop + checkpoint + encoder)")
    return wall


def _rehearsal_env() -> tuple[str, str]:
    """The rehearsal config and its out_dir, refusing anything that could land on M's."""
    os.environ.setdefault("GJ_CONFIG", DEFAULT_CONFIG)
    cfg = os.environ["GJ_CONFIG"]
    if Path(cfg).resolve() == (REPO / M_CONFIG).resolve():
        raise SystemExit("GJ_CONFIG is configs/pretrain.yaml — that is M; use --stop-dry-test")
    missing = [k for k in ("F0_CACHE_BASE", "F0_NORM_PREFIX") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set {missing}: without them F0 checks M's cache and freeze, not this one")
    out = _raw(cfg)["paths"]["out_dir"]
    if not Path(out).name.startswith("_rehearsal"):
        raise SystemExit(f"{cfg} out_dir is {out}; a rehearsal writes only to runs/_rehearsal*")
    return cfg, out


def _losses(out: Path) -> list[float]:
    return json.loads((out / "traces.json").read_text())["losses"]


def _latest(ckdir: Path) -> dict:
    m = json.loads((ckdir / "checkpoints.json").read_text())
    return max(m["entries"], key=lambda e: int(e["step"]))


# --- modes ---------------------------------------------------------------------------------


def schedules() -> None:
    from galaxy_jepa.objectives.jepa import ema_momentum, learning_rate

    cfg = os.environ.get("GJ_CONFIG", DEFAULT_CONFIG)
    j, m = _harness(cfg).to_jepa_config(), _harness(M_CONFIG).to_jepa_config()
    print(f"{'step':>9} {'lr (' + Path(cfg).name + ')':>24} {'lr (M)':>12} "
          f"{'ema (' + Path(cfg).name + ')':>26} {'ema (M)':>18}")
    same = True
    for s in SCHEDULE_STEPS:
        lr, lr_m = learning_rate(s, j), learning_rate(s, m)
        e = ema_momentum(s, j.steps, j.ema_start, j.ema_end)
        e_m = ema_momentum(s, m.steps, m.ema_start, m.ema_end)
        same &= lr == lr_m and e == e_m
        print(f"{s:>9,} {lr:>24.6e} {lr_m:>12.6e} {e:>26.12f} {e_m:>18.12f}")
    print(f"\nsteps={j.steps:,} warmup={j.warmup_steps} lr={j.lr} lr_final={j.lr_final} "
          f"ema {j.ema_start}->{j.ema_end}\nschedules {'IDENTICAL to M' if same else 'DIFFER from M'}")
    if not same:
        raise SystemExit("the rehearsal's schedule is not M's")


def config_diff() -> None:
    from galaxy_jepa.core.config import config_hash

    cfg = os.environ.get("GJ_CONFIG", DEFAULT_CONFIG)
    a, b = _flat(_raw(M_CONFIG)), _flat(_raw(cfg))
    print(f"--- {M_CONFIG}\n+++ {cfg}")
    for k in sorted(a.keys() | b.keys()):
        if a.get(k, "<absent>") != b.get(k, "<absent>"):
            print(f"  {k}\n    - {a.get(k, '<absent>')!r:.160}\n    + {b.get(k, '<absent>')!r:.160}")
    m_manifest = json.loads((M_RUN / "checkpoints" / "checkpoints.json").read_text())["config_hash"]
    for name in (M_CONFIG, cfg):
        try:
            h = config_hash(_harness(name).determining_dump())
        except SystemExit as exc:
            print(f"config_hash {name}: unavailable — {exc}")
            continue
        tag = "  == runs/m manifest" if h == m_manifest else ""
        print(f"config_hash {name}: v2:{h}{tag}")
    print(f"runs/m manifest records   : {m_manifest}")


def rehearse(stop: int, sub: str | None) -> None:
    _, out = _rehearsal_env()
    out_dir = f"{out}/{sub}" if sub else out
    _train("REHEARSAL", "_rehearsal", out_dir, stop)
    latest = _latest(Path(out_dir) / "checkpoints")
    if int(latest["step"]) != stop:
        raise SystemExit(f"stop at {stop} asked, latest checkpoint is {latest['step']}")
    print(f"REHEARSAL: landed at step {stop}: {latest['file']}")


def resume_test() -> None:
    _, out = _rehearsal_env()
    sub = Path(out) / "resume"
    if sub.exists():
        raise SystemExit(f"{sub} exists — delete it; the test must start from nothing")
    me = [sys.executable, str(Path(__file__).resolve())]
    # two child processes, one after the other: the second restarts from the manifest exactly as a
    # crash recovery would, sharing nothing in memory with the first
    for stop in (RESUME_SPLIT, 500):
        print(f"\n=== resume test: child process to step {stop} ===", flush=True)
        subprocess.run([*me, "--stop", str(stop), "--sub", "resume"], cwd=str(REPO), check=True)
    files = sorted(p.name for p in (sub / "checkpoints").glob("ckpt_*.pt"))
    want = [f"ckpt_{RESUME_SPLIT:09d}.pt", f"ckpt_{500:09d}.pt"]
    if files != want:
        raise SystemExit(f"resume test wrote {files}, expected {want}")
    got = _losses(sub)
    straight = Path(out) / "traces.json"
    if not straight.exists() or len(_losses(Path(out))) < 500:
        print(f"resume test: segments landed ({files}); no straight 500-step run at {out} to "
              "compare against — run the default mode first for the step-for-step check")
        return
    ref = _losses(Path(out))[:500]
    diff = [abs(x - y) for x, y in zip(got[:500], ref, strict=True)]
    nan = sum(math.isnan(d) for d in diff)
    worst = max((d for d in diff if not math.isnan(d)), default=0.0)
    same = sum(d == 0.0 for d in diff)
    print(f"resume test: 250 + 250 in two processes vs 500 straight — {same}/500 losses "
          f"bit-identical, worst |diff| {worst:.3e}, {nan} NaN")
    if nan or worst:
        raise SystemExit("resume is NOT step-for-step identical to the straight run")
    print("resume test PASS")


def _m_listing() -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(M_RUN)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in M_RUN.rglob("*") if p.is_file()}


def stop_dry_test() -> None:
    set_ = [k for k in _M_ENV if os.environ.get(k)]
    if set_:
        raise SystemExit(f"--stop-dry-test runs M's own config and cache; unset {set_}")
    scratch = REPO / STOPDRY
    if scratch.exists():
        raise SystemExit(f"{scratch} exists — delete it; the test must start from one checkpoint")
    if scratch.resolve().is_relative_to(M_RUN.resolve()):
        raise SystemExit("scratch dir is inside runs/m")
    before = _m_listing()

    src = M_RUN / "checkpoints"
    manifest = json.loads((src / "checkpoints.json").read_text())
    entry = max((e for e in manifest["entries"] if int(e["step"]) <= DRY_FROM),
                key=lambda e: int(e["step"]))
    ck = scratch / "checkpoints"
    ck.mkdir(parents=True)
    print(f"STOPDRY: copying {entry['file']} (step {entry['step']:,}, "
          f"{entry['bytes'] / 1e6:.0f} MB) -> {ck}")
    shutil.copyfile(src / entry["file"], ck / entry["file"])
    if _sha256(ck / entry["file"]) != entry["sha256"]:
        raise SystemExit("the copy does not hash to M's manifest entry")
    (ck / "checkpoints.json").write_text(json.dumps({**manifest, "entries": [entry]}, indent=2))

    _train("STOPDRY", "_stopdry", STOPDRY, DRY_STOP)

    import torch

    latest = _latest(ck)
    files = sorted(p.name for p in ck.glob("ckpt_*.pt"))
    want = f"ckpt_{DRY_STOP:09d}.pt"
    if int(latest["step"]) != DRY_STOP or latest["file"] != want or files[-1] != want:
        raise SystemExit(f"last checkpoint is {latest['file']} (files {files}), expected {want}")
    enc = torch.load(scratch / "encoder.pt", map_location="cpu", weights_only=False)
    if int(enc["extra"]["steps"]) != DRY_STOP:
        raise SystemExit(f"encoder.pt says steps={enc['extra']['steps']}, expected {DRY_STOP}")
    ours = torch.load(ck / want, map_location="cpu", weights_only=False, mmap=True)["encoder"]
    if any(not torch.equal(enc["state_dict"][k], v) for k, v in ours.items()):
        raise SystemExit(f"encoder.pt weights are not {want}'s encoder")
    print(f"STOPDRY: stop landed — {files}; encoder.pt is step {DRY_STOP:,} and equals {want}")

    # M ran 101,296 -> 101,308 uninterrupted, so its own 101,308 is a free resume-identity
    # reference. Reported, not asserted: it was a different process on a different day.
    theirs = torch.load(src / want, map_location="cpu", weights_only=False, mmap=True)["encoder"]
    worst = max(float((ours[k].float() - theirs[k].float()).abs().max()) for k in ours)
    lo, hi = int(entry["step"]), DRY_STOP
    a, b = _losses(scratch)[lo:hi], _losses(M_RUN)[lo:hi]
    same = sum(x == y for x, y in zip(a, b, strict=True))
    print(f"STOPDRY vs runs/m {want}: encoder max |diff| {worst:.3e}; "
          f"losses {lo:,}..{hi - 1:,}: {same}/{hi - lo} bit-identical")

    if _m_listing() != before:
        raise SystemExit("runs/m CHANGED during the stop dry test")
    print("STOPDRY: runs/m untouched (sizes and mtimes identical). PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true")
    g.add_argument("--resume-test", action="store_true")
    g.add_argument("--schedules", action="store_true")
    g.add_argument("--config-diff", action="store_true")
    g.add_argument("--stop-dry-test", action="store_true")
    ap.add_argument("--stop", type=int, default=500, help="absolute step this invocation ends at")
    ap.add_argument("--sub", default=None, help=argparse.SUPPRESS)  # the resume test's children
    args = ap.parse_args()

    # every path in m2, the configs and the corpora is REPO-relative through the cwd
    if Path.cwd().resolve() != REPO.resolve():
        raise SystemExit(f"run from {REPO}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.schedules:
        schedules()
    elif args.config_diff:
        config_diff()
    elif args.stop_dry_test:
        stop_dry_test()
    elif args.resume_test:
        resume_test()
    elif args.plan:
        _, out = _rehearsal_env()
        import m2_long_run as m2

        m2._schedule = lambda _per_epoch, total, _epochs: [min(args.stop, total)]
        sys.argv = [str(Path(m2.__file__)), "--plan"]
        m2.main(m2.Run(label="REHEARSAL", tag="_rehearsal", out_dir=out))
    else:
        rehearse(args.stop, args.sub)


if __name__ == "__main__":
    main()
