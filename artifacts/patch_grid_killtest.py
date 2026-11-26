"""Planted kill test for the patch-grid launcher (user, 25 Nov 2026: "kill -9 during training, restart, identical continuation"). Mac, scratch dirs only.

    python artifacts/patch_grid_killtest.py test [--arms jepa_p16_s0 moco_p16_s0 mae_p16_s0] [--steps 60] [--every 20] [--device mps] --out DIR
    python artifacts/patch_grid_killtest.py run ARM OUT STEPS EVERY DEVICE      # one launcher invocation (what `test` spawns)

`run` is patch_grid_run.main() itself, with three things changed from outside and nothing edited: STOP is STEPS, GJ_CHECKPOINT_EVERY is EVERY, and the
two code-of-record gates are stubbed (pins: the Mac's branch is not the box commit; the cache: the Mac has no pulled manifest). Per arm:
  A  uninterrupted; A' uninterrupted again (the control: if A ≠ A' the device is not deterministic and A vs B can't be read);
  B  the same, SIGKILLed (the whole process group, loader workers too) after its first committed checkpoint and before the second, then the same command
     again, which must resume; C  uninterrupted with checkpoints every 1.5 × EVERY (GJ_CHECKPOINT_EVERY is outside the config hash). Identical = every tensor of the final checkpoint (modules, optimiser, RNG, losses, collapse history) and encoder.pt equal to
     A's, bit for bit. → OUT/killtest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))


def run(arm: str, out: str, steps: int, every: int, device: str) -> None:
    sys.modules["patch_grid_pins"] = types.SimpleNamespace(verify=lambda: None, verify_cache=lambda *a, **k: None)
    os.environ["GJ_CHECKPOINT_EVERY"] = str(every)
    import patch_grid_run as R
    R.STOP = steps
    sys.argv = ["patch_grid_run.py", arm, "--out", out, "--workers", "4", "--device", device]
    R.main()


def same(a, b, path="") -> list[str]:
    import torch
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [path + " (keys)"]
        return [d for k in a for d in same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path + " (len)"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return []
    return [] if a == b else [path]


def compare(x: Path, y: Path, steps: int) -> dict:
    import torch
    f = f"checkpoints/ckpt_{steps:09d}.pt"
    pa, pb = (torch.load(d / f, map_location="cpu", weights_only=False) for d in (x, y))
    diffs = same({k: v for k, v in pa.items()}, {k: v for k, v in pb.items()})
    ea, eb = (torch.load(d / "encoder.pt", map_location="cpu", weights_only=False) for d in (x, y))
    ta, tb = (json.loads((d / "traces.json").read_text()) for d in (x, y))
    return {"checkpoint_identical": not diffs, "differs_at": diffs[:12], "encoder_identical": not same(ea, eb),
            "losses_identical": ta["losses"] == tb["losses"], "steps": [len(ta["losses"]), len(tb["losses"])]}


def committed(d: Path) -> int:
    try:
        return max(int(e["step"]) for e in json.loads((d / "checkpoints/checkpoints.json").read_text())["entries"])
    except (OSError, ValueError, KeyError):
        return 0


def spawn(arm, out: Path, a, log: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, __file__, "run", arm, str(out), str(a.steps), str(a.every), a.device], cwd=REPO,
                            stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)


def test(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    base = vars(a).copy()
    for arm in a.arms:
        if a.every_by_objective:
            ev = dict(x.split("=") for x in a.every_by_objective.split(","))[arm.split("_")[0]]
            a = argparse.Namespace(**{**base, "every": int(ev), "steps": int(ev) + a.tail})
        d = {k: out / f"{arm}_{k}" for k in ("A", "A2", "B", "C")}
        for k in ("A", "A2"):
            t0 = time.time()
            assert spawn(arm, d[k], a, out / f"{arm}_{k}.log").wait() == 0, f"{arm} {k} failed; see its log"
            print(f"{arm} {k}: {time.time() - t0:.0f} s", flush=True)
        if not a.no_c:  # C: another checkpoint interval (60 = 2 × 30), so a different GJ_CHECKPOINT_EVERY is shown not to move the trajectory
            c_every = a.every * 3 // 2
            assert a.steps % c_every == 0
            assert spawn(arm, d["C"], argparse.Namespace(**{**vars(a), "every": c_every}), out / f"{arm}_C.log").wait() == 0, f"{arm} C failed"
        p = spawn(arm, d["B"], a, out / f"{arm}_B.log")
        while committed(d["B"]) < a.every:  # the first checkpoint is committed; now a few steps into the next interval
            assert p.poll() is None, f"{arm} B ended before its first checkpoint"
            time.sleep(0.2)
        t_ck = time.time()
        time.sleep(a.kill_after)
        st = committed(d["B"])
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        killed = {"committed_at_kill": st, "seconds_after_first_commit": round(time.time() - t_ck, 1), "traces_after_kill": (d["B"] / "traces.json").exists()}
        assert st < a.steps and not killed["traces_after_kill"], f"{arm}: the run finished before the kill; raise --steps"
        assert spawn(arm, d["B"], a, out / f"{arm}_B.log").wait() == 0, f"{arm} B's restart failed; see its log"
        log = (out / f"{arm}_B.log").read_text()
        m = re.search(r"resuming at step (\d+)", log)
        res[arm] = {"kill": killed, "resumed_at": int(m.group(1)) if m else None, "A_vs_A2": compare(d["A"], d["A2"], a.steps), "A_vs_B": compare(d["A"], d["B"], a.steps),
                    "A_vs_C_other_interval": None if a.no_c else compare(d["A"], d["C"], a.steps),
                    "checkpoint_write_s": [float(x) for x in re.findall(r"checkpoint step \d+: \d+ MB, sha \w+, ([\d.]+)s", log)]}
        print(arm, json.dumps(res[arm]), flush=True)
    ok = all(r["resumed_at"] == r["kill"]["committed_at_kill"] and all(r["A_vs_B"][k] for k in ("checkpoint_identical", "encoder_identical", "losses_identical")) for r in res.values())
    det = all(all(r["A_vs_A2"][k] for k in ("checkpoint_identical", "encoder_identical", "losses_identical")) for r in res.values())
    verdict = "IDENTICAL" if ok else ("DIFFERS (but A ≠ A' too: the device is not deterministic; rerun on cpu)" if not det else "DIFFERS")
    (out / "killtest.json").write_text(json.dumps({"verdict": verdict, "device": a.device, "steps": a.steps, "every": a.every, "every_by_objective": a.every_by_objective, "arms": res}, indent=1))
    print("kill test:", verdict)


def main() -> None:
    if sys.argv[1:2] == ["run"]:
        arm, out, steps, every, dev = sys.argv[2:7]
        return run(arm, out, int(steps), int(every), dev)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test"])
    ap.add_argument("--arms", nargs="+", default=["jepa_p16_s0", "moco_p16_s0", "mae_p16_s0"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--kill-after", type=float, default=2.0, help="seconds after the first commit")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--every-by-objective", help="e.g. mae=2000,jepa=3150,moco=3650: each arm's own interval (overrides --every); --steps is then every + --tail")
    ap.add_argument("--tail", type=int, default=300)
    ap.add_argument("--no-c", action="store_true")
    ap.add_argument("--out", required=True)
    test(ap.parse_args())


if __name__ == "__main__":
    main()
