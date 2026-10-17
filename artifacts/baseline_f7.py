"""F7's sanity check (baselines.md §8 F7, §5.3 rows 2–3): each baseline at 1/3×, 1× and 3× its rule's
rate for 5,000 steps, from the seed-0 init and data order, `smoke: true`, scratch out_dirs.

Each arm is `baseline_run.py <arm>0 --lr <rate> --stop 5000 --out runs/_f7/<arm>_<rate>`, one child
process at a time, MoCo's three before MAE's (§5.3: the cheaper arm first). No probe runs.

**The choice, fixed here before any F7 run.** F7: the seed runs keep the rule's rate (1.875e-5 for
both arms) unless that rate is unstable; then the nearest stable of the three, stated. Chosen by
pretext loss only (MoCo's InfoNCE, MAE's masked-patch loss), never a probe. A rate is UNSTABLE if:
  - the run halted: a non-finite loss, or the hard collapse floor (F8; train_objective halts on
    either), or it did not reach step 5,000; or
  - its loss diverged: the final 500-step mean exceeds the lowest 500-step rolling mean of the same
    run by more than DIVERGE (50%) — the loss went down and then blew up.
"Nearest" is by rate ratio; both neighbours of 1× are 3× away, so a tie goes to the lower rate. If
the rule's rate is unstable the sequence stops for the user: the seed runs read the config's rate,
and departing from it is a config change, not a flag.

Recorded (runs/_f7/f7.json): per rate, the loss curve (from traces.json), the final 500-step mean,
the lowest rolling mean, halted or not, the steps reached, and the choice per arm.

Investigation code: terse, excluded from lint/CI.

    .venv/bin/python artifacts/baseline_f7.py run       # all six, then the summary
    .venv/bin/python artifacts/baseline_f7.py summary
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "runs" / "_f7"
RULE = 1.875e-5
RATES = (RULE / 3, RULE, RULE * 3)  # 6.25e-6, 1.875e-5, 5.625e-5 (§8 F7)
STEPS, WINDOW, DIVERGE = 5_000, 500, 0.5
ARMS = ("moco", "mae")


def _dir(arm: str, lr: float) -> Path:
    return ROOT / f"{arm}_{lr:.3e}"


def run() -> None:
    busy = subprocess.run(["pgrep", "-fl", "sciserver_pull|bake|ladder_v2_run|baseline_run"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        raise SystemExit(f"f7: another heavy job is running:\n{busy}")
    n, k = len(ARMS) * len(RATES), 0
    for arm in ARMS:
        for lr in (RULE, RULE / 3, RULE * 3):  # the rule's rate first
            k += 1
            d = _dir(arm, lr)
            print(f"=== {k}/{n} {arm}@{lr:.3e}", flush=True)
            r = subprocess.run([sys.executable, "artifacts/baseline_run.py", f"{arm}0", "--lr", repr(lr),
                                "--stop", str(STEPS), "--out", str(d)], cwd=REPO,
                               env={**os.environ, "GJ_CHECKPOINT_EVERY": "2500", "PYTHONUNBUFFERED": "1"})
            print(f"    {arm}@{lr:.3e}: exit {r.returncode}", flush=True)  # a failed arm is recorded, not fatal
    summary()


def _judge(d: Path) -> dict:
    if not (d / "traces.json").exists():
        return {"unstable": True, "why": "no traces (the run did not start or crashed)"}
    losses = json.loads((d / "traces.json").read_text())["losses"]
    reached = len(losses)
    finite = [x for x in losses if math.isfinite(x)]
    out: dict = {"steps": reached, "curve_every_100": [
        statistics.fmean(losses[i:i + 100]) for i in range(0, reached, 100)]}
    if reached < STEPS or len(finite) < reached:
        out.update(unstable=True, why=f"halted or non-finite: reached {reached:,} of {STEPS:,}")
        return out
    roll = [statistics.fmean(losses[i:i + WINDOW]) for i in range(0, reached - WINDOW + 1, 100)]
    final, low = statistics.fmean(losses[-WINDOW:]), min(roll)
    out.update(final_500_mean=final, lowest_rolling_500=low, rise_over_low=final / low - 1)
    bad = final > low * (1 + DIVERGE)
    out.update(unstable=bad, why=f"diverged: final {final:.4f} > lowest {low:.4f} × {1 + DIVERGE}" if bad else "")
    return out


def summary() -> None:
    rec: dict = {"f7": "baselines.md §8 F7", "smoke": True, "steps": STEPS, "window": WINDOW,
                 "diverge": DIVERGE, "rule_rate": RULE, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}
    for arm in ARMS:
        res = {f"{lr:.3e}": _judge(_dir(arm, lr)) for lr in RATES}
        stable = [lr for lr in RATES if not res[f"{lr:.3e}"]["unstable"]]
        if RULE in stable:
            choice, why = RULE, "the rule's rate is stable"
        elif stable:
            choice = min(stable, key=lambda lr: (abs(math.log(lr / RULE)), lr))
            why = "the rule's rate is UNSTABLE; nearest stable (tie -> lower)"
        else:
            choice, why = None, "no rate is stable"
        rec["arms"][arm] = {"rates": res, "choice": choice, "why": why}
        for lr in RATES:
            r = res[f"{lr:.3e}"]
            tag = "UNSTABLE: " + r["why"] if r["unstable"] else (
                f"final-500 {r['final_500_mean']:.4f}  lowest-500 {r['lowest_rolling_500']:.4f}")
            print(f"{arm:4s} {lr:.3e}{' (rule)' if lr == RULE else '       '}  {tag}")
        print(f"{arm:4s} CHOICE: {choice if choice is None else f'{choice:.3e}'} — {why}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "f7.json").write_text(json.dumps(rec, indent=1))
    print(f"record: {ROOT / 'f7.json'}")


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]]()
