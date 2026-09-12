"""Brief H3 — read the H2 arms honestly, including the trap the brief names.

Higher effective rank is **not** the objective; it is a collapse diagnostic. An arm that holds rank
while learning nothing is worse than the baseline. Two things guard against reading it that way:

* every arm's erank is printed **beside** its loss, at the same step;
* and the arms are compared at **matched loss** — for each arm, the erank at the first step whose
  loss has reached the baseline's final level. A lower peak LR trivially shows higher rank at a fixed
  step count because less has happened, and matched-loss is what separates "holds rank while
  learning" from "has not started".

One caveat the loss cannot resolve on its own: this is a latent MSE against an EMA target, so it
*shrinks* as predictor and target co-adapt, and collapse drives it towards zero. Low loss is
therefore not by itself good. The signature to want is **low loss with rank held**; the signature to
fear is **low loss with rank gone**.

    uv run python artifacts/h3_read_arms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Deliberately NOT `from f1_loader_bench import OUT`: that module imports torch at module scope, so
# reading the results loaded a second torch (~0.2 GB, plus MPS init) alongside a training arm already
# holding 8 GB of MPS driver memory on an 18 GB machine. That is what pushed the first attempt at
# this sweep into a low-memory kill. The read-out needs json and numpy and nothing else.
OUT = Path(__file__).resolve().parent / "out"

WINDOW = 25  # smooth the loss over this many steps before matching — it is noisy per-step


def load() -> list[dict]:
    path = OUT / "h2_arms.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{")]
    return [r for r in rows if not r.get("failed")]


def smooth(losses: list[float]) -> np.ndarray:
    v = np.asarray(losses, dtype=float)
    if len(v) < WINDOW:
        return v
    kernel = np.ones(WINDOW) / WINDOW
    return np.convolve(v, kernel, mode="valid")


def matched(arm: dict, target: float) -> tuple[int, float] | None:
    """The first step whose smoothed loss reaches ``target``, and the erank nearest it."""
    sm = smooth(arm["losses"])
    hit = np.argmax(sm <= target) if (sm <= target).any() else None
    if hit is None:
        return None
    step = int(hit) + WINDOW - 1
    nearest = min(arm["trace"], key=lambda r: abs(r["step"] - step))
    return step, nearest["effective_rank"]


def traces(rows: list[dict]) -> None:
    """The full per-arm trace at the smoke's own 25-step cadence — what H2 asks to be reported."""
    for r in rows:
        print(f"\n{r['arm']}  (peak {r['peak_lr']:.4g}, warmup {r['warmup']}, decay {r['decay']}, "
              f"wd->{r['wd_ramp_to']})  —  {r['why']}")
        print(f"  {'step':>5} {'lr':>10} {'wd':>6} {'loss':>8} {'std':>8} {'erank':>7} "
              f"{'cos':>7} {'halt':>5}")
        for x in r["trace"]:
            print(f"  {x['step']:>5} {x['lr']:>10.3e} {x['weight_decay']:>6.3f} {x['loss']:>8.4f} "
                  f"{x['std']:>8.4f} {x['effective_rank']:>7.2f} {x['mean_cosine']:>+7.3f} "
                  f"{str(x['would_halt']):>5}")


def main() -> None:
    rows = load()
    if not rows:
        print("no completed arms yet")
        return
    by = {r["arm"]: r for r in rows}
    base = by.get("baseline")
    if "--traces" in sys.argv:
        traces(rows)

    print(f"\n{'arm':<11} {'peak lr':>10} {'warmup':>7} {'erank@0':>8} {'@100':>7} {'@175':>7} "
          f"{'@300':>7} {'@end':>7} {'min':>6} {'loss@end':>9} {'halt':>6}")
    print("-" * 104)
    for name in ("baseline", "linear", "sqrt", "cosine", "warmup1250", "wd_ramp"):
        r = by.get(name)
        if r is None:
            continue
        t = {x["step"]: x for x in r["trace"]}

        def er(s: int, t=t) -> str:
            return f"{t[s]['effective_rank']:.2f}" if s in t else "—"

        halted = "YES" if r.get("halt_reason") else "no"
        print(f"{name:<11} {r['peak_lr']:>10.3e} {r['warmup']:>7} {er(0):>8} {er(100):>7} "
              f"{er(175):>7} {er(300):>7} {r['erank_final']:>7.2f} {r['erank_min']:>6.2f} "
              f"{r['loss_last50_mean']:>9.4f} {halted:>6}")

    if base is None:
        return
    # The baseline's DEEPEST smoothed loss, not its final one. The loss here is non-monotonic —
    # it bottoms out around step 100 and then rises as the EMA target moves — so "first step
    # reaching the final loss" is hit on the way down and matches nothing meaningful.
    target = float(smooth(base["losses"]).min())
    print(f"\nMatched-loss comparison — erank at the first step reaching the baseline's DEEPEST "
          f"smoothed loss ({target:.4f}, window {WINDOW}):")
    for name in ("baseline", "linear", "sqrt", "cosine", "warmup1250", "wd_ramp"):
        r = by.get(name)
        if r is None:
            continue
        m = matched(r, target)
        own = float(smooth(r["losses"]).min())
        if m is None:
            print(f"  {name:<11} never reached it — its own deepest smoothed loss is {own:.4f}, "
                  f"{own/target:.1f}x the baseline's. Its erank is NOT comparable: less happened.")
        else:
            step, erank = m
            print(f"  {name:<11} reached at step {step:>4}, erank there {erank:6.2f}   "
                  f"(own deepest {own:.4f})")

    print("\nThe inverse view — how far each arm got BEFORE losing rank. The step and loss at which"
          f"\nerank first fell below the frozen G5 floor of 5.0:")
    for name in ("baseline", "linear", "sqrt", "cosine", "warmup1250", "wd_ramp"):
        r = by.get(name)
        if r is None:
            continue
        crossed = next((x for x in r["trace"] if x["effective_rank"] < 5.0), None)
        if crossed is None:
            print(f"  {name:<11} never fell below 5.0 in {r['steps_run']} steps "
                  f"(min {r['erank_min']:.2f}); loss reached {r['loss_last50_mean']:.4f}")
        else:
            print(f"  {name:<11} crossed at step {crossed['step']:>4}, loss there "
                  f"{crossed['loss']:.4f}, erank {crossed['effective_rank']:.2f}")

    print("\nWhere the LR actually was, per arm, at the readings that matter:")
    for name in ("baseline", "cosine", "warmup1250"):
        r = by.get(name)
        if r is None:
            continue
        t = {x["step"]: x for x in r["trace"]}
        cells = "  ".join(
            f"s{s}={t[s]['lr']:.2e}" for s in (25, 100, 175, 300, 475) if s in t
        )
        print(f"  {name:<11} {cells}")

    print("\nWeight decay, per arm:")
    for name in ("baseline", "wd_ramp"):
        r = by.get(name)
        if r is None:
            continue
        t = {x["step"]: x for x in r["trace"]}
        cells = "  ".join(f"s{s}={t[s]['weight_decay']:.3f}" for s in (25, 175, 475) if s in t)
        print(f"  {name:<11} {cells}")


if __name__ == "__main__":
    main()
