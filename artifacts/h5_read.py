"""Read the H5 resolving run against the rule pre-registered in h5_decision_rule.md.

Deliberately imports no torch: this runs while a training arm may still hold the MPS
allocator, and a second torch alongside it is what got four H2 arms killed for low memory.

    uv run python artifacts/h5_read.py            # the read
    uv run python artifacts/h5_read.py --traces   # full per-arm traces
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ORDER = ["baseline", "proposal"]
G5_FLOOR, G5_GRACE_FRAC = 5.0, 0.10


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    recs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["arm"]] = r  # later wins, so a re-run supersedes
    return recs


def smooth(xs: list[float], w: int = 25) -> list[float]:
    return [sum(xs[max(0, i - w + 1) : i + 1]) / len(xs[max(0, i - w + 1) : i + 1])
            for i in range(len(xs))]


def at(trace: list[dict], step: int, key: str) -> float | None:
    pt = next((p for p in trace if p["step"] == step), None)
    return None if pt is None else pt[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="store_true")
    args = ap.parse_args()

    arms = load(OUT / "h5_arms.jsonl")
    probes = load(OUT / "h5_probes.jsonl")
    names = [n for n in ORDER if n in arms]
    if not names:
        print("no arms yet")
        return

    if args.traces:
        for n in names:
            a = arms[n]
            print(f"\n{n}  (peak {a['peak_lr']:.4g}, warmup {a['warmup']}, decay {a['decay']})")
            print(f"  {a['why']}")
            print("   step         lr     loss      std   erank     cos  halt")
            for p in a["trace"]:
                print(f"  {p['step']:>5}  {p['lr']:.3e}  {p['loss']:7.4f}  {p['std']:7.3f} "
                      f"{p['effective_rank']:7.2f}  {p['mean_cosine']:+.3f} {str(p['would_halt']):>5}")
        print()

    steps = arms[names[0]]["steps_run"]
    reads = [0, 175, 500, 1000, 2000, steps - (steps % 25) - 25]

    print(f"\nRESOLVING RUN — {steps} steps per arm, {len(names)} arms\n")
    print("EFFECTIVE RANK (the H2 diagnostic)")
    print(f"  {'arm':<10} {'peak lr':>10} {'warmup':>7} " + "".join(f"{'@'+str(s):>8}" for s in reads)
          + f"{'min':>8}")
    for n in names:
        a = arms[n]
        row = "".join(f"{at(a['trace'], s, 'effective_rank') or float('nan'):8.2f}" for s in reads)
        print(f"  {n:<10} {a['peak_lr']:10.3e} {a['warmup']:>7} {row}{a['erank_min']:8.2f}")

    print("\nEMBEDDING STD — first-class, not a footnote. H2: it inverts the erank ordering in")
    print("all six arms while mean-cosine carries none, so the failure mode is norm growth with")
    print("directional concentration, not shrinkage towards a point.")
    print(f"  {'arm':<10} " + "".join(f"{'@'+str(s):>9}" for s in reads) + f"{'max':>9}{'growth':>9}")
    for n in names:
        a = arms[n]
        row = "".join(f"{at(a['trace'], s, 'std') or float('nan'):9.3f}" for s in reads)
        g = a["std_final"] / a["trace"][0]["std"]
        print(f"  {n:<10} {row}{a['std_max']:9.3f}{g:8.1f}x")

    print("\nMEAN COSINE — reported because H2 found it uninformative; watch for that changing.")
    print(f"  {'arm':<10} " + "".join(f"{'@'+str(s):>9}" for s in reads))
    for n in names:
        row = "".join(f"{at(arms[n]['trace'], s, 'mean_cosine') or float('nan'):+9.3f}" for s in reads)
        print(f"  {n:<10} {row}")

    print("\nLOSS — both framings, because H2's two disagreed at 500 steps")
    print(f"  {'arm':<10} {'deepest':>9} {'at step':>8} {'@end':>9} {'last 100':>10}  shape")
    for n in names:
        a = arms[n]
        sm = smooth(a["losses"])
        deep = min(sm)
        where = sm.index(deep)
        change = sm[-1] - sm[-100]
        shape = "RISING" if change > 0.005 else ("flat" if abs(change) <= 0.005 else "descending")
        print(f"  {n:<10} {deep:9.4f} {where:>8} {sm[-1]:9.4f} {change:+10.4f}  {shape}")

    if len(names) == 2:
        b, p = arms["baseline"], arms.get("proposal")
        if p:
            sb, sp = smooth(b["losses"]), smooth(p["losses"])
            print(f"\n  at equal step {steps}: baseline {sb[-1]:.4f}  proposal {sp[-1]:.4f}"
                  f"  ({'proposal' if sp[-1] < sb[-1] else 'baseline'} lower"
                  f", {max(sb[-1], sp[-1]) / max(min(sb[-1], sp[-1]), 1e-12):.2f}x)")
            print(f"  on deepest-ever  : baseline {min(sb):.4f}  proposal {min(sp):.4f}"
                  f"  ({'proposal' if min(sp) < min(sb) else 'baseline'} lower)")
            agree = (sp[-1] < sb[-1]) == (min(sp) < min(sb))
            print(f"  the two framings {'AGREE' if agree else 'STILL DISAGREE'} at {steps} steps.")

    print("\nG5 COLLAPSE FLOOR")
    total = arms[names[0]]["config_steps"]
    grace = int(G5_GRACE_FRAC * total)
    print(f"  soft floor {G5_FLOOR} applies only from step {grace:,} (10% of {total:,}).")
    print(f"  This run is {steps:,} steps, so the soft floor CANNOT FIRE — only the hard floor")
    print("  (erank < 2.0 from step 100) can. would_halt below is arithmetic, not a pass:")
    for n in names:
        a = arms[n]
        fired = [p["step"] for p in a["trace"] if p["would_halt"]]
        below = [p["step"] for p in a["trace"] if p["effective_rank"] < G5_FLOOR]
        print(f"    {n:<10} would_halt fired: {fired or 'never'}"
              f"   |  erank below {G5_FLOOR} at steps: "
              f"{'never' if not below else f'{below[0]} onwards ({len(below)} readings)'}")

    print("\nTHE PROBE — the deciding measurement. AUC is the objective; rank and std are not.")
    if not probes:
        print("  not run yet")
    else:
        print(f"  {'arm':<10} {'AUC':>7} {'95% CI':>18} {'n_train':>8} {'n_test':>7}  {'mins':>5}")
        for n in names:
            r = probes.get(n)
            if r is None:
                print(f"  {n:<10} {'—':>7}  (not run)")
                continue
            print(f"  {n:<10} {r['auc']:7.4f}  [{r['auc_lo']:.4f}, {r['auc_hi']:.4f}] "
                  f"{r['n_train']:>8,} {r['n_test']:>7,}  {r['seconds']/60:5.1f}")
        print("  (AUC is on the high-consensus extremes, `probe.extreme_low/high` — the same")
        print("   subset the pilot's 0.905 was measured on, so the comparison is like-for-like.)")

        if all(n in probes for n in ("baseline", "proposal")):
            pb, pp = probes["baseline"], probes["proposal"]
            overlap = pb["auc_lo"] <= pp["auc_hi"] and pp["auc_lo"] <= pb["auc_hi"]
            print(f"\n  CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'} -> "
                  f"{'no separation detected' if overlap else 'separated'}")
            print("\nTHE PRE-REGISTERED RULE (artifacts/h5_decision_rule.md), applied:")
            better_rank = arms["proposal"]["erank_final"] > arms["baseline"]["erank_final"] * 1.3
            better_std = arms["proposal"]["std_final"] < arms["baseline"]["std_final"] / 1.3
            diag = "materially better" if (better_rank and better_std) else "not materially better"
            print(f"  rank/std for the proposal: {diag} "
                  f"(erank {arms['proposal']['erank_final']:.2f} vs "
                  f"{arms['baseline']['erank_final']:.2f}; std {arms['proposal']['std_final']:.2f} "
                  f"vs {arms['baseline']['std_final']:.2f})")
            if overlap:
                print("  AUC indistinguishable -> ADOPT THE PROPOSAL on the reference-recipe")
                print("  argument alone (sqrt-scaling for AdamW, the reference's relative warmup).")
                print("  Say plainly: the traces are not what decided it.")
            elif pp["auc"] > pb["auc"]:
                print("  proposal wins on AUC -> ADOPT THE PROPOSAL." + (
                    "" if (better_rank and better_std) else
                    "  Note: rank/std did not separate materially, so the diagnostic did not "
                    "predict the objective — a finding about the diagnostic."))
            else:
                print("  proposal LOSES on AUC -> KEEP THE BASELINE. This is H2's linear-arm trap")
                print("  at a longer horizon: rank held, the objective did not follow.")

    print("\nNOT LIKE-FOR-LIKE: the pilot's 0.905 was 6,000 steps on 10,000 stamps seen ~19x each;")
    print(f"this is {steps:,} steps on 827k seen once. The pilot is an existence proof, not a baseline.")
    print("Both arms are stamped smoke=True; the effect floor is open, so nothing here is a ladder verdict.\n")


if __name__ == "__main__":
    main()
