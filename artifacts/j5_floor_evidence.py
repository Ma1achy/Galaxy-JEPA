"""Brief J5 — the effect floor's evidence, and a proposal. Never a freeze.

The floor separates **clean from marginal among already-significant effects**. It does NOT do the
existence work — that is the null's job, via the family-corrected `ExistenceVerdict.exceeds_null`.
So the question this answers is not "where does signal start" but "above what does an effect stop
being borderline", and it has to be argued against two measured distributions at once: how high
chance can reach on these galaxies, and how the real features actually spread.

Reads `artifacts/out/j4_spread_controls.json`. Computes; proposes; does not freeze.
`effect_floor_freeze` stays None — the value is a scientific call and carries `frozen_by`.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/j5_floor_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

#: Every control is a negative control: under it the probe should not beat chance. They are
#: pooled because the floor is ONE number for a whole catalogue, so the relevant ceiling is the
#: highest any of them reached on any feature, not a per-feature one.
CONTROL_KEYS = (
    ("shuffled_max", "shuffled labels (max over draws)"),
    ("random_emb_max", "random embeddings (max over draws)"),
    ("untrained_encoder_auc", "untrained encoder"),
    ("noise_encoder_auc", "noise images"),
    ("sky_noise_auc", "sky-noise label"),
)


def main() -> None:
    path = OUT / "j4_spread_controls.json"
    if not path.exists():
        raise SystemExit(f"J5: {path} not found — run artifacts/j4_spread_controls.py first")
    blob = json.loads(path.read_text())
    feats = blob["features"]

    print(f"J5 source: {path.name}  checkpoint {blob['checkpoint']}  smoke={blob['smoke']}")
    print(f"           {blob['n_train']:,} train / {blob['n_test']:,} test, scheme "
          f"{blob['scheme']}, vote_count_min={blob['vote_count_min']:g}, C={blob['c']}")
    print()

    # --- 1. per feature: the real AUC and every null beside it -----------------------------
    print("REAL AUC AND ITS NULLS, per feature")
    hdr = (f"{'feature':46s} {'real AUC [95% CI]':>26s} {'shuf mean':>9s} {'shuf q99':>9s} "
           f"{'shuf max':>9s} {'rand max':>9s} {'untr':>6s} {'noise':>6s} {'sky':>6s} {'selec':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for f in feats:
        auc = f["auc"]
        shown = (f"{auc:.4f} [{f['auc_lo']:.4f},{f['auc_hi']:.4f}]" if auc is not None
                 else "undefined")
        print(f"{f['feature']:46s} {shown:>26s} {f['shuffled_mean']:9.4f} {f['shuffled_q99']:9.4f} "
              f"{f['shuffled_max']:9.4f} {f['random_emb_max']:9.4f} "
              f"{f['untrained_encoder_auc']:6.4f} {f['noise_encoder_auc']:6.4f} "
              f"{f['sky_noise_auc']:6.4f} {f['selectivity']:+7.4f}")
    print()
    for f in feats:
        print(f"  {f['feature']:46s} n_test {f['n_test']:>7,} pos {f['positives_test']:>6,} "
              f"({100 * f['positives_test'] / max(f['n_test'], 1):5.2f}%)  draws {f['n_draws']:>3}"
              f"   [{f['role']}]")
    print()

    # --- 2. the pooled null distribution ---------------------------------------------------
    print("POOLED NULL DISTRIBUTION — every control, every feature")
    pooled = []
    for key, label in CONTROL_KEYS:
        v = np.array([f[key] for f in feats], dtype=float)
        pooled.append(v)
        print(f"  {label:36s} min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}")
    allshuf = np.concatenate([np.full(1, f["shuffled_mean"]) for f in feats])
    pooled_v = np.concatenate(pooled)
    ceiling = float(pooled_v.max())
    print(f"  {'ALL CONTROLS POOLED':36s} min {pooled_v.min():.4f}  median "
          f"{np.median(pooled_v):.4f}  **max {ceiling:.4f}**")
    print(f"  shuffled-label mean across features: {allshuf.mean():.4f} "
          f"(chance is 0.5 by construction)")
    print()

    # --- 3. the real spread, and 4. the gap -------------------------------------------------
    reals = np.array([f["auc"] for f in feats if f["auc"] is not None], dtype=float)
    los = np.array([f["auc_lo"] for f in feats if f["auc"] is not None], dtype=float)
    order = np.argsort(reals)
    names = [f["feature"] for f in feats if f["auc"] is not None]
    print("REAL AUC SPREAD")
    for i in order:
        print(f"  {names[i]:46s} {reals[i]:.4f}  (lower CI {los[i]:.4f})")
    print(f"  range {reals.min():.4f} .. {reals.max():.4f}   median {np.median(reals):.4f}")
    print()
    print(f"THE GAP: weakest real {reals.min():.4f} - pooled null ceiling {ceiling:.4f} "
          f"= {reals.min() - ceiling:+.4f}")
    print(f"         weakest real's LOWER CI {los[order[0]]:.4f} vs ceiling {ceiling:.4f} "
          f"= {los[order[0]] - ceiling:+.4f}")
    print()

    # --- candidates, for the proposal to be argued against rather than asserted -------------
    print("CANDIDATE FLOORS (each is a measured quantity, not a round number)")
    gaps = sorted(reals)
    biggest, at = 0.0, None
    for a, b in zip(gaps, gaps[1:], strict=False):
        if b - a > biggest:
            biggest, at = b - a, (a + b) / 2
    print(f"  pooled null ceiling                        {ceiling:.4f}   (existence's job, not the floor's)")
    print(f"  ceiling + half the real range              {ceiling + (reals.max()-reals.min())/2:.4f}")
    print(f"  midpoint of the widest gap in the spread   {at if at is None else round(at, 4)}"
          f"   (width {biggest:.4f})")
    print(f"  median of the real spread                  {np.median(reals):.4f}")
    print(f"  the placeholder currently in probe.yaml    0.6500")
    print()
    print("J5 PROPOSES ONLY. `effect_floor_freeze` stays None; the value is Malachy's call.")


if __name__ == "__main__":
    sys.exit(main())
