"""Brief J5 — the effect floor's evidence, and a proposal. Never a freeze.

The floor separates **clean from marginal among already-significant effects**. It does NOT do the
existence work — that is the null's job, via the family-corrected `ExistenceVerdict.exceeds_null`.
So the question this answers is not "where does signal start" but "above what does an effect stop
being borderline", and it has to be argued against two measured distributions at once: how high
chance can reach on these galaxies, and how the real features actually spread.

Reads `artifacts/out/<tag>_spread_controls.json` (J4's by default). Computes; proposes; does not freeze.
`effect_floor_freeze` stays None — the value is a scientific call and carries `frozen_by`.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/j5_floor_evidence.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"

#: The four CHANCE-CALIBRATED controls: each breaks something, so under it the probe should not
#: beat chance. They are pooled because the floor is ONE number for a whole catalogue, so the
#: relevant ceiling is the highest any of them reached on any feature, not a per-feature one.
NULL_KEYS = (
    ("shuffled_max", "shuffled labels (max over draws)"),
    ("random_emb_max", "random embeddings (max over draws)"),
    ("untrained_encoder_auc", "untrained encoder"),
    ("noise_encoder_auc", "noise images"),
)

#: 3C-5 is NOT pooled — D19. It breaks nothing (real images, real encoder, real probe, a different
#: real label), so its AUC measures image-quality content rather than chance, and pooling it makes
#: the ceiling a nuisance measurement. It sat at 0.8355-0.8416 and dominated the pooled maximum,
#: which is exactly how it came to fail every feature. Reported below the pooled block, apart.
DIAGNOSTIC_KEYS = (("sky_noise_auc", "sky-noise label — DIAGNOSTIC, not pooled (D19)"),)


@dataclasses.dataclass(frozen=True)
class Source:
    """Which battery's output to read. Defaults are Brief J5's, so running bare reproduces J5.

    Parameterised, not forked, when Brief N2 arrived — and deliberately only the *input*: the
    tables below are J5's evidence and stay exactly as J5 reported them. N2's additions (the
    per-feature ceiling, the two swept forms, the seed overlay) live in `n2_floor_evidence.py`,
    so neither brief's report is quietly rewritten by the other's needs.
    """

    label: str
    tag: str


J5 = Source(label="J5", tag="j4")


def main(src: Source = J5) -> None:
    lbl = src.label
    path = OUT / f"{src.tag}_spread_controls.json"
    if not path.exists():
        raise SystemExit(f"{lbl}: {path} not found — run the controls battery first")
    blob = json.loads(path.read_text())
    feats = blob["features"]

    print(f"{lbl} source: {path.name}  checkpoint {blob['checkpoint']}  smoke={blob['smoke']}")
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
    print("POOLED NULL DISTRIBUTION — the four chance-calibrated controls, every feature")
    pooled = []
    for key, label in NULL_KEYS:
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
    for key, label in DIAGNOSTIC_KEYS:
        v = np.array([f[key] for f in feats], dtype=float)
        print(f"  {label:36s} min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}"
              f"  <- NOT in the ceiling above")
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
    print(f"{lbl} PROPOSES ONLY. `effect_floor_freeze` stays None; the value is Malachy's.")


if __name__ == "__main__":
    sys.exit(main())
