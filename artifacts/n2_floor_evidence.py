"""Brief N2 — the effect floor: evidence for both forms, then a proposal. Never a freeze.

J5's evidence is reprinted unchanged (by calling it), then N2 adds what J5 could not compute and
what the choice between the two candidate forms actually turns on.

**The two forms are not two flavours of one thing.**

* **(a) absolute AUC** — `clean = significant and real_auc >= floor`, one site, `nulls.py:264`.
  Orthogonal to the existence axis. This is what design 3B grounds: the floor "is a pre-registered
  constant, acceptable *because it no longer does the existence work*".
* **(b) margin over the feature's own binding null** — `real_auc >= ceiling_f + m`. On this
  apparatus that sits on the **same axis existence already thresholds at zero**:
  `existence_null_samples` takes a per-draw max over the four chance-calibrated controls, and the
  untrained-encoder singleton dominated every draw on J's encoder, so the null collapsed to a point
  mass and the existence test reduced to exactly `real_auc > ceiling_f`. Form (b) is therefore a
  tightening of existence, not a second and independent question — which is the property 3B
  disclaims. It is also not a one-line change: `effect_floor` has five consumers, and at
  `ladder.py:121` (entangled pair), `:161` (competitive nuisance) and `:248` (MLP decode) there is
  no defined "the feature's own binding null" to take a margin over.

**The decision rule was fixed before these numbers existed** (plan, Brief N). `range_f` is the
feature's untrained-null spread across seeds; `margin_f` is its real AUC minus its primary-seed
ceiling:

* STABLE   — `range_f < margin_f` for every feature AND `max range_f <= 0.010`
             -> the margin form's base quantity holds; recommend between (a) and (b) on the merits.
* MARGINAL — `max range_f` in 0.010-0.030, or `range_f >= margin_f` at a near-zero margin
             -> recommend the ABSOLUTE form; not because it is better, but because it does not
                inherit the instability. Name the seed-determined features.
* UNSTABLE — `max range_f >= 0.030`, or `range_f >= margin_f` at a substantial margin
             -> recommend NO floor. A bar that moves under reseeding is a broken control, and
                averaging the seeds would document the problem while hiding it.

Computes; proposes; does not freeze. `effect_floor_freeze` stays None; the value is Malachy's.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/n2_floor_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j5_floor_evidence import NULL_KEYS, OUT, Source, main as j5_tables  # noqa: E402

N2 = Source(label="N2", tag="n1")

STABLE_RANGE = 0.010
UNSTABLE_RANGE = 0.030
#: A margin below this is "near zero" — the t10-medium case, where J measured +0.0000.
NEAR_ZERO_MARGIN = 0.005


def _short(name: str) -> str:
    return name.replace("_smooth_or_features", "").replace("_arms_winding", "")[:34]


def _ceiling(f: dict, untrained: float | None = None) -> float:
    """The feature's own binding null: the supremum of `existence_null_samples` for it.

    Equal to it exactly — the per-draw max's maximum over draws is the max of the per-control
    maxima, which is what the battery records.
    """
    u = f["untrained_encoder_auc"] if untrained is None else untrained
    return max(f["shuffled_max"], f["random_emb_max"], f["noise_encoder_auc"], u)


def main() -> None:
    j5_tables(N2)
    print("\n" + "=" * 110 + "\nN2 ADDITIONS\n" + "=" * 110 + "\n")

    blob = json.loads((OUT / f"{N2.tag}_spread_controls.json").read_text())
    feats = blob["features"]
    seeds = blob.get("untrained_seeds", [])
    has_seeds = all("untrained_encoder_aucs" in f for f in feats) and len(seeds) > 1

    # --- 1. the per-feature ceiling and gap -------------------------------------------------
    print("PER-FEATURE NULL CEILING AND GAP  (form (b) consumes these; form (a) does not)")
    hdr = (f"  {'feature':36s} {'real':>7s} {'ceiling':>8s} {'binding control':>17s} "
           f"{'margin':>8s}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    rows = []
    for f in feats:
        ceil = _ceiling(f)
        which = max(NULL_KEYS, key=lambda kv: f[kv[0]])[1].split(" (")[0]
        margin = f["auc"] - ceil
        rows.append({"f": f, "name": _short(f["feature"]), "real": f["auc"],
                     "ceil": ceil, "margin": margin})
        print(f"  {_short(f['feature']):36s} {f['auc']:7.4f} {ceil:8.4f} {which:>17s} "
              f"{margin:+8.4f}")
    ceils = np.array([r["ceil"] for r in rows])
    margins = np.array([r["margin"] for r in rows])
    print(f"\n  ceiling spans {ceils.min():.4f} .. {ceils.max():.4f}  "
          f"(range {ceils.max() - ceils.min():.4f})")
    print(f"  margin  spans {margins.min():+.4f} .. {margins.max():+.4f}  "
          f"(range {margins.max() - margins.min():.4f})")
    print("  An absolute floor cannot encode a ceiling that varies by the first number; that is\n"
          "  J5's third objection, restated on this encoder.\n")

    # --- 2. the untrained null under reseeding ----------------------------------------------
    verdict = None
    if not has_seeds:
        print("UNTRAINED NULL UNDER RESEEDING: not measured in this battery "
              "(no `untrained_encoder_aucs`).\n  The decision rule cannot be evaluated; N2 must "
              "say so rather than recommend around it.\n")
    else:
        print(f"UNTRAINED NULL UNDER RESEEDING — seeds {seeds}, primary {seeds[0]}")
        hdr = (f"  {'feature':36s} " + " ".join(f"{'seed ' + str(s):>9s}" for s in seeds)
               + f" {'range':>8s} {'margin':>8s} {'range<margin?':>14s}")
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        for r in rows:
            f = r["f"]
            per = f["untrained_encoder_aucs"]
            cells = " ".join(f"{per[str(s)]:9.4f}" for s in seeds)
            rng = f["untrained_seed_range"]
            r["range"] = rng
            ok = "yes" if rng < r["margin"] else "NO — seed-determined"
            print(f"  {r['name']:36s} {cells} {rng:8.4f} {r['margin']:+8.4f} {ok:>14s}")
        ranges = np.array([r["range"] for r in rows])
        worst = max(rows, key=lambda r: r["range"])
        print(f"\n  max range over features {ranges.max():.4f} (at {worst['name']}), "
              f"median {np.median(ranges):.4f}")

        seed_determined = [r for r in rows if r["range"] >= r["margin"]]
        near_zero = [r for r in seed_determined if r["margin"] < NEAR_ZERO_MARGIN]
        substantial = [r for r in seed_determined if r["margin"] >= NEAR_ZERO_MARGIN]
        if ranges.max() >= UNSTABLE_RANGE or substantial:
            verdict = "UNSTABLE"
        elif ranges.max() > STABLE_RANGE or near_zero:
            verdict = "MARGINAL"
        else:
            verdict = "STABLE"
        print(f"\n  PRE-REGISTERED VERDICT: {verdict}")
        if seed_determined:
            print("  Seed-determined features (range >= their own margin): "
                  + ", ".join(r["name"] for r in seed_determined))
        print({
            "STABLE": "  -> the margin form's base quantity holds; choose on the merits.",
            "MARGINAL": "  -> recommend the ABSOLUTE form: it does not inherit the instability.",
            "UNSTABLE": "  -> recommend NO floor; the binding null is too variable to build on.",
        }[verdict])
        print()

    # --- 3. both forms, swept, with the distance to a flip ----------------------------------
    def sweep(name: str, grid, classify, flip_at) -> None:
        print(f"{name}")
        hdr = f"  {'value':>7s}  {'admitted':>8s}  {'nearest flip':>12s}  which features are admitted"
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        for v in grid:
            adm = [r["name"] for r in rows if classify(r, v)]
            dist = min(abs(v - flip_at(r)) for r in rows)
            flag = "  <- coincidence, not a threshold" if dist < 0.005 else ""
            print(f"  {v:7.4f}  {len(adm):>8d}  {dist:12.4f}  "
                  f"{', '.join(adm) if adm else '(none)'}{flag}")
        print()

    grid_a = sorted({round(v, 4) for v in np.arange(0.50, 0.901, 0.025)}
                    | {round(float(c), 4) for c in ceils} | {0.65})
    sweep("FORM (a) — ABSOLUTE AUC FLOOR, swept  (flip point per feature = its real AUC)",
          grid_a, lambda r, v: r["real"] >= v, lambda r: r["real"])

    grid_b = [round(v, 4) for v in np.arange(0.00, 0.1001, 0.01)]
    sweep("FORM (b) — MARGIN OVER THE FEATURE'S OWN CEILING, swept  (flip point = its margin)",
          grid_b, lambda r, m: r["margin"] >= m, lambda r: r["margin"])

    # --- 4. the seed overlay on form (b) ----------------------------------------------------
    if has_seeds:
        print("SEED OVERLAY ON FORM (b) — re-classified under EACH seed's untrained null")
        hdr = f"  {'margin':>7s}  " + "  ".join(f"{'seed ' + str(s):>9s}" for s in seeds) + "   features that change side"
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        for m in grid_b:
            sets, counts = [], []
            for s in seeds:
                adm = {r["name"] for r in rows
                       if r["real"] - _ceiling(r["f"], r["f"]["untrained_encoder_aucs"][str(s)]) >= m}
                sets.append(adm)
                counts.append(len(adm))
            unstable_f = sorted(set.union(*sets) - set.intersection(*sets))
            cells = "  ".join(f"{c:9d}" for c in counts)
            print(f"  {m:7.4f}  {cells}   {', '.join(unstable_f) if unstable_f else '(none)'}")
        print()

    print("=" * 110)
    print("N2 PROPOSES ONLY. `effect_floor_freeze` stays None; `configs/probe.yaml` untouched;")
    print("`headline=True` still refused at load. The value is Malachy's call.")
    if verdict:
        print(f"Pre-registered stability verdict on the binding null: {verdict}.")


if __name__ == "__main__":
    sys.exit(main())
