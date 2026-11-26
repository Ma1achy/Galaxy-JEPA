"""D8 (superseded) — the power picture at the vote floor actually being frozen.

Reach and per-bucket positives for the Scheme 1 deep buckets, at the frozen floor and at every
registered sweep point. Uses the real `eligible_ids`, so it measures the filter that runs rather
than a re-implementation of it — which matters here, because the reach denominator was the
question total only after this pass fixed it.

Investigation code: terse, excluded from lint/CI, reads only.

    uv run python artifacts/d8_vote_floor_recount.py
"""

from __future__ import annotations

import csv

import numpy as np

from galaxy_jepa.probing.schemes import eligible_ids, get_scheme

SWEEP = (1, 5, 11, 21, 37)
THRESHOLD = 0.5
DEEP = {
    "t09_bulge_shape_a26_boxy": "t09 boxy",
    "t10_arms_winding_a28_tight": "t10 tight",
    "t10_arms_winding_a29_medium": "t10 medium",
    "t10_arms_winding_a30_loose": "t10 loose",
    "t11_arms_number_a31_1": "t11 1-arm",
    "t11_arms_number_a34_4": "t11 4-arms",
    "t11_arms_number_a36_more_than_4": "t11 >4-arms",
}


def main() -> None:
    scheme = get_scheme("full_tree")
    with open("data/probe/metadata.csv", newline="") as fh:
        rows = {}
        for r in csv.DictReader(fh):
            rows[int(r["object_id"])] = r
    ids = sorted(rows)
    print(f"probe corpus: {len(ids)} galaxies | scheme full_tree, family {scheme.family_size()}")
    print(f"binarisation threshold {THRESHOLD} | population: full (the headline)\n")

    def val(oid, col):
        try:
            return float(rows[oid][col])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    hdr = f"{'bucket':<14}" + "".join(f"{f'>={f}':>21}" for f in SWEEP)
    print(hdr)
    print(f"{'':<14}" + "".join(f"{'eligible   pos (%)':>21}" for _ in SWEEP))
    for name, label in DEEP.items():
        spec = scheme.by_name[name]
        col = spec.require_testable()
        line = f"{label:<14}"
        for floor in SWEEP:
            keep = eligible_ids(rows, spec, ids, vote_count_min=floor)
            frac = np.array([val(o, col) for o in keep])
            pos = int((frac >= THRESHOLD).sum())
            line += f"{len(keep):>10}{pos:>7}" + f"({100 * pos / max(len(keep), 1):>4.1f}%)"
        print(line)

    print("\nquestion reach (any spec on that question shares it — the denominator is the total):")
    for q, spec_name in (
        ("t09 bulge shape", "t09_bulge_shape_a26_boxy"),
        ("t10 arms winding", "t10_arms_winding_a28_tight"),
        ("t11 arms number", "t11_arms_number_a31_1"),
    ):
        spec = scheme.by_name[spec_name]
        reach = [len(eligible_ids(rows, spec, ids, vote_count_min=f)) for f in SWEEP]
        print(f"  {q:<18}" + "".join(f"{n:>10}" for n in reach))
    print("  " + " " * 18 + "".join(f"{f'>={f}':>10}" for f in SWEEP))


if __name__ == "__main__":
    main()
