"""Brief E1 — survey the validity detector over both corpora.

Reports the invalid-fraction distribution per corpus, split edge vs interior, and
sub-classifies the interior population (thin dead column vs compact saturated core) with the
fraction that fall inside the galaxy bbox — the input to the deferred sampler-scope call.

Investigation code: terse, excluded from lint/CI, reads only.

    uv run python artifacts/e1_validity_survey.py [n_per_corpus]
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np

from galaxy_jepa.data.bbox import petrosian_box
from galaxy_jepa.data.sources import DirectorySource, load_fits_stamp
from galaxy_jepa.data.validity import RegionKind, analyse_validity, invalid_planes

PIXEL_SCALE, PETRO_K, GLOBAL_BOX_FRAC, STAMP_PX = 0.396, 2.5, 0.40, 256
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def survey(corpus: str, n: int):
    src = DirectorySource(f"data/{corpus}")
    rows = random.Random(11).sample(src.rows, min(n, len(src.rows)))

    tot, edge_f, inter_f = [], [], []
    kinds = {RegionKind.EDGE: 0, RegionKind.LINEAR: 0, RegionKind.COMPACT: 0}
    stamps_with = {RegionKind.EDGE: 0, RegionKind.LINEAR: 0, RegionKind.COMPACT: 0}
    in_box = {RegionKind.LINEAR: 0, RegionKind.COMPACT: 0}
    radii = {RegionKind.LINEAR: [], RegionKind.COMPACT: []}
    t0 = time.time()
    for r in rows:
        img = load_fits_stamp(f"data/{corpus}/{r['object_id']}.fits")
        edge, interior = invalid_planes(img)
        tot.append(float((edge | interior).mean()))
        edge_f.append(float(edge.mean()))
        inter_f.append(float(interior.mean()))

        sv = analyse_validity(img)
        seen = set()
        half = petrosian_box(
            float(r.get("petroRad_r", float("nan")) or "nan"),
            PIXEL_SCALE,
            k=PETRO_K,
            stamp_px=STAMP_PX,
            global_half_width_px=GLOBAL_BOX_FRAC * STAMP_PX,
        ).half_width_px
        for reg in sv.regions:
            kinds[reg.kind] += 1
            seen.add(reg.kind)
            if reg.kind in in_box:
                radii[reg.kind].append(reg.centroid_radius_px)
                # the box is the centred square of +-half px; a centroid inside it counts
                if reg.centroid_radius_px <= half:
                    in_box[reg.kind] += 1
        for k in seen:
            stamps_with[k] += 1
    dt = time.time() - t0

    print(f"\n=== {corpus}: {len(rows)} stamps, {dt:.1f}s ({len(rows)/dt:.0f}/s) ===")
    print(f"{'invalid fraction':<22}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}")
    for name, vals in (("total", tot), ("edge", edge_f), ("interior", inter_f)):
        print(f"{name:<22}{pct(vals,50):>9.4f}{pct(vals,90):>9.4f}{pct(vals,99):>9.4f}{max(vals):>9.4f}")

    nz = lambda v: sum(1 for x in v if x > 0)  # noqa: E731
    print(f"\nstamps carrying any invalid pixel: {nz(tot)} ({100*nz(tot)/len(rows):.1f}%)")
    for k in (RegionKind.EDGE, RegionKind.LINEAR, RegionKind.COMPACT):
        print(
            f"  {k:<9} stamps {stamps_with[k]:>5} ({100*stamps_with[k]/len(rows):>5.1f}%)"
            f"   regions {kinds[k]:>6}"
            + (
                f"   inside bbox {in_box[k]:>5} ({100*in_box[k]/max(kinds[k],1):>5.1f}%)"
                f"   median centroid r {np.median(radii[k]) if radii[k] else float('nan'):>6.1f} px"
                if k in in_box
                else ""
            )
        )
    return tot


if __name__ == "__main__":
    for corpus in ("pretrain", "probe"):
        survey(corpus, N)
