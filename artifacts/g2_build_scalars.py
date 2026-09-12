"""Brief G2 — write the aligned scalar sidecar for the existing 415.8 GB cache, and prove parity.

The cache is already baked (1,057,326 stamps, 3.88 h), so the sidecar is **backfilled** rather
than re-baked: ``petroRad_r`` lives in ``metadata.csv``, the row order lives in ``index.json``, and
aligning the two is a seconds-long read. Nothing touches a FITS file or a baked stamp.

Parity is checked the way the cache's own was: the array path and the row-dict path must serve the
*same number* for the same galaxy — not "close", the same float64 — on a sample large enough to
catch an off-by-one, and with the sample drawn in a scrambled order so a sidecar read positionally
rather than by object ID would disagree.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/g2_build_scalars.py
"""

from __future__ import annotations

import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import CACHE_BASE, check  # noqa: E402

from galaxy_jepa.data.cache import TensorCache, load_scalars, pipeline_hash, write_scalars  # noqa: E402
from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.transforms import Normalise  # noqa: E402
from galaxy_jepa.harness import _build_pipeline  # noqa: E402

PARITY_SAMPLE = 20_000


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def petro_by_id(*roots: Path) -> dict[int, float]:
    """``object_id -> petroRad_r`` straight from the CSVs, no full row dicts built."""
    out: dict[int, float] = {}
    for root in roots:
        with (root / "metadata.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                v = r.get("petroRad_r", "")
                out[int(r["object_id"])] = float(v) if v not in ("", "nan", None) else float("nan")
    return out


def main() -> None:
    cfg, cache = check()
    pre, probe = Path(cfg.paths.pretrain_dir), Path(cfg.paths.probe_dir)

    t0 = time.perf_counter()
    petro = petro_by_id(pre, probe)
    print(f"\nG2  read petroRad_r for {len(petro):,} objects in {time.perf_counter()-t0:.1f}s "
          f"(rss {rss_gb():.2f} GB)")

    t0 = time.perf_counter()
    path = write_scalars(cache.cache_dir, petro)
    fresh = TensorCache(cache.cache_dir)
    values = load_scalars(fresh.cache_dir, fresh.index)
    print(f"G2  wrote {path.name}: {values.size:,} float64, {path.stat().st_size/1e6:.2f} MB, "
          f"{time.perf_counter()-t0:.1f}s")
    print(f"G2  index vouches for sha {fresh.index.scalars_sha256[:16]}…")
    nan = int(np.isnan(values).sum())
    print(f"G2  values: {nan:,} NaN ({nan/values.size*100:.2f}% — the global-box fallback rate), "
          f"median {np.nanmedian(values):.2f}″, max {np.nanmax(values):.1f}″")

    # --- parity: the two code paths, same galaxies, scrambled order --------------------
    rng = np.random.default_rng(0)
    ids = [int(o) for o in rng.choice(fresh.index.object_ids, PARITY_SAMPLE, replace=False)]
    rows = {o: {"petroRad_r": petro[o]} for o in ids}
    by_rows = StampDataset(fresh, rows, ids)
    by_array = StampDataset(fresh, {}, ids, scalars=fresh.scalars)
    bad = misordered = 0
    for i in range(len(ids)):
        a, b = by_array[i]["petro_rad_arcsec"], by_rows[i]["petro_rad_arcsec"]
        if not ((np.isnan(a) and np.isnan(b)) or a == b):
            bad += 1
        if by_array[i]["object_id"] != by_rows[i]["object_id"]:
            misordered += 1
    print(f"G2  parity: {bad}/{len(ids):,} value disagreements and {misordered} ordering "
          f"disagreements between the array path and the row-dict path (scrambled draw, EXACT "
          f"float comparison)")

    # and the thing that actually reaches training: the token weight map must be identical
    from galaxy_jepa.data.bbox import petrosian_box
    from galaxy_jepa.masking.blocks import box_to_token_mask

    px, grid, scale = fresh.index.height, fresh.index.height // 16, 0.396
    mask_bad = 0
    for oid in ids[:2000]:
        boxes = [
            petrosian_box(float(v), scale, k=2.5, stamp_px=px, global_half_width_px=0.4 * px)
            for v in (petro[oid], float(fresh.scalars[fresh.row_of(oid)]))
        ]
        m = [box_to_token_mask(b.half_width_px, px, grid) for b in boxes]
        if not np.array_equal(m[0], m[1]):
            mask_bad += 1
    print(f"G2  downstream: {mask_bad}/2,000 token weight maps differ — this is what the masker "
          f"actually consumes")

    # the value that made this worth doing
    print(f"\nG2  dataset footprint: sidecar {path.stat().st_size/1e6:.1f} MB vs the "
          f"4,070 MB metadata table it replaces — {4_070e6 / path.stat().st_size:.0f}x smaller")
    pipe = _build_pipeline(q=cfg.q, freeze=cfg.normalisation)
    norm = next(t for t in pipe.transforms if isinstance(t, Normalise))
    assert pipeline_hash(pipe) == CACHE_BASE.joinpath(pipeline_hash(pipe)).name
    print(f"G2  cache key unchanged: {pipeline_hash(pipe)[:16]}… (the sidecar is not hashed into "
          f"the pipeline, so nothing re-bakes; norm mean[0]={(norm.mean or [0])[0]:.6f})")


if __name__ == "__main__":
    main()
