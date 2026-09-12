"""Brief E6 — bake the fp16 parity cache, then prove the cache and the raw path agree.

**The cache stores NORMALISED stamps, not raw ones.** ``bake_cache`` writes
``pipeline(image)`` — asinh stretch *and* the frozen ``Normalise`` — and ``StampDataset``
states plainly that there is "no decode, no stretch, no normalise here". So the cache is
strictly downstream of the E5 freeze and cannot honestly be baked before it. That is also why
its directory is keyed on ``pipeline_hash``: a different statistic is a different directory, so
stale constants can never be read back as fresh ones.

The pipeline is built the way a *run* builds it — config -> ``HarnessConfig`` ->
``_build_pipeline`` — so a missing or mismatched freeze stops this driver with the same loud
failure a training run would get, rather than a convenience path that quietly differs.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/e6_bake_cache.py pilot 3000   # measure before committing hours
    uv run python artifacts/e6_bake_cache.py              # the real bake, both corpora
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from galaxy_jepa.data.cache import TensorCache, bake_cache, pipeline_hash
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.harness import HarnessConfig, _build_pipeline

PARITY_SAMPLE = 500
BYTES_PER_STAMP = 3 * 256 * 256 * 2


class _Head:
    """The first n rows of a corpus — a pilot bake appends to the same cache, so it is not
    thrown away: the full bake continues from where it stopped."""

    def __init__(self, source, n: int):
        self.source, self.n = source, min(n, len(source))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.source[i]


def parity(cache: TensorCache, sources, pipeline, k: int = PARITY_SAMPLE) -> None:
    """Raw -> pipeline must equal the cache read, bit for bit in fp16."""
    by_id = {}
    for src in sources:
        for i, row in enumerate(src.rows):
            oid = int(row["object_id"])
            if oid in cache:
                by_id[oid] = (src, i)
    oids = list(by_id)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(oids), size=min(k, len(oids)), replace=False)
    print(f"\nparity: {len(pick)} stamps re-run raw -> pipeline and compared to the cache")
    bad, worst = 0, 0.0
    for j in pick:
        oid = oids[int(j)]
        src, i = by_id[oid]
        fresh = np.asarray(pipeline(src[i][0]), dtype=np.float16)
        cached = cache.get(oid)
        if not np.array_equal(fresh, cached):
            bad += 1
            worst = max(worst, float(np.max(np.abs(fresh.astype(np.float64)
                                                  - cached.astype(np.float64)))))
    print(f"  mismatches: {bad}/{len(pick)}" + (f"  worst abs diff {worst:.3e}" if bad else
                                                "  (bit-identical)"))
    if bad:
        raise SystemExit("PARITY BROKEN: the cache is not what the pipeline produces")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s", stream=sys.stdout)
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    n_pilot = int(sys.argv[2]) if pilot and len(sys.argv) > 2 else 3000

    with open("configs/pretrain.yaml") as fh:
        config = HarnessConfig(**yaml.safe_load(fh))
    pipeline = _build_pipeline(q=config.q, freeze=config.normalisation)
    key = pipeline_hash(pipeline)
    out = Path(config.paths.out_dir) / "cache"
    assert config.normalisation is not None
    nh = config.normalisation.content_hash
    print(f"pipeline_hash {key}\n  normalisation {nh[:12]} "
          f"(corpus={config.normalisation.corpus}, n={config.normalisation.n_sample}, "
          f"valid_only={config.normalisation.valid_pixels_only})")
    print(f"  cache dir {out}  (stores NORMALISED fp16 stamps; index records {nh[:12]})\n")

    pre = DirectorySource(config.paths.pretrain_dir)
    probe = DirectorySource(config.paths.probe_dir)
    total = len(pre) + len(probe)
    print(f"corpora: pretrain {len(pre)} + probe {len(probe)} = {total} stamps; "
          f"projected {total*BYTES_PER_STAMP/1e9:.0f} GB at {BYTES_PER_STAMP/1e6:.3f} MB each")

    t0 = time.time()
    if pilot:
        cache = bake_cache(_Head(pre, n_pilot), pipeline, out, normalisation_hash=nh)
        done = len(cache)
        rate = done / (time.time() - t0)
        print(f"\npilot: {done} stamps in {time.time()-t0:.0f}s -> {rate:.0f} stamps/s")
        print(f"  projects to {total/rate/3600:.2f} h and "
              f"{total*BYTES_PER_STAMP/1e9:.0f} GB for the full bake")
        parity(cache, [pre], pipeline)
        return

    bake_cache(pre, pipeline, out, normalisation_hash=nh)
    t_pre = time.time() - t0
    print(f"\npretrain baked in {t_pre/3600:.2f} h ({len(pre)/max(t_pre,1):.0f}/s)")
    cache = bake_cache(probe, pipeline, out, normalisation_hash=nh)  # same dir -> appends
    elapsed = time.time() - t0
    size = (out / key / "stamps.f16").stat().st_size
    print(f"probe appended; {len(cache)} stamps total in {elapsed/3600:.2f} h "
          f"({len(cache)/max(elapsed,1):.0f}/s)")
    print(f"cache size {size/1e9:.1f} GB at {out/key}")
    assert len(cache) == total, f"cache has {len(cache)} stamps, corpora have {total}"
    parity(cache, [pre, probe], pipeline)


if __name__ == "__main__":
    main()
