"""Write the probing path's column sidecar for an already-baked cache, and verify it.

`harness._prepare` writes this during a bake. This driver is for the cache that is already on
disk: it streams `metadata.csv` rather than building `DirectorySource.rows`, so writing the
artefact never materialises the 1.49 GB table it exists to remove.

    uv run python artifacts/i4_probe_columns.py write
    uv run python artifacts/i4_probe_columns.py verify --n 20000
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402

from galaxy_jepa.data.cache import (  # noqa: E402
    TensorCache,
    load_probe_columns,
    write_probe_columns,
)
from galaxy_jepa.probing.extract import required_columns  # noqa: E402
from galaxy_jepa.probing.schemes import get_scheme, scheme_names  # noqa: E402


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _stream(meta: Path):
    with meta.open(newline="") as fh:
        yield from csv.DictReader(fh)


def write(cfg, cache) -> None:
    meta = Path(cfg.paths.probe_dir) / "metadata.csv"
    columns = required_columns(schemes=[get_scheme(n) for n in scheme_names()])
    print(f"  {len(columns)} columns over {cache.index.n:,} indexed stamps", flush=True)
    t0 = time.perf_counter()
    path = write_probe_columns(cache.cache_dir, _stream(meta), columns)
    print(f"  wrote {path.name}: {path.stat().st_size/1e6:.1f} MB in "
          f"{time.perf_counter()-t0:.1f} s; peak RSS {_rss_gb():.2f} GB")


def verify(cfg, cache, n: int) -> None:
    """Parity against the path this replaces, on a sample, plus the resident-size comparison."""
    columns = load_probe_columns(cache.cache_dir, cache.index)
    meta = Path(cfg.paths.probe_dir) / "metadata.csv"

    checked = mismatched = absent = 0
    examples: list[str] = []
    for i, row in enumerate(_stream(meta)):
        if i >= n:
            break
        oid = int(row["object_id"])
        try:
            view = columns[oid]
        except KeyError:
            absent += 1  # in the corpus, not baked into this cache
            continue
        for col in columns.columns:
            if col == "object_id":
                continue
            raw = row.get(col)
            try:
                want = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                want = float("nan")
            got = view[col]
            checked += 1
            # exact, not approximate: float64 storage of a float64 parse, which is the whole
            # reason the sidecar is not float32
            if not (got == want or (np.isnan(got) and np.isnan(want))):
                mismatched += 1
                if len(examples) < 5:
                    examples.append(f"{oid} {col}: sidecar {got!r} != csv {want!r}")

    print(f"  parity over {min(n, i+1):,} rows x {len(columns.columns)-1} columns "
          f"= {checked:,} values")
    detail = "" if not examples else "  " + "; ".join(examples)
    print(f"    exact mismatches : {mismatched}{detail}")
    print(f"    not in the cache : {absent}")
    loaded = sum(c.nbytes for c in columns._loaded.values())
    print(f"    sidecar resident after touching every column: {loaded/1e6:.0f} MB")
    print(f"    peak RSS of this process: {_rss_gb():.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "verify"])
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    cfg, cache = check(verbose=False)
    cache = TensorCache(cache.cache_dir)  # re-open, so the index carries any new digest
    write(cfg, cache) if args.mode == "write" else verify(cfg, cache, args.n)


if __name__ == "__main__":
    main()
