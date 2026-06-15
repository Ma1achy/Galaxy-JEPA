"""Pull entrypoint — the small slice now, the full corpus later (same command).

Implements the pull half of ``docs/spec/data.md`` §3. Runs a CasJobs/SkyServer query
(``metadata``), cuts per-object FITS stamps (``FitsFrameSource``), and writes the
``DirectorySource`` layout (``metadata.csv`` + ``<object_id>.fits``) plus a ``manifest.json``
whose ``data_snapshot`` hash feeds the run-stamp. The small pull and the eventual 500k
corpus pull are the same command with a different ``--limit``.

Networked — runs in the devcontainer. The ``--check-join`` mode runs the 10-row ra/dec
agreement guard *before* any bulk pull (``docs/spec/data.md`` §3).

Usage::

    python -m galaxy_jepa.data.pull --corpus probe --limit 10 --check-join
    python -m galaxy_jepa.data.pull --corpus probe --limit 2000 --out data/probe
    python -m galaxy_jepa.data.pull --corpus pretrain --limit 2000 --out data/pretrain
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.metadata import (
    assert_radec_agree,
    join_check_sql,
    photometric_snr,
    pretrain_sql,
    probe_sql,
    run_sql,
)
from galaxy_jepa.data.sources import FitsFrameSource

logger = logging.getLogger(__name__)


def _object_id(row: dict[str, Any]) -> int:
    raw = row.get("objID", row.get("dr7objid"))
    if raw is None:
        raise KeyError(f"row has neither objID nor dr7objid: {row!r}")
    return int(raw)


def _with_derived(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``object_id`` and the image-domain ``snr_r`` (a bad mag error → NaN + warn)."""
    for row in rows:
        row["object_id"] = _object_id(row)
        try:
            row["snr_r"] = photometric_snr(float(row["modelMagErr_r"]))
        except (ValueError, KeyError, TypeError):
            logger.warning("object %s: bad modelMagErr_r; snr_r set NaN", row.get("object_id"))
            row["snr_r"] = float("nan")
    return rows


def check_join(*, limit: int = 10, data_release: int = 17) -> None:
    rows = run_sql(join_check_sql(limit), data_release=data_release)
    assert_radec_agree(rows)
    logger.info("join check OK: %d rows agree on ra/dec within tolerance", len(rows))




def pull_corpus(
    corpus: str,
    limit: int,
    out_dir: str | Path,
    *,
    stamp_px: int = 64,
    data_release: int = 17,
    mag_min: float = 14.0,
    mag_max: float = 19.0,
    workers: int = 16,
) -> Path:
    from astropy.io import fits  # lazy: only the live pull needs astropy

    sql = (
        pretrain_sql(limit, mag_min=mag_min, mag_max=mag_max)
        if corpus == "pretrain"
        else probe_sql(limit)
    )
    rows = _with_derived(run_sql(sql, data_release=data_release))
    source = FitsFrameSource(rows, stamp_px=stamp_px, data_release=data_release)

    # The bottleneck is the remote SDSS frame download, not local CPU — measured: threads
    # plateau ~16 workers and a process pool is *slower* (overhead, no GIL gain). So this is
    # I/O/server-bound: a shared thread pool over one FitsFrameSource (which keeps an
    # HTTP keep-alive session + a frame dedup cache) is the right tool. map preserves input
    # order → deterministic corpus (ORDER BY). One bad frame is logged and skipped.
    def fetch(i: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        try:
            image, row = source[i]
            return image, row
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the pull
            logger.warning("object %s fetch failed (%s); skipping", rows[i].get("object_id"), exc)
            return None, rows[i]

    fetched: list[tuple[np.ndarray, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for image, row in pool.map(fetch, range(len(source))):
            if image is not None:
                fetched.append((image, row))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for image, row in fetched:
        fits.PrimaryHDU(data=image.astype(np.float32)).writeto(
            out / f"{row['object_id']}.fits", overwrite=True
        )
        written.append(row)
    n_failed = len(rows) - len(written)
    if n_failed:
        logger.warning("%d/%d galaxies failed to fetch and were skipped", n_failed, len(rows))

    fieldnames = ["object_id"] + sorted({k for r in written for k in r} - {"object_id"})
    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    snapshot = manifest_hash((r["object_id"] for r in written), sql)
    (out / "manifest.json").write_text(
        json.dumps({"data_snapshot": snapshot, "n": len(written), "query": sql}, indent=2) + "\n"
    )
    logger.info("wrote %d stamps to %s (%s)", len(written), out, snapshot)
    summarise_pull(written)
    return out


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return out


def summarise_pull(rows: list[dict[str, Any]]) -> None:
    """Print at-scale sanity stats so the corrected join can be trusted before the curve.

    A 10-row ra/dec guard is thin; this is the first real pull on the corrected 4-table
    join, so report the ranges that would expose a silently-wrong join (e.g. all-NaN
    redshifts, absurd radii) plus the global-box fallback rate (missing/≤0 petroRad).
    """
    def rng(key: str) -> str:
        vals = _finite([r.get(key) for r in rows])
        if not vals:
            return f"{key}: NONE finite (!) of {len(rows)}"
        arr = np.asarray(vals)
        return (f"{key}: [{arr.min():.4g}, {arr.max():.4g}] "
                f"median {np.median(arr):.4g} (n={len(vals)}/{len(rows)})")

    def _bad_petro(r: dict[str, Any]) -> bool:
        good = _finite([r.get("petroRad_r")])
        return not good or good[0] <= 0

    bad_petro = sum(1 for r in rows if _bad_petro(r))
    fallback = 100.0 * bad_petro / len(rows) if rows else 0.0
    logger.info("--- pull summary (n=%d) ---", len(rows))
    for key in ("specz", "petroRad_r", "snr_r", "modelMag_r", "psfWidth_r"):
        if any(key in r for r in rows):
            logger.info("  %s", rng(key))
    logger.info("  global-box fallback rate (missing/≤0 petroRad_r): %.1f%% (%d/%d)",
                fallback, bad_petro, len(rows))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pull an SDSS corpus slice (FITS + metadata).")
    parser.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stamp-px", type=int, default=64)
    parser.add_argument("--data-release", type=int, default=17)
    parser.add_argument("--check-join", action="store_true", help="run the 10-row join guard")
    parser.add_argument("--workers", type=int, default=16, help="parallel frame-fetch threads")
    args = parser.parse_args(argv)

    if args.check_join:
        check_join(limit=args.limit if args.limit <= 50 else 10, data_release=args.data_release)
        return
    if args.out is None:
        parser.error("--out is required for a pull")
    pull_corpus(args.corpus, args.limit, args.out, stamp_px=args.stamp_px,
                data_release=args.data_release, workers=args.workers)


if __name__ == "__main__":
    main()
