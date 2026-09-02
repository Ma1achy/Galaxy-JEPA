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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.metadata import (
    AXIS_RATIO_COLS,
    FEATURED_FRACTION_COL,
    assert_radec_agree,
    axis_ratio_sql,
    join_check_sql,
    photometric_snr,
    pretrain_sql,
    probe_sql,
    run_sql,
)
from galaxy_jepa.data.sources import FitsFrameSource

logger = logging.getLogger(__name__)


def _object_id(row: dict[str, Any]) -> int:
    """The row's SDSS objID, from whichever key carries it.

    ``objID`` is what the SQL selects (the probe join aliases ``dr8objid AS objID``);
    ``object_id`` is what a corpus already written to disk carries. **Precedence matters:** the
    probe corpus carries a ``dr7objid`` column *as well*, and DR7 and DR8 object IDs are
    different numbers for the same galaxy — reading ``dr7objid`` in preference to an existing
    ``object_id`` silently rewrites the identity the FITS filenames are keyed on. So an
    already-written ``object_id`` always wins over ``dr7objid``, which is the last resort only.
    """
    raw = row.get("objID", row.get("object_id", row.get("dr7objid")))
    if raw is None:
        raise KeyError(f"row has no objID / object_id / dr7objid: {row!r}")
    return int(raw)


def with_derived_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``object_id`` and the image-domain ``snr_r`` (a bad mag error → NaN + warn).

    The **single** derivation site for the SNR nuisance column. Both pull paths route through
    it: this module's HTTP pull calls it inline, and the SciServer driver
    (``artifacts/sciserver_pull.py``) calls it on the target rows before they are handed to the
    server-side cut — so a corpus cannot end up without ``snr_r`` depending on which driver
    pulled it. :func:`backfill_derived` applies the same function to a corpus already on disk.

    Pure and token-free (the artifacts rule): it touches rows, never the network.
    """
    for row in rows:
        row["object_id"] = _object_id(row)
        try:
            row["snr_r"] = photometric_snr(float(row["modelMagErr_r"]))
        except (ValueError, KeyError, TypeError):
            logger.warning("object %s: bad modelMagErr_r; snr_r set NaN", row.get("object_id"))
            row["snr_r"] = float("nan")
    return rows


# Back-compat alias for the private name this used to carry.
_with_derived = with_derived_columns


def check_join(*, limit: int = 10, data_release: int = 17) -> None:
    rows = run_sql(join_check_sql(limit), data_release=data_release)
    assert_radec_agree(rows)
    logger.info("join check OK: %d rows agree on ra/dec within tolerance", len(rows))


_SCISERVER_POINTER = (
    "--source sciserver is driven from artifacts/sciserver_pull.py, not this package: the "
    "SciServer token stays in .env/artifacts and never enters the importable package "
    "(see docs/spec/data.md §3). Run, e.g.:\n"
    "    python artifacts/sciserver_pull.py --corpus {corpus} --limit {limit} --out {out}\n"
    "The package's pure pull helpers (chunking, corpus merge) live in galaxy_jepa.data.sciserver."
)


def pull_corpus(
    corpus: str,
    limit: int,
    out_dir: str | Path,
    *,
    source: str = "http",
    stamp_px: int = 256,
    data_release: int = 17,
    mag_min: float = 14.0,
    mag_max: float = 19.0,
    workers: int = 16,
) -> Path:
    if source == "sciserver":
        # The token-only-in-artifacts rule: the package never calls the SciServer Jobs API
        # nor handles the token. Fail loudly with a pointer to the artifacts driver.
        raise RuntimeError(_SCISERVER_POINTER.format(corpus=corpus, limit=limit, out=out_dir))
    if source != "http":
        raise ValueError(f"unknown pull source {source!r}; expected 'http' or 'sciserver'")
    from astropy.io import fits  # lazy: only the live pull needs astropy

    sql = (
        pretrain_sql(limit, mag_min=mag_min, mag_max=mag_max)
        if corpus == "pretrain"
        else probe_sql(limit)
    )
    rows = with_derived_columns(run_sql(sql, data_release=data_release))
    frames = FitsFrameSource(rows, stamp_px=stamp_px, data_release=data_release)

    # The bottleneck is the remote SDSS frame download, not local CPU — measured: threads
    # plateau ~16 workers and a process pool is *slower* (overhead, no GIL gain). So this is
    # I/O/server-bound: a shared thread pool over one FitsFrameSource (which keeps an
    # HTTP keep-alive session + a frame dedup cache) is the right tool. map preserves input
    # order → deterministic corpus (ORDER BY). One bad frame is logged and skipped.
    def fetch(i: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        try:
            image, row = frames[i]
            return image, row
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the pull
            logger.warning("object %s fetch failed (%s); skipping", rows[i].get("object_id"), exc)
            return None, rows[i]

    fetched: list[tuple[np.ndarray, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for image, row in pool.map(fetch, range(len(frames))):
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
        return (
            f"{key}: [{arr.min():.4g}, {arr.max():.4g}] "
            f"median {np.median(arr):.4g} (n={len(vals)}/{len(rows)})"
        )

    def _bad_petro(r: dict[str, Any]) -> bool:
        good = _finite([r.get("petroRad_r")])
        return not good or good[0] <= 0

    bad_petro = sum(1 for r in rows if _bad_petro(r))
    fallback = 100.0 * bad_petro / len(rows) if rows else 0.0
    logger.info("--- pull summary (n=%d) ---", len(rows))
    for key in ("specz", "petroRad_r", "snr_r", "modelMag_r", "psfWidth_r", FEATURED_FRACTION_COL):
        if any(key in r for r in rows):
            logger.info("  %s", rng(key))
    # The probe label balance — a degenerate (all-smooth / all-featured) pull would make
    # the headline AUC meaningless, so surface it at scale before training.
    fracs = _finite([r.get(FEATURED_FRACTION_COL) for r in rows])
    if fracs:
        arr = np.asarray(fracs)
        featured = int((arr >= 0.5).sum())
        extremes = int(((arr <= 0.2) | (arr >= 0.8)).sum())
        logger.info(
            "  label balance: %d/%d featured (>=0.5); %d confident extremes (<=0.2 or >=0.8)",
            featured,
            len(fracs),
            extremes,
        )
    logger.info(
        "  global-box fallback rate (missing/≤0 petroRad_r): %.1f%% (%d/%d)",
        fallback,
        bad_petro,
        len(rows),
    )


# --- metadata top-ups on an already-pulled corpus (no re-pull, no image re-cut) ---------


def read_metadata(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Read a ``DirectorySource`` corpus's ``metadata.csv`` into row dicts (order preserved)."""
    path = Path(corpus_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{corpus_dir} has no metadata.csv")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_metadata(corpus_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Rewrite ``metadata.csv`` with ``object_id`` first and every other column sorted.

    Column order matches :func:`pull_corpus`'s writer so a backfilled corpus is
    byte-comparable with a freshly-pulled one.
    """
    if not rows:
        raise ValueError("refusing to write an empty metadata.csv")
    path = Path(corpus_dir) / "metadata.csv"
    fieldnames = ["object_id"] + sorted({k for r in rows for k in r} - {"object_id"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def backfill_derived(corpus_dir: str | Path) -> int:
    """Re-derive the derived columns over a corpus already on disk. Pure — no network.

    The SciServer driver path historically wrote the raw SQL rows straight through, so corpora
    pulled that way carry ``modelMagErr_r`` but no ``snr_r`` and the nuisance battery loses one
    of its five. ``snr_r`` is a *function of a column already present*, so the fix is arithmetic
    on disk, not a re-pull. Runs :func:`with_derived_columns` — the same derivation the pull
    uses — so the two paths cannot diverge.
    """
    rows = read_metadata(corpus_dir)
    if not any("modelMagErr_r" in r for r in rows):
        raise ValueError(
            f"{corpus_dir} has no modelMagErr_r column, so snr_r cannot be derived. The "
            "unlabelled pretraining pull (metadata.PRETRAIN_SQL) omits the nuisance battery "
            "deliberately — it is never probed — so this corpus needs no backfill."
        )
    with_derived_columns(rows)
    write_metadata(corpus_dir, rows)
    logger.info("backfilled derived columns for %d rows in %s", len(rows), corpus_dir)
    return len(rows)


def merge_columns(
    corpus_dir: str | Path,
    extra: list[dict[str, Any]],
    columns: Sequence[str],
    *,
    key: str = "objID",
) -> tuple[int, int]:
    """Join ``columns`` from ``extra`` into a corpus's ``metadata.csv``, keyed on object ID.

    Pure (the caller does the networked query), so the merge is unit-testable offline. Returns
    ``(matched, missing)``; a row with no match gets empty strings rather than being dropped —
    a partial catalogue top-up must never silently shrink the corpus.
    """
    lookup: dict[int, dict[str, Any]] = {}
    for row in extra:
        raw = row.get(key, row.get("object_id"))
        if raw is None:
            raise KeyError(f"top-up row has neither {key!r} nor 'object_id': {row!r}")
        lookup[int(raw)] = row

    rows = read_metadata(corpus_dir)
    matched = 0
    for row in rows:
        source = lookup.get(int(row["object_id"]))
        if source is None:
            for col in columns:
                row.setdefault(col, "")
            continue
        matched += 1
        for col in columns:
            row[col] = source.get(col, "")
    write_metadata(corpus_dir, rows)
    missing = len(rows) - matched
    logger.info(
        "merged %s into %s: %d matched, %d unmatched", list(columns), corpus_dir, matched, missing
    )
    return matched, missing


def pull_axis_ratios(limit: int, *, data_release: int = 17) -> list[dict[str, Any]]:
    """Run the catalogue-only axis-ratio query (D13). Networked; no image re-cut."""
    return run_sql(axis_ratio_sql(limit), data_release=data_release)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pull an SDSS corpus slice (FITS + metadata).")
    parser.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--source",
        choices=["http", "sciserver"],
        default="http",
        help="http: download frames + cut locally (this package). sciserver: native cuts "
        "server-side — driven from artifacts/sciserver_pull.py (token stays in artifacts).",
    )
    parser.add_argument("--stamp-px", type=int, default=256)
    parser.add_argument("--data-release", type=int, default=17)
    parser.add_argument("--check-join", action="store_true", help="run the 10-row join guard")
    parser.add_argument(
        "--backfill-derived",
        type=Path,
        metavar="CORPUS_DIR",
        help="re-derive snr_r in an existing corpus's metadata.csv (offline, no re-pull)",
    )
    parser.add_argument(
        "--axis-ratios",
        type=Path,
        metavar="CORPUS_DIR",
        help="catalogue-only expAB_r/deVAB_r top-up for an existing probe corpus (D13); "
        "--limit must match the corpus size so the deterministic object set lines up",
    )
    parser.add_argument("--workers", type=int, default=16, help="parallel frame-fetch threads")
    args = parser.parse_args(argv)

    if args.check_join:
        check_join(limit=args.limit if args.limit <= 50 else 10, data_release=args.data_release)
        return
    if args.backfill_derived is not None:
        backfill_derived(args.backfill_derived)
        return
    if args.axis_ratios is not None:
        n_rows = len(read_metadata(args.axis_ratios))
        if args.limit < n_rows:
            parser.error(
                f"--limit {args.limit} is below the corpus size {n_rows}: the top-up query is "
                "TOP-n over the same ORDER BY, so a short limit would leave rows unmatched"
            )
        extra = pull_axis_ratios(args.limit, data_release=args.data_release)
        matched, missing = merge_columns(args.axis_ratios, extra, AXIS_RATIO_COLS)
        logger.info("axis-ratio top-up: %d matched, %d unmatched", matched, missing)
        return
    if args.source == "sciserver":
        parser.error(_SCISERVER_POINTER.format(corpus=args.corpus, limit=args.limit, out=args.out))
    if args.out is None:
        parser.error("--out is required for a pull")
    pull_corpus(
        args.corpus,
        args.limit,
        args.out,
        source=args.source,
        stamp_px=args.stamp_px,
        data_release=args.data_release,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
