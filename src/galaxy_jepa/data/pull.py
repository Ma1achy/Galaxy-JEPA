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
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

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
) -> Path:
    sql = (
        pretrain_sql(limit, mag_min=mag_min, mag_max=mag_max)
        if corpus == "pretrain"
        else probe_sql(limit)
    )
    rows = _with_derived(run_sql(sql, data_release=data_release))
    source = FitsFrameSource(rows, stamp_px=stamp_px, data_release=data_release)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for i in range(len(source)):
        image, row = source[i]
        fits.PrimaryHDU(data=image.astype(np.float32)).writeto(
            out / f"{row['object_id']}.fits", overwrite=True
        )
        written.append(row)

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
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pull an SDSS corpus slice (FITS + metadata).")
    parser.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stamp-px", type=int, default=64)
    parser.add_argument("--data-release", type=int, default=17)
    parser.add_argument("--check-join", action="store_true", help="run the 10-row join guard")
    args = parser.parse_args(argv)

    if args.check_join:
        check_join(limit=args.limit if args.limit <= 50 else 10, data_release=args.data_release)
        return
    if args.out is None:
        parser.error("--out is required for a pull")
    pull_corpus(args.corpus, args.limit, args.out, stamp_px=args.stamp_px,
                data_release=args.data_release)


if __name__ == "__main__":
    main()
