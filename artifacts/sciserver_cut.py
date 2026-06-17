"""Server-side bulk native-stamp cutter — runs INSIDE SciServer Compute (SDSS SAS mounted).

Reads ``targets.csv`` from the job's CWD (the results folder), cuts native ``STAMP_PX``
g,r,i stamps from the mounted DR17 frames in parallel across all cores, and writes the
``DirectorySource`` corpus layout into ``out/`` (``<object_id>.fits`` + ``metadata.csv``),
then tars ``out/`` -> ``corpus.tar`` for a single-file download. Only the ~MB of stamps
ever leave SciServer — never the 10 MB frames (the whole reason for the SciServer path).

``targets.csv`` columns (from metadata.pretrain_sql / probe_sql): objID, ra, dec, run,
camcol, field, rerun, petroRad_r, ... (+ the GZ2 t01 fractions for the probe corpus).
Edge cutouts use ``mode='partial'`` (zero-fill off-frame) so a galaxy near a field
boundary is kept at full size instead of dropped — the ~15% shape failures the HTTP pull hit.

Env: ``STAMP_PX`` (default 256). Prints a ``CUT ...`` summary + gal/s throughput line.
"""

from __future__ import annotations

import csv
import glob
import multiprocessing as mp
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

BANDS = ("g", "r", "i")
STAMP = int(os.environ.get("STAMP_PX", "256"))
FRAME_REL = (
    "dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/"
    "frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
)
ROOT_CANDIDATES = [
    "/home/idies/workspace/sdss_sas",
    "/home/idies/workspace/SDSS",
    "/home/idies/workspace/sdss",
    "/home/idies/workspace/SciServer/sdss_sas",
]


def find_root() -> str | None:
    for root in ROOT_CANDIDATES:
        if glob.glob(root + "/dr17"):
            return root
    hits = glob.glob("/home/idies/workspace/**/dr17/**/frame-r-*.fits.bz2", recursive=True)
    return hits[0].split("/dr17/")[0] if hits else None


_ROOT: str | None = None


def _init(root: str) -> None:
    global _ROOT
    _ROOT = root


def cut_one(row: dict) -> tuple[str, "np.ndarray | None"]:
    """Cut a native (3, STAMP, STAMP) g,r,i stamp; return (object_id, array) or (id, None)."""
    oid = str(row.get("objID") or row.get("object_id"))
    try:
        coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        run, camcol = int(row["run"]), int(row["camcol"])
        field, rerun = int(row["field"]), int(row["rerun"])
        planes = []
        for band in BANDS:
            path = f"{_ROOT}/" + FRAME_REL.format(
                rerun=rerun, run=run, camcol=camcol, field=field, band=band
            )
            with fits.open(path) as hdul:
                cut = Cutout2D(
                    hdul[0].data,
                    coord,
                    size=STAMP,
                    wcs=WCS(hdul[0].header),
                    mode="partial",
                    fill_value=0.0,
                )
            planes.append(np.asarray(cut.data, dtype=np.float32))
        arr = np.stack(planes)
        if arr.shape != (3, STAMP, STAMP):
            return oid, None
        return oid, arr
    except Exception:  # noqa: BLE001 — one bad frame must not kill the pull
        return oid, None


def main() -> None:
    root = find_root()
    print(f"[cut] SAS root={root!r} STAMP={STAMP}", flush=True)
    if root is None:
        sys.exit("FAIL — no DR17 frames on any mounted volume (is SDSS SAS ticked?)")

    with open("targets.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[cut] {len(rows)} targets", flush=True)

    os.makedirs("out", exist_ok=True)
    by_id = {str(r.get("objID") or r.get("object_id")): r for r in rows}
    ncpu = os.cpu_count() or 1
    written: list[str] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=ncpu, initializer=_init, initargs=(root,), mp_context=mp.get_context("spawn")
    ) as ex:
        for i, (oid, arr) in enumerate(ex.map(cut_one, rows, chunksize=8)):
            if arr is not None:
                fits.PrimaryHDU(data=arr).writeto(f"out/{oid}.fits", overwrite=True)
                written.append(oid)
            if (i + 1) % 1000 == 0:
                print(f"[cut] {i + 1}/{len(rows)} ({len(written)} ok)", flush=True)
    dt = time.time() - t0

    # metadata.csv in DirectorySource form: object_id + passthrough columns (objID renamed).
    src_cols = [c for c in rows[0].keys() if c not in ("objID", "object_id")]
    fieldnames = ["object_id", *src_cols]
    with open("out/metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for oid in written:
            r = by_id[oid]
            w.writerow({"object_id": oid, **{c: r.get(c, "") for c in src_cols}})

    with tarfile.open("corpus.tar", "w") as tar:
        tar.add("out", arcname=".")

    rate = len(written) / dt if dt > 0 else 0.0
    n_failed = len(rows) - len(written)
    print(
        f"CUT {len(written)}/{len(rows)} stamps in {dt:.1f}s -> {rate:.2f} gal/s "
        f"({ncpu} cores); {n_failed} failed; corpus.tar="
        f"{os.path.getsize('corpus.tar') / 1e6:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
