"""SciServer-side cutter for Almagest B2b (21 Nov 2026): a large native cutout per (galaxy, target z), shrunk to 256 px.

Runs on SciServer Compute (shipped base64 by `almagest_visibility.py b2b_submit`, through `sciserver_pull`'s job
primitives). Reads ``targets.csv`` (columns: tag, ra, dec, run, camcol, field, rerun, size_px, s_aa) and, per row
and band (g, r, i), cuts a ``size_px`` square from the DR17 frame with ``mode="strict"`` (a box that leaves the frame
is a failure, never zero-filled), then applies ``shrink_to_256``: Gaussian anti-alias σ = s_aa (input px), linear
zoom to exactly 256 px. No flux scaling, PSF, noise or k-correction here — those run locally on the 256-px result
(B2's operator, in the same order). Writes ``out/<tag>.fits`` (float32, 3 × 256 × 256, frame nanomaggies per input
pixel, i.e. surface brightness preserved) and ``corpus.tar.gz``; prints a CUT line with the failure reasons.

``shrink_to_256`` is also imported locally by the B2b PSF check, so the check exercises the shipped code.
"""

from __future__ import annotations

import csv
import glob
import multiprocessing as mp
import os
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT_PX = 256
BANDS = ("g", "r", "i")
TARGETS_CSV = os.environ.get("TARGETS_CSV", "targets.csv")
FRAME_REL = (  # verbatim from sciserver_cut.py: the DR17 tree the cached corpus was cut from
    "dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/"
    "frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
)
ROOT_CANDIDATES = [
    "/home/idies/workspace/sdss_sas",
    "/home/idies/workspace/SDSS",
    "/home/idies/workspace/sdss",
    "/home/idies/workspace/SciServer/sdss_sas",
]


def shrink_to_256(im: np.ndarray, s_aa: float) -> np.ndarray:
    """(N, N) → (256, 256): Gaussian anti-alias σ = s_aa input px (reflect edges), then linear zoom; values are
    interpolated, not summed, so a constant stays constant and flux per pixel is rescaled downstream."""
    from scipy.ndimage import gaussian_filter, zoom
    im = np.asarray(im, dtype=np.float64)
    if s_aa > 0:
        im = gaussian_filter(im, s_aa)
    n = im.shape[-1]
    if n == OUT_PX:
        return im.astype(np.float32)
    out = zoom(im, OUT_PX / n, order=1)
    assert out.shape == (OUT_PX, OUT_PX), out.shape
    return out.astype(np.float32)


def find_root() -> str | None:  # verbatim from sciserver_cut.py
    for root in ROOT_CANDIDATES:
        if glob.glob(root + "/dr17"):
            return root
    hits = glob.glob("/home/idies/workspace/**/dr17/**/frame-r-*.fits.bz2", recursive=True)
    return hits[0].split("/dr17/")[0] if hits else None


_ROOT: str | None = None


def _init(root: str) -> None:
    global _ROOT
    _ROOT = root


def cut_one(row: dict) -> tuple[str, "np.ndarray | None", str]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.nddata.utils import NoOverlapError, PartialOverlapError
    from astropy.wcs import WCS
    tag = row["tag"]
    try:
        coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        size, s_aa = int(row["size_px"]), float(row["s_aa"])
        planes = []
        for band in BANDS:
            path = f"{_ROOT}/" + FRAME_REL.format(rerun=int(row["rerun"]), run=int(row["run"]), camcol=int(row["camcol"]),
                                                  field=int(row["field"]), band=band)
            with fits.open(path) as hdul:
                cut = Cutout2D(hdul[0].data, coord, size=size, wcs=WCS(hdul[0].header), mode="strict")
            planes.append(shrink_to_256(cut.data, s_aa))
        return tag, np.stack(planes), "ok"
    except (PartialOverlapError, NoOverlapError):
        return tag, None, "outside_frame"
    except FileNotFoundError:
        return tag, None, "no_frame"
    except Exception as exc:  # noqa: BLE001 — one bad row must not kill the job; the reason is counted
        return tag, None, type(exc).__name__


def main() -> None:
    from astropy.io import fits
    root = find_root()
    print(f"[cut-b2b] SAS root={root!r}", flush=True)
    if root is None:
        sys.exit("FAIL — no DR17 frames on any mounted volume (is SDSS SAS ticked?)")
    with open(TARGETS_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[cut-b2b] {len(rows)} targets", flush=True)
    os.makedirs("out", exist_ok=True)
    why: Counter = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, initializer=_init, initargs=(root,),
                             mp_context=mp.get_context("spawn")) as ex:
        for i, (tag, arr, reason) in enumerate(ex.map(cut_one, rows, chunksize=4)):
            why[reason] += 1
            if arr is not None:
                fits.PrimaryHDU(data=arr).writeto(f"out/{tag}.fits", overwrite=True)
            if (i + 1) % 250 == 0:
                print(f"[cut-b2b] {i + 1}/{len(rows)} {dict(why)}", flush=True)
    dt = time.time() - t0
    with open("out/reasons.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reason", "n"])
        w.writerows(sorted(why.items()))
    with tarfile.open("corpus.tar.gz", "w:gz", compresslevel=1) as tar:
        tar.add("out", arcname=".")
    print(f"CUT {why['ok']}/{len(rows)} in {dt:.1f}s; reasons {dict(why)}; "
          f"corpus.tar.gz={os.path.getsize('corpus.tar.gz') / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
