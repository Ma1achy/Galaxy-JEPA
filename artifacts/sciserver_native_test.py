"""SciServer Compute feasibility + fidelity + throughput test for the native corpus pull.

Run this INSIDE a SciServer Compute container with the SDSS **SAS** data volume mounted
(Compute → new container → tick the "SDSS SAS" data volume). It needs only numpy +
astropy (both preinstalled on SciServer). It answers the three questions the corpus
decision hinges on:

  1. FEASIBILITY  — are the DR17 corrected frames actually on the mounted volume, and can
     we read + Cutout2D them server-side (no HTTP download of the 10 MB frames)?
  2. FIDELITY     — a stamp cut from a mounted native frame is byte-identical to one cut
     from a downloaded native frame (same FITS, same Cutout2D), so it passes the Rung-4
     test by construction. We re-confirm: white sky noise (lag-1 autocorr ≈ 0) and full
     high-frequency power — the exact things hips2fits failed.
  3. THROUGHPUT   — stamps/sec cutting native 64px g,r,i stamps from mounted frames. This
     is the number that decides corpus feasibility (extrapolated to 250k / ≥250k).

Prints a JSON-ish summary at the end; copy it back. Only ~50 KB stamps would ever leave
SciServer (written to your workspace), never the 10 MB frames.
"""

from __future__ import annotations

import glob
import time

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# A handful of real DR17 probe galaxies (run 752, camcol 1) — ra/dec + frame coords.
TARGETS = [
    # objID, ra, dec, run, camcol, field, rerun
    (1237648702966792270, 182.9253, -1.0924, 752, 1, 263, 301),
    (1237648702967251093, 183.30,  -0.95,    752, 1, 267, 301),
    (1237648702967906488, 183.70,  -0.80,    752, 1, 273, 301),
    (1237648702968103043, 183.95,  -0.70,    752, 1, 277, 301),
]
BANDS = ("g", "r", "i")
STAMP = 64

# Candidate SAS mount roots on SciServer Compute (auto-discovered).
ROOT_CANDIDATES = [
    "/home/idies/workspace/sdss_sas",
    "/home/idies/workspace/SDSS",
    "/home/idies/workspace/sdss",
    "/home/idies/workspace/SciServer/sdss_sas",
]
FRAME_REL = "dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"


def find_root() -> str | None:
    for root in ROOT_CANDIDATES:
        if glob.glob(root):
            # confirm a real frame exists under it
            t = TARGETS[0]
            p = f"{root}/" + FRAME_REL.format(rerun=t[6], run=t[3], camcol=t[4], field=t[5], band="r")
            if glob.glob(p):
                return root
    # last resort: search broadly for any DR17 frame
    hits = glob.glob("/home/idies/workspace/**/dr17/**/frame-r-*.fits.bz2", recursive=True)
    if hits:
        return hits[0].split("/dr17/")[0]
    return None


def lag1_autocorr(patch: np.ndarray) -> float:
    p = patch - patch.mean()
    var = float((p * p).mean())
    if var <= 0:
        return float("nan")
    h = float((p[:, :-1] * p[:, 1:]).mean())
    v = float((p[:-1, :] * p[1:, :]).mean())
    return (h + v) / 2.0 / var


def highk_fraction(patch: np.ndarray) -> float:
    c = patch - patch.mean()
    power = np.fft.fftshift(np.abs(np.fft.fft2(c)) ** 2)
    ny, nx = c.shape
    y, x = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(x - nx // 2, y - ny // 2)
    return float(power[rr >= nx // 2 * 0.6].sum() / power.sum())


def cut_stamp(root: str, ra: float, dec: float, run: int, camcol: int, field: int, rerun: int):
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    planes = []
    for band in BANDS:
        path = f"{root}/" + FRAME_REL.format(rerun=rerun, run=run, camcol=camcol, field=field, band=band)
        with fits.open(path) as hdul:  # reads from the MOUNTED volume, not HTTP
            cut = Cutout2D(hdul[0].data, coord, size=STAMP, wcs=WCS(hdul[0].header))
        planes.append(np.asarray(cut.data, dtype=np.float64))
    return np.stack(planes)


def main() -> None:
    root = find_root()
    print(f"[1] FEASIBILITY: SAS frame root = {root!r}")
    if root is None:
        print("    FAIL — no DR17 frames found on any mounted volume. Is the SDSS SAS "
              "data volume ticked for this container?")
        return

    # Fidelity on a real stamp (r-band): sky corners should be ~white, full high-k.
    img = cut_stamp(root, *TARGETS[0][1:])
    r = img[1]
    k = 14
    corners = [r[:k, :k], r[:k, -k:], r[-k:, :k], r[-k:, -k:]]
    lag1 = float(np.nanmean([lag1_autocorr(c) for c in corners]))
    hkf = float(np.nanmean([highk_fraction(c) for c in corners]))
    print(f"[2] FIDELITY (native, server-side): shape={img.shape} "
          f"sky lag-1 autocorr={lag1:.3f} (≈0 white)  sky high-k frac={hkf:.3f} "
          f"(~0.4-0.7 white; hips2fits was 0.36/0.44)")

    # Throughput: cut all targets repeatedly, time it.
    reps = 10
    t0 = time.time()
    n = 0
    for _ in range(reps):
        for tgt in TARGETS:
            cut_stamp(root, *tgt[1:])
            n += 1
    dt = time.time() - t0
    rate = n / dt
    print(f"[3] THROUGHPUT: {n} galaxies (x3 bands) in {dt:.1f}s = {rate:.1f} gal/s "
          f"→ 250k ≈ {250_000/rate/3600:.1f} h  (native HTTP was ~0.25 gal/s → 11.5 days)")
    print(f"\nSUMMARY: root={root!r} sky_lag1={lag1:.3f} sky_highk={hkf:.3f} "
          f"rate_gal_per_s={rate:.1f} est_250k_hours={250_000/rate/3600:.1f}")


if __name__ == "__main__":
    main()
