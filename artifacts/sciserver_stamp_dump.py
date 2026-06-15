"""Cut ONE native stamp server-side on SciServer and emit it base64 for byte-comparison.

Used to confirm a SciServer-cut native stamp is byte-identical to our HTTP-pulled native
stamp for the same galaxy (objID 1237648702966792270) — so the frozen asinh Q=4 transfers
without re-tuning. Prints a single STAMP_B64:<...> line (g,r,i float64 (3,64,64) .npy).
"""

from __future__ import annotations

import base64
import glob
import io

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# objID 1237648702966792270 — TARGETS[0] / fidelity natives[0]
RA, DEC, RUN, CAMCOL, FIELD, RERUN = 184.07200, -1.16268, 752, 1, 271, 301
BANDS = ("g", "r", "i")
STAMP = 64
FRAME_REL = "dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"


def find_root():
    for root in ("/home/idies/workspace/sdss_sas",):
        if glob.glob(root + "/dr17"):
            return root
    hits = glob.glob("/home/idies/workspace/**/dr17/**/frame-r-*.fits.bz2", recursive=True)
    return hits[0].split("/dr17/")[0] if hits else None


def main() -> None:
    root = find_root()
    coord = SkyCoord(RA * u.deg, DEC * u.deg)
    planes = []
    for band in BANDS:
        path = f"{root}/" + FRAME_REL.format(rerun=RERUN, run=RUN, camcol=CAMCOL, field=FIELD, band=band)
        with fits.open(path) as hdul:
            cut = Cutout2D(hdul[0].data, coord, size=STAMP, wcs=WCS(hdul[0].header))
        planes.append(np.asarray(cut.data, dtype=np.float64))
    arr = np.stack(planes)
    buf = io.BytesIO()
    np.save(buf, arr)
    print("STAMP_B64:" + base64.b64encode(buf.getvalue()).decode())


if __name__ == "__main__":
    main()
