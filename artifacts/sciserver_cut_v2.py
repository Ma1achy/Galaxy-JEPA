"""Server-side aligned stamp cutter (v2) — runs INSIDE SciServer Compute (SDSS SAS mounted).

v1 (`sciserver_cut.py`) cut each band with ``Cutout2D`` on its own frame WCS, which snaps each
band to its own integer grid: g, r, i sat up to ±0.5 px apart, and that misregistration became
M's PC1/PC2 (aa_findings.md §AA3a). v2 puts the target at the same sub-pixel point in every band.

Per object, per band b, on that band's own frame:

1.  ``(x_b, y_b) = skycoord_to_pixel(coord, WCS(hdr_b), origin=0, mode="all")`` — 0-based, pixel
    centres on integers (the convention ``Cutout2D`` itself uses).
2.  A padded cut of P = 256 + 2·MARGIN + 1 = 321 px at integer origin ``o_b = floor(x_b − 159)``
    (so the target sits at 159 ≤ x_b − o_b < 160). Off-frame pixels are filled with the band's
    valid-region median (a zero step would ring) and flagged in a valid mask.
3.  **Centre convention: the target lands on (127.5, 127.5), 0-based, in the final 256²** — the
    corner shared by the four central pixels, the array's geometric centre. The applied shift is
    ``s_b = 159.5 − (x_b − o_b)`` ∈ (−0.5, 0.5] per axis; every band, r included, is shifted.
4.  Fourier shift: a phase ramp on the padded cut. P is odd, so there is no unpaired Nyquist
    bin and the shift is exactly unitary on a real image (an even length would halve Nyquist).
    The valid mask is shifted by the same operator and thresholded at 0.5.
5.  Beside a frame edge, the EDGE_ERODE = 2 px of real pixels nearest the fill are dropped too
    (the shift's sinc tails mix the fill into them). Crop the central 256². Pad = the union of the three bands' invalid masks, set to exactly 0.0
    in all three — the validity detector needs bit-identical zeros in every channel.

Also logged per object, from v1's own ``Cutout2D`` call at v1's coordinate (``v1_ra``/``v1_dec``):
v1's integer (virtual) origin — negative where the stamp overhangs the frame — and the target's
position inside the v1 stamp, in (127, 128] per axis — so the old corpora's
processing variables are recorded exactly, for free, alongside the new ones.

Outputs in ``out/``: ``<object_id>.fits`` (float32 (3,256,256), g,r,i — v1's layout),
``cut_log.csv`` (one row per written object), ``frames.csv`` (per frame: header SHA-256),
``failed.csv`` (object_id + exception — v1 swallowed these), then ``corpus.tar.gz``.

Env: ``TARGETS_CSV``; ``VARIANTS`` (default ``v2``; the pilot adds ``nomargin`` — no pad, the
edge-ringing control — and ``bilinear`` — a bilinear shift, the resampling-fingerprint plant;
``margin64`` — the border-convergence reference; extra variants land in ``out/variants/<name>/``); ``LOG_ONLY=1`` writes logs but no stamps.
"""

from __future__ import annotations

import csv
import glob
import hashlib
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
from astropy.wcs.utils import skycoord_to_pixel

CUTTER_VERSION = "v2"
BANDS = ("g", "r", "i")
STAMP = 256
MARGIN = 32
EDGE_ERODE = 2  # px of real pixels given up beside a frame edge: the sinc tails mix in the median fill
MARGINS = {"nomargin": 0, "margin64": 64}  # pilot variants; everything else uses MARGIN
CENTRE = (STAMP - 1) / 2  # 127.5
TARGETS_CSV = os.environ.get("TARGETS_CSV", "targets.csv")
VARIANTS = tuple(os.environ.get("VARIANTS", "v2").split(","))
LOG_ONLY = os.environ.get("LOG_ONLY", "") == "1"
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
LOG_BAND_COLS = ("x", "y", "ox", "oy", "sx", "sy", "valid_frac", "edge_dist", "frame_sha",
                 "v1_ox", "v1_oy", "v1_relx", "v1_rely")


def find_root() -> str | None:
    for root in ROOT_CANDIDATES:
        if glob.glob(root + "/dr17"):
            return root
    hits = glob.glob("/home/idies/workspace/**/dr17/**/frame-r-*.fits.bz2", recursive=True)
    return hits[0].split("/dr17/")[0] if hits else None


_ROOT: str | None = None
_FRAMES: dict[str, tuple] = {}  # band -> (path, data, wcs, sha); objID order keeps fields adjacent


def _init(root: str) -> None:
    global _ROOT
    _ROOT = root


def _frame(band: str, row: dict) -> tuple:
    path = f"{_ROOT}/" + FRAME_REL.format(rerun=int(row["rerun"]), run=int(row["run"]),
                                          camcol=int(row["camcol"]), field=int(row["field"]), band=band)
    hit = _FRAMES.get(band)
    if hit is None or hit[0] != path:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float64)
            sha = hashlib.sha256(hdr.tostring().encode()).hexdigest()
            _FRAMES[band] = (path, data, WCS(hdr), sha)
    return _FRAMES[band]


def fourier_shift(a: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Move content by (+sy, +sx) px: a phase ramp, periodic on the padded cut."""
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * (ky * sy + kx * sx))).real


def bilinear_shift(a: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Pilot plant only: a bilinear resample (smooths noise by an amount that depends on |s|)."""
    out = a
    for ax, s in ((0, sy), (1, sx)):
        k = int(np.floor(s))
        f = s - k
        r0 = np.roll(out, k, axis=ax)
        out = (1 - f) * r0 + f * np.roll(r0, 1, axis=ax)
    return out


def padded_cut(data: np.ndarray, x: float, y: float, margin: int) -> tuple:
    """Cut P = 256 + 2·margin + 1 (or 256 when margin = 0) around (x, y): array, valid, origin, shift."""
    p = STAMP + 2 * margin + (1 if margin else 0)
    c = CENTRE + margin  # the target's destination in the padded cut
    ox, oy = int(np.floor(x - c + 0.5)), int(np.floor(y - c + 0.5))
    sx, sy = c - (x - ox), c - (y - oy)
    cut = np.zeros((p, p))
    valid = np.zeros((p, p), bool)
    h, w = data.shape
    ya, yb, xa, xb = max(oy, 0), min(oy + p, h), max(ox, 0), min(ox + p, w)
    if ya < yb and xa < xb:
        cut[ya - oy:yb - oy, xa - ox:xb - ox] = data[ya:yb, xa:xb]
        valid[ya - oy:yb - oy, xa - ox:xb - ox] = True
    if not valid.any():
        raise ValueError("target off frame")
    cut[~valid] = np.median(cut[valid])
    return cut, valid, ox, oy, sx, sy


def erode(ok: np.ndarray, n: int) -> np.ndarray:
    """4-neighbour binary erosion, n steps; the array's own border does not erode."""
    for _ in range(n):
        e = ok.copy()
        e[1:, :] &= ok[:-1, :]
        e[:-1, :] &= ok[1:, :]
        e[:, 1:] &= ok[:, :-1]
        e[:, :-1] &= ok[:, 1:]
        ok = e
    return ok


def aligned(data: np.ndarray, x: float, y: float, variant: str) -> tuple:
    margin = MARGINS.get(variant, MARGIN)
    cut, valid, ox, oy, sx, sy = padded_cut(data, x, y, margin)
    shift = bilinear_shift if variant == "bilinear" else fourier_shift
    img = shift(cut, sy, sx)
    ok = shift(valid.astype(float), sy, sx) > 0.5
    if not valid.all():
        ok = erode(ok, EDGE_ERODE)
    sl = slice(margin, margin + STAMP)
    return img[sl, sl], ok[sl, sl], ox, oy, sx, sy


def cut_one(row: dict) -> tuple:
    """Return (object_id, {variant: (3,256,256) float32} | None, log row | None, frames, error)."""
    oid = str(row.get("objID") or row.get("object_id"))
    try:
        coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        v1_coord = SkyCoord(float(row.get("v1_ra") or row["ra"]) * u.deg,
                            float(row.get("v1_dec") or row["dec"]) * u.deg)
        log: dict = {"object_id": oid, "cutter": CUTTER_VERSION, "ra": row["ra"], "dec": row["dec"],
                     **{k: row[k] for k in ("run", "camcol", "field", "rerun")}}
        planes: dict = {v: [] for v in VARIANTS}
        masks: dict = {v: [] for v in VARIANTS}
        frames = {}
        for band in BANDS:
            path, data, wcs, sha = _frame(band, row)
            frames[path.rsplit("/", 1)[-1]] = sha
            x, y = (float(t) for t in skycoord_to_pixel(coord, wcs, origin=0, mode="all"))
            h, w = data.shape
            v1 = Cutout2D(data, v1_coord, size=STAMP, wcs=wcs, mode="partial", fill_value=0.0)
            v1x, v1y = (float(t) for t in skycoord_to_pixel(v1_coord, wcs, origin=0, mode="all"))
            # The stamp's virtual origin, not `origin_original`: that is the overlap's origin in the
            # frame and clamps to 0 at a frame edge (repull_findings.md §Pilot result).
            v1relx, v1rely = (float(t) for t in v1.input_position_cutout)
            v1ox, v1oy = round(v1x - v1relx), round(v1y - v1rely)
            for variant in VARIANTS:
                img, ok, ox, oy, sx, sy = aligned(data, x, y, variant)
                planes[variant].append(img)
                masks[variant].append(ok)
                if variant == VARIANTS[0]:
                    log.update({f"{band}_x": repr(x), f"{band}_y": repr(y), f"{band}_ox": ox, f"{band}_oy": oy,
                                f"{band}_sx": repr(sx), f"{band}_sy": repr(sy),
                                f"{band}_valid_frac": f"{ok.mean():.6f}",
                                f"{band}_edge_dist": f"{min(x, w - 1 - x, y, h - 1 - y):.3f}",
                                f"{band}_frame_sha": sha[:16],
                                f"{band}_v1_ox": v1ox, f"{band}_v1_oy": v1oy,
                                f"{band}_v1_relx": repr(v1relx), f"{band}_v1_rely": repr(v1rely)})
        out = {}
        for variant in VARIANTS:
            arr = np.stack(planes[variant])
            pad = ~np.logical_and.reduce(masks[variant])
            arr[:, pad] = 0.0
            if variant == VARIANTS[0]:
                log["valid_frac"] = f"{1 - pad.mean():.6f}"
            out[variant] = arr.astype(np.float32)
        return oid, (None if LOG_ONLY else out), log, frames, ""
    except Exception as e:  # noqa: BLE001 — one bad frame must not kill the pull; logged, not swallowed
        return oid, None, None, {}, f"{type(e).__name__}: {e}"


def main() -> None:
    root = find_root()
    print(f"[cut] SAS root={root!r} cutter={CUTTER_VERSION} variants={VARIANTS} log_only={LOG_ONLY}", flush=True)
    if root is None:
        sys.exit("FAIL — no DR17 frames on any mounted volume (is SDSS SAS ticked?)")
    with open(TARGETS_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[cut] {len(rows)} targets", flush=True)

    os.makedirs("out", exist_ok=True)
    for v in VARIANTS[1:]:
        os.makedirs(f"out/variants/{v}", exist_ok=True)
    log_cols = ["object_id", "cutter", "ra", "dec", "run", "camcol", "field", "rerun", "valid_frac",
                *(f"{b}_{c}" for b in BANDS for c in LOG_BAND_COLS)]
    frames: dict[str, str] = {}
    n_ok, n_fail = 0, 0
    t0 = time.time()
    with open("out/cut_log.csv", "w", newline="") as lf, open("out/failed.csv", "w", newline="") as ff, \
            ProcessPoolExecutor(max_workers=os.cpu_count() or 1, initializer=_init, initargs=(root,),
                                mp_context=mp.get_context("spawn")) as ex:
        lw = csv.DictWriter(lf, fieldnames=log_cols)
        fw = csv.writer(ff)
        lw.writeheader()
        fw.writerow(["object_id", "error"])
        for i, (oid, arrs, log, fr, err) in enumerate(ex.map(cut_one, rows, chunksize=16)):
            if log is None:
                fw.writerow([oid, err])
                n_fail += 1
            else:
                if arrs is not None:
                    for v, arr in arrs.items():
                        dest = f"out/{oid}.fits" if v == VARIANTS[0] else f"out/variants/{v}/{oid}.fits"
                        fits.PrimaryHDU(data=arr).writeto(dest, overwrite=True)
                lw.writerow(log)
                frames.update(fr)
                n_ok += 1
            if (i + 1) % 1000 == 0:
                print(f"[cut] {i + 1}/{len(rows)} ({n_ok} ok, {n_fail} failed)", flush=True)
    dt = time.time() - t0
    with open("out/frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "header_sha256"])
        w.writerows(sorted(frames.items()))
    with tarfile.open("corpus.tar.gz", "w:gz", compresslevel=1) as tar:
        tar.add("out", arcname=".")
    print(f"CUT {n_ok}/{len(rows)} stamps in {dt:.1f}s -> {n_ok / dt if dt > 0 else 0:.2f} gal/s "
          f"({os.cpu_count()} cores); {n_fail} failed; corpus.tar.gz="
          f"{os.path.getsize('corpus.tar.gz') / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
