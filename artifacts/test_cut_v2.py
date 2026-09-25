"""Local unit test of the v2 cutter on synthetic frames (no SciServer, no network).

Each band gets its own TAN WCS with a different sub-pixel CRPIX, as real frames do. A pixel-
integrated Gaussian star sits exactly at the target's sky position. After ``cut_one`` the star
must sit at (127.5, 127.5) ± 0.005 px in every band; an injected (0.3, −0.2) px offset must
be recovered to ± 0.01 px; white-noise variance must survive the shift within 1%; the edge pad
must be bit-identical across channels, and the package's validity detector must find it.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sciserver_cut_v2 as C  # noqa: E402

from galaxy_jepa.data.validity import analyse_validity

H, W = 1489, 2048
RA0, DEC0 = 150.0, 2.0
SIG = 1.5
rng = np.random.default_rng(0)
CRPIX = {b: (1024.0 + rng.uniform(-3, 3), 744.0 + rng.uniform(-3, 3)) for b in C.BANDS}


def wcs(band: str) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [CRPIX[band][0] + 1, CRPIX[band][1] + 1]  # FITS 1-based
    rot = np.radians({"g": 0.02, "r": 0.0, "i": -0.03}[band])
    s = 0.396 / 3600
    w.wcs.cd = [[-s * np.cos(rot), s * np.sin(rot)], [s * np.sin(rot), s * np.cos(rot)]]
    return w


def star(x0: float, y0: float) -> np.ndarray:
    xe, ye = np.arange(W + 1) - 0.5, np.arange(H + 1) - 0.5
    fx = np.diff(erf((xe - x0) / (SIG * np.sqrt(2)))) / 2
    fy = np.diff(erf((ye - y0) / (SIG * np.sqrt(2)))) / 2
    return 1000.0 * fy[:, None] * fx[None, :]


def frames_for(ra: float, dec: float, noise: float = 0.0) -> dict:
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    out = {}
    for b in C.BANDS:
        w = wcs(b)
        x, y = skycoord_to_pixel(SkyCoord(ra, dec, unit="deg"), w, origin=0, mode="all")
        data = star(float(x), float(y)) + (rng.normal(0, noise, (H, W)) if noise else 0.0)
        out[b] = (f"frame-{b}", data, w, "0" * 64)
    return out


def run(ra: float, dec: float, noise: float = 0.0):
    C._FRAMES.clear()
    fr = frames_for(ra, dec, noise)
    C._frame = lambda band, row: fr[band]  # the one seam: frames come from memory, not the SAS
    row = {"objID": "1", "ra": repr(ra), "dec": repr(dec), "run": "1", "camcol": "1", "field": "1", "rerun": "301"}
    oid, arrs, log, _, err = C.cut_one(row)
    assert not err, err
    return arrs["v2"], log


def centroid(img: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    win = (np.abs(yy - 127.5) < 20) & (np.abs(xx - 127.5) < 20)
    m = np.where(win, img, 0)
    return float((m * xx).sum() / m.sum()), float((m * yy).sum() / m.sum())


def main() -> None:
    res = {}
    # 1. centring, every band, several sky positions (so the fractional parts vary)
    worst = 0.0
    for k in range(6):
        ra, dec = RA0 + rng.uniform(-0.05, 0.05), DEC0 + rng.uniform(-0.03, 0.03)
        arr, log = run(ra, dec)
        for bi, b in enumerate(C.BANDS):
            cx, cy = centroid(arr[bi])
            worst = max(worst, abs(cx - 127.5), abs(cy - 127.5))
            assert -0.5 < float(log[f"{b}_sx"]) <= 0.5 and -0.5 < float(log[f"{b}_sy"]) <= 0.5
    res["centring_worst_px"] = worst
    assert worst < 0.005, worst

    # 2. planted: move the star by (+0.3, −0.2) px in r only, keep the target → recovered offset
    ra, dec = RA0 + 0.01, DEC0 + 0.01
    C._FRAMES.clear()
    fr = frames_for(ra, dec)
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    x, y = skycoord_to_pixel(SkyCoord(ra, dec, unit="deg"), fr["r"][2], origin=0, mode="all")
    fr["r"] = ("frame-r", star(float(x) + 0.3, float(y) - 0.2), fr["r"][2], "0" * 64)
    C._frame = lambda band, row: fr[band]
    _, arrs, _, _, _ = C.cut_one({"objID": "2", "ra": repr(ra), "dec": repr(dec), "run": "1", "camcol": "1",
                                   "field": "1", "rerun": "301"})
    cx, cy = centroid(arrs["v2"][1])
    res["plant_recovered"] = (cx - 127.5, cy - 127.5)
    assert abs(cx - 127.5 - 0.3) < 0.01 and abs(cy - 127.5 + 0.2) < 0.01, res["plant_recovered"]

    # 3. white-noise variance preserved (interior; the shift is unitary on the odd padded cut)
    arr, _ = run(RA0 + 0.02, DEC0 - 0.01, noise=1.0)
    ratios = []
    for bi in range(3):
        a = arr[bi].astype(float).copy()
        a[108:148, 108:148] = np.nan  # the star
        ratios.append(float(np.nanvar(a)))
    res["noise_var_ratio"] = ratios
    assert all(abs(r - 1) < 0.01 for r in ratios), ratios

    # 4. near an edge: pad bit-identical in all channels, detector finds it, valid_frac agrees
    edge_x = W - 60.3  # target 60 px from the right edge → ~68 columns off frame
    w = wcs("r")
    ra_e, dec_e = (float(t) for t in w.pixel_to_world_values(edge_x, 700.2))
    arr, log = run(ra_e, dec_e, noise=1.0)
    pad = (arr == 0).all(axis=0)
    v = analyse_validity(arr)
    res["edge_pad_frac"] = float(pad.mean())
    res["detector_invalid_frac"] = v.invalid_fraction
    res["log_valid_frac"] = float(log["valid_frac"])
    assert pad[:, -40:].all() and not pad[:, :150].any()
    assert abs(v.invalid_fraction - pad.mean()) < 1e-9, (v.invalid_fraction, pad.mean())
    assert abs(float(log["valid_frac"]) - (1 - pad.mean())) < 1e-6

    # 5. v1 geometry at the frame's low corner, where `origin_original` clamps to 0: the log must
    # carry the virtual origin (negative) and the in-stamp position in (127, 128]
    ra_c, dec_c = (float(t) for t in w.pixel_to_world_values(20.4, 15.7))
    _, log = run(ra_c, dec_c)
    for b in C.BANDS:
        for a in "xy":
            rel, o = float(log[f"{b}_v1_rel{a}"]), int(log[f"{b}_v1_o{a}"])
            assert 127 < rel <= 128 and o < 0, (b, a, rel, o)
            assert abs(o + rel - float(log[f"{b}_{a}"])) < 1e-9  # origin + rel = the frame position
    res["v1_corner_rel"] = (float(log["r_v1_relx"]), float(log["r_v1_rely"]))
    print(res)
    print("CUT_V2 UNIT TEST: PASS")


if __name__ == "__main__":
    main()
