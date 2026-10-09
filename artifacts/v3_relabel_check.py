"""V3′ label re-projection to v2 geometry: the data-only checks before any aligned V3′ map (user,
2026-10-09). Nothing is fitted; no encoder is read.

  0. Consistency, all 794: dd_v3's v1 origin equals the cutter's logged r_v1_ox/oy; the cut_log's v2
     coordinate through dd_v3's r header gives the logged r_x/r_y; v1 labels reproduce the stored ones.
  a. Synthetic (50 galaxies, real r headers and GZ3D grids): an elliptical Gaussian feature rendered into
     g, r, i frames whose WCS carry known per-band shifts, cut by the v2 cutter's own `aligned`, and the
     same feature drawn as a GZ3D mask. The re-projected label's centroid must land on the feature's in
     every band. Pass (fixed before the run): max error ≤ 0.2 px.
  b. Real (500 of the 794): GZ3D's own image (what the volunteers drew on) sampled through each label
     geometry, phase-correlated with the v2 stamp's light (g+r+i) over the GZ3D footprint. Pass (fixed
     before the run): where the origin moved ≥ 0.5 px, the median of |off_v1| − |off_v2| is > 0; the
     re-projected median |offset| ≤ 0.25 px; where it moved < 0.1 px, the two agree (median |diff|
     ≤ 0.05 px). Also reported: label-weighted flux of the listed structure under each geometry.

  uv run python artifacts/v3_relabel_check.py     # -> artifacts/out/aligned/v3_relabel_check.json
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dd_v3 as V  # noqa: E402
from sciserver_cut_v2 import aligned  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "out" / "aligned" / "v3_relabel_check.json"
SEED, N_SYN, N_REAL = 20261009, 50, 500
SYN_MAX, REAL_REP_MAX, REAL_SAME_MAX = 0.2, 0.25, 0.05


def _log() -> dict[int, dict]:
    with open(REPO / "data" / "probe_v2" / "cut_log.csv") as fh:
        return {int(r["object_id"]): r for r in csv.DictReader(fh)}


def _centroid(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    r, c = np.indices(w.shape)
    return np.array([(w * r).sum(), (w * c).sum()]) / w.sum()  # (row, col) = (y, x)


def measure(a: np.ndarray, b: np.ndarray, kmax: float = 0.12) -> np.ndarray:
    """(dy, dx) such that b ≈ a moved by (dy, dx): weighted least squares on the cross-power phase."""
    yy, xx = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing="ij")
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2(w * (a - a.mean())), np.fft.fft2(w * (b - b.mean()))
    c = B * np.conj(A)
    band = (np.hypot(yy, xx) > 0) & (np.hypot(yy, xx) < kmax)
    ph, wt = -np.angle(c[band]) / (2 * np.pi), np.abs(c[band])
    m = np.stack([yy[band], xx[band]], 1) * np.sqrt(wt)[:, None]
    return np.linalg.lstsq(m, ph * np.sqrt(wt), rcond=None)[0]


def consistency(ts, union, log, org) -> dict:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    bad_origin, dxy, dorg = [], [], []
    for oid in union:
        t, r = ts[oid], log[oid]
        fw = V._frame_wcs(t)
        x0, y0, _, _ = V.stamp_origin(t, fw)
        if (x0, y0) != (int(r["r_v1_ox"]), int(r["r_v1_oy"])):
            bad_origin.append(oid)
        x, y = (float(v) for v in skycoord_to_pixel(SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg), fw, mode="all"))
        dxy.append(max(abs(x - float(r["r_x"])), abs(y - float(r["r_y"]))))
        dorg.append((org[oid][0] - x0, org[oid][1] - y0))
    v1 = np.load(V.LOCAL / "v3_patch_labels.npz")
    regress = [all(np.array_equal(v, v1[f"{oid}__{k}"]) for k, v in V.label_rec(V.reproject(ts[oid])).items())
               for oid in union[:30]]
    d = np.array(dorg)
    return {"n": len(union), "v1_origin_mismatches": bad_origin, "v2_xy_max_abs_px": float(max(dxy)),
            "v1_labels_reproduce (30)": all(regress),
            "origin_shift_v2_minus_v1_px": {"x_min_max": [float(d[:, 0].min()), float(d[:, 0].max())],
                                            "y_min_max": [float(d[:, 1].min()), float(d[:, 1].max())],
                                            "median_abs": [float(np.median(np.abs(d[:, 0]))), float(np.median(np.abs(d[:, 1])))]}}


def synthetic(ts, union, log, org, rng) -> dict:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    real_load = V.load_gz3d
    rows, err_v2, err_v1 = [], [], []
    for oid in rng.choice(union, N_SYN, replace=False):
        oid = int(oid)
        t, r = ts[oid], log[oid]
        fw = V._frame_wcs(t)
        shape = (int(fw.pixel_shape[1]), int(fw.pixel_shape[0])) if fw.pixel_shape else (1489, 2048)
        xc, yc = float(r["r_x"]) + rng.uniform(-30, 30), float(r["r_y"]) + rng.uniform(-30, 30)
        sx_, sy_, th = rng.uniform(3, 8), rng.uniform(3, 8), rng.uniform(0, np.pi)

        def prof(xr, yr):  # the feature, defined on the r frame
            dx, dy = xr - xc, yr - yc
            a = np.cos(th) * dx + np.sin(th) * dy
            b = -np.sin(th) * dx + np.cos(th) * dy
            return np.exp(-0.5 * ((a / sx_) ** 2 + (b / sy_) ** 2))

        coord = SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg)
        shifts = {"g": rng.uniform(-3, 3, 2), "r": np.zeros(2), "i": rng.uniform(-3, 3, 2)}
        cents = {}
        for band, s in shifts.items():
            wb = copy.deepcopy(fw)
            wb.wcs.crpix = wb.wcs.crpix + s  # a known per-band shift of the frame grid
            wb.wcs.set()
            xb, yb = (float(v) for v in skycoord_to_pixel(coord, wb, mode="all"))
            data = np.zeros(shape)
            ya, xa = int(yb) - 200, int(xb) - 200
            yy, xx = np.mgrid[max(ya, 0):min(ya + 400, shape[0]), max(xa, 0):min(xa + 400, shape[1])]
            xr, yr = fw.world_to_pixel(wb.pixel_to_world(xx, yy))
            data[yy, xx] = prof(xr, yr)
            img, _, _, _, _, _ = aligned(data, xb, yb, "v2")
            cents[band] = _centroid(img)
        gw, rgb, masks, votes = real_load(t)
        gy, gx = np.indices(masks["bar"].shape)
        xr, yr = fw.world_to_pixel(gw.pixel_to_world(gx, gy))
        syn = np.where(prof(xr, yr) >= 0.5, 5.0, 0.0).astype(np.float32)
        V.load_gz3d = lambda _t, gw=gw, rgb=rgb, syn=syn, votes=votes: (gw, rgb, {"bar": syn, "spiral": syn}, votes)
        try:
            lab2 = _centroid(V.reproject(t, org[oid])["bar"][3])
            lab1 = _centroid(V.reproject(t)["bar"][3])
        finally:
            V.load_gz3d = real_load
        e2 = {b: float(np.abs(lab2 - c).max()) for b, c in cents.items()}
        e1 = {b: float(np.abs(lab1 - c).max()) for b, c in cents.items()}
        err_v2 += e2.values()
        err_v1 += e1.values()
        rows.append({"id": oid, "band_shifts_px": {b: s.round(3).tolist() for b, s in shifts.items()},
                     "err_reprojected_px": e2, "err_v1_labels_px": e1,
                     "band_centroid_spread_px": float(np.ptp(np.stack(list(cents.values())), 0).max())})
    return {"n": N_SYN, "max_err_reprojected_px": float(max(err_v2)), "median_err_reprojected_px": float(np.median(err_v2)),
            "max_err_v1_labels_px": float(max(err_v1)), "median_err_v1_labels_px": float(np.median(err_v1)),
            "max_band_centroid_spread_px": float(max(r_["band_centroid_spread_px"] for r_ in rows)),
            "pass": bool(max(err_v2) <= SYN_MAX), "rows": rows}


def _sampled(rp: dict, rgb: np.ndarray) -> np.ndarray:
    gx, gy = rp["gz_xy"]
    ix, iy = np.rint(gx).astype(int), np.rint(gy).astype(int)
    n = rgb.shape[:2]
    inside = (ix >= 0) & (iy >= 0) & (ix < n[1]) & (iy < n[0])
    lum = rgb.astype(np.float64).mean(2)[np.clip(iy, 0, n[0] - 1), np.clip(ix, 0, n[1] - 1)] * inside
    return V._pool(lum)


def real(ts, union, lst, org, rng) -> dict:
    from galaxy_jepa.data.sources import load_fits_stamp
    rows = []
    for oid in sorted(int(o) for o in rng.choice(union, N_REAL, replace=False)):
        t = ts[oid]
        light = np.asarray(load_fits_stamp(REPO / "data" / "probe_v2" / f"{oid}.fits"), np.float64).sum(0)
        light_v1 = np.asarray(load_fits_stamp(REPO / "data" / "probe" / f"{oid}.fits"), np.float64).sum(0)
        _, rgb, _, _ = V.load_gz3d(t)
        rp1, rp2 = V.reproject(t), V.reproject(t, org[oid])
        fp = rp1["footprint"] & rp2["footprint"]
        rr, cc = np.where(fp)
        box = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        s1 = _sampled(rp1, rgb)
        o1, o2 = measure(s1[box], light[box]), measure(_sampled(rp2, rgb)[box], light[box])
        o11 = measure(s1[box], light_v1[box])  # reported only: M's own pairing, v1 labels on v1 stamps
        k = "spiral" if oid in lst["spiral"] or oid in lst["spiral_f25"] else "bar"
        flux = {}
        for name, rp in (("v1", rp1), ("v2", rp2)):
            f = rp[k][3] * fp
            flux[name] = float((f * light).sum() / f.sum() / light[fp].mean()) if f.sum() else float("nan")
        # shift_px is (x, y); the offsets are (y, x), as `measure` returns them
        rows.append({"id": oid, "shift_px": [org[oid][0] - rp1["x0"], org[oid][1] - rp1["y0"]],
                     "off_v1": o1.tolist(), "off_v2": o2.tolist(), "off_v1_on_v1_stamp": o11.tolist(), "structure": k, "label_flux": flux})
    sh = np.array([np.hypot(*r_["shift_px"]) for r_ in rows])
    a1 = np.array([np.hypot(*r_["off_v1"]) for r_ in rows])
    a2 = np.array([np.hypot(*r_["off_v2"]) for r_ in rows])
    big, small = sh >= 0.5, sh < 0.1
    o1s, o2s, o11s = (np.array([r_[k] for r_ in rows]) for k in ("off_v1", "off_v2", "off_v1_on_v1_stamp"))
    a11 = np.hypot(o11s[:, 0], o11s[:, 1])
    d = np.array([r_["shift_px"][::-1] for r_ in rows])  # (y, x)
    e = o1s - o2s
    fl = np.array([[r_["label_flux"]["v1"], r_["label_flux"]["v2"]] for r_ in rows])
    ok = ~np.isnan(fl).any(1)
    out = {"n": len(rows), "n_shift_ge_0.5": int(big.sum()), "n_shift_lt_0.1": int(small.sum()),
           "abs_offset_v1_px": {"median": float(np.median(a1)), "p90": float(np.percentile(a1, 90))},
           "abs_offset_v2_px": {"median": float(np.median(a2)), "p90": float(np.percentile(a2, 90))},
           "shift_ge_0.5": {"median_improvement_px": float(np.median(a1[big] - a2[big])),
                            "frac_v2_better": float((a2[big] < a1[big]).mean())},
           "shift_lt_0.1": {"median_abs_diff_px": float(np.median(np.abs(a1[small] - a2[small]))) if small.any() else None},
           "offset_difference_vs_shift_slope": float(np.polyfit(sh, a1 - a2, 1)[0]),
           "reported: off_v1 - off_v2 against the origin shift (y, x), slope and r":
               [[float(np.polyfit(d[:, ax], e[:, ax], 1)[0]), float(np.corrcoef(d[:, ax], e[:, ax])[0, 1])] for ax in (0, 1)],
           "reported: M's own pairing, v1 labels on v1 stamps": {
               "median_abs_px": float(np.median(a11)), "median_offset_yx": np.median(o11s, 0).tolist()},
           "reported: median offset (y, x)": {"v1_on_v2": np.median(o1s, 0).tolist(), "v2_on_v2": np.median(o2s, 0).tolist()},
           "label_weighted_flux (reported)": {"median_v1": float(np.median(fl[ok, 0])), "median_v2": float(np.median(fl[ok, 1])),
                                              "frac_v2_higher": float((fl[ok, 1] > fl[ok, 0]).mean()),
                                              "frac_v2_higher_shift_ge_0.5": float((fl[ok & big, 1] > fl[ok & big, 0]).mean())}}
    out["pass"] = bool(out["shift_ge_0.5"]["median_improvement_px"] > 0 and out["abs_offset_v2_px"]["median"] <= REAL_REP_MAX
                       and (out["shift_lt_0.1"]["median_abs_diff_px"] is None or out["shift_lt_0.1"]["median_abs_diff_px"] <= REAL_SAME_MAX))
    out["rows"] = rows
    return out


def main() -> dict:
    ts = {t["id"]: t for t in json.loads((V.LOCAL / "v3_targets.json").read_text())}
    lst = {k: set(v) for k, v in json.loads((V.LOCAL / "v3_list.json").read_text())["lists"].items()}
    union = sorted(set().union(*lst.values()))
    log, org = _log(), V.v2_origins()
    rng = np.random.default_rng(SEED)
    res = {"criteria (fixed before the run)": {"synthetic_max_err_px": SYN_MAX, "real_reprojected_median_abs_px": REAL_REP_MAX,
                                               "real_same_shift_median_abs_diff_px": REAL_SAME_MAX,
                                               "real_shift_ge_0.5": "median(|off_v1| - |off_v2|) > 0"},
           "consistency": consistency(ts, union, log, org)}
    res["synthetic"] = synthetic(ts, union, log, org, rng)
    res["real"] = real(ts, union, lst, org, rng)
    c = res["consistency"]
    res["pass"] = bool(not c["v1_origin_mismatches"] and c["v1_labels_reproduce (30)"] and c["v2_xy_max_abs_px"] < 0.01
                       and res["synthetic"]["pass"] and res["real"]["pass"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v)
                      for k, v in r.items()}, indent=1, default=float))
