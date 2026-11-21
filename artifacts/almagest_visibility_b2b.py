"""Almagest B2b (user brief, 21 Nov 2026): B2 redone with in-support sources, full-frame cutouts and a k-correction.

EXPLORATORY; not in the science record; feeds the 8×8 pre-registration. B2's operator and checks
(`almagest_visibility.py`, imported, unchanged) with three changes the brief asks for:

1. **Sources in support (Malmquist):** per target z, only galaxies whose simulated r (modelMag_r + 5 log10(D_L1/D_L0)
   + the k-term) stays ≤ 17.77. n is reported per z.
2. **Full-frame cutouts:** each (galaxy, z) is cut at N = round(256/f) px from its DR17 frames on SciServer
   (`sciserver_cut_b2b.py`, strict: a box leaving the frame is excluded and counted) and shrunk to exactly 256 px
   there, so the simulated galaxy fills the stamp — no footprint, no padding, so no noise-only control is needed.
3. **k-correction:** kcorrect 5.1.9 (own venv), per-galaxy NMF SED fitted at z0 to aperture g/r/i photometry of
   the source stamp (circle of 2 R_petro), reconstructed at z1; the colour term scales each band. Declared: one
   SED per galaxy (not per pixel, as full FERENGI would), Galactic extinction ignored, modelMag_r as the stand-in
   for the survey's extinction-corrected Petrosian r ≤ 17.77.

Steps (one process each; the token is needed only by submit and fetch, and stays in .env):
    .venv/bin/python artifacts/almagest_visibility_b2b.py prep     # sample, photometry, kcorrect, selection, targets, PSF check
    .venv/bin/python artifacts/almagest_visibility_b2b.py submit   # SciServer cut jobs (chunks)
    .venv/bin/python artifacts/almagest_visibility_b2b.py fetch    # download and unpack the 256-px results
    .venv/bin/python artifacts/almagest_visibility_b2b.py finish   # flux, PSF, noise; embed; scores; checks; JSON; sheet

Investigation code: terse, excluded from lint/CI. Writes runs/almagest/a1/visibility/b2b/ and docs/almagest/.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import almagest_visibility as V  # noqa: E402  (B2's operator, targets and checks, unchanged)
import sciserver_cut_b2b as CUT  # noqa: E402

AXS, S, log, ci = V.AXS, V.S, V.log, V.ci
REPO = Path(__file__).resolve().parents[1]
B2B = V.OUT / "b2b"
KC_PY = V.OUT / "kc_venv/bin/python"
R_LIMIT = 17.77
N_PLANTED = 300
AP_RAD = 2.0       # aperture radius in R_petro
CAL_FLOOR = 0.03   # fractional flux error floor per band (calibration), added in quadrature
CHUNK = 1500       # rows per SciServer job (Small domain: 60-min cap)


def sample_b2(d, N):
    """B2's source sample, re-derived with B2's own constants (identical rows)."""
    test_rows = np.flatnonzero(np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok)
    z_te = N["redshift"][test_rows]
    low = test_rows[z_te < V.Z_MAX_SOURCE]
    big = low[N["size"][low] > np.nanmedian(N["size"][low])]
    return test_rows, big[N["redshift"][big] >= V.Z_MIN_SOURCE]


def targets_b2(d, st, N, test_rows):
    rng = np.random.default_rng(V.SEED)  # B2's seed: the same 400 real galaxies per target z
    z_te = N["redshift"][test_rows]
    tgt = {}
    for z in V.Z_TARGETS:
        near = test_rows[np.abs(z_te - z) < V.Z_WIN]
        pick = rng.choice(near, min(400, len(near)), replace=False)
        sig = np.array([V.sky(st.to_linear(st.cache.get(int(d.union[i]))))[1] for i in pick])
        tgt[z] = {"n_real": int(len(near)), "psf_fwhm_arcsec": float(np.nanmedian(N["psf"][near])),
                  "sky_sigma_linear": np.median(sig, axis=0).tolist(), "n_for_sky": int(len(pick))}
    return tgt


def frame_columns(ids) -> dict:
    want = {str(int(i)) for i in ids}
    out = {}
    with open(REPO / "data/probe_v2/metadata.csv", newline="") as h:
        for r in csv.DictReader(h):
            if r["object_id"] in want:
                out[int(r["object_id"])] = {k: r[k] for k in ("ra", "dec", "run", "camcol", "field", "rerun")}
    assert len(out) == len(want), f"frame columns found for {len(out)} of {len(want)}"
    return out


def aperture(x, petro_arcsec):
    """g/r/i flux (frame nanomaggies) inside a circle of AP_RAD·R_petro, sky (border median) removed; its noise."""
    n = x.shape[-1]
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.5
    r_px = min(AP_RAD * petro_arcsec / V.PIX, n / 2 - V.BORDER)
    m = (yy ** 2 + xx ** 2) <= r_px ** 2
    bg, sig = V.sky(x)
    flux = np.array([(x[c][m] - bg[c]).sum() for c in range(3)])
    err = sig * np.sqrt(m.sum())
    return flux, err, float(r_px)


def geometry(z0, z1, cosmo):
    f = float(cosmo.angular_diameter_distance(z0) / cosmo.angular_diameter_distance(z1))
    g = float((cosmo.luminosity_distance(z0) / cosmo.luminosity_distance(z1)) ** 2)
    n_src = int(round(256 / f))
    f_eff = 256 / n_src
    s_aa = max((1.0 / f_eff - 1.0) / 2.0, 0.0)  # B2's anti-alias rule, on the realised zoom
    return f, f_eff, g, n_src, s_aa


def psf_add_px(f_eff, s_aa, psf0, psf1):
    eff2 = (f_eff * psf0) ** 2 + (V.FWHM * s_aa * f_eff * V.PIX) ** 2  # B2's formula
    return float(np.sqrt(max(psf1 ** 2 - eff2, 0.0)) / V.FWHM / V.PIX)


def finish_stamp(x256, n_src, g, dm, add_px, sig1, rng):
    """Local half of the operator on the server's 256-px shrink: sky off, flux per new pixel × g·(N/256)² ×
    10^(−0.4 Δm_band), PSF top-up, white noise to the target sky σ over the whole stamp, sky back on."""
    from scipy.ndimage import gaussian_filter
    bg, _ = V.sky(x256)
    out = np.empty_like(x256, dtype=np.float64)
    sig_add = []
    for c in range(3):
        im = (x256[c] - bg[c]) * g * (n_src / 256) ** 2 * 10 ** (-0.4 * dm[c])
        if add_px > 0:
            im = gaussian_filter(im, add_px)
        b = V.border(im[None])[0]
        sres = 1.4826 * np.median(np.abs(b - np.median(b)))
        sa = np.sqrt(max(sig1[c] ** 2 - sres ** 2, 0.0))
        out[c] = im + rng.normal(0.0, sa, im.shape) + bg[c]
        sig_add.append(float(sa))
    return out, sig_add


def psf_check_b2b(cosmo, cases=((0.03, 0.07, 1.3, 1.4), (0.03, 0.10, 1.3, 1.5), (0.02, 0.13, 1.2, 1.5),
                                (0.045, 0.07, 1.6, 1.4))) -> list[dict]:
    """A Gaussian point source of FWHM psf0 on an N-px frame cut, through the shipped shrink and the local finish,
    must come out at max(psf1, f·psf0) within ±5% (the brief's check, on B2b's operator)."""
    out = []
    rng = np.random.default_rng(0)
    for z0, z1, p0, p1 in cases:
        f, f_eff, g, n_src, s_aa = geometry(z0, z1, cosmo)
        yy, xx = np.mgrid[:n_src, :n_src] - n_src / 2 + 0.5
        s = p0 / V.FWHM / V.PIX
        big = np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s))
        x256 = np.stack([CUT.shrink_to_256(big, s_aa)] * 3)
        add = psf_add_px(f_eff, s_aa, p0, p1)
        xs, _ = finish_stamp(x256, n_src, g, np.zeros(3), add, [1e-12] * 3, rng)
        y2, x2 = np.mgrid[:256, :256] - 128 + 0.5
        w = np.clip(xs[1], 0, None); w = w / w.sum()
        cy, cx = (w * y2).sum(), (w * x2).sum()
        got = V.FWHM * np.sqrt((w * ((y2 - cy) ** 2 + (x2 - cx) ** 2)).sum() / 2) * V.PIX
        tgt = max(p1, f_eff * p0)
        out.append({"z0": z0, "z1": z1, "psf0": p0, "psf1": p1, "n_src": n_src, "out": float(got), "rel_err": float(got / tgt - 1)})
        assert abs(got / tgt - 1) < 0.05, f"PSF check failed: {out[-1]}"
    return out


def step_prep() -> None:
    from astropy.cosmology import Planck18 as cosmo
    B2B.mkdir(parents=True, exist_ok=True)
    d = AXS.Data(untrained=False)
    st = V.Stretch()
    N = {n: d.lab.nuisance_value(n, d.union).astype(float) for n in S.NUIS}
    test_rows, sample = sample_b2(d, N)
    p95 = float(np.nanpercentile(N["redshift"][test_rows], 95))
    assert all(z < p95 for z in V.Z_TARGETS), f"a target z is above the test specz p95 ({p95:.3f})"
    oids = d.union[sample].astype(np.int64)
    log(f"b2b sample (B2's): {len(sample)}")
    tgt = targets_b2(d, st, N, test_rows)
    # aperture photometry → kcorrect input
    petro = N["size"][sample]
    kin = B2B / "kc_in.csv"
    phot = []
    with open(kin, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["tag", "z0"] + [f"maggies_{b}" for b in "gri"] + [f"ivar_{b}" for b in "gri"])
        for k, i in enumerate(sample):
            x = st.to_linear(st.cache.get(int(d.union[i])).astype(np.float32))
            flux, err, rap = aperture(x, petro[k])
            tot = np.sqrt(err ** 2 + (CAL_FLOOR * np.abs(flux)) ** 2)
            phot.append((flux, rap))
            w.writerow([int(oids[k]), N["redshift"][i]] + list(flux * 1e-9) + list(1.0 / (tot * 1e-9) ** 2))
    flux_r = np.array([p[0][1] for p in phot])
    ap_mag_r = 22.5 - 2.5 * np.log10(np.clip(flux_r, 1e-6, None))
    dmag = ap_mag_r - N["magnitude"][sample]
    log(f"b2b aperture r − modelMag_r: median {np.median(dmag):+.3f}, IQR {np.subtract(*np.percentile(dmag, [75, 25])):.3f}")
    kout = B2B / "kc_out.csv"
    r = subprocess.run([str(KC_PY), str(REPO / "artifacts/b2b_kcorrect.py"), str(kin), str(kout)]
                       + [f"{z}" for z in V.Z_TARGETS], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    log(r.stdout.strip())
    kc = {row["tag"]: row for row in csv.DictReader(open(kout, newline=""))}
    fr = frame_columns(oids)
    rows, sel = [], {}
    rng = np.random.default_rng(V.SEED + 77)
    for z in V.Z_TARGETS:
        keep = []
        for k, i in enumerate(sample):
            z0 = float(N["redshift"][i])
            f, f_eff, g, n_src, s_aa = geometry(z0, z, cosmo)
            dm = [float(kc[str(oids[k])][f"dm_{b}_{z:.2f}"]) for b in "gri"]
            r_sim = float(N["magnitude"][i] - 2.5 * np.log10(g) + dm[1])  # −2.5 log g = 5 log(D_L1/D_L0)
            if r_sim <= R_LIMIT:
                keep.append(k)
                rows.append({"tag": f"{oids[k]}_z{int(round(z * 100)):02d}", **fr[int(oids[k])], "size_px": n_src,
                             "s_aa": f"{s_aa:.6f}"})
        sel[f"z={z:.2f}"] = keep
        log(f"b2b z={z:.2f}: {len(keep)} of {len(sample)} stay at r ≤ {R_LIMIT}")
    planted = np.sort(rng.choice(len(sample), N_PLANTED, replace=False))
    rows += [{"tag": f"{oids[k]}_p", **fr[int(oids[k])], "size_px": 256, "s_aa": "0"} for k in planted]
    with open(B2B / "targets.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    psf = psf_check_b2b(cosmo)
    log("b2b psf check", json.dumps(psf))
    fitres = np.array([[float(kc[str(o)][f"fitres_{b}"]) for b in "gri"] for o in oids])
    prep = {"n_sample": int(len(sample)), "sample_rows": sample.tolist(), "oids": oids.tolist(), "targets": {str(k): v for k, v in tgt.items()},
            "selection": {k: v for k, v in sel.items()}, "n_selected": {k: len(v) for k, v in sel.items()},
            "planted": planted.tolist(), "n_rows": len(rows), "psf_check": psf, "specz_p95_test": p95,
            "aperture_r_minus_modelmag_r": {"median": float(np.median(dmag)), "iqr": float(np.subtract(*np.percentile(dmag, [75, 25])))},
            "kcorrect_fit_residual_mag_abs_median": [float(v) for v in np.median(np.abs(fitres), 0)],
            "declared": {"r_limit": R_LIMIT, "magnitude": "modelMag_r (stand-in for extinction-corrected Petrosian r)",
                         "aperture": f"{AP_RAD} R_petro circle, capped at {256 // 2 - V.BORDER} px", "cal_floor": CAL_FLOOR,
                         "kcorrect": "5.1.9, responses sdss_g0/r0/i0, one SED per galaxy, no Galactic extinction",
                         "cutout": "strict (whole box inside the frame) or excluded"}}
    (B2B / "prep.json").write_text(json.dumps(prep, indent=1, default=float))
    log(f"→ {B2B}/targets.csv ({len(rows)} rows), prep.json")


def _sciserver():
    import sciserver_pull as P
    from SciServer import Files, Jobs
    P.authenticate()
    P.CUTTER = Path(CUT.__file__)
    return P, Files, Jobs


def step_submit() -> None:
    P, Files, Jobs = _sciserver()
    rows = list(csv.DictReader(open(B2B / "targets.csv", newline="")))
    domain, image, sdss = P._pick_domain(Jobs, "Small")
    chunks = []
    for k in range(0, len(rows), CHUNK):
        chunks.append(P._submit_chunk(Jobs, Files, corpus="b2b", k=k // CHUNK, rows=rows[k:k + CHUNK], stamp_px=256,
                                      domain=domain, image=image, sdss=sdss))
    (B2B / "jobs.json").write_text(json.dumps({"chunks": chunks, "domain": domain.get("name"), "time": time.ctime()}, indent=1))
    log(f"b2b submitted {len(chunks)} job(s) for {len(rows)} rows")


def step_fetch() -> None:
    P, Files, Jobs = _sciserver()
    st = json.loads((B2B / "jobs.json").read_text())
    for ch in st["chunks"]:
        dest = B2B / "cut" / f"chunk{ch['k']}"
        if (dest / "reasons.csv").exists():
            continue
        P._fetch_chunk(Jobs, Files, ch, dest, poll_s=30, max_wait_min=240)
    log("b2b fetched all chunks")


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"prep": step_prep, "submit": step_submit, "fetch": step_fetch}.get(
        step, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()


if __name__ == "__main__":
    main()
