"""Almagest B2b k-correction (21 Nov 2026): runs in its own venv (runs/almagest/a1/visibility/kc_venv, kcorrect 5.1.9),
not the project's, so uv.lock is untouched.

    runs/almagest/a1/visibility/kc_venv/bin/python artifacts/b2b_kcorrect.py IN.csv OUT.csv 0.07 0.10 0.13

IN.csv: tag, z0, maggies_g/r/i, ivar_g/r/i (aperture photometry of the source stamp). Per galaxy, fit kcorrect's
NMF SED coefficients at z0 from g, r, i, then reconstruct the model's g, r, i maggies at each target z with the same
coefficients. `reconstruct` carries no distance dimming (checked 21 Nov: Δm − 5 log(D_L1/D_L0) ≈ −Δm_dist), so
dm_<band>_<z> = −2.5 log10(m(z1) / m(z0)) is the colour (K) term alone; the caller adds 5 log10(D_L1/D_L0).
Also writes the fit's residual per band at z0 (mag) as a fit-quality column.
"""
import csv
import sys

import numpy as np
import kcorrect.kcorrect as K


def main() -> None:
    src, dst, zs = sys.argv[1], sys.argv[2], [float(z) for z in sys.argv[3:]]
    rows = list(csv.DictReader(open(src, newline="")))
    z0 = np.array([float(r["z0"]) for r in rows])
    m = np.array([[float(r[f"maggies_{b}"]) for b in "gri"] for r in rows])
    iv = np.array([[float(r[f"ivar_{b}"]) for b in "gri"] for r in rows])
    kc = K.Kcorrect(responses=["sdss_g0", "sdss_r0", "sdss_i0"])
    co = kc.fit_coeffs(redshift=z0, maggies=m, ivar=iv)
    m0 = kc.reconstruct(redshift=z0, coeffs=co)
    out = {"tag": [r["tag"] for r in rows]}
    for j, b in enumerate("gri"):
        out[f"fitres_{b}"] = -2.5 * np.log10(m0[:, j] / m[:, j])
    for z in zs:
        m1 = kc.reconstruct(redshift=np.full(len(rows), z), coeffs=co)
        for j, b in enumerate("gri"):
            out[f"dm_{b}_{z:.2f}"] = -2.5 * np.log10(m1[:, j] / m0[:, j])
    keys = list(out)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(rows)):
            w.writerow([out[k][i] if k == "tag" else f"{out[k][i]:.6f}" for k in keys])
    print(f"kcorrect {K.__name__}: {len(rows)} galaxies, {len(zs)} targets -> {dst}")


if __name__ == "__main__":
    main()
