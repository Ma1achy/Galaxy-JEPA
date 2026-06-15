"""Rung-4 fidelity gate — native frame stamp vs hips2fits cutout.

Decision gate (NOT production): adopting hips2fits means abandoning the no-rebin rule,
which is a Rung-4 protection. Resampling smooths high spatial frequencies and correlates
pixel noise — exactly where a Rung-4 ("absent from the pixels") result lives. So this
tests, on genuinely FAINT, STRUCTURED spirals (not bright ellipticals), the three things
Rung-4 depends on:

  1. high-frequency power — does resampling attenuate the high-k end (= resolution loss)?
  2. adjacent-pixel noise correlation — native ~independent; resampling correlates it
     (a correlated noise field is learnable as fake structure by JEPA).
  3. faint-feature contrast — is the faint structure still there, at the same contrast?

PASS = high-freq content + faint-feature contrast preserved within tolerance, introduced
noise correlation small enough not to be learnable.
"""

from __future__ import annotations

import time

import numpy as np
from astropy import units as u
from astroquery.hips2fits import hips2fits

from galaxy_jepa.data.metadata import run_sql
from galaxy_jepa.data.sources import FitsFrameSource

# Faint (16.3 < r < 17.6), extended-but-on-frame (4 < R_petro < 9"), high spiral confidence.
SELECT_SQL = """\
SELECT TOP 8 g.dr8objid AS objID, g.ra, g.dec,
    p.petroRad_r, p.modelMag_r, p.run, p.camcol, p.field, p.rerun,
    g.t04_spiral_a08_spiral_fraction AS f_spiral,
    g.t01_smooth_or_features_a02_features_or_disk_fraction AS f_feat
FROM zoo2MainSpecz AS g
JOIN PhotoObjAll AS p ON p.objID = g.dr8objid
WHERE p.modelMag_r BETWEEN 16.3 AND 17.6
  AND p.petroRad_r BETWEEN 4 AND 9
  AND g.t04_spiral_a08_spiral_fraction > 0.5
ORDER BY g.dr8objid"""


def radial_power(img: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged power spectrum P(k); index = integer radius in Fourier space."""
    a = img - img.mean()
    power = np.fft.fftshift(np.abs(np.fft.fft2(a)) ** 2)
    ny, nx = a.shape
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.hypot(x - nx // 2, y - ny // 2).astype(int)
    return np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def lag1_autocorr_2d(patch: np.ndarray) -> float:
    """Mean lag-1 (horizontal+vertical) autocorrelation coefficient of a 2D patch."""
    p = patch - patch.mean()
    var = float((p * p).mean())
    if var <= 0:
        return float("nan")
    h = float((p[:, :-1] * p[:, 1:]).mean())
    v = float((p[:-1, :] * p[1:, :]).mean())
    return (h + v) / 2.0 / var


def sky_lag1(img: np.ndarray, k: int = 14) -> float:
    corners = [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    return float(np.nanmean([lag1_autocorr_2d(c) for c in corners]))


def sky_stats(img: np.ndarray, k: int = 14) -> tuple[float, float]:
    corners = np.concatenate(
        [img[:k, :k].ravel(), img[:k, -k:].ravel(), img[-k:, :k].ravel(), img[-k:, -k:].ravel()]
    )
    med = float(np.median(corners))
    return med, float(np.median(np.abs(corners - med)))  # sky median, sky MAD


def faint_contrast(img: np.ndarray, inner: float = 8.0, outer: float = 20.0) -> float:
    """95th-percentile brightness in the faint annulus, above sky, in sky-MAD units."""
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.hypot(x - (nx - 1) / 2, y - (ny - 1) / 2)
    ann = img[(r >= inner) & (r < outer)]
    med, mad = sky_stats(img)
    if mad <= 0:
        return float("nan")
    return (float(np.percentile(ann, 95)) - med) / mad


def hips_r(ra: float, dec: float) -> np.ndarray:
    im = hips2fits.query(
        hips="CDS/P/SDSS9/r", width=64, height=64, ra=ra * u.deg, dec=dec * u.deg,
        fov=64 * 0.396 * u.arcsec, projection="TAN", format="fits",
    )
    return np.asarray(im[0].data, dtype=np.float64)


def main() -> None:
    rows = run_sql(SELECT_SQL)
    print(f"selected {len(rows)} faint structured spirals "
          f"(r in [16.3,17.6], R_petro in [4,9]\", f_spiral>0.5)")
    src = FitsFrameSource(rows, stamp_px=64)

    hi_k = slice(20, 31)  # high-frequency band (Nyquist ~32 for 64 px)
    per_gal = []
    natives, hipses = [], []
    for i, row in enumerate(rows):
        nat = np.asarray(src[i][0][1], dtype=np.float64)  # native r-band
        t = time.time()
        hip = hips_r(float(row["ra"]), float(row["dec"]))
        natives.append(nat)
        hipses.append(hip)

        pn, ph = radial_power(nat), radial_power(hip)
        hf_ratio = float(ph[hi_k].sum() / pn[hi_k].sum())  # <1 ⇒ hips attenuates high-freq
        nat_lag1, hip_lag1 = sky_lag1(nat), sky_lag1(hip)
        c_nat, c_hip = faint_contrast(nat), faint_contrast(hip)
        contrast_ratio = c_hip / c_nat if c_nat else float("nan")
        per_gal.append((hf_ratio, nat_lag1, hip_lag1, c_nat, c_hip, contrast_ratio))
        print(f"  obj {row['objID']} r={float(row['modelMag_r']):.1f} "
              f"Rp={float(row['petroRad_r']):.1f}\" f_sp={float(row['f_spiral']):.2f} "
              f"[hips {time.time()-t:.2f}s]: hf={hf_ratio:.3f} "
              f"lag1 nat={nat_lag1:.3f}/hips={hip_lag1:.3f} "
              f"contrast nat={c_nat:.1f}/hips={c_hip:.1f} ({contrast_ratio:.2f}x)")

    arr = np.array(per_gal, dtype=float)
    hf, nlag, hlag, cn, ch, cr = (arr[:, j] for j in range(6))
    print("\n=== VERDICT against the three criteria (median over galaxies) ===")
    print(f"1. HIGH-FREQ POWER  hips/native (high-k) = {np.nanmedian(hf):.3f}  "
          f"[range {np.nanmin(hf):.3f}-{np.nanmax(hf):.3f}]   (1.0 = no attenuation)")
    print(f"2. NOISE LAG-1 AUTOCORR  native={np.nanmedian(nlag):.3f}  hips={np.nanmedian(hlag):.3f}  "
          f"introduced Δ={np.nanmedian(hlag)-np.nanmedian(nlag):+.3f}  (0 = independent pixels)")
    print(f"3. FAINT-FEATURE CONTRAST  hips/native = {np.nanmedian(cr):.3f}  "
          f"[range {np.nanmin(cr):.3f}-{np.nanmax(cr):.3f}]   (1.0 = fully preserved)")

    np.savez("/workspaces/Galaxy-JEPA/artifacts/fidelity_arrays.npz",
             natives=np.array(natives), hipses=np.array(hipses), per_gal=arr)
    _figure(natives, hipses)


def _figure(natives: list, hipses: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(4, len(natives))
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for j in range(n):
        nat, hip = natives[j], hipses[j]

        def show(ax, img, title):
            lo, hi = np.percentile(img, 2), np.percentile(img, 99)
            ax.imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), origin="lower", cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        show(axes[0, j], nat, f"native #{j}")
        show(axes[1, j], hip, f"hips2fits #{j}")
        axes[2, j].loglog(radial_power(nat)[1:], color="tab:blue", label="native")
        axes[2, j].loglog(radial_power(hip)[1:], color="tab:red", label="hips")
        axes[2, j].set_title("radial power P(k)", fontsize=8)
        if j == 0:
            axes[2, j].legend(fontsize=7)
    fig.suptitle("Rung-4 fidelity: native vs hips2fits (rows: native / hips / power spectrum)")
    fig.tight_layout()
    fig.savefig("/workspaces/Galaxy-JEPA/artifacts/fidelity_compare.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
