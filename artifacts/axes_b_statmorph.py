"""Part B, statmorph (Rodriguez-Gomez et al. 2019; statmorph 0.7.3, BSD) on the r-band stamps. Runs in its own venv so uv.lock is untouched:

    runs/almagest/axes_phys/sm_venv/bin/python artifacts/axes_b_statmorph.py bench 60
    runs/almagest/axes_phys/sm_venv/bin/python artifacts/axes_b_statmorph.py run --workers 2     # 20,000 test galaxies, fixed seed; resumable

Set-up, fixed before any measurement is compared with an embedding: r band of the native FITS stamp (nanomaggies, 0.396"/px, sky-subtracted upstream);
sky level and σ from a 3σ-clipped median of the pixels outside 2 × petroRad_r; sources at 1.5 σ in a 3-px-FWHM smoothed image, ≥ 10 px, deblended
(32 levels, 0.005); the target is the segment at the centre pixel (or nothing: counted as a failure); other segments masked. gain = 900 e⁻ per nanomaggy
(SDSS r: ~4.7 e⁻/DN × ~190 DN/nMgy; enters only the S/N and the fit weights). PSF: Gaussian of the field's psfWidth_r, for the Sérsic fit only.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/almagest/axes_phys/b"
N, SEED, PX, GAIN = 20_000, 20261125, 0.396, 900.0
KEYS = ("asymmetry", "gini", "m20", "smoothness", "sersic_n", "concentration", "sersic_rhalf", "sersic_ellip", "sn_per_pixel", "flag", "flag_sersic", "rpetro_circ", "ellipticity_asymmetry")


def sample():
    f = OUT / "statmorph_ids.csv"
    if not f.exists():
        raise SystemExit("statmorph_ids.csv is missing: axes_b.py ids writes it from the frozen test split")
    return pd.read_csv(f, dtype={"object_id": str})


def one(r):
    import statmorph
    from photutils.segmentation import deblend_sources, detect_sources
    from scipy.ndimage import gaussian_filter
    warnings.simplefilter("ignore")
    t0, out = time.time(), {"object_id": r["object_id"]}
    try:
        im = fits.getdata(REPO / "data/probe_v2" / f"{r['object_id']}.fits")[1].astype(np.float64)
        bad = ~np.isfinite(im)
        yy, xx = np.mgrid[:256, :256] - 127.5
        rad = np.hypot(xx, yy)
        sky = (rad > min(2 * r["petroRad_r"] / PX, 100)) & ~bad
        _, med, sd = sigma_clipped_stats(im[sky], sigma=3, maxiters=5)
        im = np.where(bad, 0, im - med)
        sm = gaussian_filter(im, 3 / 2.3548)
        _, _, sds = sigma_clipped_stats(sm[sky], sigma=3, maxiters=5)
        seg = detect_sources(sm, 1.5 * sds, 10)
        if seg is None:
            return {**out, "error": "no source"}
        seg = deblend_sources(sm, seg, 10, n_levels=32, contrast=0.005, progress_bar=False)
        c = seg.data[126:130, 126:130]
        lab = np.bincount(c[c > 0]).argmax() if (c > 0).any() else 0
        if not lab:
            return {**out, "error": "no segment at the centre"}
        mask = bad | ((seg.data > 0) & (seg.data != lab))
        s = r["psfWidth_r"] / PX / 2.3548
        k = np.arange(-12, 13)
        psf = np.exp(-(k[:, None] ** 2 + k[None] ** 2) / 2 / s**2)
        m = statmorph.source_morphology(
            im, (seg.data == lab).astype(int), mask=mask, gain=GAIN, psf=psf / psf.sum())[0]
        out |= {k_: float(getattr(m, k_)) for k_ in KEYS}
        out |= {"sky_sd": float(sd), "npix": int((seg.data == lab).sum())}
    except Exception as e:  # recorded and counted, never silently dropped
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    out["secs"] = round(time.time() - t0, 3)
    return out


def swap_free():
    try:
        return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("free = ")[1].split("M")[0])
    except (IndexError, ValueError):
        return 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "run"])
    ap.add_argument("n", nargs="?", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    S = sample()
    if a.mode == "bench":
        R = [one(r) for r in S.iloc[:a.n].to_dict("records")]
        d = pd.DataFrame(R)
        print(f"{len(d)} galaxies: {d.secs.median():.2f} s median, {d.secs.mean():.2f} mean; load {os.getloadavg()[0]:.1f}")
        print("errors:", d.error.value_counts().to_dict() if "error" in d else {}, "flag:", d.flag.value_counts().to_dict(), "flag_sersic:", d.flag_sersic.value_counts().to_dict())
        print(d[list(KEYS[:6]) + ["sn_per_pixel"]].describe().round(3).to_string())
        return
    f = OUT / "statmorph.jsonl"
    done = {json.loads(x)["object_id"] for x in f.read_text().splitlines()} if f.exists() else set()
    todo = [r for r in S.to_dict("records") if r["object_id"] not in done]
    print(f"{len(done):,} done, {len(todo):,} to go; load {os.getloadavg()[0]:.1f}", flush=True)
    t0, low = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=500) as pool, f.open("a") as fh:
        for k, o in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
            fh.write(json.dumps(o) + "\n")
            if k % 50 == 0:
                fh.flush()
                low = low + 1 if swap_free() < 300 else 0
                if low >= 3:
                    print("swap guard: free swap under 300 MB, stopping (resumable)", flush=True)
                    pool.terminate()
                    sys.exit(75)
            if k % 200 == 0:
                print(f"{len(done) + k}/{len(S)} {(time.time() - t0) / k:.2f} s each", flush=True)


if __name__ == "__main__":
    main()
