"""Seeded synthetic fixture corpus — the offline data the integration tier runs on.

Implements the fixture half of ``docs/spec/testing.md`` §3 ("Tiny committed fixtures").
Produces a dozen tiny 3-channel (g, r, i) FITS galaxy stamps plus a ``metadata.csv``,
fully deterministic from ``seed`` so the fixture is reproducible without committing
binaries. Real cached cutouts drop into the same layout at the eyeball gate.

Two stamps are special and load-bearing for the stretch-sanity check
(``docs/spec/validation.md`` ``T2.stretch-sanity``):

* a **blank-sky control** (``kind="sky"``) — pure noise floor, no source;
* a **faint-arm exemplar** (``kind="faint_arm"``) — a low-surface-brightness spiral
  arm and *no* bright bulge, so it isolates the question the stretch exists to answer:
  does the faint arm survive stretch+normalise while the sky floor stays controlled?

Run as ``python -m tests.fixtures.generate`` (or import :func:`generate_fixture_corpus`)
to materialise a corpus.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PIXEL_SCALE = 0.396  # SDSS native arcsec/pixel
SKY_STD = 0.02  # sky-noise floor (calibrated-flux units)
BANDS = ("g", "r", "i")
_BAND_GAIN = np.array([0.8, 1.0, 1.2])  # mild colour: i brighter than g


def _radius_angle(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    cy = cx = (size - 1) / 2.0
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return r, theta


def _bulge(size: int, amp: float, scale: float) -> np.ndarray:
    r, _ = _radius_angle(size)
    return amp * np.exp(-((r / scale) ** 2))


def _spiral_arm(size: int, amp: float, r0: float, width: float, pitch: float) -> np.ndarray:
    """A faint two-arm logarithmic spiral confined to an annulus around ``r0``."""
    r, theta = _radius_angle(size)
    annulus = np.exp(-(((r - r0) / width) ** 2))
    ridge = 0.5 + 0.5 * np.cos(2.0 * (theta - pitch * np.log(r + 1.0)))
    return amp * annulus * ridge


def _stamp(rng: np.random.Generator, size: int, *, bulge: float, arm: float) -> np.ndarray:
    """Build a single-plane flux image, then broadcast to 3 colour channels."""
    plane = rng.normal(0.0, SKY_STD, size=(size, size))
    if bulge > 0:
        plane = plane + _bulge(size, amp=bulge, scale=size / 6.0)
    if arm > 0:
        plane = plane + _spiral_arm(size, amp=arm, r0=size / 3.0, width=size / 10.0, pitch=3.0)
    return np.clip(plane, 0.0, None)[None, :, :] * _BAND_GAIN[:, None, None]


def generate_fixture_corpus(
    out_dir: str | Path, *, n: int = 12, seed: int = 0, size: int = 64
) -> Path:
    """Write ``n`` deterministic FITS stamps + ``metadata.csv`` under ``out_dir``."""
    from astropy.io import fits  # lazy: only materialising a corpus needs the data extra,

    # so importing this module (which conftest does at collection) stays dev-only.
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for i in range(n):
        rng = np.random.default_rng(seed * 1000 + i)
        if i == 0:
            kind, bulge, arm = "sky", 0.0, 0.0
        elif i == 1:
            kind, bulge, arm = "faint_arm", 0.0, 8.0 * SKY_STD
        else:
            kind, bulge, arm = "spiral", float(rng.uniform(0.4, 1.0)), 4.0 * SKY_STD

        image = _stamp(rng, size, bulge=bulge, arm=arm)
        object_id = 1000 + i
        fits.PrimaryHDU(data=image.astype(np.float32)).writeto(
            root / f"{object_id}.fits", overwrite=True
        )
        rows.append(
            {
                "object_id": object_id,
                "ra": round(float(150.0 + i * 0.01), 6),
                "dec": round(float(2.0 + i * 0.01), 6),
                "z": round(float(rng.uniform(0.02, 0.15)), 4),
                "mag_r": round(float(rng.uniform(15.0, 17.7)), 3),
                "petroRad": round(float(rng.uniform(3.0, 12.0)), 3),
                "snr": round(float(rng.uniform(10.0, 40.0)), 2),
                "psf": round(float(rng.uniform(1.0, 1.8)), 3),
                "pixel_scale": PIXEL_SCALE,
                "kind": kind,
            }
        )

    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


if __name__ == "__main__":
    target = Path(__file__).parent / "synthetic"
    generate_fixture_corpus(target)
    print(f"wrote fixture corpus to {target}")
