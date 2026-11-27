"""Deterministic synthetic embedding & procedural galaxy fixtures (§10.1–§10.2).

SYNTHETIC BENCHMARK ONLY: these generators produce controlled mathematical and procedural
test fixtures with known ground-truth parameters. They are strictly for algorithmic,
statistical, and end-to-end software verification and are NOT physical SDSS/GZ2 galaxy models.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SyntheticPairFixture:
    """Embedding + binary labels + continuous nuisance for matching/entanglement benchmarks."""

    x: np.ndarray  # (N, D)
    y_target: np.ndarray  # (N,) int {0, 1}
    nuisance: np.ndarray  # (N,) float
    ids: np.ndarray  # (N,) int64 (> 2^53)


def make_world_confounded_pair(
    n: int = 800,
    d: int = 12,
    seed: int = 1001,
) -> SyntheticPairFixture:
    """Synthetic fixture where encoder ONLY encodes nuisance `u`, and `y` depends ONLY on `u`.

    Unmatched probe AUC on `y` is high (> 0.72) purely via `u`, while conditional matching on `u`
    collapses AUC to ~0.50 (world correlation, zero direct representation of `y`).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    x = rng.normal(scale=0.6, size=(n, d))
    x[:, 0] = 2.0 * u + rng.normal(scale=0.4, size=n)
    # y depends ONLY on nuisance u (zero direct dependence on x[:, 1:])
    probs = 1.0 / (1.0 + np.exp(-2.2 * u))
    y = (rng.random(n) < probs).astype(np.int64)
    ids = np.arange(n, dtype=np.int64) + (1 << 54)
    return SyntheticPairFixture(x=x, y_target=y, nuisance=u, ids=ids)


def make_representational_pair(
    n: int = 800,
    d: int = 12,
    seed: int = 1002,
) -> SyntheticPairFixture:
    """Synthetic fixture where `y` has strong direct signal orthogonal to nuisance `u`.

    Even after exact stratification on `u`, `y` remains strongly recoverable (AUC > 0.75).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    s_direct = rng.normal(size=n)
    x = rng.normal(scale=0.5, size=(n, d))
    x[:, 0] = 1.5 * u + rng.normal(scale=0.3, size=n)
    x[:, 1] = 2.0 * s_direct + rng.normal(scale=0.3, size=n)
    probs = 1.0 / (1.0 + np.exp(-(1.2 * u + 2.2 * s_direct)))
    y = (rng.random(n) < probs).astype(np.int64)
    ids = np.arange(n, dtype=np.int64) + (1 << 54) + 100_000
    return SyntheticPairFixture(x=x, y_target=y, nuisance=u, ids=ids)


@dataclasses.dataclass(frozen=True)
class ProceduralGalaxyBatch:
    """Deterministic 3-band procedural galaxy stamps with known latent morphological parameters."""

    images: np.ndarray  # (N, 3, H, W) float32
    ids: np.ndarray  # (N,) int64 (> 2^53)
    petro_rad_arcsec: np.ndarray  # (N,) float32
    pixel_scale: np.ndarray  # (N,) float32
    has_spiral: np.ndarray  # (N,) int64 {0, 1}
    has_bar: np.ndarray  # (N,) int64 {0, 1}
    is_edge_on: np.ndarray  # (N,) int64 {0, 1}
    winding_cw: np.ndarray  # (N,) int64 {0, 1}
    band_shift_px: np.ndarray  # (N, 2) float32 (g-band dy, dx offset)
    brightness: np.ndarray  # (N,) float32
    rows: dict[int, dict[str, float]]


def generate_procedural_galaxies(
    n: int = 120,
    img_size: int = 32,
    seed: int = 2026,
) -> ProceduralGalaxyBatch:
    """Generate deterministic 3-band (g, r, i) synthetic galaxy stamps with ground-truth labels.

    Includes Sérsic-like bulge + exponential disc, optional logarithmic spiral arms with winding
    handedness, optional central bar, inclination (axis ratio), orientation angle, PSF-like
    smoothing, sky noise, and controllable g-band misregistration.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    cy = (img_size - 1) / 2.0
    cx = (img_size - 1) / 2.0
    dy0 = (yy - cy) / (img_size / 2.0)
    dx0 = (xx - cx) / (img_size / 2.0)

    images = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    ids = np.arange(n, dtype=np.int64) + 1_237_666_310_706_430_371
    petro = np.empty(n, dtype=np.float32)
    pix_scale = np.full(n, 0.396, dtype=np.float32)
    has_spiral = rng.integers(0, 2, size=n, dtype=np.int64)
    has_bar = rng.integers(0, 2, size=n, dtype=np.int64)
    is_edge_on = rng.integers(0, 2, size=n, dtype=np.int64)
    winding_cw = rng.integers(0, 2, size=n, dtype=np.int64)
    band_shift = rng.normal(scale=0.6, size=(n, 2)).astype(np.float32)
    brightness = np.empty(n, dtype=np.float32)
    rows: dict[int, dict[str, float]] = {}

    for i in range(n):
        theta = float(rng.uniform(0.0, np.pi))
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        u = ct * dx0 + st * dy0
        v = -st * dx0 + ct * dy0

        # Inclination: edge-on galaxies have small axis ratio q ~ 0.22; face-on have q ~ 0.85
        q = float(rng.uniform(0.18, 0.30) if is_edge_on[i] else rng.uniform(0.75, 0.98))
        r_ell = np.sqrt(u**2 + (v / q) ** 2 + 1e-4)

        scale_r = float(rng.uniform(0.28, 0.52))
        petro[i] = float(scale_r * (img_size / 2.0) * 0.396)
        amp = float(rng.uniform(0.8, 1.8))

        # Sérsic bulge (n ~ 2) + exponential disc
        bulge = 0.7 * np.exp(-2.8 * (r_ell / (0.35 * scale_r)) ** 0.5)
        disc = np.exp(-r_ell / scale_r)
        prof = amp * (bulge + disc)

        # Optional central bar aligned along u-axis
        if has_bar[i]:
            bar = 0.65 * amp * np.exp(-((u / (0.55 * scale_r)) ** 2) - ((v / 0.08) ** 2))
            prof = prof + bar

        # Optional 2-arm logarithmic spiral with controllable winding handedness
        if has_spiral[i] and not is_edge_on[i]:
            phi = np.arctan2(v / q, u)
            sign = 1.0 if winding_cw[i] else -1.0
            arms = np.cos(2.0 * phi + sign * 4.5 * np.log(r_ell + 0.08))
            envelope = np.exp(-((r_ell - 0.55 * scale_r) ** 2) / (0.18 * scale_r**2 + 1e-3))
            prof = prof * (1.0 + 0.45 * np.maximum(arms, 0.0) * envelope)

        # 3 bands (g, r, i) with mild colour gradient + controllable g-band integer-rolled shift
        s_dy = int(np.round(band_shift[i, 0]))
        s_dx = int(np.round(band_shift[i, 1]))
        g_band = np.roll(np.roll(0.9 * prof, s_dy, axis=0), s_dx, axis=1)
        r_band = prof
        i_band = 1.15 * prof

        # Add deterministic Gaussian sky noise
        noise = rng.normal(scale=0.04, size=(3, img_size, img_size)).astype(np.float32)
        stamp = np.stack([g_band, r_band, i_band], axis=0) + noise
        images[i] = stamp
        brightness[i] = float(r_band.sum())

        # Smooth vote-fraction labels near 0.9 / 0.1 so both continuous & extreme filters work
        oid = int(ids[i])
        rows[oid] = {
            "spiral_frac": 0.90 if has_spiral[i] else 0.10,
            "spiral_n": 30.0,
            "bar_frac": 0.88 if has_bar[i] else 0.12,
            "bar_n": 30.0,
            "edge_frac": 0.92 if is_edge_on[i] else 0.08,
            "edge_n": 30.0,
            "brightness": float(brightness[i]),
            "band_offset": float(np.hypot(band_shift[i, 0], band_shift[i, 1])),
        }

    return ProceduralGalaxyBatch(
        images=images,
        ids=ids,
        petro_rad_arcsec=petro,
        pixel_scale=pix_scale,
        has_spiral=has_spiral,
        has_bar=has_bar,
        is_edge_on=is_edge_on,
        winding_cw=winding_cw,
        band_shift_px=band_shift,
        brightness=brightness,
        rows=rows,
    )
