"""Stretch-sanity check — Tier-2 ``T2.stretch-sanity`` (docs/spec/validation.md).

Implements the cheap pre-pretraining check from ``docs/spec/data.md`` §2: on known
faint-arm galaxies, confirm faint arms **survive** stretch + normalise while the
**sky-noise floor stays controlled**. Pairs with the collapse monitor — if the encoder
starts modelling noise, the stretch is too aggressive.

Returns metrics (not a bare bool) so the contact sheet can show *how* comfortably the
margin holds, and reusable on real galaxies as well as the fixtures.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class StretchSanity:
    arm_peak: float
    sky_p99: float
    margin: float
    sky_std: float
    passes: bool


def stretch_sanity(
    faint_arm_images: Sequence[Array],
    sky_control_images: Sequence[Array],
    pipeline,
    *,
    margin_floor: float = 0.5,
    sky_std_ceiling: float = 2.0,
) -> StretchSanity:
    """Compare faint-arm survival against the controlled sky floor after the pipeline."""
    if not faint_arm_images or not sky_control_images:
        raise ValueError("need at least one faint-arm exemplar and one sky control")

    arm_peak = max(float(pipeline(img).max()) for img in faint_arm_images)
    sky_stack = np.stack([pipeline(img) for img in sky_control_images])
    sky_p99 = float(np.percentile(sky_stack, 99))
    sky_std = float(np.std(sky_stack))

    margin = arm_peak - sky_p99
    passes = margin > margin_floor and sky_std < sky_std_ceiling
    return StretchSanity(arm_peak, sky_p99, margin, sky_std, passes)


# --- per-galaxy spatially-separated metric (the real T2.stretch-sanity measurement) ---
#
# The exemplar-vs-control form above cannot separate signal from noise *on the same
# galaxy*, which is the whole problem: faint outer arms and sky noise live in the same
# brightness range (the reason we pull FITS, not cutouts — docs/spec/data.md §2). So the
# measurement is done per galaxy, in two disjoint regions, on a post stretch+normalise
# image: faint structure in an annulus around the galaxy, sky noise in the blank corners.


@dataclass(frozen=True)
class ZoneMetrics:
    """Two spatially-separated numbers for one galaxy (post stretch+normalise).

    ``faint_retention`` — median pixel value in the outskirt **annulus** (``NaN`` when no
    annulus could be defined); ``sky_floor`` — MAD of the blank-sky **corner** patches;
    ``faint_valid`` — whether the annulus existed. The honest signal quantity is the
    **gap** ``faint_retention - sky_floor`` (real structure above noise), not the
    retention height alone — the annulus median includes lifted inter-arm sky noise.
    """

    faint_retention: float  # annulus median ABOVE the sky background (NaN if invalid)
    sky_floor: float
    faint_valid: bool


def _radial_grid(height: int, width: int) -> Array:
    """Pixel distance from the stamp centre, shape ``(H, W)``."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    return np.hypot(x - cx, y - cy)


def galaxy_zone_metrics(
    normalised_image: Array,
    petro_rad_arcsec: float | None,
    pixel_scale: float,
    *,
    k: float = 2.5,
    sky_patch_px: int = 12,
    object_id: object = None,
) -> ZoneMetrics:
    """Faint-retention (annulus) and sky-noise floor (corners) for one galaxy.

    Operates on a **post stretch+normalise** ``(C, H, W)`` image (``docs/spec/data.md``
    §2.3, ``T2.stretch-sanity``). The annulus is ``R_petro ≤ r < min(k·R_petro,
    stamp/2)`` px — real galaxy outskirts, not the bright core, not blank sky — using the
    same ``R_petro``→px scaling as :func:`bbox.petrosian_box` (``k`` default 2.5). A
    missing / non-finite / non-positive ``petroRad``, or one so large the annulus is
    empty, **excludes the galaxy from faint-retention** with a loud log — never a
    fabricated annulus. The sky floor is geometry-independent (always computed).
    """
    img = np.asarray(normalised_image, dtype=np.float64)
    if img.ndim != 3:
        raise ValueError(f"expected a (C, H, W) image, got shape {img.shape}")
    if pixel_scale <= 0:
        raise ValueError(f"pixel_scale must be > 0, got {pixel_scale!r}")
    _, height, width = img.shape

    # Sky-noise floor: robust MAD over the four blank-sky corner patches.
    p = sky_patch_px
    corners = np.concatenate(
        [
            img[:, :p, :p].ravel(), img[:, :p, -p:].ravel(),
            img[:, -p:, :p].ravel(), img[:, -p:, -p:].ravel(),
        ]
    )
    sky_median = float(np.median(corners))
    sky_floor = float(np.median(np.abs(corners - sky_median)))  # MAD = robust noise floor

    stamp_half = min(height, width) / 2.0
    if petro_rad_arcsec is None or not math.isfinite(petro_rad_arcsec) or petro_rad_arcsec <= 0:
        logger.warning(
            "petroRad missing/invalid (%r) for object %r; excluding from faint-retention.",
            petro_rad_arcsec, object_id,
        )
        return ZoneMetrics(float("nan"), sky_floor, False)

    inner_px = float(petro_rad_arcsec) / pixel_scale
    outer_px = min(k * inner_px, stamp_half)
    if inner_px >= stamp_half:
        logger.warning(
            "petroRad %.2f″ → inner radius %.1f px ≥ stamp half %.1f px for object %r; "
            "annulus empty, excluding from faint-retention.",
            petro_rad_arcsec, inner_px, stamp_half, object_id,
        )
        return ZoneMetrics(float("nan"), sky_floor, False)

    r = _radial_grid(height, width)
    mask = (r >= inner_px) & (r < outer_px)
    # Signal level *above the sky background*, so it is comparable to the sky floor and the
    # gap (faint - sky) is a true signal-over-noise margin — not a global-mean-centred level.
    faint_retention = float(np.median(img[:, mask]) - sky_median)
    return ZoneMetrics(faint_retention, sky_floor, True)
