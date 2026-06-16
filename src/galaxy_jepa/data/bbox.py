"""Per-galaxy Petrosian bounding box — the masking sampling prior.

Implements ``docs/masking.md`` §3.1 (the Paper-1 default box) and ``docs/spec/data.md`` §3.

Computes the **geometric prior** only — the box in pixels from ``petroRad`` + the cutout's
arcsec/pixel scale. The β-biased *sampling* that consumes it stays in ``objectives/``
(keeps the ``docs/spec/data.md`` "masking is not a data transform" boundary).

Box half-width is ``k · R_petro`` in pixels (``k ≈ 2.5``, tunable from the contact sheet),
clamped to the frame. A missing / non-finite / non-positive ``petroRad`` — common on the
faint end of the pretraining corpus — falls back to the **global-average box** with a
**loud log**, never a silent default (``docs/architecture.md``: fail loudly). The
fallback *rate* is a quantity to watch at the eyeball gate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_K = 2.5


@dataclass(frozen=True)
class Box:
    """A centred square box, ``half_width_px`` either side of the stamp centre."""

    half_width_px: float
    used_fallback: bool


def petrosian_box(
    petro_rad_arcsec: float | None,
    pixel_scale: float,
    *,
    k: float = DEFAULT_K,
    stamp_px: int,
    global_half_width_px: float,
    object_id: object = None,
) -> Box:
    """Per-galaxy box half-width in pixels, or the global-box fallback (logged)."""
    if pixel_scale <= 0:
        raise ValueError(f"pixel_scale must be > 0, got {pixel_scale!r}")
    if petro_rad_arcsec is None or not math.isfinite(petro_rad_arcsec) or petro_rad_arcsec <= 0:
        logger.warning(
            "petroRad missing/invalid (%r) for object %r; falling back to the global "
            "average box (half-width %.1f px).",
            petro_rad_arcsec,
            object_id,
            global_half_width_px,
        )
        return Box(half_width_px=float(global_half_width_px), used_fallback=True)

    half = k * float(petro_rad_arcsec) / pixel_scale
    half = min(half, stamp_px / 2.0)  # clamp to the frame
    return Box(half_width_px=half, used_fallback=False)


def fallback_rate(boxes: Iterable[Box]) -> float:
    """Fraction of boxes that fell back to the global average — watch this at the gate."""
    boxes = list(boxes)
    if not boxes:
        return 0.0
    return sum(b.used_fallback for b in boxes) / len(boxes)
