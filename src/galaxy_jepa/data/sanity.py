"""Stretch-sanity check — Tier-2 ``T2.stretch-sanity`` (docs/spec/validation.md).

Implements the cheap pre-pretraining check from ``docs/spec/data.md`` §2: on known
faint-arm galaxies, confirm faint arms **survive** stretch + normalise while the
**sky-noise floor stays controlled**. Pairs with the collapse monitor — if the encoder
starts modelling noise, the stretch is too aggressive.

Returns metrics (not a bare bool) so the contact sheet can show *how* comfortably the
margin holds, and reusable on real galaxies as well as the fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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
