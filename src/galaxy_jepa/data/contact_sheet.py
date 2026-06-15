"""Contact sheet for the eyeball gate (docs/spec/data.md §2; the empirical leash).

Renders a grid of stretched + normalised cutouts so a human can decide the asinh scale
*by looking* — the stretch params are not frozen until the cutouts have been seen.

Each cutout is overlaid with its **Petrosian sampling box** (half-width ``k · R_petro``,
``k`` default 2.5 and tunable from what the cutouts show): the box is load-bearing for how
much faint outskirt the masking prior includes, so it must be visible — is it cropping the
arms, or floating in sky? The sheet also reports the **stretch-sanity** result and the
**global-box fallback rate** (``docs/data.md`` §3.1 faint-end caveat).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from galaxy_jepa.data.bbox import DEFAULT_K, fallback_rate, petrosian_box
from galaxy_jepa.data.sources import NATIVE_PIXEL_SCALE, DataSource
from galaxy_jepa.data.transforms import AsinhStretch

Array = np.ndarray


def _radius(meta: dict) -> float | None:
    for key in ("petroRad_r", "petroRad"):
        if key in meta:
            return meta[key]
    return None


def _to_rgb(stretched: Array) -> Array:
    """(C, H, W) → (H, W, 3) display image, percentile-clipped to [0, 1] (display only)."""
    chans = stretched[:3] if stretched.shape[0] >= 3 else np.repeat(stretched[:1], 3, axis=0)
    rgb = np.transpose(chans, (1, 2, 0))
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def build_contact_sheet(
    source: DataSource,
    pipeline,
    out_path: str | Path,
    *,
    k: float = DEFAULT_K,
    pixel_scale: float = NATIVE_PIXEL_SCALE,
    global_half_width_px: float | None = None,
    max_cutouts: int = 64,
) -> Path:
    """Render the contact sheet to ``out_path`` (PNG) and return the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = min(len(source), max_cutouts)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)

    boxes = []
    for idx in range(n):
        image, meta = source[idx]
        stamp_px = image.shape[-1]
        if global_half_width_px is None:
            global_half_width_px = 0.7 * stamp_px / 2.0  # v1-style provisional global box
        rgb = _to_rgb(pipeline(image))
        box = petrosian_box(
            _radius(meta), pixel_scale, k=k, stamp_px=stamp_px,
            global_half_width_px=global_half_width_px, object_id=meta.get("object_id"),
        )
        boxes.append(box)

        ax = axes[idx // cols][idx % cols]
        ax.imshow(rgb, origin="lower")
        centre = (stamp_px - 1) / 2.0
        edge = "yellow" if box.used_fallback else "cyan"
        ax.add_patch(
            Rectangle(
                (centre - box.half_width_px, centre - box.half_width_px),
                2 * box.half_width_px, 2 * box.half_width_px,
                fill=False, edgecolor=edge, linewidth=1.2,
            )
        )
        rad = _radius(meta)
        ax.set_title(
            f"r={meta.get('modelMag_r', meta.get('mag_r', float('nan'))):.1f}  "
            f"R={rad:.1f}\"" + ("  [fallback]" if box.used_fallback else ""),
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    qval = next(
        (t.q for t in getattr(pipeline, "transforms", ()) if isinstance(t, AsinhStretch)),
        None,
    )
    qstr = f"asinh Q={qval:g}, " if qval is not None else ""
    fig.suptitle(
        f"Galaxy-JEPA contact sheet — {qstr}k={k}, global-box fallback rate "
        f"{fallback_rate(boxes):.1%} (cyan=per-galaxy box, yellow=fallback)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
