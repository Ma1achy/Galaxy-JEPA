"""Position angle and axis ratio from a stamp's own pixels — the label for Brief R3.

R3 asks whether the geometry machinery can find non-Euclidean structure at all, and orientation
is the natural test: it is plainly visible, pretraining applies no rotation or flip augmentation
(a divergence from D10, recorded there), so the encoder is free to carry it — and it is circular.

**Two choices made here, both deliberate.**

* **The angle comes from the pixels, not from SDSS.** ``expPhi_r`` is measured east of north,
  while a stamp's pixel axes are rotated from north by a per-field offset. Intensity-weighted
  second moments of the stamp itself give orientation exactly as the encoder sees it, with no
  frame convention to get wrong and nothing to pull.
* **It is 180°-periodic.** An ellipse's major axis at 10° and at 190° is the same image. Anything
  circular built on it uses the doubled angle, so twelve bins over [0°, 180°) trace one full loop.

Measured on the *preprocessed* stamp (stretched and normalised, as cached), because that is the
image the encoder receives; a stretch re-weights the outskirts but does not rotate anything.
"""

from __future__ import annotations

import numpy as np

__all__ = ["second_moment_orientation"]


def second_moment_orientation(image: np.ndarray, *, radius_px: float) -> tuple[float, float]:
    """``(theta_deg, axis_ratio)`` from background-subtracted second moments inside a circle.

    ``image`` is ``(H, W)`` or ``(C, H, W)`` (channels summed — the encoder sees them all), with
    the galaxy at the centre, as every stamp here is. ``theta_deg`` is the major axis measured
    from the +x (column) axis towards +y (row), in [0, 180). ``axis_ratio`` is minor/major in
    (0, 1]. Returns ``(nan, nan)`` when nothing positive survives the background cut.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.sum(axis=0)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = r2 <= radius_px**2
    # Background from the ring outside the aperture: the stretch leaves sky at a non-zero level.
    ring = (r2 > radius_px**2) & (r2 <= (1.5 * radius_px) ** 2)
    bkg = float(np.median(img[ring])) if ring.any() else float(np.median(img))
    wgt = np.where(inside, np.clip(img - bkg, 0.0, None), 0.0)
    total = wgt.sum()
    if total <= 0:
        return float("nan"), float("nan")
    mx = (wgt * xx).sum() / total
    my = (wgt * yy).sum() / total
    dx, dy = xx - mx, yy - my
    mxx = (wgt * dx * dx).sum() / total
    myy = (wgt * dy * dy).sum() / total
    mxy = (wgt * dx * dy).sum() / total
    theta = 0.5 * np.degrees(np.arctan2(2.0 * mxy, mxx - myy)) % 180.0
    tr, det = mxx + myy, mxx * myy - mxy**2
    disc = np.sqrt(max(tr**2 / 4.0 - det, 0.0))
    lam_max, lam_min = tr / 2.0 + disc, max(tr / 2.0 - disc, 0.0)
    q = float(np.sqrt(lam_min / lam_max)) if lam_max > 0 else float("nan")
    return float(theta), q
