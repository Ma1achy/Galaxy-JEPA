"""Transform pipeline — the parity-locked front of the data stack.

Implements ``docs/spec/data.md`` §1–§2 ("The parity rule" and "Format + stretch —
decided: FITS + asinh") and ``docs/architecture.md`` "The data stack".

Three pieces, all :class:`~galaxy_jepa.core.config.Configurable` so the *whole* pipeline
serialises into the run-stamp and its ``config_hash`` is the parity guarantee:

* :class:`AsinhStretch` — the dynamic-range stretch. Its parameters (softening ``q`` +
  per-channel ``flux_scale``) are config and therefore stamped — never a notebook
  constant. **Provisional defaults until the eyeball gate** (the asinh scale is a
  look-at-the-images decision, ``docs/spec/data.md`` §2).
* :class:`Normalise` — a *stateful* transform: per-channel mean/std fitted **after** the
  stretch, **once on the pretraining corpus**, then frozen and carried everywhere. The
  fitted statistics are constructor fields, so they enter the config hash too.
* :class:`Pipeline` — an ordered, composable list. The parity rule is enforced by
  sharing *one* fitted ``Pipeline`` across both corpora and every baseline.

Masking is **not** here — it is part of the JEPA objective and lives in ``objectives/``
(``docs/spec/data.md`` preamble).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from galaxy_jepa.core.config import Configurable

# Images flow through the pipeline as float arrays shaped ``(C, H, W)`` (channels-first,
# to match torch). Transforms preserve that shape.
Array = np.ndarray


@runtime_checkable
class Transform(Protocol):
    """A pure, shape-preserving ``(C, H, W) -> (C, H, W)`` array transform."""

    def __call__(self, image: Array) -> Array: ...


class AsinhStretch(Configurable):
    """Per-channel asinh dynamic-range stretch.

    ``out_c = arcsinh(q * x_c / a_c) / arcsinh(q)`` where ``a_c`` is the per-channel
    ``flux_scale`` and ``q`` the softening. The map sends flux ``a_c`` to 1 and boosts
    the faint end relative to a linear stretch (the low-surface-brightness range the
    Rung-4 measurement depends on; ``docs/spec/data.md`` §2). Larger ``q`` ⇒ a stronger
    faint-end boost.

    Defaults are **provisional** — frozen only after the eyeball gate on real cutouts.
    """

    def __init__(self, q: float = 8.0, flux_scale: tuple[float, ...] = (1.0, 1.0, 1.0)):
        if q <= 0:
            raise ValueError(f"asinh softening q must be > 0, got {q!r}")
        if not flux_scale or any(s <= 0 for s in flux_scale):
            raise ValueError(f"flux_scale must be non-empty and all > 0, got {flux_scale!r}")
        self.q = float(q)
        self.flux_scale = tuple(float(s) for s in flux_scale)

    def __call__(self, image: Array) -> Array:
        scale = np.asarray(self.flux_scale, dtype=np.float64).reshape(-1, 1, 1)
        if image.shape[0] != scale.shape[0]:
            raise ValueError(
                f"AsinhStretch expects {scale.shape[0]} channels (flux_scale), "
                f"got image with {image.shape[0]}"
            )
        x = np.asarray(image, dtype=np.float64) / scale
        return np.arcsinh(self.q * x) / np.arcsinh(self.q)


class Normalise(Configurable):
    """Per-channel mean/std normalisation — a *stateful* transform.

    Constructed unfitted; :meth:`fit` returns a frozen instance whose ``mean``/``std``
    are captured config (so they are stamped and travel with the pipeline). Calling an
    unfitted ``Normalise`` is a loud error — never a silent identity.
    """

    def __init__(
        self,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ):
        if (mean is None) != (std is None):
            raise ValueError("mean and std must both be given or both omitted")
        if std is not None and any(s <= 0 for s in std):
            raise ValueError(f"std must be all > 0, got {std!r}")
        self.mean = None if mean is None else tuple(float(m) for m in mean)
        self.std = None if std is None else tuple(float(s) for s in std)

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    @classmethod
    def fit(cls, images: Array) -> Normalise:
        """Fit per-channel mean/std over a stack of ``(C, H, W)`` images (``(N, C, H, W)``)."""
        arr = np.asarray(images, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError(f"fit expects a stack shaped (N, C, H, W), got {arr.shape}")
        mean = arr.mean(axis=(0, 2, 3))
        std = arr.std(axis=(0, 2, 3))
        if np.any(std <= 0):
            raise ValueError("a channel has zero variance; cannot normalise (fail loud)")
        return cls(mean=tuple(mean.tolist()), std=tuple(std.tolist()))

    def __call__(self, image: Array) -> Array:
        if not self.fitted:
            raise RuntimeError(
                "Normalise called before fit(); the normalisation statistic must be "
                "fitted once on the pretraining corpus and frozen (docs/spec/data.md §2)."
            )
        assert self.mean is not None and self.std is not None  # for type-checkers
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1, 1, 1)
        std = np.asarray(self.std, dtype=np.float64).reshape(-1, 1, 1)
        return (np.asarray(image, dtype=np.float64) - mean) / std


class Pipeline(Configurable):
    """An ordered list of transforms applied left-to-right.

    The parity guarantee is structural: share *one* fitted ``Pipeline`` across the
    pretraining corpus, the probing corpus, and every baseline. Two corpora with the
    same pipeline have, by construction, the same ``config_hash``.
    """

    def __init__(self, transforms: tuple[Transform, ...]):
        self.transforms = tuple(transforms)

    def __call__(self, image: Array) -> Array:
        for transform in self.transforms:
            image = transform(image)
        return image
