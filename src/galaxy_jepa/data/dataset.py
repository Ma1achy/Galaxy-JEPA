"""Torch dataset over the fp16 cache — zero per-batch preprocessing.

Wraps a :class:`~galaxy_jepa.data.cache.TensorCache` (the pre-baked, parity-locked fp16
stamps) and the per-galaxy metadata, restricted to a *split* (a set of object IDs from
``data/orchestrate.py``). Each item is already stretched+normalised on disk, so
``__getitem__`` only copies one small fp16 stamp out of the memmap and attaches the few
scalars the objective/probe need — no FITS read, no asinh, no normalise in the hot loop.

Carried alongside the image:

* ``petro_rad_arcsec`` + ``pixel_scale`` — the per-galaxy Petrosian box for the bbox-biased
  masking (``data/bbox.py`` / ``docs/masking.md``); pretraining only.
* ``label`` — the binary smooth-vs-featured target (``data/metadata.featured_label``);
  probing only, requested via ``label_fraction_col``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from galaxy_jepa.data.cache import TensorCache
from galaxy_jepa.data.metadata import featured_label
from galaxy_jepa.data.sources import NATIVE_PIXEL_SCALE

__all__ = ["StampDataset", "rows_by_id"]


def rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index metadata rows by ``object_id`` for O(1) lookup by the dataset."""
    return {int(r["object_id"]): dict(r) for r in rows}


class StampDataset(Dataset):
    """Indexable view of one split's pre-baked stamps + the scalars the model needs.

    ``object_ids`` is intersected with what the cache actually holds (a split may name
    galaxies not yet baked); the dataset covers only the present ones, in the given order.
    """

    def __init__(
        self,
        cache: TensorCache,
        rows: Mapping[int, Mapping[str, Any]],
        object_ids: Sequence[int],
        *,
        pixel_scale: float = NATIVE_PIXEL_SCALE,
        label_fraction_col: str | None = None,
    ):
        self.cache = cache
        self.rows = rows
        self.pixel_scale = float(pixel_scale)
        self.label_fraction_col = label_fraction_col
        self.object_ids: list[int] = [int(o) for o in object_ids if int(o) in cache]

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        oid = self.object_ids[index]
        # copy the single stamp out of the read-only memmap (≈0.4 MB) — np.array makes a
        # writable owned copy (torch needs writable); keep fp16, the model upcasts on-device.
        # No decode, no stretch, no normalise here.
        image = torch.from_numpy(np.array(self.cache.get(oid)))
        row = self.rows.get(oid, {})
        petro = row.get("petroRad_r", float("nan"))
        item: dict[str, Any] = {
            "image": image,
            "object_id": oid,
            "petro_rad_arcsec": float(petro) if petro is not None else float("nan"),
            "pixel_scale": self.pixel_scale,
        }
        if self.label_fraction_col is not None:
            frac = row.get(self.label_fraction_col, float("nan"))
            item["label"] = featured_label(float(frac))
            item["featured_fraction"] = float(frac)
        return item
