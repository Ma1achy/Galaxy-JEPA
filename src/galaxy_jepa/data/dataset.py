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

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from galaxy_jepa.data.cache import TensorCache
from galaxy_jepa.data.metadata import featured_label
from galaxy_jepa.data.sources import NATIVE_PIXEL_SCALE

__all__ = ["ResumableShuffle", "StampDataset", "rows_by_id"]


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
        scalars: np.ndarray | None = None,
    ):
        self.cache = cache
        self.rows = rows
        self.pixel_scale = float(pixel_scale)
        self.label_fraction_col = label_fraction_col
        self.object_ids: list[int] = [int(o) for o in object_ids if int(o) in cache]
        # `scalars` is the cache's index-aligned petroRad_r array (data.cache.write_scalars).
        # Given it, the dataset needs nothing from `rows` for pretraining, so the 4.07 GB
        # metadata table stops being resident for the whole run: gathered here into one array
        # in *this split's* order, it is 3.2 MB and `__getitem__` becomes an array index.
        self._petro: np.ndarray | None = (
            None
            if scalars is None
            else np.asarray(scalars, dtype=np.float64)[[cache.row_of(o) for o in self.object_ids]]
        )

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        oid = self.object_ids[index]
        # copy the single stamp out of the read-only memmap (≈0.4 MB) — np.array makes a
        # writable owned copy (torch needs writable); keep fp16, the model upcasts on-device.
        # No decode, no stretch, no normalise here.
        image = torch.from_numpy(np.array(self.cache.get(oid)))
        if self._petro is not None:
            petro = float(self._petro[index])
            row: Mapping[str, Any] = {}
        else:
            row = self.rows.get(oid, {})
            raw = row.get("petroRad_r", float("nan"))
            petro = float(raw) if raw is not None else float("nan")
        item: dict[str, Any] = {
            "image": image,
            "object_id": oid,
            "petro_rad_arcsec": petro,
            "pixel_scale": self.pixel_scale,
        }
        if self.label_fraction_col is not None:
            frac = row.get(self.label_fraction_col, float("nan"))
            item["label"] = featured_label(float(frac))
            item["featured_fraction"] = float(frac)
        return item


class ResumableShuffle(Sampler[int]):
    """An endless shuffled index stream whose position is a single integer.

    The pretrain loop needs two things a plain ``DataLoader(shuffle=True)`` cannot give together:
    an endless stream (it is step-bounded, not epoch-bounded, and ``_cycle`` re-iterates) and a
    **resumable** one. With ``shuffle=True`` the order comes from the global torch RNG at each
    ``iter()``, so a resumed run cannot land where the interrupted one stopped — it would train on
    a different sequence from the same step and the trajectory would silently diverge. A resume
    that runs but diverges is worse than no resume.

    Here the order of epoch *e* is ``default_rng([seed, e]).permutation(n)``, a pure function of
    the seed, so position ``i`` names exactly one index for all time and resuming is
    ``start=steps_done * batch_size``. Together with the masker's per-step seeding
    (``Jepa.loss_step(seed=cfg.seed + step)``) and the absence of dropout in the encoder, this
    makes the whole trajectory a function of ``(seed, step)`` — which is what lets the resume
    identity be *proven* rather than hoped for.
    """

    def __init__(self, n: int, *, seed: int, start: int = 0):
        if n <= 0:
            raise ValueError("cannot shuffle an empty dataset")
        self.n, self.seed, self.start = int(n), int(seed), int(start)
        self._epoch: int | None = None
        self._perm: np.ndarray | None = None

    def permutation(self, epoch: int) -> np.ndarray:
        if self._epoch != epoch:  # one 6.5 MB permutation cached per epoch (~26k steps)
            self._epoch, self._perm = (
                epoch,
                np.random.default_rng([self.seed, epoch]).permutation(self.n),
            )
        assert self._perm is not None
        return self._perm

    def __iter__(self) -> Iterator[int]:
        i = self.start
        while True:
            epoch, offset = divmod(i, self.n)
            yield int(self.permutation(epoch)[offset])
            i += 1
