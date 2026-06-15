"""Data sources — the only place that touches raw bytes (and, later, the network).

Implements ``docs/spec/data.md`` §3 ("The CasJobs join") and ``docs/architecture.md``
"The data stack". This module defines the :class:`DataSource` contract and the offline
pieces: :func:`load_fits_stamp` (pure file IO) and :class:`FixtureSource` (reads the
seeded test fixtures). Tests run entirely against these — **no network**.

The networked ``FitsFrameSource`` (``astroquery.sdss`` full-frame FITS → per-object
``astropy.nddata.Cutout2D``) and the CasJobs / SkyServer metadata join are built **next**,
after the query + columns are confirmed (the empirical-leash pause). They will live here
too, so the network boundary stays inside this one module — see ``docs/spec/data.md`` §3
for why ``galaxy-datasets`` (lossy 8-bit JPG) cannot be that source.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from astropy.io import fits

Array = np.ndarray


@runtime_checkable
class DataSource(Protocol):
    """A finite, indexable source of ``(image, metadata)`` pairs.

    ``image`` is a float ``(C, H, W)`` array of calibrated flux (pre-stretch);
    ``metadata`` is the per-galaxy row (object ID, the nuisance columns, pixel scale).
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Array, dict[str, Any]]: ...


def load_fits_stamp(path: str | Path) -> Array:
    """Load a per-galaxy FITS stamp as a float ``(C, H, W)`` array.

    The fixture (and, later, the cut SDSS) stamp stores channels-first flux in the
    primary HDU. Fails loudly on a malformed shape rather than guessing an axis order.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a (C, H, W) stamp, got shape {data.shape}")
    return data


class FixtureSource:
    """Reads a seeded fixture corpus (``metadata.csv`` + ``<object_id>.fits`` stamps).

    The offline stand-in for the real SDSS source, so the integration tier exercises the
    full pipeline with no network (``docs/spec/testing.md`` §3).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"no metadata.csv under fixture root {self.root}")
        with meta_path.open(newline="") as handle:
            self.rows: list[dict[str, Any]] = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Array, dict[str, Any]]:
        row = self.rows[index]
        image = load_fits_stamp(self.root / f"{row['object_id']}.fits")
        return image, self._typed(row)

    def __iter__(self) -> Iterator[tuple[Array, dict[str, Any]]]:
        for i in range(len(self)):
            yield self[i]

    def stack(self) -> Array:
        """Return all stamps stacked as ``(N, C, H, W)`` — for fitting normalisation."""
        return np.stack([image for image, _ in self])

    @staticmethod
    def _typed(row: dict[str, str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"object_id"}:
                out[key] = int(value)
            elif key in {"kind"}:
                out[key] = value
            else:
                out[key] = float(value)
        return out
