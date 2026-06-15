"""Data sources — the only place that touches raw bytes and the network.

Implements ``docs/spec/data.md`` §3 ("The CasJobs join") and ``docs/architecture.md``
"The data stack". Defines the :class:`DataSource` contract and:

* :func:`load_fits_stamp` — pure file IO;
* :class:`DirectorySource` — reads a corpus directory (``metadata.csv`` + ``<objID>.fits``)
  with **no network**. The seeded test fixtures *and* a real pull share this one consumer
  — that shared path is the parity guarantee (``docs/spec/data.md`` §1);
* :class:`FitsFrameSource` — the **networked** SDSS source (``astroquery.sdss`` frame
  FITS → per-object ``astropy.nddata.Cutout2D``). ``astroquery`` is imported lazily inside
  the class so the module imports without it, and so tests (which use
  :class:`DirectorySource`) never hit the network. ``galaxy-datasets`` cannot be this
  source — it serves lossy 8-bit JPG (``docs/spec/data.md`` §2).

Install the networked dependencies (astropy + astroquery + pillow) with::

    uv sync --extra data
"""

from __future__ import annotations

import csv
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray

# SDSS frame FITS are calibrated in nanomaggies at the native pixel scale.
NATIVE_PIXEL_SCALE = 0.396  # arcsec/pixel
DEFAULT_BANDS = ("g", "r", "i")

# Calibrated frame FITS on the SDSS Science Archive Server (bz2-compressed nanomaggies).
_FRAME_URL = (
    "https://dr{dr}.sdss.org/sas/dr{dr}/eboss/photoObj/frames/"
    "{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
)


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

    The stamp stores channels-first flux in the primary HDU. Fails loudly on a malformed
    shape rather than guessing an axis order. ``astropy`` is imported lazily so the module
    (and ``import galaxy_jepa.data``) loads without the ``data`` extra installed.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a (C, H, W) stamp, got shape {data.shape}")
    return data


class DirectorySource:
    """Reads a corpus directory: ``metadata.csv`` + ``<object_id>.fits`` stamps.

    The single offline consumer for both the seeded fixtures and a real pull
    (``docs/spec/testing.md`` §3): the integration tier exercises the full pipeline with
    no network, and a real pull writes exactly this layout.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"no metadata.csv under corpus root {self.root}")
        with meta_path.open(newline="") as handle:
            self.rows: list[dict[str, Any]] = [self._typed(r) for r in csv.DictReader(handle)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Array, dict[str, Any]]:
        row = self.rows[index]
        image = load_fits_stamp(self.root / f"{row['object_id']}.fits")
        return image, row

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
            if key == "object_id":
                out[key] = int(value)
            elif key == "kind":
                out[key] = value
            else:
                out[key] = float(value) if value not in ("", "nan") else float("nan")
        return out


# Backwards-compatible alias: the fixtures are just a DirectorySource corpus.
FixtureSource = DirectorySource


class FitsFrameSource:
    """Networked SDSS source: frame FITS → per-object cutout (``docs/spec/data.md`` §3).

    Not exercised by tests. Each row carries ``ra``/``dec`` + ``run``/``camcol``/
    ``field``/``rerun``, so it downloads the calibrated frame FITS straight from the SDSS
    Science Archive Server (more reliable than ``astroquery.get_images``, whose
    coordinate path re-resolves run/camcol/field and is flaky), then cuts a fixed-size
    stamp centred on the galaxy at the **native 0.396″/px** scale — no rebin (rebinning
    interacts with the Rung-4 resolution question; keep it out of the data layer).
    """

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        bands: tuple[str, ...] = DEFAULT_BANDS,
        stamp_px: int = 64,
        data_release: int = 17,
        timeout: int = 120,
        frame_cache_size: int = 24,
    ):
        self.rows = list(rows)
        self.bands = bands
        self.stamp_px = stamp_px
        self.data_release = data_release
        self.timeout = timeout
        # Download is server/IO-bound, so a thread pool over one shared source is the right
        # tool — these back an HTTP keep-alive session (avoid re-handshaking dr17.sdss.org)
        # and a small LRU frame cache (consecutive ORDER-BY-objID galaxies share a field
        # frame, so cache it instead of re-downloading). Both are guarded for thread use.
        self._frame_cache_size = frame_cache_size
        self._frame_cache: OrderedDict[tuple, tuple[Array, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._session: Any = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Array, dict[str, Any]]:
        row = self.rows[index]
        return self._fetch_stamp(row), row

    def _get_session(self) -> Any:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import requests
                    from requests.adapters import HTTPAdapter

                    session = requests.Session()
                    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
                    session.mount("https://", adapter)
                    self._session = session
        return self._session

    def _get_frame(self, key: tuple, url: str) -> tuple[Array, Any]:
        """Return ``(frame_data, wcs)`` from the LRU cache, downloading on a miss."""
        with self._lock:
            cached = self._frame_cache.get(key)
            if cached is not None:
                self._frame_cache.move_to_end(key)
                return cached

        # Download outside the lock so concurrent fetches of *different* frames overlap.
        import bz2
        import io

        from astropy.io import fits
        from astropy.wcs import WCS

        resp = self._get_session().get(url, timeout=self.timeout)
        resp.raise_for_status()
        with fits.open(io.BytesIO(bz2.decompress(resp.content))) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            wcs = WCS(hdul[0].header)

        with self._lock:
            self._frame_cache[key] = (data, wcs)
            self._frame_cache.move_to_end(key)
            while len(self._frame_cache) > self._frame_cache_size:
                self._frame_cache.popitem(last=False)
        return data, wcs

    def _fetch_stamp(self, row: dict[str, Any]) -> Array:
        # Lazy imports: keep the module importable (and tests offline) without the extra.
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.nddata import Cutout2D

        coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        run, camcol = int(row["run"]), int(row["camcol"])
        field, rerun = int(row["field"]), int(row["rerun"])
        planes: list[Array] = []
        for band in self.bands:
            url = _FRAME_URL.format(
                dr=self.data_release, rerun=rerun, run=run, camcol=camcol,
                field=field, band=band,
            )
            data, wcs = self._get_frame((rerun, run, camcol, field, band), url)
            cut = Cutout2D(data, coord, size=self.stamp_px, wcs=wcs)
            planes.append(np.asarray(cut.data, dtype=np.float64))
        return np.stack(planes)
