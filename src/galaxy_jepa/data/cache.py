"""fp16 pre-baked tensor cache — the parity-locked pipeline run **once**, on disk.

The training dataloader must never read FITS and run asinh+normalise per batch: that
per-step CPU cost dominates and starves the device (``docs/spec/data.md``; the vertical-
slice plan). Instead the frozen :class:`~galaxy_jepa.data.transforms.Pipeline` runs **once**
over the whole pulled corpus and the result is written as **fp16** to a memory-mapped
array, which the dataloader then reads with *zero* per-batch preprocessing. fp16 halves the
working set versus fp32 — decisive on an 18 GB unified-memory machine.

**The cache contract (corpus-stable, incremental):**

* The cache lives under ``base_dir/<pipeline_hash>/`` where ``pipeline_hash`` is the
  ``config_hash`` of the (fitted) pipeline. A different Q / flux-scale / normalisation
  statistic ⇒ a different hash ⇒ a different directory: **stale stats can never be mixed
  with fresh ones**, and invalidation is automatic.
* Normalisation statistics are fitted **once, before the pilot**, on a seeded subsample
  (:func:`fit_normalise`) and frozen into the pipeline. Because the pilot and the full run
  share that one frozen pipeline, they share the ``pipeline_hash`` — so topping the pilot's
  ~30k corpus up to ~100k **appends** new stamps to the *same* cache and **reuses every
  pilot stamp**. The bake is incremental; nothing is ever re-baked. (If the stats were
  re-fit on the top-up, the hash would move, the cache would invalidate, and — worse — the
  pilot encoder would have trained under different preprocessing than the full run.)

This module is numpy-only (no torch, no astropy beyond the source it is handed), so it
stays import-light; the torch ``Dataset`` that consumes it lives in ``data/dataset.py``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from galaxy_jepa.core.config import config_hash
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline
from galaxy_jepa.data.validity import validity_mask

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.json"
_DATA_FILE = "stamps.f16"


class _Source(Protocol):
    """The minimal source contract the cache needs: indexable ``(image, row)`` pairs."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, Any]]: ...


def pipeline_hash(pipeline: Pipeline) -> str:
    """The cache key: a stable ``config_hash`` over the fitted pipeline's config tree."""
    return config_hash(pipeline.to_config())


@dataclasses.dataclass(frozen=True)
class NormaliseFit:
    """The fitted statistic, and the one it would have been without the validity mask.

    Both come out of the same streaming pass — the naive figures cost nothing extra and are
    the evidence that excluding padding was worth doing, so they are returned rather than
    recomputed later by a script that might drift from this one.
    """

    valid: Normalise
    naive: Normalise
    n_stamps: int
    valid_pixel_fraction: float

    def contamination(self) -> list[tuple[float, float, float, float]]:
        """Per channel: ``(naive mean, valid mean, naive std, valid std)``."""
        assert self.valid.mean is not None and self.valid.std is not None
        assert self.naive.mean is not None and self.naive.std is not None
        return list(
            zip(self.naive.mean, self.valid.mean, self.naive.std, self.valid.std, strict=True)
        )


def fit_normalise(
    source: _Source,
    stretch: AsinhStretch,
    *,
    n_sample: int = 8000,
    seed: int = 0,
    valid_only: bool = True,
) -> NormaliseFit:
    """Fit per-channel mean/std on a seeded post-stretch subsample, streaming (low memory).

    Computes the statistic from running per-channel sums rather than stacking the sample, so
    fitting on ~5–10k 256² stamps does not blow past RAM.

    ``valid_only`` excludes the constant regions :mod:`galaxy_jepa.data.validity` detects —
    about one stamp in seven carries cutout padding, which is a constant dragging the mean
    toward the pad value and deflating the variance. The divisor becomes **per channel**,
    since a channel no longer contributes a fixed pixel count.

    **This function no longer decides anything.** Its result is pinned into a
    ``NormalisationFreeze`` and loaded from there by every run; calling it again is fitting, not
    loading, and the harness refuses to do that. Note the reason: the subsample is
    ``rng.choice(len(source), ...)``, so it moves when the corpus grows — which is precisely how
    the statistic drifted silently between the 10k pilot and the 827k corpus.
    """
    n = len(source)
    if n == 0:
        raise ValueError("cannot fit normalisation on an empty source")
    rng = np.random.default_rng(seed)
    k = min(n_sample, n)
    idx = rng.choice(n, size=k, replace=False)

    vsum = vsumsq = nsum = nsumsq = None
    vpixels: np.ndarray | None = None
    npixels = 0
    for i in idx:
        raw = source[int(i)][0]
        stretched = np.asarray(stretch(raw), dtype=np.float64)
        c = stretched.shape[0]
        if vsum is None:
            vsum, vsumsq = np.zeros(c), np.zeros(c)
            nsum, nsumsq = np.zeros(c), np.zeros(c)
            vpixels = np.zeros(c)
        assert vsumsq is not None and nsum is not None and nsumsq is not None
        assert vpixels is not None
        nsum += stretched.sum(axis=(1, 2))
        nsumsq += (stretched**2).sum(axis=(1, 2))
        npixels += stretched.shape[1] * stretched.shape[2]

        keep = validity_mask(raw) if valid_only else np.ones(stretched.shape[1:], dtype=bool)
        masked = stretched * keep
        vsum += masked.sum(axis=(1, 2))
        vsumsq += (masked**2).sum(axis=(1, 2))
        vpixels += float(keep.sum())

    assert vsum is not None and vsumsq is not None and nsum is not None and nsumsq is not None
    assert vpixels is not None
    valid = _moments(vsum, vsumsq, vpixels)
    naive = _moments(nsum, nsumsq, np.full_like(nsum, float(npixels)))
    frac = float(vpixels.sum() / max(npixels * len(vsum), 1))
    logger.info(
        "fit Normalise on %d/%d stamps (valid_only=%s, %.4f%% of pixels kept): mean=%s std=%s",
        k,
        n,
        valid_only,
        100 * frac,
        valid.mean,
        valid.std,
    )
    return NormaliseFit(valid=valid, naive=naive, n_stamps=k, valid_pixel_fraction=frac)


def _moments(csum: np.ndarray, csumsq: np.ndarray, pixels: np.ndarray) -> Normalise:
    mean = csum / pixels
    std = np.sqrt(np.clip(csumsq / pixels - mean**2, 1e-12, None))
    return Normalise(mean=tuple(mean.tolist()), std=tuple(std.tolist()))


@dataclasses.dataclass(frozen=True)
class CacheIndex:
    """The on-disk index sidecar: shape, dtype, the cache key, and the row order."""

    pipeline_hash: str
    channels: int
    height: int
    width: int
    dtype: str
    object_ids: list[int]
    #: ``NormalisationFreeze.content_hash`` of the statistic these stamps were baked under.
    #: ``pipeline_hash`` already covers the *values*; this names the **artefact** they came
    #: from, so a baked stamp can be traced to a provenance record and not merely to a number.
    #: Empty means the cache predates the freeze (D16) and its statistic is unrecorded.
    normalisation_hash: str = ""

    @property
    def n(self) -> int:
        return len(self.object_ids)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


def _read_index(cache_dir: Path) -> CacheIndex | None:
    path = cache_dir / _INDEX_FILE
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return CacheIndex(
        pipeline_hash=raw["pipeline_hash"],
        channels=raw["channels"],
        height=raw["height"],
        width=raw["width"],
        dtype=raw["dtype"],
        object_ids=[int(o) for o in raw["object_ids"]],
        # tolerant: a cache baked before D16 has no such field, and says so by being empty
        normalisation_hash=str(raw.get("normalisation_hash", "")),
    )


def _write_index(cache_dir: Path, index: CacheIndex) -> None:
    (cache_dir / _INDEX_FILE).write_text(json.dumps(dataclasses.asdict(index), indent=2) + "\n")


def _assert_untorn(data_path: Path, index: CacheIndex | None, dtype: np.dtype[Any]) -> None:
    """Refuse to append to a cache whose data file and index disagree about its length.

    An interrupted bake is ordinary — a four-hour job gets Ctrl-C'd — and it leaves
    ``stamps.f16`` longer than the index, because the index is only written at the end. Appending
    to that is not merely wasteful: the next bake writes past the orphaned tail while the index
    numbers those rows as the stamps it just baked, so every row after the interruption point
    reads one galaxy's pixels under another galaxy's object ID. Silent, and fatal to everything
    downstream. The index is the commit point, so recovery is to discard the uncommitted tail —
    but that is the operator's call to make, not a default.
    """
    if not data_path.exists():
        return
    row = int(dtype.itemsize)
    if index is not None:
        row *= index.channels * index.height * index.width
        expected = index.n * row
    else:
        expected = 0
    actual = data_path.stat().st_size
    if actual == expected:
        return
    orphaned = actual - expected
    raise RuntimeError(
        f"torn cache at {data_path.parent}: the data file holds {actual} bytes but the index "
        f"accounts for {expected} ({orphaned:+d}, "
        f"{abs(orphaned) / max(row, 1):.0f} rows). An interrupted bake leaves exactly this, and "
        "appending would number the orphaned rows as newly baked stamps — every one of them "
        "mislabelled. Nothing committed lives past the index, so recovery is to truncate to it:\n"
        f"    python -c \"open({str(data_path)!r},'r+b').truncate({expected})\"\n"
        "or delete the directory and bake again."
    )


def bake_cache(
    source: _Source,
    pipeline: Pipeline,
    base_dir: str | Path,
    *,
    normalisation_hash: str,
    dtype: type = np.float16,
    log_every: int = 2000,
) -> TensorCache:
    """Bake the frozen pipeline over ``source`` into the hash-keyed fp16 cache (incremental).

    Object IDs already present in the cache are **skipped** (their FITS is never even read),
    so a 30k→100k top-up appends only the new ~70k under the same frozen stats. Returns a
    :class:`TensorCache` reader over the resulting cache.

    ``normalisation_hash`` is the ``NormalisationFreeze.content_hash`` the stamps are baked
    under. It is **required** because the cache stores *normalised* stamps: a cache that could
    not name its statistic's provenance record would be a parity lock with no key. It is recorded
    in the index and checked on every top-up.
    """
    key = pipeline_hash(pipeline)
    cache_dir = Path(base_dir) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / _DATA_FILE

    index = _read_index(cache_dir)
    if index is not None and index.pipeline_hash != key:  # pragma: no cover - dir is keyed by hash
        raise RuntimeError(
            f"cache at {cache_dir} has pipeline_hash {index.pipeline_hash} != {key}; won't mix"
        )
    if index is not None and index.normalisation_hash != normalisation_hash:
        raise RuntimeError(
            f"cache at {cache_dir} was baked under normalisation "
            f"{index.normalisation_hash[:12] or '(unrecorded, pre-D16)'} but this run carries "
            f"{normalisation_hash[:12]}. The values may hash alike and the provenance still "
            "differ — re-bake into a fresh directory rather than mixing two records."
        )
    existing = set(index.object_ids) if index is not None else set()
    object_ids = list(index.object_ids) if index is not None else []
    shape = index.shape if index is not None else None
    np_dtype: np.dtype[Any] = np.dtype(dtype)
    _assert_untorn(data_path, index, np_dtype)

    n_new = 0
    with data_path.open("ab") as fh:
        for i in range(len(source)):
            image, row = source[i]
            oid = int(row["object_id"])
            if oid in existing:
                continue
            baked = np.asarray(pipeline(image), dtype=np_dtype)
            if shape is None:
                shape = (baked.shape[0], baked.shape[1], baked.shape[2])
            elif baked.shape != shape:
                raise ValueError(f"stamp {oid} shape {baked.shape} != cache shape {shape}")
            fh.write(baked.tobytes())
            object_ids.append(oid)
            existing.add(oid)
            n_new += 1
            if log_every and n_new % log_every == 0:
                logger.info("baked %d new stamps (%d total)", n_new, len(object_ids))

    if shape is None:
        raise ValueError("nothing to bake: source is empty and cache did not exist")
    new_index = CacheIndex(
        pipeline_hash=key,
        normalisation_hash=normalisation_hash,
        channels=shape[0],
        height=shape[1],
        width=shape[2],
        dtype=np_dtype.name,
        object_ids=object_ids,
    )
    _write_index(cache_dir, new_index)
    logger.info("cache %s: %d new, %d total stamps", key[:12], n_new, len(object_ids))
    return TensorCache(cache_dir)


class TensorCache:
    """Read-only memmap view over a baked fp16 cache (``base_dir/<pipeline_hash>/``).

    ``stamps.f16`` is a flat ``(N, C, H, W)`` fp16 array in the row order recorded by the
    index; this maps each ``object_id`` to its row so a split (a set of IDs) reads straight
    out of the memmap with no per-item decode.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        index = _read_index(self.cache_dir)
        if index is None:
            raise FileNotFoundError(f"no cache index under {self.cache_dir}")
        self.index = index
        self._row_of: dict[int, int] = {oid: r for r, oid in enumerate(index.object_ids)}
        self.data: np.ndarray = np.memmap(
            self.cache_dir / _DATA_FILE,
            dtype=np.dtype(index.dtype),
            mode="r",
            shape=(index.n, *index.shape),
        )

    def __len__(self) -> int:
        return self.index.n

    @property
    def object_ids(self) -> list[int]:
        return self.index.object_ids

    def __contains__(self, object_id: int) -> bool:
        return int(object_id) in self._row_of

    def get(self, object_id: int) -> np.ndarray:
        """Return the baked fp16 ``(C, H, W)`` stamp for ``object_id`` (a memmap view)."""
        row = self._row_of.get(int(object_id))
        if row is None:
            raise KeyError(f"object {object_id} not in cache {self.cache_dir}")
        return self.data[row]

    def present(self, object_ids: Iterable[int]) -> list[int]:
        """The subset of ``object_ids`` that are actually baked, preserving input order."""
        return [int(o) for o in object_ids if int(o) in self._row_of]

    def stack(self, object_ids: Sequence[int]) -> np.ndarray:
        """Stack the baked stamps for ``object_ids`` into ``(len, C, H, W)`` (fp16)."""
        rows = [self._row_of[int(o)] for o in object_ids]
        return np.asarray(self.data[rows])
