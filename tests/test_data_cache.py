"""Integration tests for the fp16 pre-bake cache + dataset (docs/spec/data.md, slice plan).

The cache is the parity-locked pipeline run once on disk, so the load-bearing guarantees:

* a baked stamp equals the frozen ``Pipeline`` output (within fp16 tolerance) — the cache
  is not silently lossy beyond the fp16 it advertises;
* the bake is **incremental and hash-keyed**: re-baking the same corpus adds nothing, and
  a top-up appends only new IDs under the same stats; a changed Q lands in a *different*
  cache directory (automatic invalidation, never a silent mix);
* the streaming ``fit_normalise`` matches a full-stack ``Normalise.fit`` (low-memory, same answer).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from galaxy_jepa.data.cache import (
    _assert_untorn,
    _read_index,
    bake_cache,
    fit_normalise,
    pipeline_hash,
)
from galaxy_jepa.data.dataset import StampDataset, rows_by_id
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline

pytestmark = pytest.mark.integration


# the cache stores normalised stamps, so every bake names the freeze that made them
NORM_HASH = "test-normalisation-hash"


def _fitted_pipeline(source, q: float = 4.0) -> Pipeline:
    stretch = AsinhStretch(q=q)
    return Pipeline((stretch, fit_normalise(source, stretch, n_sample=64, seed=0).valid))


def _frozen_pipeline(source: DirectorySource, *, q: float = 4.0, seed: int = 0) -> Pipeline:
    stretch = AsinhStretch(q=q)
    norm = fit_normalise(source, stretch, n_sample=10_000, seed=seed).valid  # >= n → all
    return Pipeline((stretch, norm))


def test_streaming_fit_matches_full_stack(pretraining_corpus):
    source = DirectorySource(pretraining_corpus)
    stretch = AsinhStretch(q=4.0)
    # valid_only=False so the comparison is like for like: Normalise.fit has no validity mask
    streamed = fit_normalise(source, stretch, n_sample=10_000, seed=0, valid_only=False).valid
    stacked = Normalise.fit(np.stack([stretch(img) for img, _ in source]))
    assert streamed.mean == pytest.approx(stacked.mean, rel=1e-9, abs=1e-9)
    assert streamed.std == pytest.approx(stacked.std, rel=1e-9, abs=1e-9)


def test_bake_round_trips_within_fp16_tol(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    pipeline = _frozen_pipeline(source)
    cache = bake_cache(source, pipeline, tmp_path / "cache", normalisation_hash=NORM_HASH)

    assert len(cache) == len(source)
    assert cache.index.dtype == "float16"
    assert cache.index.shape == (3, 64, 64)
    # every baked stamp equals the pipeline output within fp16 precision
    for image, row in source:
        expected = pipeline(image).astype(np.float32)
        got = np.asarray(cache.get(int(row["object_id"])), dtype=np.float32)
        assert np.allclose(got, expected, atol=2e-3, rtol=2e-3)


def test_bake_is_incremental_and_idempotent(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    pipeline = _frozen_pipeline(source)
    base = tmp_path / "cache"
    first = bake_cache(source, pipeline, base, normalisation_hash=NORM_HASH)
    n1 = len(first)
    # re-baking the identical corpus appends nothing (every ID already present)
    second = bake_cache(source, pipeline, base, normalisation_hash=NORM_HASH)
    assert len(second) == n1
    assert second.object_ids == first.object_ids


def test_changed_q_lands_in_a_separate_cache(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    base = tmp_path / "cache"
    p4 = _frozen_pipeline(source, q=4.0)
    p8 = _frozen_pipeline(source, q=8.0)
    assert pipeline_hash(p4) != pipeline_hash(p8)
    c4 = bake_cache(source, p4, base, normalisation_hash=NORM_HASH)
    c8 = bake_cache(source, p8, base, normalisation_hash=NORM_HASH)
    assert c4.cache_dir != c8.cache_dir  # automatic invalidation by hash-keyed dir


def test_dataset_yields_baked_tensors(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    pipeline = _frozen_pipeline(source)
    cache = bake_cache(source, pipeline, tmp_path / "cache", normalisation_hash=NORM_HASH)
    rows = rows_by_id([row for _, row in source])
    ids = [int(r["object_id"]) for r in rows.values()]

    ds = StampDataset(cache, rows, ids)
    assert len(ds) == len(ids)
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 64, 64)
    assert item["image"].dtype.is_floating_point
    assert "petro_rad_arcsec" in item and "pixel_scale" in item


def test_dataset_label_path():
    # A hand-built cache row + a row carrying the t01 fraction → binary label derivation.
    from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL

    class _Cache:
        def __contains__(self, oid):
            return True

        def get(self, oid):
            return np.zeros((3, 8, 8), dtype=np.float16)

    rows = {7: {"object_id": 7, "petroRad_r": 5.0, FEATURED_FRACTION_COL: 0.92}}
    ds = StampDataset(_Cache(), rows, [7], label_fraction_col=FEATURED_FRACTION_COL)
    item = ds[0]
    assert item["label"] == 1  # featured (>= 0.5)
    assert item["featured_fraction"] == pytest.approx(0.92)


class TestTheCacheKnowsWhatMadeIt:
    """The cache stores *normalised* stamps, so it has to name the statistic that normalised them.

    ``pipeline_hash`` covers the mean/std *values*; ``normalisation_hash`` names the provenance
    record they came from. Two freezes can carry identical numbers and different stories — a
    re-fit that landed on the same constants, say — and a cache that could not tell them apart
    would be a parity lock with no key.
    """

    def test_the_index_records_the_normalisation_artefact(self, tmp_path, pretraining_corpus):
        source = DirectorySource(pretraining_corpus)
        pipeline = _fitted_pipeline(source)
        cache = bake_cache(source, pipeline, tmp_path / "cache", normalisation_hash="abc123")
        assert cache.index.normalisation_hash == "abc123"
        assert json.loads((cache.cache_dir / "index.json").read_text())["normalisation_hash"] == (
            "abc123"
        )

    def test_topping_up_under_a_different_freeze_is_refused(self, tmp_path, pretraining_corpus):
        source = DirectorySource(pretraining_corpus)
        pipeline = _fitted_pipeline(source)
        base = tmp_path / "cache"
        bake_cache(source, pipeline, base, normalisation_hash="abc123")
        with pytest.raises(RuntimeError, match="baked under normalisation"):
            bake_cache(source, pipeline, base, normalisation_hash="def456")

    def test_a_cache_baked_before_the_freeze_reads_back_as_unrecorded(self, tmp_path):
        """Tolerant, not silent: an old index has no such field and says so by being empty."""
        d = tmp_path / "old"
        d.mkdir()
        (d / "index.json").write_text(
            json.dumps(
                {
                    "pipeline_hash": "h",
                    "channels": 3,
                    "height": 4,
                    "width": 4,
                    "dtype": "float16",
                    "object_ids": [],
                }
            )
        )
        assert _read_index(d) is not None
        assert _read_index(d).normalisation_hash == ""


class TestATornCacheIsNeverAppendedTo:
    """A four-hour bake gets interrupted; the index is only written at the end.

    That leaves ``stamps.f16`` longer than the index accounts for. Appending would number the
    orphaned rows as the stamps just baked, so every row past the interruption point would read
    one galaxy's pixels under another galaxy's object ID — silent, and fatal downstream. This is
    the defect that actually happened, mid-E6, at 18,455 rows against an index claiming 3,000.
    """

    def test_appending_to_a_torn_cache_raises_rather_than_mislabelling(
        self, tmp_path, pretraining_corpus
    ):
        source = DirectorySource(pretraining_corpus)
        pipeline = _fitted_pipeline(source)
        base = tmp_path / "cache"
        cache = bake_cache(source, pipeline, base, normalisation_hash="n")
        data = cache.cache_dir / "stamps.f16"

        committed = data.stat().st_size
        with data.open("ab") as fh:  # an interrupted bake, exactly
            fh.write(b"\x00" * (2 * 3 * 4 * 4 * 2))
        with pytest.raises(RuntimeError, match="torn cache"):
            bake_cache(source, pipeline, base, normalisation_hash="n")

        # and the message's own recovery restores a cache that bakes again
        with data.open("r+b") as fh:
            fh.truncate(committed)
        again = bake_cache(source, pipeline, base, normalisation_hash="n")
        assert len(again) == len(cache)

    def test_a_data_file_with_no_index_at_all_is_refused(self, tmp_path):
        d = tmp_path / "cache" / "somehash"
        d.mkdir(parents=True)
        (d / "stamps.f16").write_bytes(b"\x00" * 4096)
        with pytest.raises(RuntimeError, match="torn cache"):
            _assert_untorn(d / "stamps.f16", None, np.dtype(np.float16))
