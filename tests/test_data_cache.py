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
import math

import numpy as np
import pytest

from galaxy_jepa.data.cache import (
    TensorCache,
    _assert_untorn,
    _read_index,
    bake_cache,
    fit_normalise,
    load_probe_columns,
    pipeline_hash,
    write_probe_columns,
    write_scalars,
)
from galaxy_jepa.data.dataset import StampDataset, rows_by_id
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
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


class TestTheScalarSidecarIsAlignedOrRefused:
    """The 8.46 MB array that replaced a 4.07 GB metadata table (Brief G2).

    ``StampDataset`` needs exactly one number per stamp beyond the pixels — ``petroRad_r``, for
    the bbox-biased masking. It used to read it out of ``rows_by_id`` over *both* corpora,
    1,057,326 rows, measured at 4.07 GB resident. Alignment with the index's row order is the
    entire contract, and a silent misalignment would hand every later galaxy another galaxy's
    Petrosian radius — so these pin that it is checked rather than trusted.
    """

    def _cache(self, probing_corpus, tmp_path):
        source = DirectorySource(probing_corpus)
        pipeline = _fitted_pipeline(source)
        cache = bake_cache(source, pipeline, tmp_path / "cache", normalisation_hash=NORM_HASH)
        return source, cache

    def test_the_values_are_identical_to_what_the_row_dict_path_served(
        self, pretraining_corpus, tmp_path
    ):
        """Parity, as with the cache itself: the cheap path must not be a different path."""
        source, cache = self._cache(pretraining_corpus, tmp_path)
        rows = rows_by_id(source.rows)
        write_scalars(cache.cache_dir, {o: float(r["petroRad_r"]) for o, r in rows.items()})
        fresh = TensorCache(cache.cache_dir)
        # reversed, so a sidecar read positionally rather than by object ID would disagree
        ids = sorted(rows, reverse=True)
        from_rows = StampDataset(fresh, rows, ids)
        from_array = StampDataset(fresh, {}, ids, scalars=fresh.scalars)
        assert len({r["petroRad_r"] for r in source.rows}) > 1, "a constant would prove nothing"
        for i in range(len(ids)):
            assert from_array[i]["object_id"] == from_rows[i]["object_id"]
            # exact, not approximate: the array path must serve the same float, not a near one
            assert from_array[i]["petro_rad_arcsec"] == from_rows[i]["petro_rad_arcsec"]

    def test_a_missing_galaxy_is_a_misalignment_and_is_refused(self, pretraining_corpus, tmp_path):
        source, cache = self._cache(pretraining_corpus, tmp_path)
        petro = {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
        petro.pop(next(iter(petro)))
        with pytest.raises(ValueError, match="misalignment, not"):
            write_scalars(cache.cache_dir, petro)

    def test_a_missing_radius_is_a_value_not_a_misalignment(self, pretraining_corpus, tmp_path):
        """``petrosian_box`` already falls back to the global box for a NaN, so NaN must pass."""
        source, cache = self._cache(pretraining_corpus, tmp_path)
        write_scalars(cache.cache_dir, {int(r["object_id"]): float("nan") for r in source.rows})
        assert np.isnan(TensorCache(cache.cache_dir).scalars).all()

    def test_a_sidecar_the_index_does_not_vouch_for_is_not_read(self, pretraining_corpus, tmp_path):
        _source, cache = self._cache(pretraining_corpus, tmp_path)
        with pytest.raises(FileNotFoundError, match="no scalar sidecar recorded"):
            _ = TensorCache(cache.cache_dir).scalars

    def test_a_tampered_sidecar_is_refused(self, pretraining_corpus, tmp_path):
        source, cache = self._cache(pretraining_corpus, tmp_path)
        write_scalars(
            cache.cache_dir, {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
        )
        path = cache.cache_dir / "petro_rad_arcsec.f64"
        values = np.fromfile(path, dtype=np.float64)
        values[0] += 1.0
        path.write_bytes(values.tobytes())
        with pytest.raises(RuntimeError, match="digest"):
            _ = TensorCache(cache.cache_dir).scalars

    def test_a_truncated_sidecar_is_refused_before_the_digest(self, pretraining_corpus, tmp_path):
        source, cache = self._cache(pretraining_corpus, tmp_path)
        write_scalars(
            cache.cache_dir, {int(r["object_id"]): float(r["petroRad_r"]) for r in source.rows}
        )
        path = cache.cache_dir / "petro_rad_arcsec.f64"
        path.write_bytes(path.read_bytes()[:-8])
        with pytest.raises(RuntimeError, match="Misaligned by construction"):
            _ = TensorCache(cache.cache_dir).scalars


class TestTheProbeColumnSidecar:
    """The probing path's equivalent of the scalar sidecar (Brief I).

    ``evaluate_probe`` built ``rows_by_id`` over 230,358 rows x 132 columns — measured at
    1.49 GB, doubled to 2.99 GB by ``LabelProvider``'s copy — to read columns that fit in
    677 MB as arrays. It was killed for memory before embedding a stamp. Same discipline as
    ``write_scalars``: the digest lives in the index, and anything that does not match is
    refused rather than used.
    """

    @staticmethod
    def _baked(probing_corpus, tmp_path):
        source = DirectorySource(probing_corpus)
        pipeline = Pipeline([AsinhStretch(q=4.0)])
        cache = bake_cache(source, pipeline, tmp_path / "cache", normalisation_hash="test")
        return source, cache

    @pytest.mark.invariant
    def test_values_round_trip_exactly(self, probing_corpus, tmp_path):
        """float64 storage of a float64 parse — exact, not approximate.

        The confident-extremes cut is a comparison against exactly these bits, so a parity
        claim that needed a tolerance would not be a parity claim.
        """
        source, cache = self._baked(probing_corpus, tmp_path)
        cols = ["object_id", "petroRad_r", "snr", "psf"]
        write_probe_columns(cache.cache_dir, source.rows, cols)
        columns = load_probe_columns(cache.cache_dir, TensorCache(cache.cache_dir).index)

        for row in source.rows:
            oid = int(row["object_id"])
            if oid not in columns:
                continue
            assert columns[oid]["snr"] == float(row["snr"])
            assert columns[oid]["petroRad_r"] == float(row["petroRad_r"])

    @pytest.mark.invariant
    def test_a_missing_object_is_a_misalignment_and_is_refused(self, probing_corpus, tmp_path):
        """Every later row would carry another galaxy's votes: a value may be absent, a row
        may not."""
        source, cache = self._baked(probing_corpus, tmp_path)
        write_probe_columns(cache.cache_dir, source.rows, ["object_id", "snr"])
        index = TensorCache(cache.cache_dir).index
        # a sidecar one row short of the index it claims to align to
        (cache.cache_dir / "probe_columns.f64").write_bytes(
            np.zeros((1, index.n - 1), dtype=np.float64).tobytes()
        )
        with pytest.raises(RuntimeError, match="Misaligned by construction"):
            load_probe_columns(cache.cache_dir, index)

    @pytest.mark.invariant
    def test_a_changed_file_under_an_unchanged_index_is_refused(self, probing_corpus, tmp_path):
        source, cache = self._baked(probing_corpus, tmp_path)
        write_probe_columns(cache.cache_dir, source.rows, ["object_id", "snr"])
        index = TensorCache(cache.cache_dir).index
        path = cache.cache_dir / "probe_columns.f64"
        values = np.fromfile(path, dtype=np.float64)
        values[0] += 1.0
        path.write_bytes(values.tobytes())
        with pytest.raises(RuntimeError, match="refusing to read it"):
            load_probe_columns(cache.cache_dir, index)

    @pytest.mark.invariant
    def test_a_cache_without_one_says_so_rather_than_returning_empty(
        self, probing_corpus, tmp_path
    ):
        _, cache = self._baked(probing_corpus, tmp_path)
        with pytest.raises(FileNotFoundError, match="no probe-column sidecar"):
            load_probe_columns(cache.cache_dir, cache.index)

    def test_an_absent_or_unparseable_cell_is_nan_not_a_failure(self, probing_corpus, tmp_path):
        """Absent, blank or unreadable is a missing *measurement*; the floors already treat it so.

        Both cases at once: this fixture corpus carries no vote-fraction column at all, and
        ``snr`` is blanked. A sidecar naming a column the corpus lacks is the ordinary case
        when the schemes widen ahead of a pull, and it must be NaN rather than a crash.
        """
        source, cache = self._baked(probing_corpus, tmp_path)
        rows = [{**r, "snr": ""} for r in source.rows]
        cols = ["object_id", "snr", FEATURED_FRACTION_COL]
        write_probe_columns(cache.cache_dir, rows, cols)
        columns = load_probe_columns(cache.cache_dir, TensorCache(cache.cache_dir).index)
        oid = int(source.rows[0]["object_id"])
        assert math.isnan(columns[oid]["snr"])  # present but blank
        assert math.isnan(columns[oid][FEATURED_FRACTION_COL])  # never in the corpus

    def test_only_the_columns_touched_are_materialised(self, probing_corpus, tmp_path):
        """The memory claim: a single-feature probe must not pay for the other eighty columns."""
        source, cache = self._baked(probing_corpus, tmp_path)
        cols = ["object_id", "petroRad_r", "snr", "psf"]
        write_probe_columns(cache.cache_dir, source.rows, cols)
        columns = load_probe_columns(cache.cache_dir, TensorCache(cache.cache_dir).index)
        assert columns._loaded == {}
        _ = columns[int(source.rows[0]["object_id"])]["snr"]
        assert set(columns._loaded) == {"snr"}

    def test_rows_from_another_corpus_are_refused(self, probing_corpus, tmp_path):
        _, cache = self._baked(probing_corpus, tmp_path)
        with pytest.raises(ValueError, match="aligned to nothing"):
            write_probe_columns(
                cache.cache_dir, [{"object_id": -1, "snr": 0.5}], ["object_id", "snr"]
            )
