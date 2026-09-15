"""Invariant tests for the embedding matrix's index — the ladder's hottest inner loop.

`EmbeddingMatrix.index` is read once per element by `feature_ids`, and `build_feature_controls`
calls that eight times per feature. As a plain property it rebuilt the whole dict on every read,
which made the filter quadratic in (number of ids x matrix size): measured at 225 s for 40,000
ids against 8.6 ms hoisted, about 2.8 h of one six-hour ladder run (Brief K3-iv). These pin the
cache so it cannot silently come undone.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, feature_ids

pytestmark = pytest.mark.invariant


def _matrix(n: int) -> EmbeddingMatrix:
    return EmbeddingMatrix(
        object_ids=np.arange(n, dtype=np.int64),
        x=np.zeros((n, 4), dtype=np.float32),
        encoder_name="test",
    )


def test_index_is_built_once_not_per_read():
    """The structural guarantee: two reads are the *same object*, so nothing can rebuild it.

    This is the whole fix. A timing bound would also catch a regression, but identity catches it
    deterministically and in the fast gate.
    """
    matrix = _matrix(1_000)
    assert matrix.index is matrix.index
    assert matrix.index == {i: i for i in range(1_000)}


def test_caching_the_index_does_not_unfreeze_the_matrix():
    """``cached_property`` writes into ``__dict__``; the declared fields must stay frozen."""
    matrix = _matrix(10)
    _ = matrix.index
    with pytest.raises(dataclasses.FrozenInstanceError):
        matrix.encoder_name = "mutated"  # type: ignore[misc]


def test_feature_ids_does_not_rebuild_the_index_per_element():
    """The filter must be linear in the id list, not in (ids x matrix size).

    Without the cache this exact call measured ~10 s; with it, single-digit milliseconds. The
    bound is deliberately loose — two orders of magnitude above the cached cost and an order
    below the quadratic one — so a slow machine cannot fail it but the regression cannot pass.
    """
    matrix = _matrix(20_000)
    labels = LabelProvider({}, vote_count_min=1)  # no scheme -> eligible() is the identity
    ids = list(range(10_000))
    start = time.perf_counter()
    present = feature_ids(matrix, labels, "featured", ids)
    assert time.perf_counter() - start < 2.0
    assert present == ids
