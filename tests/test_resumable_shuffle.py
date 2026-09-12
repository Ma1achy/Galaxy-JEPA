"""The data order must be a pure function of the seed, or a resume cannot be identical.

``DataLoader(shuffle=True)` draws its order from the global torch RNG at each ``iter()``, so a
resumed run lands on a different sequence from the same step: the weights restore, the data does
not, and the trajectory diverges while looking continuous. :class:`ResumableShuffle` makes position
``i`` name exactly one index for all time, which is the last piece of "the trajectory is a function
of ``(seed, step)``" — the claim the mid-run checkpoint's identity proof rests on (Brief G1).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from galaxy_jepa.data.dataset import ResumableShuffle

pytestmark = pytest.mark.invariant


def _take(sampler: ResumableShuffle, n: int) -> list[int]:
    return list(itertools.islice(iter(sampler), n))


def test_resuming_at_a_position_continues_the_same_stream():
    whole = _take(ResumableShuffle(11, seed=3), 40)
    assert _take(ResumableShuffle(11, seed=3, start=17), 23) == whole[17:]


def test_every_epoch_is_a_complete_permutation():
    n = 9
    stream = _take(ResumableShuffle(n, seed=0), 3 * n)
    for e in range(3):
        assert sorted(stream[e * n : (e + 1) * n]) == list(range(n))


def test_consecutive_epochs_are_shuffled_differently():
    n = 64
    stream = _take(ResumableShuffle(n, seed=0), 2 * n)
    assert stream[:n] != stream[n:], "reusing one permutation would make every epoch identical"


def test_the_seed_determines_the_order_and_nothing_else_does():
    first = _take(ResumableShuffle(20, seed=5), 60)
    np.random.seed(999)  # global state must not reach it
    assert _take(ResumableShuffle(20, seed=5), 60) == first
    assert _take(ResumableShuffle(20, seed=6), 60) != first


def test_the_stream_is_endless():
    """The pretrain loop is step-bounded, not epoch-bounded: it must never run out."""
    assert len(_take(ResumableShuffle(4, seed=0), 500)) == 500


def test_an_empty_dataset_is_a_loud_error_not_an_empty_stream():
    with pytest.raises(ValueError, match="empty dataset"):
        ResumableShuffle(0, seed=0)
