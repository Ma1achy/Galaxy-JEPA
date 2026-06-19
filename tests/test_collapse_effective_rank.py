"""Invariant: the extracted ``effective_rank`` helper is identical to the inline computation.

The collapse-monitor effective-rank kernel was factored out so the probing eigen-analysis
reuses the *same* definition (the second-consumer rule). This pins that the refactor changed
nothing — the monitor's ``effective_rank`` still equals the kernel applied to the centred
matrix's singular values, on a fixed matrix.
"""

from __future__ import annotations

import pytest
import torch

from galaxy_jepa.callbacks.collapse import collapse_signals, effective_rank

pytestmark = pytest.mark.invariant


def test_effective_rank_matches_collapse_monitor():
    x = torch.randn(40, 12, generator=torch.Generator().manual_seed(3))
    centred = x - x.mean(dim=0, keepdim=True)
    direct = effective_rank(torch.linalg.svdvals(centred))
    assert collapse_signals(x).effective_rank == pytest.approx(direct)


def test_effective_rank_of_rank_one_is_one():
    # a perfectly collapsed (rank-1) spectrum has effective rank → 1
    svals = torch.tensor([5.0, 0.0, 0.0, 0.0])
    assert effective_rank(svals) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_of_flat_spectrum_is_dimension():
    # a flat spectrum spreads variance across all directions → effective rank = the dimension
    svals = torch.ones(8)
    assert effective_rank(svals) == pytest.approx(8.0)
