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


def test_the_svd_runs_on_the_cpu_so_no_blanket_mps_fallback_is_needed(monkeypatch):
    """``aten::_linalg_svd.U`` has no MPS kernel — the one op in the training path that doesn't.

    Pinned because the alternative is ``PYTORCH_ENABLE_MPS_FALLBACK=1``, and a blanket fallback
    would quietly relocate *any* future unimplemented op to the CPU instead of raising. Keeping
    this one relocation explicit is what makes "nothing falls back silently" checkable. The
    matrix is at most (batch, embed_dim), so the move costs nothing. Brief F2.3.
    """
    seen: list[str] = []
    real = torch.linalg.svdvals

    def spy(t, *a, **k):
        seen.append(t.device.type)
        return real(t, *a, **k)

    monkeypatch.setattr(torch.linalg, "svdvals", spy)
    collapse_signals(torch.randn(16, 8, generator=torch.Generator().manual_seed(1)))
    assert seen == ["cpu"], f"the collapse SVD must be handed a CPU tensor, got {seen}"
