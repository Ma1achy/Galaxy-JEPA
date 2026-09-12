"""Invariant: the extracted ``effective_rank`` helper is identical to the inline computation.

The collapse-monitor effective-rank kernel was factored out so the probing eigen-analysis
reuses the *same* definition (the second-consumer rule). This pins that the refactor changed
nothing — the monitor's ``effective_rank`` still equals the kernel applied to the centred
matrix's singular values, on a fixed matrix.
"""

from __future__ import annotations

import pytest
import torch

from galaxy_jepa.callbacks.collapse import (
    CollapseFloorFreeze,
    CollapseMonitor,
    CollapseSignals,
    collapse_signals,
    effective_rank,
)

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


class TestThePreRegisteredCollapseFloor:
    """The G5 kill criterion: stated before the run, enforced by the monitor, not by judgement.

    The floor exists so a dead representation costs a monitor interval rather than thirty hours.
    It is deliberately conservative — half the pilot's ~10.3, the only effective rank in this
    project tied to a working probe — so these pin both directions: it fires on genuine collapse
    and it does *not* fire on a run in the regime that produced AUC 0.905.
    """

    def _floor(self, **overrides) -> CollapseFloorFreeze:
        base = dict(
            derived_from="pilot trace", frozen_at="2026-09-12", frozen_by="tests", rationale="r"
        )
        base.update(overrides)
        return CollapseFloorFreeze(**base)

    def _drive(self, monitor: CollapseMonitor, ranks, *, first_step: int, every: int = 100):
        """Feed the monitor readings with a chosen effective rank, and report when it halts."""
        for i, rank in enumerate(ranks):
            step = first_step + i * every
            monitor.history.append(
                {"step": step, "std": 1.0, "effective_rank": float(rank), "mean_cosine": 0.5}
            )
            signals = CollapseSignals(std=1.0, effective_rank=float(rank), mean_cosine=0.5, n=64)
            if monitor.should_halt(signals):
                return step
        return None

    def test_a_run_in_the_pilots_regime_is_never_killed(self):
        monitor = CollapseMonitor(floor=self._floor(), total_steps=50_000)
        assert self._drive(monitor, [10.3] * 20, first_step=0) is None

    def test_the_brief_f_smoke_trace_would_not_have_been_killed_inside_the_grace(self):
        """erank 4.1 at step 175 is below the floor but well inside a 5,000-step grace."""
        monitor = CollapseMonitor(floor=self._floor(), total_steps=50_000)
        assert self._drive(monitor, [4.1] * 10, first_step=0, every=25) is None

    def test_a_sustained_breach_past_the_grace_halts(self):
        monitor = CollapseMonitor(floor=self._floor(), total_steps=10_000)
        halted_at = self._drive(monitor, [4.0] * 5, first_step=1_000)
        assert halted_at == 1_200  # the third consecutive reading, not the first
        assert "below the pre-registered floor" in (monitor.halt_reason or "")

    def test_one_noisy_reading_cannot_end_a_ten_hour_job(self):
        monitor = CollapseMonitor(floor=self._floor(), total_steps=10_000)
        assert self._drive(monitor, [9.0, 3.0, 9.0, 9.0, 3.0, 9.0], first_step=5_000) is None

    def test_the_hard_floor_applies_as_soon_as_the_warmup_is_over(self):
        """Effective rank under 2 is one direction; waiting out the grace would waste hours."""
        monitor = CollapseMonitor(floor=self._floor(), total_steps=1_000_000)
        halted_at = self._drive(monitor, [1.4] * 4, first_step=100)
        assert halted_at == 300
        assert "hard floor" in (monitor.halt_reason or "")

    def test_without_a_floor_only_the_unambiguous_failures_halt(self):
        """A run that forfeits the tripwire keeps the old behaviour — and is stamped for it."""
        monitor = CollapseMonitor(floor=None, total_steps=10_000)
        assert self._drive(monitor, [1.1] * 10, first_step=9_000) is None
        assert monitor.should_halt(
            CollapseSignals(std=0.0, effective_rank=1.0, mean_cosine=1.0, n=8)
        )
        assert "std" in (monitor.halt_reason or "")

    def test_the_floor_rides_in_the_config_hash(self):
        """A criterion that does not reach the artefact is a note, not a pre-registration."""
        from galaxy_jepa.core.config import config_hash
        from galaxy_jepa.harness import HarnessConfig, PathsConfig

        paths = PathsConfig(pretrain_dir="a", probe_dir="b", out_dir="c")
        loose = HarnessConfig(paths=paths)
        pinned = HarnessConfig(paths=paths, collapse_floor=self._floor())
        assert config_hash(loose.determining_dump()) != config_hash(pinned.determining_dump())
        assert config_hash(pinned.determining_dump()) != config_hash(
            HarnessConfig(
                paths=paths, collapse_floor=self._floor(min_effective_rank=6.0)
            ).determining_dump()
        )

    def test_the_history_survives_a_resume_so_the_consecutive_rule_is_not_reset(self):
        """Restarting must not buy a collapsing run another three readings of grace."""
        prior = [
            {"step": 5_000, "std": 1.0, "effective_rank": 3.0, "mean_cosine": 0.5},
            {"step": 5_100, "std": 1.0, "effective_rank": 3.0, "mean_cosine": 0.5},
        ]
        monitor = CollapseMonitor(floor=self._floor(), total_steps=10_000, history=prior)
        assert self._drive(monitor, [3.0], first_step=5_200) == 5_200
