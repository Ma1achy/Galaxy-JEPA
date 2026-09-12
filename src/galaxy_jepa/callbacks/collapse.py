"""Collapse monitor — the key risk watch for the JEPA pretrain (T3.collapse-monitor).

``docs/spec/validation.md`` Tier 3: representation variance / rank / std must stay above a
collapse floor. The bbox-biased masking removes the easy sky-prediction task, which shifts
the loss landscape and can move the collapse / EMA sweet spot (``docs/masking.md`` §7) — so
this watch is not optional bookkeeping, it is *the* thing the vertical-slice pilot reads.

Three complementary signals on the (frozen monitor-slice) embeddings ``(N, D)``:

* **std** — mean per-dimension standard deviation. Collapse to a point ⇒ → 0.
* **effective rank** — ``exp(entropy(singular-value distribution))`` of the centred
  embeddings. A healthy representation spreads variance across many directions (erank ≫ 1);
  a collapsed one concentrates it in one (erank → 1).
* **mean pairwise cosine** — average cosine similarity between distinct embeddings.
  Everything collapsing to one direction ⇒ → 1.

The monitor *reports the trace* (the deliverable); it only **halts** on the unambiguous
failures — a NaN/Inf embedding, or std driven essentially to zero — so a merely-undertrained
pilot is read, not aborted.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import torch

from galaxy_jepa.core.config import FrozenChoice

logger = logging.getLogger(__name__)

__all__ = [
    "CollapseFloorFreeze",
    "CollapseMonitor",
    "CollapseSignals",
    "collapse_signals",
    "effective_rank",
]


class CollapseFloorFreeze(FrozenChoice):
    """The pre-registered kill criterion: what counts as collapse, decided *before* the run.

    A criterion chosen after seeing the curve is not a criterion. So this is a ``FrozenChoice``
    like the effect floor and the normalisation statistic — hashed into ``config_hash``, stamped
    onto every artefact, enforced by :class:`CollapseMonitor` rather than by judgement at hour 30.

    **What the thresholds are anchored on, and what they are not.** The one place in this project
    where a collapse signal is tied to a *scientific outcome* is the pilot: effective rank held
    ≈ 10.2–10.6 out to 6,000 steps and the frozen probe reached AUC 0.905.
    :attr:`min_effective_rank` is half that — comfortably below a configuration known to work, so
    a run in the pilot's regime is never killed. It is deliberately **not** set near the 4.1 the
    Brief F smoke plateaued at,
    because that would be reading the threshold off the curve it is meant to judge.

    :attr:`grace_fraction` comes from the schedule's own shape, not from any observed trace: the
    EMA momentum ramps from ``ema_start`` to ``ema_end`` across the *whole* run, so early steps
    have a fast-moving target and an unsettled representation is expected rather than alarming.
    A tenth of the schedule is the grace. :attr:`hard_floor` is the unambiguous case — an
    effective rank below 2 is essentially a single direction — and applies as soon as the LR
    warmup is over. :attr:`consecutive_readings` is there so one noisy monitor batch cannot end a
    ten-hour job.

    **What is not known.** The pilot is *one* trace, on 10,000 stamps of a different corpus, and
    the Brief F smoke ran 300 steps. Neither says what 827k stamps should look like. So this floor
    is conservative about declaring failure and explicitly **provisional**: it is a tripwire
    against wasting days on a dead run, not a claim about where healthy training sits. Re-derive
    it from the first real trace, deliberately, and record that as a new freeze.
    """

    #: Sustained effective rank below this, after the grace period, means stop and retune.
    min_effective_rank: float = 5.0
    #: Grace period as a fraction of ``steps`` — the EMA ramp's own timescale.
    grace_fraction: float = 0.10
    #: Unambiguous collapse: essentially one direction. Applies from ``hard_floor_after_step``.
    hard_floor: float = 2.0
    hard_floor_after_step: int = 100
    #: Consecutive monitor readings required before either floor fires.
    consecutive_readings: int = 3


def effective_rank(svals: torch.Tensor) -> float:
    """``exp(entropy)`` of the normalised singular-value distribution.

    The collapse-monitor effective-rank kernel, factored out so the probing eigen-analysis
    (the concept-direction Gram spectrum, ``probing/entanglement.py``) reads the *same*
    definition the pretraining monitor reports — the repo's second-consumer rule (the
    second consumer has now arrived). A flat spectrum spreads variance across many
    directions (erank ≫ 1); a collapsed one concentrates it in one (erank → 1).
    """
    p = svals / svals.sum().clamp_min(1e-12)
    entropy = float(-(p * (p.clamp_min(1e-12)).log()).sum())
    return math.exp(entropy)


@dataclasses.dataclass(frozen=True)
class CollapseSignals:
    """One reading of the three collapse signals on a batch of embeddings."""

    std: float
    effective_rank: float
    mean_cosine: float
    n: int

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.std, self.effective_rank, self.mean_cosine))


def collapse_signals(embeddings: torch.Tensor) -> CollapseSignals:
    """Compute the three collapse signals for ``(N, D)`` embeddings."""
    if embeddings.dim() != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {tuple(embeddings.shape)}")
    x = embeddings.detach().float()
    n = x.shape[0]

    std = float(x.std(dim=0, unbiased=False).mean())

    centred = x - x.mean(dim=0, keepdim=True)
    # singular values of the centred matrix → normalised distribution → entropy → exp.
    # The SVD runs on the CPU **deliberately**: `aten::_linalg_svd.U` has no MPS kernel, so on the
    # project's own device this line is the one op in the whole training path that cannot execute
    # there. Relocating it here — explicitly, for a matrix that is at most (batch, embed_dim) —
    # keeps `PYTORCH_ENABLE_MPS_FALLBACK` **unset**, which is what makes "nothing falls back
    # silently" a fact rather than a hope: with the blanket variable set, any future unimplemented
    # op would quietly move to the CPU instead of raising. Brief F2.3.
    erank = effective_rank(torch.linalg.svdvals(centred.cpu()))

    normed = torch.nn.functional.normalize(x, dim=1)
    sim = normed @ normed.t()
    off_diag = sim.sum() - torch.diagonal(sim).sum()
    mean_cosine = float(off_diag / max(n * (n - 1), 1))

    return CollapseSignals(std=std, effective_rank=erank, mean_cosine=mean_cosine, n=n)


class CollapseMonitor:
    """Tracks the collapse signals across pretraining and decides the hard-halt condition."""

    def __init__(
        self,
        *,
        std_floor: float = 1e-4,
        floor: CollapseFloorFreeze | None = None,
        total_steps: int | None = None,
        history: list[dict[str, float]] | None = None,
    ):
        self.std_floor = std_floor
        #: The pre-registered rank criterion. ``None`` keeps the historical behaviour — halt only
        #: on the unambiguous failures — and a run without it forfeits the tripwire, which
        #: ``harness`` records in ``escape_hatches_used`` rather than leaving implied.
        self.floor = floor
        self.total_steps = total_steps
        #: Seeded on resume, so the consecutive-readings rule is not reset by a restart.
        self.history: list[dict[str, float]] = [dict(r) for r in (history or [])]
        self.halt_reason: str | None = None

    def update(self, step: int, embeddings: torch.Tensor) -> CollapseSignals:
        signals = collapse_signals(embeddings)
        self.history.append(
            {
                "step": step,
                "std": signals.std,
                "effective_rank": signals.effective_rank,
                "mean_cosine": signals.mean_cosine,
            }
        )
        return signals

    def should_halt(self, signals: CollapseSignals) -> bool:
        """Halt on a non-finite embedding, a std collapsed to ~0, or the pre-registered floor."""
        if not signals.is_finite:
            self.halt_reason = "non-finite embedding"
            return True
        if signals.std < self.std_floor:
            self.halt_reason = f"std {signals.std:.2e} below the {self.std_floor:.0e} floor"
            return True
        return self._rank_floor_breached()

    def _rank_floor_breached(self) -> bool:
        """The G5 criterion, applied to the *history* so a single noisy batch cannot fire it."""
        floor = self.floor
        if floor is None or not self.history:
            return False
        step = int(self.history[-1]["step"])
        window = self.history[-floor.consecutive_readings :]
        if len(window) < floor.consecutive_readings:
            return False
        ranks = [r["effective_rank"] for r in window]
        if step >= floor.hard_floor_after_step and all(r < floor.hard_floor for r in ranks):
            self.halt_reason = (
                f"effective rank {ranks[-1]:.2f} below the hard floor {floor.hard_floor} for "
                f"{len(ranks)} consecutive readings at step {step} — a single direction, not "
                "slow training"
            )
            return True
        grace = int(floor.grace_fraction * (self.total_steps or 0))
        if step >= grace and all(r < floor.min_effective_rank for r in ranks):
            self.halt_reason = (
                f"effective rank {ranks[-1]:.2f} below the pre-registered floor "
                f"{floor.min_effective_rank} for {len(ranks)} consecutive readings at step "
                f"{step} (grace was {grace} steps). Stop and retune EMA / masking ratio rather "
                "than spend more hours; the floor is half the pilot's ~10.3, which worked"
            )
            return True
        return False

    def trace(self) -> dict[str, list[float]]:
        """The recorded trace as column lists — for plotting / the pilot read-out."""
        if not self.history:
            return {"step": [], "std": [], "effective_rank": [], "mean_cosine": []}
        return {key: [row[key] for row in self.history] for key in self.history[0]}
