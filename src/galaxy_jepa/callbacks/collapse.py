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
import math

import torch

__all__ = ["CollapseSignals", "collapse_signals", "effective_rank", "CollapseMonitor"]


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
    erank = effective_rank(torch.linalg.svdvals(centred))

    normed = torch.nn.functional.normalize(x, dim=1)
    sim = normed @ normed.t()
    off_diag = sim.sum() - torch.diagonal(sim).sum()
    mean_cosine = float(off_diag / max(n * (n - 1), 1))

    return CollapseSignals(std=std, effective_rank=erank, mean_cosine=mean_cosine, n=n)


class CollapseMonitor:
    """Tracks the collapse signals across pretraining and decides the hard-halt condition."""

    def __init__(self, *, std_floor: float = 1e-4):
        self.std_floor = std_floor
        self.history: list[dict[str, float]] = []

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
        """Halt only on the unambiguous failures: non-finite, or std collapsed to ~0."""
        return (not signals.is_finite) or signals.std < self.std_floor

    def trace(self) -> dict[str, list[float]]:
        """The recorded trace as column lists — for plotting / the pilot read-out."""
        if not self.history:
            return {"step": [], "std": [], "effective_rank": [], "mean_cosine": []}
        return {key: [row[key] for row in self.history] for key in self.history[0]}
