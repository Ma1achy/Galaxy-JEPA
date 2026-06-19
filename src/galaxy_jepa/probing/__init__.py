"""Probing — the frozen-encoder nameability ladder, controls battery, and uncertainty geometry.

The encoder-agnostic measurement layer (``docs/probing-harness-design.md``): L2-logistic
concept directions (``logistic``), the cost spine that extracts embeddings once and slices
them per feature (``extract``), the controls battery (``controls`` + ``nulls``), the gated
cascade that assigns each feature its rung (``ladder`` + ``gates``), the entanglement geometry
(``entanglement`` + ``matching``), and the non-circular uncertainty geometry (``uncertainty``).
``run.run_probing`` is the stamped entry point.

The freeze boundary is structural: this package receives a frozen ``Encoder`` and asserts
``assert_frozen`` on entry — it imports ``models`` (for the untrained-encoder control) but
never ``objectives``. The five flagged statistical decisions are parameterised in
``config.ProbingConfig`` with placeholder defaults, pending the stats-grounding session.
"""

from __future__ import annotations

from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.ladder import LadderResult, RungVerdict, run_ladder
from galaxy_jepa.probing.run import ProbingReport, run_probing

__all__ = [
    "ProbingConfig",
    "ProbingReport",
    "run_probing",
    "run_ladder",
    "LadderResult",
    "RungVerdict",
]
