"""Probing — the frozen-encoder nameability ladder, controls battery, and uncertainty geometry.

The encoder-agnostic measurement layer (``docs/probing-harness-design.md``): L2-logistic
concept directions (``logistic``), the cost spine that extracts embeddings once and slices
them per feature (``extract``), the controls battery (``controls`` + ``nulls``), the gated
cascade that assigns each feature its rung (``ladder`` + ``gates``), the entanglement geometry
(``entanglement`` + ``matching``), the non-circular uncertainty geometry (``uncertainty``), and
the two-scheme feature experiment with its conditional populations (``schemes``, D14).
``run.run_probing`` is the stamped entry point; ``harness.probe_frozen_checkpoint`` is what
hands it a real frozen checkpoint.

The freeze boundary is structural: this package receives a frozen ``Encoder`` and asserts
``assert_frozen`` on entry — it imports ``models`` (for the untrained-encoder control) but
never ``objectives``. The five statistical decisions are **grounded** and carried in
``config.ProbingConfig`` (see its docstring for what remains open).
"""

from __future__ import annotations

from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.ladder import LadderResult, RungVerdict, run_ladder
from galaxy_jepa.probing.run import ProbingReport, run_probing
from galaxy_jepa.probing.schemes import FeatureScheme, FeatureSpec, get_scheme, scheme_names

__all__ = [
    "ProbingConfig",
    "ProbingReport",
    "run_probing",
    "run_ladder",
    "LadderResult",
    "RungVerdict",
    "FeatureScheme",
    "FeatureSpec",
    "get_scheme",
    "scheme_names",
]
