"""The concrete control gates (design 3A / docs/spec/gates.md §2), built from the primitives.

The keystone: a rung verdict is **not** a human reading numbers — it is a pre-registered,
deterministic function of the controls, emitted in code, applied identically to every feature
(``docs/spec/gates.md`` §2 puts these concrete gates in ``probing/``). The primitives live in
``core/gates.py`` (``MetricGate`` / ``gate.all`` / ``GateResult``, verdict-separate-from-run-
status); this module composes them into the named control gates and wires the **thresholds
from ``ProbingConfig``** — never hard-coded.

Every gate consumes a per-feature ``metrics: Mapping[str, float]`` the ladder assembles. The
contract (the metric names the run must produce):

* ``auc`` — the canonical linear-probe ROC-AUC.
* ``exceeds_null`` — 1.0 iff the family-corrected existence test passed (``nulls.py``), else 0.
* ``selectivity`` — real-label AUC − shuffled-label control AUC (Hewitt–Liang).
* ``nuisance_cleared`` — 1.0 iff no nuisance is competitive, or a competitive one survived
  matching (``matching.py``); else 0.
* ``entangled`` — 1.0 iff the eigen-triangulation marks the feature entangled (R2), else 0.

A failing gate is a ``GateResult(passed=False)`` *finding*, recorded with the verdict — not an
error. Evaluation raises only if the run never produced a referenced metric (a structural bug).
"""

from __future__ import annotations

import dataclasses

from galaxy_jepa.core.gates import Gate, MetricGate
from galaxy_jepa.core.gates import all as gate_all
from galaxy_jepa.probing.config import ProbingConfig

__all__ = ["ProbingGates", "build_gates", "EXISTENCE_METRIC_FLOOR"]

# A 0/1 indicator metric passes at ≥ 0.5 — clean for an exact-boolean gate without a float
# ``eq`` (which ``MetricGate`` forbids as a latent bug).
EXISTENCE_METRIC_FLOOR = 0.5


@dataclasses.dataclass(frozen=True)
class ProbingGates:
    """The named control gates for one run, with thresholds bound from ``ProbingConfig``."""

    existence: Gate  # exceeds the 5-null max (corrected) AND clears the effect floor (3B)
    selectivity: Gate  # beats the shuffled-label control by the floor (Hewitt–Liang)
    nuisance_clearance: Gate  # no nuisance competitive, or matched-survived (3D)
    clean: Gate  # not entangled → R1 (else R2) (2A)

    def rung_inputs(self) -> tuple[Gate, ...]:
        """The gates whose conjunction a clean linear rung (R1) requires."""
        return (self.existence, self.selectivity, self.nuisance_clearance, self.clean)


def build_gates(config: ProbingConfig) -> ProbingGates:
    """Construct the control gates with the (flagged) thresholds from ``config``.

    Finalising a flagged threshold in the stats grounding is a config edit here — the gate
    structure does not change.
    """
    return ProbingGates(
        existence=gate_all(
            MetricGate("exceeds_null", gte=EXISTENCE_METRIC_FLOOR),
            MetricGate("auc", gte=config.effect_floor),  # FLAGGED value (effect floor, 3B)
        ),
        selectivity=MetricGate("selectivity", gte=config.selectivity_floor),
        nuisance_clearance=MetricGate("nuisance_cleared", gte=EXISTENCE_METRIC_FLOOR),
        clean=MetricGate("entangled", lt=EXISTENCE_METRIC_FLOOR),
    )
