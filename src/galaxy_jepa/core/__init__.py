"""The small framework — the contracts the rest of Galaxy-JEPA is built against.

Implements ``docs/architecture.md`` "Sequencing — start minimal, grow on demand":
``core/`` is deliberately tiny on day one — the :class:`~galaxy_jepa.core.encoder.Encoder`
Protocol, the auto-capture config base + provenance stamping, and the ``Gate`` types.
Everything else is a plain module. The registry and the full Tier-1 validator suite
arrive only when a second consumer makes them pay (the "second consumer" test).

See ``docs/spec/`` for the component contracts.
"""

from galaxy_jepa.core import gates
from galaxy_jepa.core.config import (
    Configurable,
    RunConfig,
    RunStamp,
    class_ref,
    code_sha,
    config_hash,
    resolve_ref,
    write_stamp,
)
from galaxy_jepa.core.encoder import DEFAULT_LAYER, Encoder, assert_frozen, is_frozen
from galaxy_jepa.core.gates import AllGate, AnyGate, Gate, GateResult, MetricGate

__all__ = [
    "DEFAULT_LAYER",
    "AllGate",
    "AnyGate",
    "Configurable",
    "Encoder",
    "Gate",
    "GateResult",
    "MetricGate",
    "RunConfig",
    "RunStamp",
    "assert_frozen",
    "class_ref",
    "code_sha",
    "config_hash",
    "gates",
    "is_frozen",
    "resolve_ref",
    "write_stamp",
]
