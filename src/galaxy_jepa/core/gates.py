"""Controls as gates — composable, with the verdict recorded separately from run status.

Implements ``docs/architecture.md`` "Controls as gates" and ``docs/spec/gates.md``.

A :class:`Gate` is a first-class, composable object. :class:`MetricGate` carries
**exactly one** comparator (enforced); :func:`all` / :func:`any` build composites that
**reject empty children** (a degenerate logic error, not a no-op). Evaluating a gate
produces a :class:`GateResult` tree.

The load-bearing distinction: **the verdict is separate from run status.** Evaluating
a gate *never raises* because the bar was not met — a feature failing its controls is a
*finding*, recorded as ``passed=False``, not an error that aborts the run. (Evaluation
*does* fail loudly on a structural mistake — a metric the run never produced — because
that is a bug, not a finding.)

The concrete control gates (``selectivity``, ``negative_control``,
``nuisance_clearance``) are compositions of these primitives, defined in
``docs/spec/gates.md`` and built in ``probing/`` with the probing code. Note
``bbox_degradation`` (β=0 ⇒ I-JEPA) is **not** a control gate — it is a deterministic
property test (``tests/``), not a metric threshold.
"""

from __future__ import annotations

import builtins
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping

_COMPARATORS: dict[str, str] = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


@dataclasses.dataclass(frozen=True)
class GateResult:
    """A node in the verdict tree. Leaves describe a metric check; composites hold children."""

    label: str
    passed: bool
    detail: str | None = None
    children: tuple[GateResult, ...] = ()

    def render(self, indent: int = 0) -> str:
        """Render the verdict tree as an indented, tickable string."""
        mark = "PASS" if self.passed else "FAIL"
        line = f"{'  ' * indent}[{mark}] {self.label}"
        if self.detail:
            line += f" ({self.detail})"
        lines = [line]
        for child in self.children:
            lines.append(child.render(indent + 1))
        return "\n".join(lines)


class Gate(ABC):
    """Something that evaluates a metrics mapping to a :class:`GateResult`."""

    @abstractmethod
    def evaluate(self, metrics: Mapping[str, float]) -> GateResult: ...


@dataclasses.dataclass(frozen=True)
class MetricGate(Gate):
    """A single metric compared against one threshold.

    Exactly one of ``gt``/``gte``/``lt``/``lte`` must be set; anything else is a
    degenerate gate and raises at construction.
    """

    metric: str
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    def __post_init__(self) -> None:
        set_comparators = [c for c in _COMPARATORS if getattr(self, c) is not None]
        if len(set_comparators) != 1:
            raise ValueError(
                f"MetricGate({self.metric!r}) must set exactly one comparator "
                f"({'/'.join(_COMPARATORS)}); got {set_comparators or 'none'}."
            )

    def evaluate(self, metrics: Mapping[str, float]) -> GateResult:
        if self.metric not in metrics:
            raise KeyError(
                f"Gate references metric {self.metric!r}, which the run did not produce. "
                f"Available metrics: {sorted(metrics)}."
            )
        value = metrics[self.metric]
        comparator = next(c for c in _COMPARATORS if getattr(self, c) is not None)
        threshold = getattr(self, comparator)
        passed = {
            "gt": value > threshold,
            "gte": value >= threshold,
            "lt": value < threshold,
            "lte": value <= threshold,
        }[comparator]
        detail = f"{self.metric}={value:g} {_COMPARATORS[comparator]} {threshold:g}"
        return GateResult(label=self.metric, passed=passed, detail=detail)


@dataclasses.dataclass(frozen=True)
class AllGate(Gate):
    """Passes iff every child passes. Rejects an empty child list."""

    children: tuple[Gate, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("all(...) requires at least one child gate (empty is a logic error).")

    def evaluate(self, metrics: Mapping[str, float]) -> GateResult:
        results = tuple(child.evaluate(metrics) for child in self.children)
        passed = builtins.all(r.passed for r in results)
        return GateResult(label="all", passed=passed, children=results)


@dataclasses.dataclass(frozen=True)
class AnyGate(Gate):
    """Passes iff at least one child passes. Rejects an empty child list."""

    children: tuple[Gate, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("any(...) requires at least one child gate (empty is a logic error).")

    def evaluate(self, metrics: Mapping[str, float]) -> GateResult:
        results = tuple(child.evaluate(metrics) for child in self.children)
        passed = builtins.any(r.passed for r in results)
        return GateResult(label="any", passed=passed, children=results)


def all(*gates: Gate) -> AllGate:
    """Compose gates that must *all* pass (e.g. ``gate.all(...)``)."""
    return AllGate(tuple(gates))


def any(*gates: Gate) -> AnyGate:
    """Compose gates of which *any* may pass."""
    return AnyGate(tuple(gates))
