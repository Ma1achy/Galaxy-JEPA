"""The effect floor as a write-once, hashed record — the pre-registration made mechanical.

The floor (D22) is the one number in the battery that is a call rather than a derivation, and an
inline YAML value can be moved by anyone at any time with nothing on disk saying it moved. This
module takes both freedoms away. :func:`freeze_effect_floor` computes the value from a *named
rule* over *named inputs* (never typed in) and writes it exactly once, by exclusive create: a
second write fails even with identical content, because "re-freezing to the same number" and
"re-freezing after looking" are indistinguishable from the file. :func:`load_effect_floor`
refuses a record whose hash, or whose value under its own rule, no longer matches.

Scoring reads the floor through :func:`resolve_effect_floor`, called by ``run_ladder`` and by
``run_probing`` (so the stamp carries the resolved record), and hands it on as an *object*:
``nulls.existence_verdicts`` takes the floor as a required :class:`EffectFloorRecord` — re-checked
intact on the way in — rather than as a float, so a direct caller cannot score past the gate by
passing a number.

**Only a smoke may bypass the gate.** A non-smoke run scores against an intact record or not at
all: no ``effect_floor_file``, a missing one, or an edited one refuses the run. A smoke without
an intact record scores at its inline ``effect_floor`` under a :class:`FloorBypass`, and
everything it writes carries :data:`FLOOR_BYPASSED` — the stamp's ledger, every existence entry
in ``ladder_summary.json``, and the summary's ``effect_floor`` block — while every path in the
package that reports a verdict (``ProbingReport.rung_table``, ``population_comparison``, the
figures) refuses it. ``effect_floor_open`` used to be a declarable hatch that let any run fall
back past a missing or edited file; it is now only the ledger name a bypass is stamped under, and
declaring it is refused at load (``docs/spec/escape-hatches.md``).

**Rounding is half-up, decided.** :func:`_widest_gap_midpoint` states the midpoint to four places
with ``ROUND_HALF_UP``. N2's widest gap is (0.6538, 0.7995], whose exact midpoint 0.72665 sits on
the tie: half-up gives **0.7267**, the value recorded by hand in D22 (``DECISIONS.md``) and
frozen in ``configs/probe.yaml``; half-even would give 0.7266. The rule has to reproduce the call
it mechanises, or it is a different rule — so half-up it is, and changing it would refuse every
record written under it.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import warnings
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, ClassVar

from galaxy_jepa.core.config import HashedFrozenChoice, code_sha
from galaxy_jepa.probing.config import EffectFloorFreeze, ProbingConfig

__all__ = [
    "FLOOR_BYPASSED",
    "RULES",
    "EffectFloorRecord",
    "FloorBypass",
    "assert_reportable",
    "effect_floor_gate",
    "freeze_effect_floor",
    "load_effect_floor",
    "resolve_effect_floor",
]

#: The ledger name a run scored without an intact floor record is stamped under. Stamped, never
#: declared: ``ProbingConfig`` refuses it in ``escape_hatches``, because only a smoke may bypass.
OPEN_HATCH = "effect_floor_open"

#: The marker every output of a bypassed run carries, and every verdict reader refuses on.
FLOOR_BYPASSED = "FLOOR BYPASSED"

#: Decimal places the floor is stated to — the precision the N2 AUCs were read and argued at.
_PLACES = Decimal("0.0001")


def _widest_gap_midpoint(inputs: Mapping[str, float]) -> float:
    """Midpoint of the widest gap between consecutive real AUCs (the N2 rule, Brief O0).

    Every value inside that gap gives the same partition of the features, so its centre is the
    point furthest from any feature's flip. Computed in decimal on the inputs as written and
    stated to four places, half-up, so the value is the one a reader recomputes by hand rather
    than whatever the binary midpoint happens to round to. A tie for the widest gap has no
    centre the rule can name, so it is refused rather than broken arbitrarily.
    """
    aucs = sorted(Decimal(repr(v)) for v in inputs.values())
    if len(aucs) < 2:
        raise ValueError(f"widest_gap_midpoint needs at least two AUCs, got {len(aucs)}")
    gaps = [(hi - lo, lo, hi) for lo, hi in zip(aucs, aucs[1:], strict=False)]
    widest = max(width for width, _, _ in gaps)
    at_widest = [(lo, hi) for width, lo, hi in gaps if width == widest]
    if len(at_widest) > 1:
        raise ValueError(
            f"widest_gap_midpoint is undefined here: {len(at_widest)} gaps tie at width "
            f"{widest} ({at_widest}). The rule names one gap; picking between them would be a "
            "second call made after seeing the numbers."
        )
    ((lo, hi),) = at_widest
    return float(((lo + hi) / 2).quantize(_PLACES, rounding=ROUND_HALF_UP))


#: Rule name -> the function that turns the inputs into the floor. A record names its rule and
#: :meth:`EffectFloorRecord.assert_intact` re-runs it, so a value can never outlive its reason.
#: A rule's code is part of what a record means: changing one refuses every record written by it.
RULES: dict[str, Callable[[Mapping[str, float]], float]] = {
    "widest_gap_midpoint": _widest_gap_midpoint,
}


def _apply_rule(rule: str, inputs: Mapping[str, float]) -> float:
    if rule not in RULES:
        raise ValueError(f"unknown effect-floor rule {rule!r}; registered: {sorted(RULES)}")
    bad = {k: v for k, v in inputs.items() if not (math.isfinite(v) and 0.0 <= v <= 1.0)}
    if bad:
        raise ValueError(f"effect-floor inputs must be AUCs in [0, 1]; got {bad}")
    return RULES[rule](inputs)


class EffectFloorRecord(HashedFrozenChoice, EffectFloorFreeze):
    """The effect-floor freeze with its derivation attached: rule, inputs, and a content hash.

    It *is* an :class:`EffectFloorFreeze`, so once resolved it sits in
    ``ProbingConfig.effect_floor_freeze`` and is hashed and stamped with everything else — the
    path it was read from is not identity (D15), but this record is. The hash covers every field
    bar itself, provenance included: the record is written once, so no field has an honest
    reason to change afterwards.
    """

    rule: str
    #: Feature -> real AUC, for the pre-declared spread features the rule is applied over.
    inputs: dict[str, float]
    code_sha: str
    code_dirty: bool

    RECORD_NAME: ClassVar[str] = "the effect-floor record"
    MADE: ClassVar[str] = "frozen"
    REMEDY: ClassVar[str] = (
        "A frozen floor is not re-frozen or corrected in place; a new call is a new record, and "
        "only a smoke may score without one"
    )

    def determining_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"content_hash"})

    def assert_intact(self) -> None:
        super().assert_intact()
        # The hash proves nobody edited the record; this proves the record's number is its
        # rule's number, so an input or a rule cannot be swapped under a value either.
        recomputed = _apply_rule(self.rule, self.inputs)
        if recomputed != self.value:
            raise ValueError(
                f"the effect-floor record states value={self.value} but its rule {self.rule!r} "
                f"gives {recomputed} on its own inputs. The value must be the rule's output; a "
                "record where they differ was not produced by freeze_effect_floor as it stands."
            )


def freeze_effect_floor(
    path: str | Path,
    *,
    rule: str,
    inputs: Mapping[str, float],
    frozen_by: str,
    derived_from: str,
    rationale: str,
    frozen_at: str | None = None,
) -> EffectFloorRecord:
    """Compute the floor by ``rule`` over ``inputs`` and write the record to ``path``, ONCE.

    There is no value argument: the number is the rule's, never typed in. The write is an
    exclusive create, so a record that already exists — identical or not — is refused rather than
    overwritten, and there is no recompute path to fall back on. Everything is validated before
    the file is created, so a refused rule leaves nothing on disk.
    """
    inputs = {str(k): float(v) for k, v in inputs.items()}
    sha, dirty = code_sha()
    draft = EffectFloorRecord(
        value=_apply_rule(rule, inputs),
        rule=rule,
        inputs=inputs,
        derived_from=derived_from,
        frozen_at=frozen_at or datetime.date.today().isoformat(),
        frozen_by=frozen_by,
        rationale=rationale,
        code_sha=sha,
        code_dirty=dirty,
        content_hash="",
    )
    record = draft.model_copy(update={"content_hash": draft.expected_hash()})
    record.assert_intact()
    text = json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as fh:
            fh.write(text)
    except FileExistsError:
        raise FileExistsError(
            f"{path} already holds an effect-floor record, and a floor is frozen once. Writing it "
            "again — even byte-identical — would make 'frozen before the headline' rest on nobody "
            "having re-run this after looking. Nothing here overwrites: a new call is a new record "
            "at a new path, and the run that uses it says so."
        ) from None
    # Read-only on disk too: not the guard (the hash is), but an edit should take a decision.
    os.chmod(path, 0o444)
    return record


def load_effect_floor(path: str | Path) -> EffectFloorRecord:
    """Read a floor record and refuse it unless its hash and its rule both still hold."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no effect-floor record at {path}. Freeze one with freeze_effect_floor before "
            "scoring against it; a missing floor is not a floor of zero."
        )
    record = EffectFloorRecord.model_validate(json.loads(path.read_text()))
    record.assert_intact()
    return record


@dataclasses.dataclass(frozen=True)
class FloorBypass:
    """A smoke scoring without an intact floor record — the floor it used, and that it bypassed.

    The only floor object other than :class:`EffectFloorRecord` that ``existence_verdicts``
    accepts, and it marks every verdict it touches :data:`FLOOR_BYPASSED`. Built by
    :func:`effect_floor_gate`, which refuses to build one for anything but a smoke.
    """

    value: float
    reason: str


def effect_floor_gate(config: ProbingConfig) -> EffectFloorRecord | FloorBypass:
    """The floor object scoring hands to ``existence_verdicts``, from a *resolved* config.

    The intact record when the config carries one; a :class:`FloorBypass` at the inline value
    when it is a smoke that does not; otherwise a refusal — reached only by a caller that skipped
    :func:`resolve_effect_floor`, since that already refuses a non-smoke without a record.
    """
    freeze = config.effect_floor_freeze
    if isinstance(freeze, EffectFloorRecord):
        return freeze
    if config.smoke:
        return FloorBypass(
            value=config.effect_floor,
            reason=(
                f"smoke without an intact effect-floor record "
                f"(effect_floor_file={config.effect_floor_file!r})"
            ),
        )
    raise ValueError(
        "refusing to score: this config carries no intact effect-floor record and is not a smoke. "
        "Resolve it with resolve_effect_floor (which loads effect_floor_file) before scoring."
    )


def assert_reportable(floor: EffectFloorRecord | FloorBypass | None, what: str) -> None:
    """Refuse to report ``what`` off verdicts scored without an intact floor record.

    ``None`` — a result that does not say which floor it was scored at — is refused too: a
    verdict reader cannot tell it from a bypass, so it does not get the benefit of the doubt.
    """
    if not isinstance(floor, EffectFloorRecord):
        reason = floor.reason if isinstance(floor, FloorBypass) else "no floor record attached"
        raise ValueError(
            f"{FLOOR_BYPASSED}: refusing to report {what}. These verdicts were scored without an "
            f"intact effect-floor record ({reason}), so their clean-vs-marginal line is not the "
            "pre-registered one. A smoke exercises the plumbing; it is never a result."
        )


def resolve_effect_floor(config: ProbingConfig) -> ProbingConfig:
    """The one place scoring learns its effect floor, and the gate: a record, or a smoke.

    With ``effect_floor_file`` set, the record is loaded and its value and the record itself
    replace the inline ones, so every reader below (``existence_verdicts``, ``build_gates``) and
    the stamp see the same floor. An inline ``effect_floor`` that was stated alongside must agree
    with the record, or the pair is refused. A non-smoke run with no file, a missing file or an
    edited one is refused; a smoke is handed back untouched to score at its inline floor, and
    :func:`effect_floor_gate` marks it :data:`FLOOR_BYPASSED`. Idempotent: resolving an
    already-resolved config re-checks the file and changes nothing.
    """
    if config.effect_floor_file is None:
        if config.smoke:
            return config
        raise ValueError(
            "refusing to score: no effect_floor_file. Only a smoke may score without a frozen "
            "effect-floor record; freeze one with freeze_effect_floor and point the config at it "
            "(configs/probe.yaml uses configs/effect_floor.json). An inline effect_floor_freeze "
            "is not the gate — it is a number a text editor can move."
        )
    try:
        record = load_effect_floor(config.effect_floor_file)
    except (OSError, ValueError) as exc:
        if not config.smoke:
            raise ValueError(
                f"refusing to score: effect_floor_file={config.effect_floor_file!r} did not load "
                f"as an intact record ({type(exc).__name__}: {exc}). Only a smoke may score "
                f"without it, and its outputs are marked {FLOOR_BYPASSED!r}."
            ) from exc
        warnings.warn(
            f"effect_floor_file={config.effect_floor_file!r} did not load ({exc}); this smoke "
            f"scores at the inline effect_floor={config.effect_floor} and its outputs are marked "
            f"{FLOOR_BYPASSED!r}.",
            stacklevel=2,
        )
        return config
    if "effect_floor" in config.model_fields_set and config.effect_floor != record.value:
        raise ValueError(
            f"the config states effect_floor={config.effect_floor} but its floor record "
            f"{config.effect_floor_file!r} says {record.value}. The record is the floor; drop the "
            "inline line or make it the record's value, rather than keeping two that disagree."
        )
    current = config.effect_floor_freeze
    if current is not None and not (
        isinstance(current, EffectFloorRecord) and current.content_hash == record.content_hash
    ):
        raise ValueError(
            "the config already carries a different effect-floor freeze than "
            f"{config.effect_floor_file!r}; a run has one floor record, not two."
        )
    return config.model_copy(update={"effect_floor": record.value, "effect_floor_freeze": record})
