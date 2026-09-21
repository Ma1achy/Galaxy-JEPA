"""The early-stopping rule for a probe-point trajectory, as three named outcomes.

Brief M briefed this as "stop when consensus AUC improves by less than ~0.002 across two
consecutive probe intervals". A **decline** satisfies that wording — it improves by less than
0.002 — and the first implementation encoded it as a signed comparison, so Brief O2's curve
(0.9678 at 2 epochs, 0.9609 at 4) stopped with the label ``FLAT``. Stopping was correct; the label
asserted the opposite of what happened.

M's own outcome list separated "turns over" from "flattens". The rule text did not, so this module
does: **FLAT** and **DECLINING** are both stop conditions, and they are different findings that
argue different compute cases. A run that has stopped improving might be worth more compute spent
elsewhere; a run that is getting worse says something about the schedule.

Pure: takes probe points, returns a verdict. No torch, no I/O — it decides when to end a 46-hour
job and names the finding, which is not something to leave untestable in a driver script.
"""

from __future__ import annotations

import dataclasses

#: M4, stated before the numbers: the ΔAUC below which an interval counts as no longer moving.
FLAT_DELTA = 0.002

RISING = "RISING"
FLAT = "FLAT"
DECLINING = "DECLINING"
UNSETTLED = "UNSETTLED"
INSUFFICIENT = "INSUFFICIENT"


@dataclasses.dataclass(frozen=True)
class StoppingVerdict:
    """What the rule decided, with the label kept distinct from the decision.

    ``stop`` is the action; ``label`` is the finding. They are recorded separately because
    ``FLAT`` and ``DECLINING`` both stop a run and mean different things about it.
    """

    stop: bool
    label: str
    detail: str

    @property
    def reason(self) -> str:
        return f"{self.label} — {self.detail}" if self.detail else self.label


def read_curve(curve: list[dict], *, flat_delta: float = FLAT_DELTA) -> StoppingVerdict:
    """Apply the rule to the last three probe points of ``curve``.

    The intervals are UNEQUAL by design (0.5, 1, 2, 4, 6, 8, 10 epochs), so a raw delta favours
    stopping late, when intervals are widest — a 2-epoch gap has four times the room to improve
    that a 0.5-epoch gap has. The slope per epoch is therefore reported alongside, and if the two
    disagree the run CONTINUES: the rule may stop a run early, it may never extend it.
    """
    if len(curve) < 3:
        return StoppingVerdict(False, INSUFFICIENT, f"{len(curve)} probe point(s) — needs 3")

    a, b, c = curve[-3:]
    d1, d2 = b["auc"] - a["auc"], c["auc"] - b["auc"]
    s1 = d1 / (b["epoch"] - a["epoch"])
    s2 = d2 / (c["epoch"] - b["epoch"])
    detail = (
        f"ΔAUC {d1:+.4f} then {d2:+.4f} (band ±{flat_delta}); slope/epoch {s1:+.4f} then {s2:+.4f}"
    )

    # Order matters: a decline is a decline whatever the earlier interval did.
    if d2 < -flat_delta:
        return StoppingVerdict(True, DECLINING, detail)

    if abs(d1) < flat_delta and abs(d2) < flat_delta:
        if d2 > d1:  # still accelerating within the band — not yet a plateau
            return StoppingVerdict(
                False, UNSETTLED, f"within the band but still rising within intervals — {detail}"
            )
        return StoppingVerdict(True, FLAT, detail)

    if d2 > flat_delta:
        return StoppingVerdict(False, RISING, detail)

    return StoppingVerdict(False, UNSETTLED, f"still improving — {detail}")
