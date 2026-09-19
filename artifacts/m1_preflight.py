"""Brief M1 — the pre-flight for the long λ=0 baseline run.

Same gate as J1, different answers. The checks themselves are J1's and are not restated here;
what M declares is the recipe they are asserted against, because D21 turned SIGReg off and M
moved the budget from 1.97 epochs to 10.

Three things differ from Brief J, and each is a fact the run would otherwise assume:

* **SIGReg is OFF.** J1 asserted `sigreg_lambda > 0`, because under D18 it had to be on. M
  asserts the opposite. The check is not "is it non-default" but "is it what THIS brief runs
  with" — either direction is an error, and a silent flip is exactly the class of defect the
  gate exists for.
* **The soft rank floor is ACTIVE.** Under D18 it was inert — effective rank was
  constraint-satisfied by the penalty, so the criterion could never fire and its silence had to
  be understood rather than assumed. At λ=0 it is a real halt path, and M is the first run in
  the project long enough to pass the grace period (0.10 × steps) at all. Brief L is the
  evidence 2.5 is not set too high: effective rank rose 8.1 → 18.6 across 10,500 steps, away
  from the floor, and it did not fire.
* **The hash chain gains a link.** `steps` and `checkpoint_every` are both determining, so
  moving the horizon moves the hash. Stripping the budget recovers D21's recipe chain, which
  still walks back to D17's `538bf997880a8767` — unchanged by any of this, because stripping the
  SIGReg keys removes `sigreg_lambda` whatever its value. Four strips, each naming exactly one
  thing that moved.

Raises on the first failure. Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/m1_preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j1_preflight import Expect, main as preflight  # noqa: E402

#: The stock budget M displaces. Strip it and D21's recipe chain is reachable again.
STOCK_BUDGET = {"steps": 50000, "checkpoint_every": 1500}

M = Expect(
    label="M1",
    shipped_hash="046659910b5fd543",  # 10 epochs at a quarter-epoch cadence, smoke on
    smoke_hash="7ecf5dce5a1f60ba",  # minus the budget: D21 as shipped
    recipe_hash="de87b8f9704b7e2d",  # minus smoke: D21's recipe
    sigreg_on=False,  # D21 — and asserting it, not assuming it
    steps_per_s=1.5150,  # Brief L's λ=0 arm, measured on THIS driver's path over 10,500 steps
    decisions="D17+D21",
    stock_budget=STOCK_BUDGET,
)


def main() -> None:
    preflight(M)


if __name__ == "__main__":
    main()
