"""Ladder v2's encoder/output contract (docs/ladder_v2/plan.md, Task 1b.1).

The ladder drivers were written for M: a literal `runs/m/encoder.pt`, untagged records in
`artifacts/out`, the v1 untrained bank. Ladder v2 runs the same drivers on A1, A2 and A3. Rather than
edit every call site, the shared sites read this contract — `j4_spread_controls.prepare` and `OUT`,
through which `r_nonlinear` and every Brief R–AA driver pass:

    LADDER_ENCODER         the checkpoint every prepare() reads, whatever literal the driver passes
    LADDER_OUT             the output directory (per encoder; drivers read each other's records by
                           name, so one encoder's records share one directory)
    LADDER_UNTRAINED_BANK  the untrained bank (D23's bar) the ladder loads, in place of probe.yaml's
    LADDER_SPLIT=identical the identical set: v1's stride and test split, minus the galaxies the
                           active probe corpus lacks (decision 1)

**Unset, nothing changes**: every accessor returns the value the driver had before, so M's runs stay
reproducible (the E10 inertness test). Set, `LADDER_ENCODER` and `LADDER_OUT` must come together —
a ladder entry that wrote into `artifacts/out` would overwrite M's records.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The probe galaxies probe_v2 lacks (`aligned_score.NINE`, declared 2026-10-06). The identical set
#: may drop these and nothing else.
NINE = frozenset({1237651273510813752, 1237655472895099027, 1237658423543595058, 1237658802034900998,
                  1237661976553652249, 1237662263251042419, 1237662264854904841, 1237662528990674970,
                  1237665329849958421})


def active() -> bool:
    on = [k for k in ("LADDER_ENCODER", "LADDER_OUT") if os.environ.get(k)]
    if len(on) == 1:
        raise SystemExit(f"ladder_env: {on[0]} is set without its pair; set LADDER_ENCODER and "
                         "LADDER_OUT together, or neither")
    return len(on) == 2


def encoder(default: str | Path | None) -> Path | None:
    if active():  # checks the pair first: a lone variable refuses here, whichever is read first
        return Path(os.environ["LADDER_ENCODER"])
    return Path(default) if default else None


def out(default: Path) -> Path:
    return Path(os.environ["LADDER_OUT"]) if active() else default


def untrained_bank(default: str) -> str:
    v = os.environ.get("LADDER_UNTRAINED_BANK")
    return v if v else default


def identical_split() -> bool:
    v = os.environ.get("LADDER_SPLIT", "")
    if v not in ("", "identical"):
        raise SystemExit(f"ladder_env: LADDER_SPLIT={v!r}; the only split is 'identical'")
    return v == "identical"


def tag(default: str) -> str:
    return os.environ.get("LADDER_TAG") or default
