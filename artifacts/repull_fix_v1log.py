"""Correct the v1 geometry in a cut_log written by cutter 327c4213 (repull_findings.md §Pilot result).

That cutter logged v1's origin as ``Cutout2D.origin_original``, which clamps to 0 where the v1 stamp
overhangs the frame. The frame position v1x = logged origin + logged rel survives either way, and the
stamp's virtual origin is ceil(v1x − 128) — exact against astropy's ``input_position_cutout`` on
20,000 random positions, clipped ones included. Idempotent: an already-correct row maps to itself.

  uv run python artifacts/repull_fix_v1log.py <corpus dir> [...]

Keeps the original as ``cut_log.uncorrected.csv`` and notes the correction in ``manifest.json``.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

FORMULA = "v1_o = ceil(v1_o_logged + v1_rel_logged - 128); v1_rel = v1_o_logged + v1_rel_logged - v1_o"


def fix(d: Path) -> int:
    log, orig = d / "cut_log.csv", d / "cut_log.uncorrected.csv"
    if not orig.exists():
        shutil.copy2(log, orig)
    with orig.open(newline="") as fh:
        rd = csv.DictReader(fh)
        cols, rows = rd.fieldnames, list(rd)
    n = 0
    for r in rows:
        for b in "gri":
            for a in "xy":
                v1 = int(r[f"{b}_v1_o{a}"]) + float(r[f"{b}_v1_rel{a}"])
                o = math.ceil(v1 - 128)
                n += o != int(r[f"{b}_v1_o{a}"])
                r[f"{b}_v1_o{a}"], r[f"{b}_v1_rel{a}"] = o, repr(v1 - o)
                assert 127 < v1 - o <= 128
    tmp = log.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(log)
    mf = d / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        m["v1_log_corrected"] = {"formula": FORMULA, "origins_changed": n, "original": orig.name}
        mf.write_text(json.dumps(m, indent=1))
    return n


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "band-axis origins corrected:", fix(Path(p)))
