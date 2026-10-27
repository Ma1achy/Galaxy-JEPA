"""SAME under a real effect (baselines_comparison.md §F, 25 Oct): the calibration measured false
BETTER/WORSE calls and power, not how often SAME is called when an effect of SESOI or more exists.
Decision layer, three pairs, the §C constants; the CI is approximated by its point, as in `calibrate`.
Two SAME rules: the current one (|D̄| ≤ b and b ≤ SESOI) and the equivalence form (|D̄| + b ≤ SESOI).
No real result is read.   .venv/bin/python artifacts/baselines_false_same.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import baselines_comparison as B  # noqa: E402

w = B._w()
rng = np.random.default_rng(20261025)
R = 60_000
out = []
for r in B.REGIMES + [(0.010, 0.006, 1, "se"), (0.015, 0.006, 1, "se")]:
    row = {"tau": r[0], "sigma": r[1], "ratio": r[2], "het": r[3]}
    for d in (0.0, 0.02, 0.03):
        se_fam, sj, D, Dj, split_f, _ = B._stats(*B._draw(rng, R, 3, w, *r, d), w)
        b = B.K_FAM * se_fam
        row[f"same_current_{d}"] = float(((~split_f) & (np.abs(D) <= b) & (b <= B.SESOI)).mean())
        row[f"same_equivalence_{d}"] = float(((~split_f) & (np.abs(D) + b <= B.SESOI)).mean())
    out.append(row)
    print(json.dumps(row), flush=True)
(B.HERE / "out" / "baselines_false_same_3seed.json").write_text(json.dumps(out, indent=1))
