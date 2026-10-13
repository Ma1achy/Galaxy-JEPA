"""Ladder v2, E2: every Scheme-1 answer's held-out AUC on the full identical split — scoring only.

    LADDER_* env set by the runner; .venv/bin/python artifacts/ladder_v2_e2.py <tag>

E2's only implementation (`aligned_score.py c2`) scores the family in `powered.json`, fixed from M1/M2,
so under decision 4 it cannot run (second approval, 13 October, item 1). This reads E2's protocol
unchanged from `aligned_c2` — `fit_scores` (the E1 bank, `_fit` C from probe.yaml, the uncapped split),
`auc`, and `boot_aucs` under `weights` (2,000 Poisson galaxy draws, seed fixed there) — for all 37
answers, with no powered filter, no multiplicity correction and no aggregation across answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_env as L  # noqa: E402
from aligned_c2 import N_BOOT, SEED, auc, boot_aucs, fit_scores, weights  # noqa: E402
from j4_spread_controls import OUT  # noqa: E402


def main() -> None:
    tag = sys.argv[1]
    if not (L.active() and L.identical_split()):
        raise SystemExit("ladder_v2_e2: runs only under ladder v2 (LADDER_* and LADDER_SPLIT=identical)")
    sc, meta, test_ids = fit_scores([tag])
    w, pos = weights(test_ids)
    answers = {}
    for f, y in sorted(meta["y"].items()):
        s = sc[tag][f]
        b = boot_aucs(s, y, w[:, [pos[int(o)] for o in meta["rows"][f]]])
        lo, hi = np.percentile(b, [2.5, 97.5])
        answers[f] = {"auc": auc(s, y), "ci95": [float(lo), float(hi)], "n_test": int(len(y)),
                      "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    rec = {"encoder": tag, "n_test_split": len(test_ids), "n_boot": N_BOOT, "boot_seed": SEED,
           "n_answers": len(answers), "answers": answers}
    (OUT / "e2_answers.json").write_text(json.dumps(rec, indent=1))
    print(f"E2 {tag}: {len(answers)} answers scored", file=sys.stderr)


if __name__ == "__main__":
    main()
