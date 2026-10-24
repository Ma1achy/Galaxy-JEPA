"""The baselines comparison's answer set, from label counts only (baselines_comparison.md §A).

Reads the probe corpus's GZ2 labels on ladder v2's identical split, exactly as the trajectory probe
selects them (`prepare(None, 0)` under LADDER_SPLIT=identical, then `labels.eligible` and
`binary_label` per answer), and counts each class in probe-train and probe-test. **No encoder is
loaded, no bank or probe output is opened**: the rule cannot see any representation.

Investigation code: terse, excluded from lint/CI.

    .venv/bin/python artifacts/baselines_answer_set.py   # -> artifacts/out/baselines_answer_set.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_run  # noqa: E402,F401  — A1's v2 environment, before F0 reads it

os.environ.setdefault("LADDER_SPLIT", "identical")
from j4_spread_controls import OUT, prepare  # noqa: E402

MIN_TEST_MINORITY = 200  # §A: the rule, fixed before the counts were printed
MAX_SE = 0.02            # Hanley–McNeil SE at AUC_REF must not exceed this
AUC_REF = 0.75


def hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(v)


def main() -> None:
    s = prepare(None, 0, label="baselines-answer-set", sources=1)
    out = {}
    for f in s.labels.features:
        row = {}
        for part, ids in (("train", s.train_ids), ("test", s.test_ids)):
            el = s.labels.eligible(f, [int(o) for o in ids])
            y = s.labels.binary_label(f, el)
            row[f"{part}_pos"], row[f"{part}_neg"] = int(y.sum()), int(len(y) - y.sum())
        mino = min(row["test_pos"], row["test_neg"])
        se = hanley_mcneil_se(AUC_REF, row["test_pos"], row["test_neg"]) if mino else math.inf
        row["test_minority"], row["se_at_ref"] = mino, se
        row["in_set"] = bool(mino >= MIN_TEST_MINORITY and se <= MAX_SE
                             and min(row["train_pos"], row["train_neg"]) > 0)
        out[f] = row
    ids_sha = hashlib.sha1(json.dumps([sorted(map(int, s.train_ids)), sorted(map(int, s.test_ids))])
                           .encode()).hexdigest()
    rec = {"rule": {"min_test_minority": MIN_TEST_MINORITY, "max_se": MAX_SE, "auc_ref": AUC_REF},
           "split": "identical", "n_train": len(s.train_ids), "n_test": len(s.test_ids),
           "split_ids_sha1": ids_sha, "answers": out,
           "answer_set": sorted(f for f, r in out.items() if r["in_set"])}
    path = OUT / "baselines_answer_set.json"
    path.write_text(json.dumps(rec, indent=1))
    for f, r in sorted(out.items(), key=lambda kv: kv[1]["test_minority"]):
        print(f"{f:48s} train {r['train_pos']:>7,}/{r['train_neg']:>7,}  test {r['test_pos']:>6,}/"
              f"{r['test_neg']:>6,}  minority {r['test_minority']:>6,}  SE {r['se_at_ref']:.4f}  "
              f"{'IN' if r['in_set'] else 'out'}")
    print(f"{len(rec['answer_set'])} of {len(out)} in the set; train {rec['n_train']:,} test "
          f"{rec['n_test']:,}; wrote {path}")


if __name__ == "__main__":
    main()
