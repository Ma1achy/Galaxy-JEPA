"""A3 report (aligned_comparison.md, Declared A3, items 2-3). Criterion 2's fit unchanged; no rescoring.

Run from the A3 worktree (a36994f) under M's default config, as the c2 step ran; the A3 bank was
embedded by aligned_c2.embed under the v2 env, as A1's and A2's were (runs/aligned_score/embed.sh).

    uv run python artifacts/aligned_a3_report.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "artifacts")
import aligned_score as S  # patches C2.prepare to the identical set, as the scoring run
C2 = S.C2
ENC = ("a1", "a2", "a3", "m1", "m2")
sc, meta, test_ids = C2.fit_scores(list(ENC))
pt = {e: {f: C2.auc(sc[e][f], meta["y"][f]) for f in meta["y"] if f in sc[e]} for e in ENC}
allf = sorted(set.intersection(*(set(pt[e]) for e in ENC)))
ans = S._powered()
stored = json.loads((S.RES / "c2.json").read_text())["per_answer_all (D_j, exploratory beyond powered)"]
repro = max(abs(pt[e][f] - stored[f][e]) for f in allf for e in ("a1", "a2", "m1", "m2"))
rows = {}
for f in allf:
    a = [pt[e][f] for e in ("a1", "a2", "a3")]
    rows[f] = {**{e: pt[e][f] for e in ENC},
               "three_way": (abs(a[0]-a[1]) + abs(a[0]-a[2]) + abs(a[1]-a[2])) / 3,
               "a1a2": abs(a[0]-a[1]), "m": abs(pt["m1"][f]-pt["m2"][f]),
               "hi": ("a1","a2","a3")[int(np.argmax(a))], "lo": ("a1","a2","a3")[int(np.argmin(a))]}
def mean(k, fs): return float(np.mean([rows[f][k] for f in fs]))
out = {"n_test": len(test_ids), "n_answers": len(allf), "n_powered": len(ans),
       "max_abs_diff_vs_stored_c2 (a1,a2,m1,m2)": repro, "per_answer": rows,
       "summary": {sub: {"three_way": mean("three_way", fs), "a1a2": mean("a1a2", fs), "m1m2": mean("m", fs),
                         **{f"mean_{e}": mean(e, fs) for e in ENC},
                         "highest": {e: sum(rows[f]["hi"] == e for f in fs) for e in ("a1","a2","a3")},
                         "lowest": {e: sum(rows[f]["lo"] == e for f in fs) for e in ("a1","a2","a3")}}
                   for sub, fs in (("all", allf), ("powered", ans))}}
(S.RES / "a3_report.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["summary"], indent=1)); print("n", out["n_answers"], out["n_powered"], "repro", repro)
for f in allf:
    r = rows[f]; print(f"{f:42s} {'*' if f in ans else ' '} " + " ".join(f"{r[e]:.4f}" for e in ENC) + f"  3w {r['three_way']:.4f}  m {r['m']:.4f}")
