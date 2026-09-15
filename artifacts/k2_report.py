"""Render K2's trajectory as the two tables the findings doc carries. Read-only."""
from __future__ import annotations
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    path = OUT / "k2_trajectory.json"
    if not path.exists():
        path = OUT / "k2_trajectory.partial.json"
    blob = json.loads(path.read_text())
    recs = blob["checkpoints"] if isinstance(blob, dict) else blob
    print(f"source {path.name}  n={len(recs)}")
    if isinstance(blob, dict):
        print(f"  n_train {blob['n_train']:,}  n_test {blob['n_test']:,}  "
              f"label {blob['label_col']}  C={blob['c']}  "
              f"nuisance n_train {blob['nuisance_n_train']}")
    print()
    print("| step | consensus AUC [95% CI] | all held-out | ambiguous | pred loss | SIGReg | erank | std | cos |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        star = "*" if r["collapse_step"] != r["step"] else ""
        print(f"| {r['step']:,} | {r['auc']:.4f} [{r['auc_lo']:.4f}, {r['auc_hi']:.4f}] | "
              f"{r['auc_all']:.4f} | {r['auc_ambiguous']:.4f} | {r['prediction_loss']:.4f} | "
              f"{r['sigreg_loss']:.4f} | {r['effective_rank']:.2f}{star} | {r['std']:.4f}{star} | "
              f"{r['mean_cosine']:+.4f}{star} |")
    names = list(recs[0]["nuisance_aucs"])
    print()
    print("| step | morphology | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for r in recs:
        print(f"| {r['step']:,} | {r['auc']:.4f} | "
              + " | ".join(f"{r['nuisance_aucs'][n]:.4f}" for n in names) + " |")
    print()
    a = [r["auc"] for r in recs]
    print(f"morphology  peak {max(a):.4f} at step {recs[a.index(max(a))]['step']:,}  "
          f"final {a[-1]:.4f}  drop {a[-1] - max(a):+.4f}")
    for n in names:
        v = [r["nuisance_aucs"][n] for r in recs]
        print(f"  {n:10s} {v[0]:.4f} -> {v[-1]:.4f}  ({v[-1] - v[0]:+.4f})")


if __name__ == "__main__":
    sys.exit(main())
