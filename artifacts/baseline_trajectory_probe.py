"""D12's baselines — the trajectory probe at 0.5 / 1 / 2 / 4 epochs, on E2's protocol (baselines.md §1.1).

The baselines are matched to A1 on pretrain_v2 (user, 2026-10-13), so m2's k2 probe cannot serve:
it is a v1 like-for-like guard (J4(A)'s split counts) and refuses probe_v2. The decision is E2's
protocol instead, read unchanged from `aligned_c2` as ladder v2's E2 reads it: the full-split bank
(`embed`, uncapped probe-train), `fit_scores` (`_fit`, C from probe.yaml), `auc`, and `boot_aucs`
under `weights` (2,000 Poisson galaxy draws, seed fixed there) — for all 37 Scheme-1 answers, with no
powered filter, no multiplicity correction and no aggregation over answers (in particular none over
an M-derived list such as `powered`).

Trajectories only (spec 1C): reported, never used to choose a checkpoint. The record carries no
headline field and the driver prints no AUC; baseline_run.py logs only how many answers were scored.

Called by baseline_run.py as a subprocess, one checkpoint per call, under its environment:
`import baseline_run` comes first, so GJ_CONFIG / F0_CACHE_BASE / F0_NORM_PREFIX are held to A1's
before F0 reads them. The split is ladder v2's identical set (`j4_spread_controls._identical_split`,
decision 1: v1's stride and test split, minus the nine galaxies probe_v2 lacks), held here through
`LADDER_SPLIT=identical`, so a baseline's trajectory is scored on exactly the galaxies E2 scored A1–A3
on. tests/test_trajectory_split.py checks the two splits byte for byte.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/baseline_trajectory_probe.py --run runs/moco_s0 --steps 12663 --tag moco_s01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_run  # noqa: E402,F401  — the v2 environment guard, before F0 reads it

# the ladder's split, not the config's own: prepare() reads this at call time (ladder_env)
if os.environ.setdefault("LADDER_SPLIT", "identical") != "identical":
    raise SystemExit(f"trajectory: LADDER_SPLIT={os.environ['LADDER_SPLIT']!r}; the baselines are "
                     "scored on ladder v2's identical split")
from aligned_c2 import N_BOOT, SEED, _bank, auc, boot_aucs, embed, fit_scores, weights  # noqa: E402
from i3_loss_usability import frozen_from  # noqa: E402
from j4_spread_controls import OUT  # noqa: E402


def score(run: Path, step: int, tag: str) -> dict:
    """One checkpoint through E1 (the bank) and E2 (per-answer AUC with a galaxy-bootstrap CI)."""
    ckpt = run / "checkpoints" / f"ckpt_{step:09d}.pt"
    if not ckpt.exists():
        raise SystemExit(f"trajectory: no checkpoint at step {step} ({ckpt})")
    t0 = time.perf_counter()
    # re-exported so load_frozen_encoder stays the only way in; kept, as the bank hashes it
    enc = OUT / f"{tag}_scratch" / f"enc_{step:09d}.pt"
    _, got = frozen_from(ckpt, enc)
    if got != step:
        raise SystemExit(f"trajectory: {ckpt.name} says step {got}")
    bank_tag = f"{tag}_{step:09d}"
    if _bank(bank_tag).exists():
        raise SystemExit(f"trajectory: {_bank(bank_tag)} exists; remove it to re-embed this checkpoint")
    embed(bank_tag, str(enc))
    sc, meta, test_ids = fit_scores([bank_tag])
    w, pos = weights(test_ids)
    answers = {}
    for f, y in sorted(meta["y"].items()):
        s = sc[bank_tag][f]
        b = boot_aucs(s, y, w[:, [pos[int(o)] for o in meta["rows"][f]]])
        lo, hi = np.percentile(b, [2.5, 97.5])
        answers[f] = {"auc": auc(s, y), "ci95": [float(lo), float(hi)], "n_test": int(len(y)),
                      "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    traces = json.loads((run / "traces.json").read_text())
    window = traces["losses"][max(0, step - 100):step]  # a trailing 100-step mean: batch 32 is noisy
    return {"step": step, "checkpoint": str(ckpt), "smoke": True, "protocol": "E2",
            "n_test_split": len(test_ids), "n_boot": N_BOOT, "boot_seed": SEED,
            "n_answers": len(answers), "answers": answers,
            "pretext_loss": float(np.mean(window)) if window else None,
            "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    records = []
    for step in args.steps:
        records.append(score(args.run, step, args.tag))
        print(f"trajectory {args.tag}: step {step:,} — {records[-1]['n_answers']} answers scored",
              file=sys.stderr, flush=True)
    path = OUT / f"{args.tag}_trajectory.json"
    path.write_text(json.dumps({"run": str(args.run), "smoke": True, "protocol": "E2",
                                "checkpoints": records}, indent=1))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
