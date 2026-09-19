"""Brief L1a — extract each checkpoint's frozen embeddings ONCE and bank them.

K2 embedded 74,829 stamps per checkpoint at ~620 s a time, read three AUCs off each matrix, and
threw the matrix away. L1 needs the same matrices for the MLP capacity sweep, so this brief pays
83 minutes to re-make what it already had. **That is the lesson this driver exists to stop
repeating**: an extraction pass costing ten minutes per checkpoint must bank its output, for the
same reason `keep=3` had to go and for the same reason J4 writes after every feature.

Banked, the downstream work becomes free and re-runnable: the MLP sweep is 45 s per checkpoint on
CPU (measured), needs no encoder and no MPS, and can therefore run while a training arm has the
GPU. That is what makes L1b and L2 overlap at all.

The split is K2's, imported rather than re-derived — same `_probe_rows` sidecar, same
`assign_three_way`, same H5 stride, and the same two assertions against J4(A) (consensus train
25,305, held-out 34,829) fired before the first encode. The checkpoint -> frozen-encoder
conversion is `i3_loss_usability.frozen_from`, so `load_frozen_encoder` stays the only route to a
probe and the freeze boundary still runs through disk.

Idempotent: a checkpoint whose `.npz` is already present and well-formed is skipped, so an
interrupted pass resumes instead of restarting.

    uv run python artifacts/l1a_cache_embeddings.py                       # the K2 trajectory
    uv run python artifacts/l1a_cache_embeddings.py --run runs/l2 \
        --steps 1500 3000 6000 10500 --tag l2                             # the lambda=0 arm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402
from i3_loss_usability import frozen_from  # noqa: E402
from k2_trajectory_probe import (  # noqa: E402
    STEPS,
    J4_N_TEST_ALL,
    J4_N_TRAIN_EXTREMES,
    _capped_train,
    _featured,
    _release,
)

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.harness import _probe_rows  # noqa: E402
from galaxy_jepa.probing.extract import EmbeddingMatrix, extract_matrix  # noqa: E402
from galaxy_jepa.probing.logistic import _extremes, probe_auc_ci  # noqa: E402

#: The SSD, not `artifacts/out` — 920 MB of float32 belongs beside the 416 GB cache, not on the
#: internal container that holds 34 GB of headroom and the macOS swap file (Brief J0).
BANK = REPO / "runs" / "l1_embeddings"

#: K2's own consensus AUC per step, so a bad cache is caught at write time on the first
#: checkpoint rather than an hour later. L1b re-checks all of them.
#:
#: **Only ever applied to `--tag full`** — the lambda=0.05 trajectory K2 measured. A different arm
#: is a different encoder and MUST NOT reproduce these; checking it against them would abort a
#: perfectly good pass, and passing would mean something had gone badly wrong.
K2_TAG = "full"
K2_CONSENSUS = {
    1500: 0.9477, 3000: 0.9470, 6000: 0.9448, 10500: 0.9412,
    18000: 0.9376, 27000: 0.9295, 37500: 0.9284, 50000: 0.9278,
}


def _bank_path(tag: str, step: int) -> Path:
    return BANK / tag / f"emb_{step:09d}.npz"


def _load(path: Path, n: int) -> EmbeddingMatrix | None:
    """A banked matrix, or None if it is absent or does not describe this split."""
    if not path.exists():
        return None
    with np.load(path) as z:
        oids, x = z["object_ids"], z["x"]
    if oids.shape[0] != n or x.shape[0] != n:
        print(f"  {path.name}: banked {oids.shape[0]:,} rows, split wants {n:,} — re-extracting",
              file=sys.stderr)
        return None
    return EmbeddingMatrix(object_ids=oids, x=x, encoder_name=f"banked:{path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=REPO / "runs" / "full")
    ap.add_argument("--steps", type=int, nargs="*", default=list(STEPS))
    ap.add_argument("--max-train", type=int, default=40_000)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    cfg, cache = check(verbose=False)
    hp = cfg.probe
    device = cfg.runtime.resolved_device()
    (BANK / args.tag).mkdir(parents=True, exist_ok=True)

    rows, corpus_ids = _probe_rows(cache, cfg.paths.probe_dir)
    probe_ids = cache.present(sorted(corpus_ids))
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _capped_train(sorted(split.train), args.max_train)
    test_ids = sorted(split.test)
    union = sorted({*train_ids, *test_ids})
    if len(test_ids) != J4_N_TEST_ALL:
        raise SystemExit(
            f"L1a: held-out split is {len(test_ids):,}, J4(A) measured on {J4_N_TEST_ALL:,} — "
            "not the same split, so a banked matrix would not be comparable to K2."
        )
    print(f"L1a: {len(train_ids):,} train + {len(test_ids):,} test = {len(union):,} stamps, "
          f"{len(args.steps)} checkpoints -> {BANK / args.tag}", file=sys.stderr)

    ds = StampDataset(cache, rows, union)
    manifest: list[dict] = []
    checked_first = False
    t_start = time.perf_counter()
    for step in args.steps:
        ckpt = args.run / "checkpoints" / f"ckpt_{step:09d}.pt"
        path = _bank_path(args.tag, step)
        t0 = time.perf_counter()
        matrix = _load(path, len(union))
        if matrix is None:
            if not ckpt.exists():
                raise SystemExit(f"L1a: no checkpoint at step {step} ({ckpt})")
            frozen, got = frozen_from(ckpt, BANK / args.tag / "scratch_encoder.pt")
            if got != step:
                raise SystemExit(f"L1a: {ckpt.name} says step {got}")
            matrix = extract_matrix(frozen, ds, device=device)
            np.savez(path, object_ids=matrix.object_ids, x=matrix.x)
            del frozen
            _release(device)
            how = f"extracted {time.perf_counter() - t0:.0f}s"
        else:
            how = "banked already"

        # The faithfulness gate, on the FIRST checkpoint of the pass: if the cache does not
        # reproduce K2's linear number the bank is wrong, and finding that out after eight
        # extractions rather than one is an hour of nothing. L1b re-checks every step.
        note = ""
        if not checked_first and args.tag == K2_TAG and step in K2_CONSENSUS:
            tr = _extremes(_featured(matrix, rows, train_ids, hp.label_col),
                           low=hp.extreme_low, high=hp.extreme_high)
            te = _extremes(_featured(matrix, rows, test_ids, hp.label_col),
                           low=hp.extreme_low, high=hp.extreme_high)
            if len(tr.y) != J4_N_TRAIN_EXTREMES:
                raise SystemExit(f"L1a: consensus train {len(tr.y):,} != {J4_N_TRAIN_EXTREMES:,}")
            auc, _, _ = probe_auc_ci(tr, te, c=hp.c, seed=cfg.seed)
            want = K2_CONSENSUS[step]
            if abs(auc - want) > 5e-5:
                raise SystemExit(
                    f"L1a: banked step {step} probes to {auc:.4f}, K2 measured {want:.4f}. "
                    "The cache does not reproduce the trajectory — L1 would be void."
                )
            note = f"  reproduces K2 {auc:.4f}"
            checked_first = True

        rec = {"step": step, "checkpoint": str(ckpt), "path": str(path),
               "n": int(matrix.x.shape[0]), "dim": int(matrix.x.shape[1]),
               "bytes": path.stat().st_size, "seconds": time.perf_counter() - t0}
        manifest.append(rec)
        (BANK / args.tag / "manifest.json").write_text(json.dumps(
            {"run": str(args.run), "tag": args.tag, "device": device, "smoke": True,
             "n_train": len(train_ids), "n_test": len(test_ids), "max_train": args.max_train,
             "label_col": hp.label_col, "checkpoints": manifest}, indent=2))
        print(f"  step {step:6,d}  {matrix.x.shape[0]:,}x{matrix.x.shape[1]}  "
              f"{rec['bytes'] / 1e6:6.1f} MB  {how}{note}", file=sys.stderr, flush=True)
        del matrix
        _release(device)

    scratch = BANK / args.tag / "scratch_encoder.pt"
    scratch.unlink(missing_ok=True)
    total = sum(r["bytes"] for r in manifest)
    print(f"\nbanked {len(manifest)} matrices, {total / 1e9:.2f} GB, "
          f"{time.perf_counter() - t_start:.0f}s -> {BANK / args.tag}", file=sys.stderr)


if __name__ == "__main__":
    main()
