"""H5's probe — the deciding measurement, run without the 4 GB metadata table.

`harness.evaluate_probe` is the production path and was the first choice, but it builds
`rows_by_id(DirectorySource(probe_dir).rows)`: `csv.DictReader` over 230,358 rows x ~150
columns, the same multi-gigabyte table the *training* path already fled from via the scalar
sidecar. On this 18 GB machine it was killed before embedding a single stamp.

So this driver keeps every measurement primitive and drops only the parts that are not the
measurement. It uses the **same** functions the production path uses — `assign_three_way` for
the split, `extract_embeddings` through the frozen encoder, `_extremes` for the consensus cut,
`probe_auc_ci` for the AUC and its bootstrap interval — with the same `ProbeConfig` values read
from the same `configs/pretrain.yaml`. What it does not do is materialise 150 columns per galaxy
when `StampDataset` reads exactly one of them, and it does not write the UMAP or the explorer
blobs, which are figures rather than results.

Two deliberate, stated departures, applied **identically to both arms** so the comparison is
untouched:
  * the probe-train split is deterministically capped (`--max-train`). A 384-dimensional logistic
    probe is thoroughly converged long before 161k examples — the pilot fit on 1,835 — and the
    cap is what makes two probes affordable here.
  * the probe-test split is **not** capped. That is the measurement.

    uv run python artifacts/h5_probe_lean.py --arm baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from f1_loader_bench import OUT  # noqa: E402

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.logistic import (  # noqa: E402
    _extremes,
    extract_embeddings,
    probe_auc_ci,
)

RUNS = Path(__file__).resolve().parent.parent / "runs" / "h5"
PROBES = OUT / "h5_probes.jsonl"


def probe(arm: str, max_train: int, runs: Path = RUNS) -> dict:
    cfg, cache = check(verbose=False)
    pc = cfg.probe
    device = cfg.runtime.resolved_device()
    ckpt = runs / arm / "encoder.pt"

    # only the id and the one label column this probe reads — not the other ~150
    meta = Path(cfg.paths.probe_dir) / "metadata.csv"
    df = pd.read_csv(meta, usecols=["object_id", pc.label_col])
    probe_ids = [int(o) for o in df["object_id"]]
    rows = {
        int(o): {pc.label_col: float(f)}
        for o, f in zip(df["object_id"], df[pc.label_col], strict=True)
    }
    del df

    # the same deterministic split the production path builds — same seed, same ratios
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = sorted(split.train)
    if max_train and len(train_ids) > max_train:
        # deterministic stride, so both arms see the identical subset
        stride = len(train_ids) / max_train
        train_ids = [train_ids[int(i * stride)] for i in range(max_train)]
    test_ids = sorted(split.test)

    frozen = load_frozen_encoder(ckpt)  # eval() + requires_grad_(False); assert_frozen downstream
    t0 = time.perf_counter()
    train_ds = StampDataset(cache, rows, train_ids, label_fraction_col=pc.label_col)
    test_ds = StampDataset(cache, rows, test_ids, label_fraction_col=pc.label_col)
    print(f"  {arm}: embedding {len(train_ds):,} train + {len(test_ds):,} test on {device}",
          file=sys.stderr, flush=True)
    train_full = extract_embeddings(frozen, train_ds, device=device)
    test_full = extract_embeddings(frozen, test_ds, device=device)

    train_emb = _extremes(train_full, low=pc.extreme_low, high=pc.extreme_high)
    test_emb = _extremes(test_full, low=pc.extreme_low, high=pc.extreme_high)
    auc, lo, hi = probe_auc_ci(train_emb, test_emb, c=pc.c, seed=cfg.seed)

    # the whole held-out set too, so the consensus headline is never the only number reported
    auc_all, lo_all, hi_all = probe_auc_ci(train_emb, test_full, c=pc.c, seed=cfg.seed)
    amb = (test_full.fraction > pc.extreme_low) & (test_full.fraction < pc.extreme_high)
    from galaxy_jepa.probing.logistic import Embeddings

    ambiguous = Embeddings(test_full.x[amb], test_full.y[amb], test_full.fraction[amb])
    auc_amb, lo_amb, hi_amb = probe_auc_ci(train_emb, ambiguous, c=pc.c, seed=cfg.seed)

    rec = {
        "arm": arm, "checkpoint": str(ckpt), "smoke": True, "device": device,
        "runs": str(runs),
        "auc": auc, "auc_lo": lo, "auc_hi": hi,
        "n_train": int(len(train_emb.y)), "n_test": int(len(test_emb.y)),
        "auc_all": auc_all, "auc_all_lo": lo_all, "auc_all_hi": hi_all,
        "n_test_all": int(len(test_full.y)),
        "auc_ambiguous": auc_amb, "auc_amb_lo": lo_amb, "auc_amb_hi": hi_amb,
        "n_ambiguous": int(len(ambiguous.y)),
        "train_pool": int(len(train_ds)), "max_train": max_train,
        "label_col": pc.label_col, "extreme_low": pc.extreme_low, "extreme_high": pc.extreme_high,
        "c": pc.c, "seconds": time.perf_counter() - t0,
    }
    with PROBES.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--max-train", type=int, default=40000)
    # Brief I reuses this driver verbatim so its numbers sit alongside H5's 0.9043 / 0.9358.
    # Only *where the checkpoint lives* is a parameter; the measurement is untouched, and the
    # default still reproduces the H5 invocation exactly.
    ap.add_argument("--runs", type=Path, default=RUNS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("PROBE " + json.dumps(probe(args.arm, args.max_train, args.runs)))


if __name__ == "__main__":
    main()
