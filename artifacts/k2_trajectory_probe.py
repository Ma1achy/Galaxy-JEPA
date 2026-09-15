"""Brief K2 — probe the trajectory, to separate two suspects for the degradation.

J4 measured featured-ness at 0.9278 on the 50,000-step encoder, against 0.9470 on I2's
3,000-step SIGReg arm. Training 16.7x longer made the representation measurably worse. Two
candidate mechanisms are confounded in that single comparison:

  * **SIGReg saturating** — the isotropy penalty keeps pushing after the representation is
    already isotropic, and past that point it can only cost structure.
  * **The cosine decay** — this run is the schedule's FIRST real exercise. At step 3,000 the
    LR had never left 99.7% of peak, so the 3,000-step arms measured a constant-LR regime.
    D17 flagged the decay as the weakest third of that decision.

34 checkpoints exist on disk, so the question is answerable without training anything. Probe
eight of them along the trajectory on the SAME split and probe config J4(A) used, so every
number sits directly alongside 0.9470 and 0.9278:

  monotonic decline from early   -> implicates SIGReg
  flat, then a late drop         -> implicates the decay
  neither                        -> say so; a third explanation is possible and inventing an
                                    attribution would be worse than an ambiguous curve

AND the nuisances along the same trajectory. If nuisance AUC climbs while morphology falls,
the representation is not simply degrading — it is re-allocating onto image properties, which
is a mechanism story rather than a curve.

One trajectory is not a controlled comparison and n=8 is small. This driver measures; it does
not conclude.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/k2_trajectory_probe.py
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

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.metadata import featured_label  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.harness import _probe_rows, build_label_provider  # noqa: E402
from galaxy_jepa.probing.config import ProbingConfig  # noqa: E402
from galaxy_jepa.probing.controls import _safe_auc  # noqa: E402
from galaxy_jepa.probing.extract import EmbeddingMatrix, extract_matrix  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, _extremes, probe_auc_ci  # noqa: E402
from galaxy_jepa.probing.schemes import get_scheme  # noqa: E402

OUT = REPO / "artifacts" / "out"
RUN = REPO / "runs" / "full"

#: Eight of the 34, declared here rather than chosen after seeing a curve. 1500 is the
#: earliest that exists and the nearest to where the prediction term bottoms (~step 2,000);
#: 3000 is I2's stopping point, so it doubles as a free replication check against 0.9470;
#: 50000 is J4(A)'s, so it must reproduce 0.9278. The six between are spread roughly
#: geometrically, which is where a decay-driven effect would show its shape.
STEPS: tuple[int, ...] = (1500, 3000, 6000, 10500, 18000, 27000, 37500, 50000)

#: J4(A)'s numbers, for the like-for-like assertion. If the split this driver builds does not
#: reproduce these counts, the comparison to 0.9470 / 0.9278 is void and the run should say so
#: loudly rather than quietly report a different measurement.
J4_N_TEST_ALL = 34_829
J4_N_TRAIN_EXTREMES = 25_305


def _capped_train(train_ids: list[int], max_train: int) -> list[int]:
    """H5's deterministic stride, verbatim — the same galaxies H5 and J4 probed."""
    if not max_train or len(train_ids) <= max_train:
        return train_ids
    stride = len(train_ids) / max_train
    return [train_ids[int(i * stride)] for i in range(max_train)]


def _release(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.empty_cache()


def _featured(matrix: EmbeddingMatrix, rows, ids: list[int], col: str) -> Embeddings:
    """The H5-protocol featured-ness Embeddings, sliced out of the label-free matrix.

    `h5_probe_lean.py` gets these from `extract_embeddings`, which welds the label to the
    dataset. Here one label-free pass has to serve the morphology probe AND five nuisance
    probes, so the label is attached after the fact — through `featured_label`, the same
    threshold `StampDataset` applies, so the vectors are identical to H5's.

    `matrix.index` is bound ONCE. It is a plain property that rebuilds a 74,829-entry dict on
    every read, and the obvious `if int(o) in matrix.index` inside a comprehension re-reads it
    per element — 40,000 x 74,829 insertions, which is what a first attempt at this driver spent
    twenty minutes on. `probing.extract.feature_ids` has the identical shape and is production
    code; recorded for Brief K3, not fixed here.
    """
    index = matrix.index
    present = [int(o) for o in ids if int(o) in index]
    frac = np.asarray([float(rows[o][col]) for o in present], dtype=np.float64)
    y = np.asarray([featured_label(f) for f in frac], dtype=np.int64)
    rowsel = np.asarray([index[o] for o in present], dtype=np.int64)
    return Embeddings(x=matrix.x[rowsel], y=y, fraction=frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--max-train", type=int, default=40_000)
    ap.add_argument("--steps", type=int, nargs="*", default=list(STEPS))
    ap.add_argument("--tag", default="k2")
    args = ap.parse_args()

    cfg, cache = check(verbose=False)
    pc = ProbingConfig(**yaml.safe_load((REPO / "configs/probe.yaml").read_text()))
    hp = cfg.probe  # the harness's single-feature probe block — H5/J4(A)'s exact config
    device = cfg.runtime.resolved_device()
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "k2_scratch"

    traces = json.loads((args.run / "traces.json").read_text())
    ct = traces["collapse_trace"]
    ct_steps = [int(x) for x in ct["step"]]

    def _collapse_at(step: int) -> int:
        """Index of the last collapse reading at or before ``step``.

        The monitor fires every 100 steps and the run's last reading is 49,900, so the final
        checkpoint (50,000) has no exact entry. Taking the nearest reading *at or below* is the
        like-for-like rule Brief I already had to learn once: i2's `erank_final` is its step-2975
        reading, and reading it against j3's step-3000 value is a comparison of two different
        steps. The reported `collapse_step` says which one this is.
        """
        below = [i for i, x in enumerate(ct_steps) if x <= step]
        if not below:
            raise SystemExit(f"K2: no collapse reading at or before step {step}")
        return below[-1]

    rows, corpus_ids = _probe_rows(cache, cfg.paths.probe_dir)
    probe_ids = cache.present(sorted(corpus_ids))
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _capped_train(sorted(split.train), args.max_train)
    test_ids = sorted(split.test)
    union = sorted({*train_ids, *test_ids})
    if len(test_ids) != J4_N_TEST_ALL:
        raise SystemExit(
            f"K2: held-out split is {len(test_ids):,}, J4(A) measured on {J4_N_TEST_ALL:,}. "
            "Not the same split — the comparison to 0.9470 / 0.9278 would be void."
        )
    print(f"K2 split: {len(train_ids):,} train + {len(test_ids):,} test = {len(union):,} stamps "
          f"per checkpoint, {len(args.steps)} checkpoints", file=sys.stderr)

    ds = StampDataset(cache, rows, union)
    labels = build_label_provider(
        rows,
        scheme=get_scheme(pc.scheme_name),
        vote_count_min=pc.vote_count_min,
        consensus_gate=pc.consensus_gate,
    )
    # The nuisance panel is built over the WHOLE split, not a feature's eligible subset as
    # `build_feature_controls` does. A nuisance is a property of the representation, not of a
    # morphology question, and holding n fixed across checkpoints is what makes the trajectory
    # readable. Stated rather than defaulted: these numbers are therefore NOT comparable to
    # J4's per-feature `nuisance_aucs`, which carry that feature's eligibility.
    #
    # Both the validity mask and the median split are functions of the metadata alone, so they
    # are drawn ONCE here — identical labels at every checkpoint. Re-deriving them per encoder
    # would let the median move between checkpoints and quietly make the curve incomparable.
    nuis: dict[str, dict] = {}
    for name in labels.nuisances:
        keep_tr = labels.nuisance_valid(name, train_ids)
        keep_te = labels.nuisance_valid(name, test_ids)
        ids_tr = [o for o, k in zip(train_ids, keep_tr, strict=True) if k]
        ids_te = [o for o, k in zip(test_ids, keep_te, strict=True) if k]
        nuis[name] = {
            "keep_tr": keep_tr, "keep_te": keep_te,
            "y_tr": labels.nuisance_label(name, ids_tr),
            "y_te": labels.nuisance_label(name, ids_te),
            "n_tr": len(ids_tr), "n_te": len(ids_te),
        }
    dropped = {k: len(train_ids) - v["n_tr"] for k, v in nuis.items() if v["n_tr"] < len(train_ids)}
    print(f"K2 nuisances: {list(nuis)}  flagged-out of train: {dropped or 'none'}",
          file=sys.stderr)

    # The like-for-like assertion on the TRAIN side, checked before the first encode rather than
    # after it: the consensus cut is a function of the vote fractions alone, so a dummy matrix
    # over the same ids answers it in a second instead of ten minutes.
    probe_dummy = EmbeddingMatrix(
        object_ids=np.asarray(union, dtype=np.int64),
        x=np.zeros((len(union), 1), dtype=np.float32),
        encoder_name="dry",
    )
    n_consensus = len(
        _extremes(
            _featured(probe_dummy, rows, train_ids, hp.label_col),
            low=hp.extreme_low, high=hp.extreme_high,
        ).y
    )
    if n_consensus != J4_N_TRAIN_EXTREMES:
        raise SystemExit(
            f"K2: consensus train is {n_consensus:,}, H5/J4 fit on {J4_N_TRAIN_EXTREMES:,} — "
            "not the same galaxies, comparison to 0.9470 / 0.9278 void."
        )
    print(f"K2 like-for-like: consensus train {n_consensus:,}, held-out {len(test_ids):,} — "
          f"both match J4(A)", file=sys.stderr, flush=True)
    del probe_dummy

    records: list[dict] = []
    t_start = time.perf_counter()
    for step in args.steps:
        ckpt = args.run / "checkpoints" / f"ckpt_{step:09d}.pt"
        if not ckpt.exists():
            raise SystemExit(f"K2: no checkpoint at step {step} ({ckpt})")
        t0 = time.perf_counter()
        frozen, got = frozen_from(ckpt, scratch / f"enc_{step:09d}.pt")
        if got != step:
            raise SystemExit(f"K2: {ckpt.name} says step {got}")
        matrix = extract_matrix(frozen, ds, device=device)
        t_emb = time.perf_counter() - t0
        _release(device)

        train_full = _featured(matrix, rows, train_ids, hp.label_col)
        test_full = _featured(matrix, rows, test_ids, hp.label_col)
        train_emb = _extremes(train_full, low=hp.extreme_low, high=hp.extreme_high)
        test_emb = _extremes(test_full, low=hp.extreme_low, high=hp.extreme_high)
        assert len(train_emb.y) == n_consensus  # pre-checked against J4(A) above
        auc, lo, hi = probe_auc_ci(train_emb, test_emb, c=hp.c, seed=cfg.seed)
        auc_all, lo_all, hi_all = probe_auc_ci(train_emb, test_full, c=hp.c, seed=cfg.seed)
        amb = (test_full.fraction > hp.extreme_low) & (test_full.fraction < hp.extreme_high)
        ambiguous = Embeddings(test_full.x[amb], test_full.y[amb], test_full.fraction[amb])
        auc_amb, lo_amb, hi_amb = probe_auc_ci(train_emb, ambiguous, c=hp.c, seed=cfg.seed)

        # The same real embeddings, relabelled — so a nuisance AUC and the morphology AUC at
        # this checkpoint are read off one representation, not two.
        nuisance: dict[str, float] = {}
        for name, n in nuis.items():
            nz_tr = Embeddings(
                train_full.x[n["keep_tr"]], n["y_tr"], train_full.fraction[n["keep_tr"]]
            )
            nz_te = Embeddings(
                test_full.x[n["keep_te"]], n["y_te"], test_full.fraction[n["keep_te"]]
            )
            nuisance[name] = _safe_auc(nz_tr, nz_te, c=hp.c)

        # The loss terms at this step. A single step's value is noise at batch 32, so a
        # trailing 100-step mean is the reported figure and the point value rides alongside.
        w = slice(max(0, step - 100), step)
        i = _collapse_at(step)
        rec = {
            "step": step, "checkpoint": str(ckpt), "smoke": True,
            "auc": auc, "auc_lo": lo, "auc_hi": hi,
            "auc_all": auc_all, "auc_all_lo": lo_all, "auc_all_hi": hi_all,
            "auc_ambiguous": auc_amb, "auc_amb_lo": lo_amb, "auc_amb_hi": hi_amb,
            "n_train": int(len(train_emb.y)), "n_test": int(len(test_emb.y)),
            "n_test_all": int(len(test_full.y)), "n_ambiguous": int(len(ambiguous.y)),
            "loss": float(np.mean(traces["losses"][w])),
            "prediction_loss": float(np.mean(traces["prediction_losses"][w])),
            "sigreg_loss": float(np.mean(traces["sigreg_losses"][w])),
            "prediction_loss_point": float(traces["prediction_losses"][step - 1]),
            "sigreg_loss_point": float(traces["sigreg_losses"][step - 1]),
            "collapse_step": ct_steps[i],
            "effective_rank": float(ct["effective_rank"][i]),
            "std": float(ct["std"][i]),
            "mean_cosine": float(ct["mean_cosine"][i]),
            "nuisance_aucs": nuisance,
            "embed_seconds": t_emb, "seconds": time.perf_counter() - t0,
        }
        records.append(rec)
        # Banked after EVERY checkpoint — the `keep=3` lesson, and J4's.
        (OUT / f"{args.tag}_trajectory.partial.json").write_text(json.dumps(records, indent=2))
        nz = "  ".join(f"{k} {v:.4f}" for k, v in nuisance.items())
        print(f"  step {step:6,d}  auc {auc:.4f} [{lo:.4f},{hi:.4f}]  all {auc_all:.4f}  "
              f"pred {rec['prediction_loss']:.4f}  sig {rec['sigreg_loss']:.4f}  "
              f"erank {rec['effective_rank']:.2f}  std {rec['std']:.4f}  "
              f"cos {rec['mean_cosine']:+.4f}\n        nuisance: {nz}  [{t_emb:.0f}s]",
              file=sys.stderr, flush=True)
        del matrix, train_full, test_full, train_emb, test_emb, ambiguous, frozen
        _release(device)

    path = OUT / f"{args.tag}_trajectory.json"
    path.write_text(json.dumps({
        "run": str(args.run), "device": device, "smoke": True,
        "n_train": len(train_ids), "n_test": len(test_ids), "max_train": args.max_train,
        "label_col": hp.label_col, "c": hp.c,
        "extreme_low": hp.extreme_low, "extreme_high": hp.extreme_high,
        "nuisance_n_train": {k: v["n_tr"] for k, v in nuis.items()},
        "nuisance_n_test": {k: v["n_te"] for k, v in nuis.items()},
        "seconds": time.perf_counter() - t_start, "checkpoints": records,
    }, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
