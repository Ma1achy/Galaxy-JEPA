"""Brief J4/J5 — signs of life across the difficulty range, and the effect floor's evidence.

Deliberately **below the ladder**. J4 says not to compute rung verdicts, and the code agrees:
`run_probing` -> `run_ladder` -> `existence_verdicts` calls `nulls.assert_null_resolution`, which
REFUSES at `n_null_draws: 50` against Scheme 1's BY family of 37 (it needs >= 3,109, or every
feature fails existence and the catalogue reads like a scientific null). So the battery cannot
legitimately run at this budget, and the instruction and the invariant point the same way. What
is assembled here is everything underneath that gate: the real AUC per feature with its bootstrap
interval, and `build_feature_controls`' five negative-control nulls + selectivity + the nuisance
panel. No verdicts, no existence p-values, no effect floor applied.

The nulls drawn here are CHARACTERISATION for a floor *proposal*, not an existence test. That is
why 50-200 draws is honest and 3,109 is not needed: nothing here is compared against a
family-corrected bar.

Every primitive is the production one — `assign_three_way` for the split, `_probe_rows` for the
I4 column sidecar, `extract_matrix` / `untrained_encoder_matrix` / `noise_through_encoder_matrix`
for the three embedding sources, `build_feature_controls` for the battery, `probe_auc_ci` for the
AUC and its interval. The train split is capped on the SAME deterministic stride
`h5_probe_lean.py` uses, so the featured-ness number lands next to H5's on the same galaxies; the
test split is uncapped, because that is the measurement.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/j4_spread_controls.py --checkpoint runs/full/encoder.pt
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

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.harness import _probe_rows, build_label_provider  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing.config import ProbingConfig  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix, feature_embeddings  # noqa: E402
from galaxy_jepa.probing.logistic import probe_auc_ci  # noqa: E402
from galaxy_jepa.probing.schemes import get_scheme  # noqa: E402

OUT = REPO / "artifacts" / "out"

#: The spread, drawn from Scheme 1 — where every one of the 37 answers is binary, so
#: `GradedExistenceTestUndecided` cannot fire. Chosen to span the difficulty range, declared here
#: rather than picked after seeing the numbers. `draws` is the null-characterisation budget: the
#: two ends of the range get more, because they are what a floor is argued between.
SPREAD: tuple[tuple[str, str, int], ...] = (
    ("t01_smooth_or_features_a02_features_or_disk", "clean binary — the comparison point", 50),
    ("t02_edgeon_a04_yes", "clean binary — strong visual signal", 50),
    ("t10_arms_winding_a28_tight", "graded axis (1/3) — ordered question", 50),
    ("t10_arms_winding_a29_medium", "graded axis (2/3)", 50),
    ("t10_arms_winding_a30_loose", "graded axis (3/3)", 50),
    ("t09_bulge_shape_a26_boxy", "deep + confused — 89.8% of positives on <=2 votes", 50),
)


def _capped_train(train_ids: list[int], max_train: int) -> list[int]:
    """H5's deterministic stride, verbatim — so the subset is the same galaxies as H5 probed."""
    if not max_train or len(train_ids) <= max_train:
        return train_ids
    stride = len(train_ids) / max_train
    return [train_ids[int(i * stride)] for i in range(max_train)]


def _release(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: <out_dir>/encoder.pt")
    ap.add_argument("--max-train", type=int, default=40_000, help="H5's cap; 0 for no cap")
    ap.add_argument("--tag", default="j4")
    args = ap.parse_args()

    cfg, cache = check(verbose=False)
    pc = ProbingConfig(**yaml.safe_load((REPO / "configs/probe.yaml").read_text()))
    if pc.headline:
        raise SystemExit("J4: probe.yaml claims headline — the floor is open, this is not a result")
    device = cfg.runtime.resolved_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg.paths.out_dir) / "encoder.pt"
    OUT.mkdir(parents=True, exist_ok=True)

    rows, corpus_ids = _probe_rows(cache, cfg.paths.probe_dir)
    probe_ids = cache.present(sorted(corpus_ids))
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _capped_train(sorted(split.train), args.max_train)
    test_ids = sorted(split.test)
    union = sorted({*train_ids, *test_ids})
    print(f"J4 split: {len(train_ids):,} train (of {len(split.train):,}) + {len(test_ids):,} test "
          f"= {len(union):,} stamps to embed x3 sources", file=sys.stderr)

    ds = StampDataset(cache, rows, union)
    frozen = load_frozen_encoder(ckpt)  # eval() + requires_grad_(False); assert_frozen downstream
    t0 = time.perf_counter()

    # Three embedding sources, one pass each over the SAME ids so every control is co-indexed
    # with the real probe. The MPS pool is released between passes (the Brief I lesson).
    real = extract_matrix(frozen, ds, device=device)
    print(f"  real      {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    _release(device)
    untrained = ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=cfg.seed)
    print(f"  untrained {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    _release(device)
    noise = ctl.noise_through_encoder_matrix(frozen, ds, device=device, seed=cfg.seed)
    print(f"  noise     {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    _release(device)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained, noise=noise)

    labels = build_label_provider(
        rows,
        scheme=get_scheme(pc.scheme_name),
        vote_count_min=pc.vote_count_min,
        consensus_gate=pc.consensus_gate,
    )

    records = []
    for i, (feature, role, draws) in enumerate(SPREAD):
        if feature not in labels.features:
            raise SystemExit(f"J4: {feature!r} is not in scheme {pc.scheme_name!r}")
        tr = feature_embeddings(real, labels, feature, train_ids)
        te = feature_embeddings(real, labels, feature, test_ids)
        try:
            auc, lo, hi = probe_auc_ci(tr, te, c=pc.c, seed=cfg.seed)
        except ValueError as exc:  # a single-class split is an outcome, not a crash
            auc = lo = hi = None
            reason = str(exc)[:80]
        else:
            reason = None
        fc = ctl.build_feature_controls(
            feature,
            real_auc=auc if auc is not None else 0.5,
            train_ids=train_ids,
            test_ids=test_ids,
            controls=controls,
            labels=labels,
            sky_label_col="snr_r",
            c=pc.c,
            n_draws=draws,
            seed=cfg.seed + i,
        )
        rec = {
            "feature": feature, "role": role, "smoke": True, "checkpoint": str(ckpt),
            "auc": auc, "auc_lo": lo, "auc_hi": hi, "undefined_reason": reason,
            "n_train": int(len(tr.y)), "n_test": int(len(te.y)),
            "positives_train": int(tr.y.sum()), "positives_test": int(te.y.sum()),
            "n_draws": draws,
            "shuffled_mean": float(np.mean(fc.shuffled_nulls)),
            "shuffled_sd": float(np.std(fc.shuffled_nulls)),
            "shuffled_max": float(np.max(fc.shuffled_nulls)),
            "shuffled_q95": float(np.quantile(fc.shuffled_nulls, 0.95)),
            "shuffled_q99": float(np.quantile(fc.shuffled_nulls, 0.99)),
            "random_emb_mean": float(np.mean(fc.random_embedding_nulls)),
            "random_emb_max": float(np.max(fc.random_embedding_nulls)),
            "untrained_encoder_auc": fc.untrained_encoder_auc,
            "noise_encoder_auc": fc.noise_encoder_auc,
            "sky_noise_auc": fc.sky_noise_auc,
            "selectivity": fc.selectivity,
            "nuisance_aucs": dict(fc.nuisance_aucs),
        }
        records.append(rec)
        # Written after EVERY feature, not once at the end. The first run of this took 2h48m for
        # two features on a thrashing machine, and an end-of-run write means an interrupted pass
        # yields nothing at all — the same "a long job must bank its work" lesson as `keep=3`.
        (OUT / f"{args.tag}_spread_controls.partial.json").write_text(json.dumps(records, indent=2))
        shown = f"{auc:.4f} [{lo:.4f},{hi:.4f}]" if auc is not None else f"undefined ({reason})"
        print(f"  {feature:48s} {shown}  sel {fc.selectivity:+.4f}  "
              f"untrained {fc.untrained_encoder_auc:.4f}  noise {fc.noise_encoder_auc:.4f}",
              file=sys.stderr)

    path = OUT / f"{args.tag}_spread_controls.json"
    path.write_text(json.dumps({
        "checkpoint": str(ckpt), "device": device, "smoke": True,
        "n_train": len(train_ids), "n_test": len(test_ids), "max_train": args.max_train,
        "scheme": pc.scheme_name, "vote_count_min": pc.vote_count_min, "c": pc.c,
        "seconds": time.perf_counter() - t0, "features": records,
    }, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)

    # J5's three quantities, pooled — reported, never turned into a floor here.
    real_aucs = [r["auc"] for r in records if r["auc"] is not None]
    nulls = [r["shuffled_max"] for r in records] + [r["random_emb_max"] for r in records] \
        + [r["untrained_encoder_auc"] for r in records] + [r["noise_encoder_auc"] for r in records]
    print(f"J5 real AUC spread     : {min(real_aucs):.4f} .. {max(real_aucs):.4f} "
          f"(n={len(real_aucs)})")
    print(f"J5 pooled null ceiling : {max(nulls):.4f} (max over every control on every feature)")
    print(f"J5 gap                 : {min(real_aucs) - max(nulls):+.4f} at the weakest feature")
    print("J5 the floor is NOT set here. effect_floor_freeze stays None; the value is Malachy's.")


if __name__ == "__main__":
    main()
