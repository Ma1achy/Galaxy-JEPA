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
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    ("t01_smooth_or_features_a02_features_or_disk", "clean binary — the comparison point", 200),
    ("t02_edgeon_a04_yes", "clean binary — strong visual signal", 50),
    ("t10_arms_winding_a28_tight", "graded axis (1/3) — ordered question", 50),
    ("t10_arms_winding_a29_medium", "graded axis (2/3)", 50),
    ("t10_arms_winding_a30_loose", "graded axis (3/3)", 50),
    ("t09_bulge_shape_a26_boxy", "deep + confused — 89.8% of positives on <=2 votes", 200),
)


@dataclasses.dataclass(frozen=True)
class Spec:
    """What a caller of this battery declares. Defaults are Brief J's, so running bare reproduces J.

    The second consumer arrived (Brief N, on M's settled encoder), so the driver is parameterised
    rather than forked — the same move `j1_preflight` -> `m1_preflight` already made. J's record is
    what this file documents; a new brief supplies its own `Spec` and leaves that record alone.
    """

    label: str
    tag: str
    #: The pooled-quantity lines belong to the floor-evidence phase, which has its own name.
    floor_label: str = "J5"
    checkpoint: str | None = None  # None -> <out_dir>/encoder.pt
    spread: tuple[tuple[str, str, int], ...] = SPREAD
    #: Extra seeds for the untrained-encoder control, measured ALONGSIDE the primary one and never
    #: entering `FeatureControls`. The primary seed is always `cfg.seed`, so the bar, the
    #: selectivity and everything comparable to J are unchanged in construction. Brief J measured
    #: none: that single scalar per feature *is* the existence bar (the other three controls never
    #: cleared 0.5552, so `existence_null_samples` collapses to it), and its variability under
    #: reseeding was never measured. A floor built on one draw of a random network is a floor built
    #: on a coin flip, so N1 measures the spread before N2 proposes anything on top of it.
    extra_untrained_seeds: tuple[int, ...] = ()


J = Spec(label="J4", tag="j4")


def _capped_train(train_ids: list[int], max_train: int) -> list[int]:
    """H5's deterministic stride, verbatim — so the subset is the same galaxies as H5 probed."""
    if not max_train or len(train_ids) <= max_train:
        return train_ids
    stride = len(train_ids) / max_train
    return [train_ids[int(i * stride)] for i in range(max_train)]


def _release(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.empty_cache()


@dataclasses.dataclass(frozen=True)
class Setup:
    """Everything a battery needs before it touches an encoder — one construction, one split."""

    cfg: Any
    pc: ProbingConfig
    device: str
    ckpt: Path
    rows: dict
    labels: Any
    train_ids: list[int]
    test_ids: list[int]
    union: list[int]
    ds: StampDataset


def prepare(checkpoint: str | None, max_train: int, *, label: str, sources: int) -> Setup:
    """The split, the labels and the dataset — **the one site**, so two briefs cannot diverge.

    Extracted when Brief O1 arrived needing the *identical* train/test ids N1 probed: a matched
    AUC is only comparable to its unmatched partner if both were read off the same galaxies, and
    a second copy of this fifteen lines is a silent way to lose that. Second consumer, so an
    abstraction — not before.
    """
    cfg, cache = check(verbose=False)
    pc = ProbingConfig(**yaml.safe_load((REPO / "configs/probe.yaml").read_text()))
    if pc.headline:
        raise SystemExit(f"{label}: probe.yaml claims headline — this battery is not a result")
    device = cfg.runtime.resolved_device()
    ckpt = Path(checkpoint) if checkpoint else Path(cfg.paths.out_dir) / "encoder.pt"
    OUT.mkdir(parents=True, exist_ok=True)

    rows, corpus_ids = _probe_rows(cache, cfg.paths.probe_dir)
    probe_ids = cache.present(sorted(corpus_ids))
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _capped_train(sorted(split.train), max_train)
    test_ids = sorted(split.test)
    union = sorted({*train_ids, *test_ids})
    print(f"{label} split: {len(train_ids):,} train (of {len(split.train):,}) + "
          f"{len(test_ids):,} test = {len(union):,} stamps to embed x{sources} sources",
          file=sys.stderr)
    labels = build_label_provider(
        rows,
        scheme=get_scheme(pc.scheme_name),
        vote_count_min=pc.vote_count_min,
        consensus_gate=pc.consensus_gate,
    )
    return Setup(cfg, pc, device, ckpt, rows, labels,
                 train_ids, test_ids, union, StampDataset(cache, rows, union))


def main(spec: Spec = J) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=spec.checkpoint, help="default: <out_dir>/encoder.pt")
    ap.add_argument("--max-train", type=int, default=40_000, help="H5's cap; 0 for no cap")
    ap.add_argument("--tag", default=spec.tag)
    args = ap.parse_args()
    lbl = spec.label

    setup = prepare(args.checkpoint, args.max_train, label=lbl, sources=3)
    cfg, pc, device, ckpt = setup.cfg, setup.pc, setup.device, setup.ckpt
    rows, train_ids, test_ids, ds = setup.rows, setup.train_ids, setup.test_ids, setup.ds
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

    labels = setup.labels

    records = []
    for i, (feature, role, draws) in enumerate(spec.spread):
        if feature not in labels.features:
            raise SystemExit(f"{lbl}: {feature!r} is not in scheme {pc.scheme_name!r}")
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

    # --- the untrained-encoder control under reseeding ---------------------------------------
    # `existence_null_samples` takes a per-draw max over the four chance-calibrated controls, but
    # the two resamplable ones never cleared 0.5552 on J's encoder while the untrained singleton
    # reached 0.7908 — so the null collapses to a point mass at that singleton and the existence
    # test reduces, exactly, to `real_auc > untrained_encoder_auc`. One draw of a random ViT is
    # therefore the whole bar. Measured here, never used here: the primary seed stays `cfg.seed`,
    # these extra passes are recorded alongside it and enter no `FeatureControls` and no verdict.
    if spec.extra_untrained_seeds:
        seeds = (cfg.seed, *spec.extra_untrained_seeds)
        by_feature: dict[str, dict[int, float]] = {
            r["feature"]: {cfg.seed: r["untrained_encoder_auc"]} for r in records
        }
        for s_extra in spec.extra_untrained_seeds:
            _release(device)
            t_seed = time.perf_counter()
            mat = ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=s_extra)
            for feature, _role, _draws in spec.spread:
                tr_u = feature_embeddings(mat, labels, feature, train_ids)
                te_u = feature_embeddings(mat, labels, feature, test_ids)
                by_feature[feature][s_extra] = ctl._safe_auc(tr_u, te_u, c=pc.c)
            del mat
            _release(device)
            print(f"  untrained seed {s_extra:<4d} {time.perf_counter() - t_seed:6.0f}s",
                  file=sys.stderr)
        for rec in records:
            per_seed = by_feature[rec["feature"]]
            # The primary seed must still be the primary: this is the assertion, not a comment.
            assert per_seed[cfg.seed] == rec["untrained_encoder_auc"], rec["feature"]
            vals = [per_seed[k] for k in seeds]
            rec["untrained_encoder_aucs"] = {str(k): per_seed[k] for k in seeds}
            rec["untrained_seed_min"] = float(min(vals))
            rec["untrained_seed_median"] = float(np.median(vals))
            rec["untrained_seed_max"] = float(max(vals))
            rec["untrained_seed_range"] = float(max(vals) - min(vals))
        (OUT / f"{args.tag}_spread_controls.partial.json").write_text(json.dumps(records, indent=2))

    path = OUT / f"{args.tag}_spread_controls.json"
    path.write_text(json.dumps({
        "checkpoint": str(ckpt), "device": device, "smoke": True,
        "n_train": len(train_ids), "n_test": len(test_ids), "max_train": args.max_train,
        "scheme": pc.scheme_name, "vote_count_min": pc.vote_count_min, "c": pc.c,
        "seconds": time.perf_counter() - t0,
        "untrained_seeds": [cfg.seed, *spec.extra_untrained_seeds],
        "features": records,
    }, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)

    # J5's three quantities, pooled — reported, never turned into a floor here.
    real_aucs = [r["auc"] for r in records if r["auc"] is not None]
    nulls = [r["shuffled_max"] for r in records] + [r["random_emb_max"] for r in records] \
        + [r["untrained_encoder_auc"] for r in records] + [r["noise_encoder_auc"] for r in records]
    print(f"{spec.floor_label} real AUC spread     : {min(real_aucs):.4f} .. {max(real_aucs):.4f} "
          f"(n={len(real_aucs)})")
    print(f"{spec.floor_label} pooled null ceiling : {max(nulls):.4f} "
          "(max over every control on every feature)")
    print(f"{spec.floor_label} gap                 : {min(real_aucs) - max(nulls):+.4f} "
          "at the weakest feature")
    print(f"{spec.floor_label} the floor is NOT set here. effect_floor_freeze stays None; "
          "the value is Malachy's.")


if __name__ == "__main__":
    main()
