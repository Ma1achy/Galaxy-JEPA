"""Brief L1b — is the information lost, or just less linearly accessible?

Effective rank rose 24.3 -> 57.6 across the run while consensus AUC fell 0.9477 -> 0.9278. Those
may be one fact seen twice: information spread isotropically across more directions is less
accessible to a fixed-capacity L2 probe **without being gone**. The distinction changes what the
whole ladder measures, because R1 means "clean LINEAR direction".

So: run the bottom-of-ladder bounded-capacity MLP at the same eight checkpoints, on the same
split and the same protocol as the linear probe, and put the two curves side by side.

READ (written before the numbers, `artifacts/k_findings.md` carries it too):
  MLP flat while linear declines  -> information intact; the problem is linear accessibility
  both decline together           -> information genuinely destroyed
  MLP declines MORE               -> something else; say so, do not force a story

TWO GUARDS, both pre-registered:
  * PROBE-ADEQUACY. At the peak checkpoint the best sub-ceiling MLP AUC must be >= the linear
    AUC. `mlp_epochs: 200` is 200 FULL-BATCH Adam steps, a weak recipe against sklearn's
    lbfgs-to-convergence. An MLP that cannot match the linear probe where the representation is
    at its best is too blunt to answer the question, and the result is UNINTERPRETABLE rather
    than negative.
  * CEILING SENSITIVITY. `selectivity_ceiling`'s predicate is FLAGGED in the spec as undecided.
    The curve is re-read with the ceiling one width lower and one higher; if the conclusion moves,
    it belongs to the flagged predicate and not to the data.

Everything here is production machinery — `capacity_sweep`, `selectivity_ceiling`,
`shuffled_label_nulls`, `probe_auc_ci` — run on the matrices L1a banked. No encoder, no MPS, CPU
only, so this can run while a training arm holds the GPU.

    uv run python artifacts/l1b_mlp_ladder.py --tag full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402
from k2_trajectory_probe import J4_N_TEST_ALL, _capped_train, _featured  # noqa: E402
from l1a_cache_embeddings import BANK, K2_CONSENSUS, K2_TAG  # noqa: E402

from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.harness import _probe_rows, build_label_provider  # noqa: E402
from galaxy_jepa.probing import mlp as mlp_mod  # noqa: E402
from galaxy_jepa.probing.config import ProbingConfig  # noqa: E402
from galaxy_jepa.probing.controls import shuffled_label_nulls  # noqa: E402
from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, _extremes, probe_auc_ci  # noqa: E402
from galaxy_jepa.probing.schemes import get_scheme  # noqa: E402

OUT = REPO / "artifacts" / "out"


def _sweep(train: Embeddings, test: Embeddings, pc: ProbingConfig, seed: int) -> dict:
    """One checkpoint's capacity sweep + ceiling, with `ladder.py`'s exact conventions."""
    t = time.perf_counter()
    linear, lo, hi = probe_auc_ci(train, test, c=pc.c, seed=seed, n_boot=pc.n_boot)
    t_linear = time.perf_counter() - t
    t = time.perf_counter()
    nulls = shuffled_label_nulls(train, test, n_draws=pc.n_null_draws, c=pc.c, seed=seed)
    threshold = max(0.5, float(np.quantile(nulls, pc.ceiling_null_quantile)))
    t_nulls = time.perf_counter() - t
    t = time.perf_counter()

    rng = np.random.default_rng(seed)  # ladder.py's convention, verbatim
    ctrl_train = Embeddings(train.x, rng.permutation(train.y), train.fraction)
    ctrl_test = Embeddings(test.x, test.y, test.fraction)  # real test labels keep AUC defined
    rows = mlp_mod.capacity_sweep(
        train, test, ctrl_train, ctrl_test,
        widths=pc.mlp_widths, depth=pc.mlp_depth, weight_decay=pc.mlp_weight_decay,
        epochs=pc.mlp_epochs, lr=pc.mlp_lr, seed=seed,
    )
    t_sweep = time.perf_counter() - t
    ceiling = mlp_mod.selectivity_ceiling(rows, null_threshold=threshold)

    def best_below(c: int | None) -> tuple[float, int] | tuple[None, None]:
        valid = [r for r in rows if c is None or r.width < c]
        if not valid:
            return None, None
        top = max(valid, key=lambda r: r.real_auc)
        return top.real_auc, top.width

    # CEILING SENSITIVITY: the same read with the ceiling one swept width either side.
    widths = list(pc.mlp_widths)
    if ceiling is None:
        lower, higher = widths[-1], None  # "one lower" = exclude the top width; "higher" = all
    else:
        i = widths.index(ceiling)
        lower = widths[i - 1] if i > 0 else widths[0]
        higher = widths[i + 1] if i + 1 < len(widths) else None
    best, at = best_below(ceiling)
    best_lo, _ = best_below(lower)
    best_hi, _ = best_below(higher)
    return {
        "linear_auc": linear, "linear_lo": lo, "linear_hi": hi,
        "n_train": int(len(train.y)), "n_test": int(len(test.y)),
        "positives_train": int(train.y.sum()), "positives_test": int(test.y.sum()),
        "null_threshold": threshold,
        "shuffled_mean": float(np.mean(nulls)), "shuffled_max": float(np.max(nulls)),
        "ceiling_width": ceiling,
        "mlp_best_sub_ceiling": best, "mlp_best_width": at,
        "mlp_best_ceiling_lower": best_lo, "mlp_best_ceiling_higher": best_hi,
        # The sweep is CAPACITY-limited rather than ceiling-limited when its best sub-ceiling
        # width is the widest one swept: the MLP is still improving where the range ends, so the
        # number is a LOWER BOUND on nonlinear accessibility, not an estimate of it. Recorded per
        # checkpoint because the read changes if it is true at one end of the trajectory and not
        # the other.
        "capacity_limited": at == max(pc.mlp_widths),
        "seconds": {"linear": t_linear, "nulls": t_nulls, "sweep": t_sweep},
        "sweep": [{"width": r.width, "real": r.real_auc, "control": r.control_auc} for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--max-train", type=int, default=40_000)
    ap.add_argument("--feature", default="t02_edgeon_a04_yes",
                    help="the robustness feature, probed under the SCHEME convention; '' to skip")
    args = ap.parse_args()

    cfg, cache = check(verbose=False)
    pc = ProbingConfig(**yaml.safe_load((REPO / "configs/probe.yaml").read_text()))
    hp = cfg.probe
    manifest = json.loads((BANK / args.tag / "manifest.json").read_text())

    rows_meta, corpus_ids = _probe_rows(cache, cfg.paths.probe_dir)
    probe_ids = cache.present(sorted(corpus_ids))
    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _capped_train(sorted(split.train), args.max_train)
    test_ids = sorted(split.test)
    if len(test_ids) != J4_N_TEST_ALL:
        raise SystemExit(f"L1b: held-out split is {len(test_ids):,}, expected {J4_N_TEST_ALL:,}")
    labels = build_label_provider(
        rows_meta, scheme=get_scheme(pc.scheme_name),
        vote_count_min=pc.vote_count_min, consensus_gate=pc.consensus_gate,
    ) if args.feature else None

    records: list[dict] = []
    t_start = time.perf_counter()
    for entry in manifest["checkpoints"]:
        step = entry["step"]
        t0 = time.perf_counter()
        with np.load(entry["path"]) as z:
            matrix = EmbeddingMatrix(object_ids=z["object_ids"], x=z["x"],
                                     encoder_name=f"banked:{step}")
        # t01 under the H5/K2 protocol — the `_extremes` consensus cut, so the linear column
        # lands directly on K2's curve.
        tr = _extremes(_featured(matrix, rows_meta, train_ids, hp.label_col),
                       low=hp.extreme_low, high=hp.extreme_high)
        te = _extremes(_featured(matrix, rows_meta, test_ids, hp.label_col),
                       low=hp.extreme_low, high=hp.extreme_high)
        rec = {"step": step, "feature": "t01_consensus", **_sweep(tr, te, pc, cfg.seed)}
        # The reproduction check belongs to K2's own trajectory alone (see `l1a.K2_TAG`): another
        # arm is another encoder, and holding it to these numbers would be a category error.
        want = K2_CONSENSUS.get(step) if args.tag == K2_TAG else None
        if want is not None and abs(rec["linear_auc"] - want) > 5e-5:
            raise SystemExit(
                f"L1b: banked step {step} probes linearly to {rec['linear_auc']:.4f}, K2 measured "
                f"{want:.4f}. The cache does not reproduce the trajectory — L1 would be void."
            )
        rec["reproduces_k2"] = want
        records.append(rec)

        # The robustness feature under its OWN convention: the scheme's full labelled set, which
        # is where its linear number (J4's 0.7320 at step 50,000) comes from. Each feature
        # compares its MLP against its own linear probe under one convention, never across two.
        if labels is not None:
            ftr = feature_embeddings(matrix, labels, args.feature, train_ids)
            fte = feature_embeddings(matrix, labels, args.feature, test_ids)
            records.append({"step": step, "feature": args.feature, "reproduces_k2": None,
                            **_sweep(ftr, fte, pc, cfg.seed)})

        (OUT / f"l1_mlp_ladder_{args.tag}.partial.json").write_text(json.dumps(records, indent=2))
        for r in records[-2:] if labels is not None else records[-1:]:
            best = r["mlp_best_sub_ceiling"]
            ceil = r["ceiling_width"]
            # `best` is None when the ceiling fires at the SMALLEST width: no capacity in the
            # sweep is selective, so there is no admissible MLP reading at all. That is a result
            # about the representation -- it is memorisable at every capacity offered -- and it
            # must print as one rather than crash the run that found it. Brief L hit this on the
            # first lambda=0 checkpoint, after eight lambda=0.05 checkpoints never did.
            head = f"  step {step:6,d}  {r['feature']:18s} linear {r['linear_auc']:.4f}  "
            tail = (f"{'  CAPACITY-LIMITED' if r['capacity_limited'] else ''}  "
                    f"[{time.perf_counter() - t0:.0f}s: lin {r['seconds']['linear']:.0f} "
                    f"null {r['seconds']['nulls']:.0f} sweep {r['seconds']['sweep']:.0f}]")
            if best is None:
                print(head + f"NO ADMISSIBLE MLP -- ceiling fires at the smallest width "
                             f"(w{ceil}); the shuffled-label control clears the threshold "
                             f"{r['null_threshold']:.4f} at every capacity" + tail,
                      file=sys.stderr, flush=True)
            else:
                print(head + f"MLP<ceiling {best:.4f} @w{r['mlp_best_width']}  "
                             f"ceiling {ceil if ceil is not None else 'none (whole range valid)'}  "
                             f"delta {best - r['linear_auc']:+.4f}" + tail,
                      file=sys.stderr, flush=True)
        del matrix

    path = OUT / f"l1_mlp_ladder_{args.tag}.json"
    path.write_text(json.dumps({
        "tag": args.tag, "run": manifest["run"], "smoke": True,
        "widths": list(pc.mlp_widths), "mlp_depth": pc.mlp_depth, "mlp_epochs": pc.mlp_epochs,
        "mlp_lr": pc.mlp_lr, "mlp_weight_decay": pc.mlp_weight_decay,
        "n_null_draws": pc.n_null_draws, "ceiling_null_quantile": pc.ceiling_null_quantile,
        "c": pc.c, "seconds": time.perf_counter() - t_start, "checkpoints": records,
    }, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)

    # --- the probe-adequacy gate, at the peak checkpoint -----------------------------------
    t01 = [r for r in records if r["feature"] == "t01_consensus"]
    peak = max(t01, key=lambda r: r["linear_auc"])
    n_inadmissible = sum(1 for r in t01 if r["mlp_best_sub_ceiling"] is None)
    if n_inadmissible:
        print(f"\nSELECTIVITY: {n_inadmissible} of {len(t01)} t01 checkpoints have NO admissible\n"
              "  MLP -- the shuffled-label control clears the threshold at every width. On those\n"
              "  checkpoints the MLP measures memorisation, not retained information, and no\n"
              "  nonlinear reading may be taken from them. This is the R3 guardrail doing its job.",
              file=sys.stdout)
    if peak["mlp_best_sub_ceiling"] is None:
        print(f"\nPROBE-ADEQUACY GATE at the peak (step {peak['step']:,}): NOT APPLICABLE -- the\n"
              "  peak checkpoint has no admissible MLP, so the instrument cannot be assessed\n"
              "  against it and the nonlinear trajectory is UNINTERPRETABLE here.")
        return
    gap = peak["mlp_best_sub_ceiling"] - peak["linear_auc"]
    print(f"\nPROBE-ADEQUACY GATE at the peak (step {peak['step']:,}): "
          f"MLP {peak['mlp_best_sub_ceiling']:.4f} vs linear {peak['linear_auc']:.4f} "
          f"= {gap:+.4f}")
    if gap < 0:
        print("  FAILS. The MLP cannot match the linear probe where the representation is at its\n"
              "  best, so it is too blunt to answer L1. The trajectory below is UNINTERPRETABLE\n"
              "  as evidence about retained information — report it as an instrument failure.")
    else:
        print("  passes — the MLP is at least as strong as the linear probe at the peak.")


if __name__ == "__main__":
    main()
