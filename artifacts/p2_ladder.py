"""Brief P2 — Scheme 1, all 37 answers, on M's 4-epoch checkpoint.

The catalogue. Rungs emitted by the hard gate, deterministically: existence (D23's untrained-z),
the frozen effect floor at 0.7267, selectivity, nuisance clearance.

**Why this drives `run_ladder` directly rather than `run_probing`.** `run_probing` extracts over the
whole probe corpus and makes its own three-way split. The effect floor was frozen against the
40,000/34,829 split, O1's matched battery and N1's controls were measured on it, and the untrained
bank is keyed to it. Moving the split would mean the frozen floor no longer corresponds to the
evidence that justified it. The power cost of the smaller test set is **reported, not avoided** —
that is what the per-feature `resolvable_margin` is for.

**Both populations, verdict read from FULL.** The full population is gated only by
`vote_count_min = 1`, which is frozen; the conditional population adds `consensus_gate = 0.5`, which
is still OPEN (spec register item 2). Resting a 37-feature headline on an unfrozen knob would undo
the project's discipline at its most exposed point, and it is also D14's own principle —
majority-gating imposes the tree's logic before testing whether the logic holds. The conditional
ladder still runs, and what is reported is **which features CHANGE verdict between the two**: a
feature that is R4 in full and clears in conditional says the signal lives in the majority-route
galaxies, which is itself a D14 finding.

The caveat that travels with that choice: a full-population verdict answers *"can the encoder read
this among galaxies where the question was ever asked"*, not *"among galaxies that genuinely have
the parent property"*. For deep features the full population includes minority-route galaxies and
may dilute the signal.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/p2_ladder.py --plan
    uv run python artifacts/p2_ladder.py --features 3      # a dry run, to confirm the cost model
    uv run python artifacts/p2_ladder.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j4_spread_controls import OUT, _release, prepare  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import entanglement as ent  # noqa: E402
from galaxy_jepa.probing import ladder as ladder_mod  # noqa: E402
from galaxy_jepa.probing import nulls as nz  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402

MAX_TRAIN = 40_000
RECORD = OUT / "p2_ladder.json"

# The O2 range travels with M's headline wherever it appears. Training-draw variance with the
# splits held fixed — NOT an interval on the headline AUC, and n=2 is a range, not a variance.
HEADLINE = "M 4-epoch consensus AUC 0.9646; range across two training draws [0.9609, 0.9646]"


def _rung_row(f: str, v) -> dict:
    return {
        "feature": f,
        "rung": v.rung,
        "mechanism": v.mechanism,
        "auc": v.metrics.get("auc"),
        "n_test": v.n_test,
        "positives_test": v.positives_test,
        "real_se": v.real_se,
        "resolvable_margin": v.resolvable_margin,
        "underpowered": v.underpowered,
        "matched_auc": None if v.matched is None else v.matched.matched_auc,
        "matched_n_test": None if v.matched is None else v.matched.n_matched_test,
        "matched_share": None if v.matched is None else v.matched.share_test,
        "matched_degenerate": None if v.matched is None else v.matched.degenerate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/m/encoder.pt")
    ap.add_argument("--features", type=int, default=0, help="cap for a dry run; 0 = all 37")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    setup = prepare(args.checkpoint, MAX_TRAIN, label="P2", sources=3)
    cfg, pc, device = setup.cfg, setup.pc, setup.device
    labels, train_ids, test_ids, ds = setup.labels, setup.train_ids, setup.test_ids, setup.ds

    if pc.existence_method != nz.EXISTENCE_UNTRAINED_Z:
        raise SystemExit(f"P2: probe.yaml says existence_method={pc.existence_method!r}; D23 "
                         f"expects 'untrained_z'")
    print(f"P2 features   : {len(labels.features)} (scheme {pc.scheme_name!r})", file=sys.stderr)
    print(f"P2 existence  : {pc.existence_method}, K={pc.n_untrained_seeds}, "
          f"floor {pc.effect_floor} (frozen)", file=sys.stderr)
    print(f"P2 headline   : {HEADLINE}", file=sys.stderr)
    print("P2 verdict pop: full (vote_count_min frozen); conditional reported alongside",
          file=sys.stderr)

    # The bank must exist and be big enough BEFORE any compute is spent.
    bank = ladder_mod._load_untrained_bank(pc, labels.features)
    if bank is None:
        raise SystemExit("P2: no untrained bank")
    k = min(len(v) for v in bank.values())
    nz.assert_untrained_bank_resolution(k)
    print(f"P2 bank       : K={k} seeds x {len(bank)} features", file=sys.stderr)
    if args.plan:
        print("\nP2 --plan: nothing run", file=sys.stderr)
        return

    if args.features:
        labels = dataclasses.replace(labels, features=list(labels.features)[: args.features])
        print(f"P2 DRY RUN    : {len(labels.features)} features", file=sys.stderr)

    frozen = load_frozen_encoder(setup.ckpt)
    t0 = time.perf_counter()
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

    out: dict[str, object] = {"headline": HEADLINE, "checkpoint": str(setup.ckpt),
                              "n_train": len(train_ids), "n_test": len(test_ids),
                              "existence_method": pc.existence_method, "K": k,
                              "effect_floor": pc.effect_floor, "smoke": True}

    results = {}
    for population in ("full", "conditional"):
        lab = labels if population == "full" else labels.with_population("conditional")
        t1 = time.perf_counter()
        res = ladder_mod.run_ladder(
            controls, lab, train_ids, test_ids, config=pc, sky_label_col="snr_r"
        )
        results[population] = res
        rows = [_rung_row(f, v) for f, v in res.verdicts.items()]
        out[population] = rows
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["rung"]] = counts.get(r["rung"], 0) + 1
        under = sum(1 for r in rows if r["underpowered"])
        print(f"\nP2 {population:<12s} {time.perf_counter() - t1:6.0f}s  "
              f"{dict(sorted(counts.items()))}  underpowered {under}/{len(rows)}",
              file=sys.stderr)

    # What CHANGED between the populations — the D14 finding, not two columns to eyeball.
    full_v, cond_v = results["full"].verdicts, results["conditional"].verdicts
    changed = [
        {"feature": f, "full": full_v[f].rung, "conditional": cond_v[f].rung,
         "auc_full": full_v[f].metrics.get("auc"), "auc_cond": cond_v[f].metrics.get("auc"),
         "positives_full": full_v[f].positives_test, "positives_cond": cond_v[f].positives_test}
        for f in full_v
        if f in cond_v and full_v[f].rung != cond_v[f].rung
    ]
    out["population_changes"] = changed
    print(f"\nP2 population : {len(changed)} of {len(full_v)} features change rung", file=sys.stderr)
    for c in changed:
        print(f"    {c['feature']:<48s} {c['full']} -> {c['conditional']}", file=sys.stderr)

    geo = results["full"].entanglement
    if geo is not None:
        out["entanglement"] = {
            "names": list(geo.names),
            "gram_effective_rank": geo.gram_effective_rank,
            "embedding_effective_rank": geo.embedding_effective_rank,
            "span_ratio": geo.span_ratio,
            "mp_significant": bool(geo.mp.significant),
            "top_loadings": ent.component_loadings(geo.names, geo.gram_eigenvectors)
            if geo.gram_eigenvectors.size
            else [],
            "cosine": [[float(v) for v in row] for row in geo.cosine],
        }
        out["pair_verdicts"] = [dataclasses.asdict(pv) for pv in results["full"].pair_verdicts]
        print(f"P2 entangle   : erank {geo.gram_effective_rank:.2f} of "
              f"{geo.embedding_effective_rank:.2f} (span {geo.span_ratio:.3f}), "
              f"MP significant {geo.mp.significant}", file=sys.stderr)

    RECORD.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {RECORD}", file=sys.stderr)
    print("P2 reports the catalogue. Power travels with every rung; an underpowered R4 means "
          "'cannot resolve at this N', never 'absent'.", file=sys.stderr)


if __name__ == "__main__":
    main()
