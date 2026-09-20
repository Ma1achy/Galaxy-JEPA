"""Brief O1 — matched evaluation: does the morphology survive holding the observing conditions fixed?

The objection this exists to answer, stated plainly: *"you are not detecting spiral arms, you are
detecting that nearby bright well-resolved galaxies look different."* On M's encoder the nuisance
panel reads magnitude 0.9033, size 0.9061, SNR 0.8687, redshift 0.8371, PSF 0.8186 against
featured-ness at 0.8845 and every other morphology feature below 0.66 — and PSF moved 0.5813 ->
0.8186 between J's encoder and M's, so the recipe fix improved the confounds FASTER than it improved
morphology. Until this is measured, no rung verdict can be published: a ladder built on unmatched
AUCs would have to be withdrawn.

**What the matching primitive actually is.** `matching.stratified_match` is quantile-binned
stratification with class-balanced downsampling — NOT propensity scoring, NOT caliper
nearest-neighbour. It bins the nuisance into `n_strata` quantile strata (5) and keeps, within each,
the per-stratum minority count of each class. So the "tolerance" is the stratum width, which is a
property of the data rather than a parameter, and it is reported here as measured quantile edges.

**Three things the production path does not give, which this driver supplies.**

* `MatchedVerdict` carries only `matched_auc` and `survived`; `matched_auc` computes the index
  arrays and throws their sizes away. **Survivor counts are the difference between "the signal was
  confound" and "matching ate the sample"**, so `stratified_match` is called here directly and
  `n_matched` is recorded. Degenerate matching silently returns 0.5 (`matching.py:89-93`), and
  without the counts that 0.5 is indistinguishable from a real collapse.
* Production matches on the single worst competing nuisance. The objection is a JOINT one, so a
  2-D magnitude x size match is run as well — one nuisance at a time does not answer "nearby AND
  bright AND well-resolved".
* **`ladder._nuisance_clearance` does not apply `labels.nuisance_valid`** although
  `controls.build_feature_controls` does, so production strata for `size` are built over the 2,143
  `petrorad_suspect` rows the flag exists to exclude — rows that are systematically bright, nearby
  and featured. This driver applies the filter. The production defect is logged, not fixed here:
  it is a change to the verdict path and belongs in its own brief.

**The read is pre-registered** (see `VERDICT` below) and was fixed before any number existed. Note
what it does NOT use: `survive_threshold=config.effect_floor`. With the floor frozen at 0.7267,
four of these six features sit below it *unmatched* and would fail a floor-based gate by arithmetic
whatever matching did. A vacuous bar is not a bar.

No rungs, no existence p-values, no verdicts beyond the matched read itself.

    uv run python artifacts/o1_matched.py --checkpoint runs/m/encoder.pt --tag o1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j4_spread_controls import OUT, SPREAD, prepare  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import matching as match  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix, feature_embeddings, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_auc_ci  # noqa: E402

#: Singly, then the joint one the objection actually names.
NUISANCES = ("magnitude", "size", "snr", "redshift", "psf")
JOINT = ("magnitude", "size")

N_STRATA = 5
#: PRE-REGISTERED, fixed before any number existed. `A` unmatched AUC, `C` its untrained-encoder
#: null; `M` matched AUC, `C_m` the null RE-MEASURED on the same matched rows so the comparison is
#: like-for-like. A declared choice, not a derived one.
RETAIN_FRACTION = 0.5
MIN_MATCHED_TEST = 500
MIN_MATCHED_SHARE = 0.10

VERDICT = """SURVIVES  : (M - C_m) >= 0.5 * (A - C)  AND  M's CI lower bound > C_m
COLLAPSES : (M - C_m) <= 0        OR  M's CI contains C_m
PARTIAL   : between the two — real, but substantially confounded
UNRESOLVED: < 500 matched test galaxies, or < 10% of the unmatched test set surviving
            (a statement about the SAMPLE, not the signal — never folded into COLLAPSES)"""


def _release(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.empty_cache()


def _quantile_edges(v: np.ndarray) -> list[float]:
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return []
    return [round(float(q), 6) for q in np.quantile(finite, np.linspace(0, 1, N_STRATA + 1))]


def _joint_match(v1: np.ndarray, v2: np.ndarray, y: np.ndarray, *, seed: int) -> np.ndarray:
    """`stratified_match` in two dimensions: 5x5 quantile cells, classes balanced within each.

    The production primitive takes one vector. "Nearby bright well-resolved" is a conjunction, and
    matching each of its parts in turn leaves the conjunction unmatched — so the cells are the
    cross-product, and the sample cost of that is a finding rather than a nuisance.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(v1) & np.isfinite(v2)
    idx_all = np.flatnonzero(ok)
    if idx_all.size == 0:
        return np.asarray([], dtype=np.int64)
    e1 = np.quantile(v1[idx_all], np.linspace(0, 1, N_STRATA + 1))
    e2 = np.quantile(v2[idx_all], np.linspace(0, 1, N_STRATA + 1))
    e1[-1] = e2[-1] = np.inf
    kept: list[int] = []
    for a in range(N_STRATA):
        in_a = idx_all[(v1[idx_all] >= e1[a]) & (v1[idx_all] < e1[a + 1])]
        for b in range(N_STRATA):
            cell = in_a[(v2[in_a] >= e2[b]) & (v2[in_a] < e2[b + 1])]
            pos, neg = cell[y[cell] == 1], cell[y[cell] == 0]
            take = min(len(pos), len(neg))
            if take == 0:
                continue
            kept.extend(rng.choice(pos, take, replace=False).tolist())
            kept.extend(rng.choice(neg, take, replace=False).tolist())
    return np.asarray(sorted(kept), dtype=np.int64)


def _take(e: Embeddings, idx: np.ndarray) -> Embeddings:
    return Embeddings(e.x[idx], e.y[idx], e.fraction[idx])


def _auc_ci(tr: Embeddings, te: Embeddings, *, c: float, n_boot: int, seed: int):
    """AUC with its interval, or `None` when the probe has no two-class problem to solve."""
    try:
        return probe_auc_ci(tr, te, c=c, n_boot=n_boot, seed=seed)
    except ValueError:
        return None


def _verdict(a, c_un, m, lo, c_m, n_te, n_te_un) -> tuple[str, float | None]:
    if m is None or n_te < MIN_MATCHED_TEST or n_te < MIN_MATCHED_SHARE * n_te_un:
        return "UNRESOLVED", None
    unmatched_margin, matched_margin = a - c_un, m - c_m
    retained = matched_margin / unmatched_margin if unmatched_margin > 0 else 0.0
    if matched_margin <= 0 or lo <= c_m:
        return "COLLAPSES", retained
    if matched_margin >= RETAIN_FRACTION * unmatched_margin:
        return "SURVIVES", retained
    return "PARTIAL", retained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: <out_dir>/encoder.pt")
    ap.add_argument("--max-train", type=int, default=40_000)
    ap.add_argument("--tag", default="o1")
    args = ap.parse_args()

    s = prepare(args.checkpoint, args.max_train, label="O1", sources=2)
    cfg, pc, labels = s.cfg, s.pc, s.labels
    print("O1 pre-registered read (fixed before any number):\n" + VERDICT + "\n", file=sys.stderr)

    frozen = load_frozen_encoder(s.ckpt)
    t0 = time.perf_counter()
    real = extract_matrix(frozen, s.ds, device=s.device)
    print(f"  real      {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    _release(s.device)
    untrained = ctl.untrained_encoder_matrix(frozen.config, s.ds, device=s.device, seed=cfg.seed)
    print(f"  untrained {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    _release(s.device)
    # The matched null re-uses the SAME row indices as the real probe, so the two matrices must be
    # row-for-row co-indexed. Asserted, not assumed.
    assert real.object_ids == untrained.object_ids, "control matrix is not co-indexed with the real"

    records = []
    for feature, role, _draws in SPREAD:
        tr = feature_embeddings(real, labels, feature, s.train_ids)
        te = feature_embeddings(real, labels, feature, s.test_ids)
        ids_tr = feature_ids(real, labels, feature, s.train_ids)
        ids_te = feature_ids(real, labels, feature, s.test_ids)
        assert len(ids_tr) == len(tr.y) and len(ids_te) == len(te.y), feature
        u_tr = feature_embeddings(untrained, labels, feature, s.train_ids)
        u_te = feature_embeddings(untrained, labels, feature, s.test_ids)

        base = _auc_ci(tr, te, c=pc.c, n_boot=pc.n_boot, seed=cfg.seed)
        a = base[0] if base else 0.5
        c_un = ctl._safe_auc(u_tr, u_te, c=pc.c)
        rec = {"feature": feature, "role": role, "smoke": True, "checkpoint": str(s.ckpt),
               "unmatched_auc": a, "unmatched_lo": base[1] if base else None,
               "unmatched_hi": base[2] if base else None, "unmatched_null": c_un,
               "n_test_unmatched": int(len(te.y)), "matched": {}}
        print(f"\n{feature}  unmatched {a:.4f} vs null {c_un:.4f}  (margin {a - c_un:+.4f})",
              file=sys.stderr)

        for name in (*NUISANCES, "magnitude_x_size"):
            joint = name == "magnitude_x_size"
            cols = JOINT if joint else (name,)
            # the flag filter production forgets — applied to ids AND rows together
            keep_tr = np.ones(len(ids_tr), dtype=bool)
            keep_te = np.ones(len(ids_te), dtype=bool)
            for col in cols:
                keep_tr &= labels.nuisance_valid(col, ids_tr)
                keep_te &= labels.nuisance_valid(col, ids_te)
            f_ids_tr = [o for o, k in zip(ids_tr, keep_tr, strict=True) if k]
            f_ids_te = [o for o, k in zip(ids_te, keep_te, strict=True) if k]
            i_tr, i_te = np.flatnonzero(keep_tr), np.flatnonzero(keep_te)
            v_tr = [labels.nuisance_value(col, f_ids_tr) for col in cols]
            v_te = [labels.nuisance_value(col, f_ids_te) for col in cols]
            ftr, fte = _take(tr, i_tr), _take(te, i_te)

            if joint:
                m_tr = _joint_match(v_tr[0], v_tr[1], ftr.y, seed=cfg.seed)
                m_te = _joint_match(v_te[0], v_te[1], fte.y, seed=cfg.seed + 1)
            else:  # seed / seed+1 exactly as `matching.matched_auc` splits them
                m_tr = match.stratified_match(v_tr[0], ftr.y, n_strata=N_STRATA, seed=cfg.seed)
                m_te = match.stratified_match(v_te[0], fte.y, n_strata=N_STRATA, seed=cfg.seed + 1)

            mm = None
            if m_tr.size and m_te.size:
                mm = _auc_ci(_take(ftr, m_tr), _take(fte, m_te),
                             c=pc.c, n_boot=pc.n_boot, seed=cfg.seed)
            c_m = (ctl._safe_auc(_take(_take(u_tr, i_tr), m_tr), _take(_take(u_te, i_te), m_te),
                                 c=pc.c) if (m_tr.size and m_te.size) else 0.5)
            m_auc, m_lo = (mm[0], mm[1]) if mm else (None, None)
            verdict, retained = _verdict(a, c_un, m_auc, m_lo, c_m, int(m_te.size), len(te.y))
            rec["matched"][name] = {
                "matched_auc": m_auc, "matched_lo": m_lo, "matched_hi": mm[2] if mm else None,
                "matched_null": c_m, "n_matched_train": int(m_tr.size),
                "n_matched_test": int(m_te.size),
                "share_test": round(float(m_te.size) / max(len(te.y), 1), 4),
                "retained": retained, "verdict": verdict,
                "strata_edges": ([_quantile_edges(v_te[0]), _quantile_edges(v_te[1])] if joint
                                 else _quantile_edges(v_te[0])),
                "nuisance_flag_filtered": int(len(ids_te) - len(f_ids_te)),
            }
            shown = f"{m_auc:.4f}" if m_auc is not None else "  n/a "
            ret = f"{retained:+.2f}" if retained is not None else "  -  "
            print(f"    {name:18s} matched {shown} vs null {c_m:.4f}  "
                  f"n_te {m_te.size:>6,} ({100 * m_te.size / max(len(te.y), 1):5.1f}%)  "
                  f"retained {ret}  {verdict}", file=sys.stderr)
        records.append(rec)
        (OUT / f"{args.tag}_matched.partial.json").write_text(json.dumps(records, indent=2))

    path = OUT / f"{args.tag}_matched.json"
    path.write_text(json.dumps({
        "checkpoint": str(s.ckpt), "device": s.device, "smoke": True,
        "n_train": len(s.train_ids), "n_test": len(s.test_ids), "n_strata": N_STRATA,
        "retain_fraction": RETAIN_FRACTION, "min_matched_test": MIN_MATCHED_TEST,
        "min_matched_share": MIN_MATCHED_SHARE, "read": VERDICT,
        "seconds": time.perf_counter() - t0, "features": records,
    }, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)
    print("O1 reports the matched read. No rungs assigned; the floor is not used as a survival bar.")


if __name__ == "__main__":
    sys.exit(main())
