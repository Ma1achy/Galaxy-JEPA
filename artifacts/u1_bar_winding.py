"""Brief U1 — bar × {winding, arm count}: the encoder's cosines against the same-corpus votes.

Pre-registered in artifacts/u_findings.md §U1 (hashed before this ran). Full population, M's
4-epoch checkpoint, P2's split; directions are the ladder's canonical probe (`probe_direction`,
fitted on train).

  * encoder cosine   — bar's direction · partner's direction (unit vectors), as Brief P's matrix
  * human, test      — pairwise-complete vote-fraction correlation over the galaxies eligible for
                       BOTH answers (`entanglement.human_vote_correlation`, P's same-corpus matrix)
  * human, tree-zero — the same over ALL galaxies with an unreached question's stored 0.0 kept:
                       the construction in which the decision tree itself ties bar to every spiral
                       answer. Diagnostic only
  * null             — isotropic random directions (cosine sd 1/√384), and, as a check on the
                       embedding's anisotropy, bar's direction against partner directions fitted
                       on SHUFFLED partner labels; the band is the wider of the two
  * untrained        — the same cosines on R's three untrained draws, descriptive

    uv run python artifacts/u1_bar_winding.py  -> artifacts/out/u1_bar_winding.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import entanglement as ent  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, probe_direction  # noqa: E402

BAR = "t03_bar_a06_bar"
PARTNERS = (
    "t10_arms_winding_a28_tight",
    "t10_arms_winding_a29_medium",
    "t10_arms_winding_a30_loose",
    "t11_arms_number_a31_1",
    "t11_arms_number_a32_2",
    "t11_arms_number_a33_3",
    "t11_arms_number_a34_4",
    "t11_arms_number_a36_more_than_4",
)
N_SHUFFLE = 30
Z99 = 2.5758
HUMAN_STRONG = 0.30
OUT = R.OUT / "u1_bar_winding.json"


def direction(matrix, lab, f, ids, c, y=None) -> np.ndarray:
    tr = feature_embeddings(matrix, lab, f, ids)
    if y is not None:
        tr = Embeddings(tr.x, y, tr.fraction)
    return probe_direction(tr, name=f, c=c).w_unit


def classify(h: float, cos: float, band: float) -> str:
    strong, outside = abs(h) >= HUMAN_STRONG, abs(cos) > band
    if strong and not outside:
        return "HUMAN-ONLY"
    if strong and outside:
        return "SHARED" if np.sign(h) == np.sign(cos) else "CONTRARY"
    return "ENCODER-ONLY" if outside else "NEITHER"


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="U1", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    real, untrained = R.load_matrices(setup, frozen)
    lab, c, ids = setup.labels, setup.pc.c, setup.train_ids
    feats = (BAR, *PARTNERS)
    w = {f: direction(real, lab, f, ids, c) for f in feats}
    wu = [{f: direction(u, lab, f, ids, c) for f in feats} for u in untrained]

    human, overlap = ent.human_vote_correlation(lab, list(feats), setup.union)
    allv = {f: np.asarray(lab.vote_fraction(f, setup.union), dtype=float) for f in feats}

    iso = 1.0 / np.sqrt(real.x.shape[1])
    rng = np.random.default_rng(setup.pc.seed)
    rows, shuffled_sd = [], []
    for j, p in enumerate(PARTNERS, start=1):
        y = feature_embeddings(real, lab, p, ids).y
        null = [float(w[BAR] @ direction(real, lab, p, ids, c, y=rng.permutation(y)))
                for _ in range(N_SHUFFLE)]
        shuffled_sd.append(float(np.std(null, ddof=1)))
        tz = np.isfinite(allv[BAR]) & np.isfinite(allv[p])
        rows.append({
            "partner": p,
            "cosine": float(w[BAR] @ w[p]),
            "human": float(human[0, j]) if np.isfinite(human[0, j]) else None,
            "n_overlap": int(overlap[0, j]),
            "human_tree_zero": float(np.corrcoef(allv[BAR][tz], allv[p][tz])[0, 1]),
            "shuffled_null_sd": shuffled_sd[-1],
            "shuffled_null_max_abs": float(np.max(np.abs(null))),
            "untrained_cosines": [float(d[BAR] @ d[p]) for d in wu],
        })
    sd = max(iso, float(np.median(shuffled_sd)))
    band = Z99 * sd
    for r in rows:
        r["class"] = classify(r["human"] if r["human"] is not None else 0.0, r["cosine"], band)
    strong = [r for r in rows if r["human"] is not None and abs(r["human"]) >= HUMAN_STRONG]
    kinds = {r["class"] for r in strong}
    if not strong:
        verdict = "UNTESTABLE — no human-strong pair on the pairwise-complete matrix"
    elif kinds == {"HUMAN-ONLY"}:
        verdict = "SUPPORTED — labelling bleed"
    elif kinds == {"SHARED"}:
        verdict = "REJECTED — the encoder carries the human association"
    else:
        verdict = "MIXED — reported per pair"
    out = {"isotropic_sd": iso, "shuffled_sd_median": float(np.median(shuffled_sd)),
           "null_sd_used": sd, "band_99": band, "human_strong": HUMAN_STRONG,
           "check_p_bar_tight": w[BAR] @ w[PARTNERS[0]], "rows": rows, "verdict": verdict}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"U1 null sd: isotropic {iso:.4f}, shuffled-label median {np.median(shuffled_sd):.4f}"
          f" -> band ±{band:.3f} (99%)", file=sys.stderr)
    print(f"{'partner':<32s} {'cos':>7s} {'human':>7s} {'n':>7s} {'treeZ':>7s} "
          f"{'untrained':>22s}  class", file=sys.stderr)
    for r in rows:
        h = f"{r['human']:+.3f}" if r["human"] is not None else "   -  "
        u = " ".join(f"{v:+.2f}" for v in r["untrained_cosines"])
        print(f"{r['partner']:<32s} {r['cosine']:+.3f} {h:>7s} {r['n_overlap']:>7d} "
              f"{r['human_tree_zero']:+.3f} {u:>22s}  {r['class']}", file=sys.stderr)
    print(f"U1 verdict: {verdict}", file=sys.stderr)


if __name__ == "__main__":
    main()
