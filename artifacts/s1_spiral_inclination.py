"""Brief S1 — is the direction of spiral's bend inclination?

Pre-registered in artifacts/s_findings.md §S1 (hashed before this ran). The bend direction is the
second eigenvector of R2's cross-fitted path covariance, estimated on TRAIN galaxies; every number
below is measured on TEST galaxies. Partial correlation with expAB_r after removing 20 vote-fraction
bin means from both; share of the U-amplitude explained after removing inclination's within-f
slope; shuffled-expAB_r null (B = 10,000); full population and the well-voted half; M and the three
banked untrained seeds. Exploratory nuisances are computed for M and labelled as such.

    uv run python artifacts/s1_spiral_inclination.py   -> artifacts/out/s1_spiral_inclination.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import geometry as geo  # noqa: E402
from galaxy_jepa.probing.extract import feature_ids  # noqa: E402

FEATURE = "t04_spiral_a08_spiral"
EDGES = R.R2_EDGES  # R2's 10 bins, for the direction and the U-amplitude
N_PARTIAL_BINS = 20  # bin-mean removal for the partial correlation
B_PERM = 10_000
N_BOOT = 2_000
MID = (0.3, 0.7)  # bins centred 0.35-0.65; the rest are the ends
NUISANCES = ("size", "magnitude", "snr", "psf")


def bend_direction(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """PC2 of the cross-fitted path covariance (R2's construction), oriented toward the middle."""
    raw = geo._bin_of(f, EDGES)
    keep = geo._surviving(raw, len(EDGES) - 1, geo.MIN_OCCUPANCY)
    group = geo._group_index(raw, keep)
    rng = np.random.default_rng(0)
    s = np.zeros((x.shape[1], x.shape[1]))
    for _ in range(geo.N_SPLITS):
        a, b = geo._half_centroids(x, group, keep.size, rng)
        s += geo._cross_cov(a, b)
    evals, evecs = np.linalg.eigh(s / geo.N_SPLITS)
    d = evecs[:, np.argsort(evals)[::-1][1]]
    centres = 0.5 * (EDGES[keep] + EDGES[keep + 1])
    proj = np.array([x[group == g].mean(axis=0) @ d for g in range(keep.size)])
    mid = (centres > MID[0]) & (centres < MID[1])
    return d if proj[mid].mean() >= proj[~mid].mean() else -d


def demean(v: np.ndarray, bins: np.ndarray) -> np.ndarray:
    sums = np.bincount(bins, weights=v, minlength=N_PARTIAL_BINS)
    counts = np.bincount(bins, minlength=N_PARTIAL_BINS)
    return v - (sums / np.maximum(counts, 1))[bins]


def u_amplitude(s: np.ndarray, f: np.ndarray) -> float:
    """Mean of the middle bin means minus mean of the end bin means (R2's bins, >= 100 each)."""
    raw = geo._bin_of(f, EDGES)
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    means = {k: s[raw == k].mean() for k in range(len(centres)) if (raw == k).sum() >= 100}
    mid = [m for k, m in means.items() if MID[0] < centres[k] < MID[1]]
    end = [m for k, m in means.items() if not MID[0] < centres[k] < MID[1]]
    return float(np.mean(mid) - np.mean(end))


def partial(s: np.ndarray, q: np.ndarray, f: np.ndarray, *, perm: bool, seed: int = 0) -> dict:
    ok = np.isfinite(q) & np.isfinite(s)
    s, q, f = s[ok], q[ok], f[ok]
    bins = np.clip((f * N_PARTIAL_BINS).astype(int), 0, N_PARTIAL_BINS - 1)
    rs, rq = demean(s, bins), demean(q, bins)
    r = float(np.corrcoef(rs, rq)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(N_BOOT, s.size))
    boot = np.array([np.corrcoef(rs[i], rq[i])[0, 1] for i in idx])
    out = {"n": int(s.size), "partial_r": r, "ci": [float(np.quantile(boot, 0.025)),
                                                     float(np.quantile(boot, 0.975))],
           "raw_r": float(np.corrcoef(s, q)[0, 1])}
    if perm:
        rs_n = (rs - rs.mean()) / rs.std()
        null = np.empty(B_PERM)
        for b in range(B_PERM):
            rq_p = demean(q[rng.permutation(q.size)], bins)
            null[b] = rs_n @ ((rq_p - rq_p.mean()) / rq_p.std()) / rs.size
        out["p_perm"] = float((1 + np.sum(np.abs(null) >= abs(r))) / (B_PERM + 1))
        out["null_sd"] = float(null.std())
        beta = float(rs @ rq / (rq @ rq))
        a = u_amplitude(s, f)
        a_adj = u_amplitude(s - beta * (q - q.mean()), f)
        out.update({"beta": beta, "u_amplitude": a, "u_amplitude_adj": a_adj,
                    "share_explained": float(1 - a_adj / a)})
    return out


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="S1", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    lab = ctx.labels["full"]
    ab_all = R._axis_ratio(ctx)
    pos = ctx.real.index
    spec = lab.scheme.by_name[FEATURE]

    def reach(ids: list[int]) -> np.ndarray:
        return sum(lab._column(ids, c) for c in spec.reach_count_cols())

    ids_tr = feature_ids(ctx.real, lab, FEATURE, setup.train_ids)
    ids_te = feature_ids(ctx.real, lab, FEATURE, setup.test_ids)
    med = float(np.nanmedian(np.concatenate([reach(ids_tr), reach(ids_te)])))
    pops = {"full": (np.ones(len(ids_tr), bool), np.ones(len(ids_te), bool)),
            "well_voted": (reach(ids_tr) >= med, reach(ids_te) >= med)}
    f_tr, f_te = lab.vote_fraction(FEATURE, ids_tr), lab.vote_fraction(FEATURE, ids_te)
    ab_te = ab_all[[pos[o] for o in ids_te]]
    print(f"S1 {FEATURE}: train {len(ids_tr):,}  test {len(ids_te):,}  median reach {med:.0f}  "
          f"expAB_r finite on test {np.isfinite(ab_te).mean():.1%}", file=sys.stderr)

    rec: dict[str, object] = {"feature": FEATURE, "median_reach": med, "prereg_sha": None,
                              "matrices": {}, "exploratory_nuisances": {}}
    for m in [ctx.real, *ctx.untrained]:
        zm = ctx.z(m)
        x_tr, x_te = zm.x[zm.rows_for(ids_tr)], zm.x[zm.rows_for(ids_te)]
        rec["matrices"][m.encoder_name] = {}
        for pop, (k_tr, k_te) in pops.items():
            d = bend_direction(x_tr[k_tr], f_tr[k_tr])
            s = x_te[k_te] @ d
            res = partial(s, ab_te[k_te], f_te[k_te], perm=True)
            rec["matrices"][m.encoder_name][pop] = res
            print(f"  {m.encoder_name:<14s} {pop:<10s} partial r {res['partial_r']:+.4f} "
                  f"[{res['ci'][0]:+.4f},{res['ci'][1]:+.4f}] p {res['p_perm']:.5f}  "
                  f"share {res['share_explained']:+.3f}  (U {res['u_amplitude']:+.3f} -> "
                  f"{res['u_amplitude_adj']:+.3f})  raw r {res['raw_r']:+.4f}  n {res['n']:,}",
                  file=sys.stderr)
            if m is ctx.real:
                for nz in NUISANCES:
                    valid = np.asarray(lab.nuisance_valid(nz, ids_te), bool)
                    q = np.where(valid, lab.nuisance_value(nz, ids_te), np.nan)[k_te]
                    rec["exploratory_nuisances"].setdefault(pop, {})[nz] = partial(
                        s, q, f_te[k_te], perm=False)
    (R.OUT / "s1_spiral_inclination.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
